# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import concurrent.futures
import json
import logging
import os
import random
import threading
from contextlib import ExitStack
from enum import Enum
from math import ceil, floor
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4
import base64
from io import BytesIO
import re
import aiohttp
from PIL import Image
import ray
import ray.actor
from qwen_vl_utils import fetch_image
from json.decoder import JSONDecoder
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")

_IMAGE_SPECIAL_TOKEN_IDS = {151652, 151653, 151655, 151656}  # vision_start, vision_end, image_pad, video_pad

_IMAGE_PAD_TOKEN_ID = 151655  # <|image_pad|>
def _compress_image_pad_tokens(token_ids: list[int]) -> list[int]:
    """Compress consecutive <|image_pad|> tokens into a single one.

    When passing prompt_ids to vLLM via TokensPrompt, vLLM expects each image
    position to have a single <|image_pad|> marker token that it will expand
    internally based on the actual image dimensions. If the processor has
    already expanded these tokens (e.g., 95 consecutive <|image_pad|>), vLLM's
    _apply_prompt_updates will fail with a token/placeholder count mismatch.
    """
    result = []
    in_image_pad_run = False
    for tid in token_ids:
        if tid == _IMAGE_PAD_TOKEN_ID:
            if not in_image_pad_run:
                result.append(tid)
                in_image_pad_run = True
        else:
            in_image_pad_run = False
            result.append(tid)
    return result

# Local code execution directory
LOCAL_CODE_DIR = "/tmp/code_dirs"
SANDBOX_EXECUTION_TIMEOUT = 180

def execute_code_locally(
    instance_id: str,
    code_str: str,
    timeout: int = SANDBOX_EXECUTION_TIMEOUT,
    code_dir: str = LOCAL_CODE_DIR
) -> tuple[bool, dict]:
    """Execute Python code locally in a subprocess with resource limits.

    This function saves the code to a local file and executes it in a subprocess,
    then cleans up the file after execution. It's faster than the sandbox approach
    as it runs locally without network overhead.

    Args:
        instance_id: Unique identifier for the execution instance.
        code_str: The Python code string to execute.
        timeout: Maximum execution time in seconds.
        code_dir: Directory to save temporary code files.

    Returns:
        Tuple of (success, result_dict) where result_dict contains:
        - stdout: The standard output from code execution
        - stderr: The standard error from code execution
        - error: Error message if execution failed
    """
    import subprocess
    import sys

    # Ensure code directory exists
    os.makedirs(code_dir, exist_ok=True)

    # Create unique script filename
    script_filename = f"execution_{instance_id}_{uuid4().hex[:8]}.py"
    script_path = os.path.join(code_dir, script_filename)

    # Write code to file
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(code_str)

    process = None
    try:
        # Execute with timeout using sys.executable to ensure correct Python environment
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=code_dir,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            success = process.returncode == 0
            result_dict = {
                "stdout": stdout.replace("[OUTPUT_TEXT]", "").strip(),
                "stderr": stderr,
            }

            if not success:
                result_dict["error"] = f"Execution failed with return code {process.returncode}"

            return success, result_dict

        except subprocess.TimeoutExpired:
            # Force-kill the process and its children on timeout
            process.kill()
            process.communicate()  # drain resources to avoid zombie processes
            return False, {"stdout": "", "stderr": "", "error": "Code execution timeout"}

    except Exception as e:
        return False, {"stdout": "", "stderr": "", "error": str(e)}
    finally:
        # Clean up: delete the script file after execution
        try:
            if os.path.exists(script_path):
                os.remove(script_path)
        except Exception as e:
            logger.warning(f"Failed to remove script file {script_path}: {e}")

_local_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_executor_lock = threading.Lock()

def _get_local_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the shared thread pool executor."""
    global _local_executor
    if _local_executor is None:
        with _executor_lock:
            if _local_executor is None:
                _local_executor = concurrent.futures.ThreadPoolExecutor(max_workers=64)
    return _local_executor

async def execute_code_locally_async(
    instance_id: str,
    code_str: str,
    timeout: int = SANDBOX_EXECUTION_TIMEOUT,
    code_dir: str = LOCAL_CODE_DIR
) -> tuple[bool, dict]:
    """Async wrapper for execute_code_locally.

    This allows the local execution to be used in async contexts without blocking
    the event loop, by running the synchronous subprocess in an executor.
    """
    loop = asyncio.get_running_loop()
    executor = _get_local_executor()
    try:
        # Add timeout control via asyncio.wait_for
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                lambda: execute_code_locally(instance_id, code_str, timeout, code_dir)
            ),
            timeout=timeout + 10  # extra 10s buffer
        )
        return result
    except asyncio.TimeoutError:
        return False, {"stdout": "", "stderr": "", "error": "Code execution timeout"}

# Image-related operations
target_path = "/tmp/temp_processed_images"
def replace_temp_images_path(code_str: str) -> str:
    code_str = re.sub(r'(f["\'])([^"\']*?)/?temp_processed_images', rf'\1{target_path}', code_str)
    code_str = re.sub(r'(?<!f)(["\'])([^"\']*?)/?temp_processed_images', rf'\1{target_path}', code_str)
    return code_str

def resize_to_multiple_of_28(img, max_side=1260):
    """Resize keeping aspect ratio so the longest side <= max_side, then snap
    both dimensions down to a multiple of 28 (>= 28)."""
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    w, h = w * scale, h * scale
    new_w = max(int(round(w / 28)) * 28, 28)
    new_h = max(int(round(h / 28)) * 28, 28)
    return img.resize((new_w, new_h), Image.LANCZOS)

def extract_image_paths(text: str, image_extensions: set = None):
    if os.path.isfile(text.strip()): return [text.strip()], [text.strip()]
    """Extract all image paths from a string."""
    if image_extensions is None: image_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff', 'svg'}
    ext_pattern = '|'.join(image_extensions)
    paths = []
    # 1. Extract image paths in quotes (including list formats); match paths in single or double quotes
    quoted_pattern = rf"['\"]([^'\"]+\.(?:{ext_pattern}))['\"]"
    for match in re.finditer(quoted_pattern, text, re.IGNORECASE): paths.append(match.group(1))
    # 2. Extract absolute paths: /xxx/yyy.jpg or /xxx/yyy_zzz.jpg (filenames may contain . and _)
    # Modified: allow more characters in the filename, correctly handling cases like test.jpg_das.jpg
    # Key change: use a negative lookahead so the extension is not followed by alphanumerics or _-
    abs_path_pattern = rf'(?<!\.)\/[\w\-]+(?:/[\w\-\.]+)+\.(?:' + ext_pattern + r')(?![\w\-])'
    for match in re.finditer(abs_path_pattern, text, re.IGNORECASE):
        path = match.group()
        if path not in paths: paths.append(path)
    # 3. Extract relative paths: ./xxx/yyy.jpg or ./xxx/yyy_zzz.jpg
    rel_path_pattern = rf'\./[\w\-/\.]+\.(?:' + ext_pattern + r')(?![\w\-])'
    for match in re.finditer(rel_path_pattern, text, re.IGNORECASE):
        path = match.group()
        if path not in paths: paths.append(path)
    # 3.5 Extract relative paths with a directory but no ./ prefix: xxx/yyy.jpg or xxx/yyy_zzz.jpg
    # Path format like: test/ims.3.s.0g.png
    rel_path_no_dot_pattern = rf'(?<![\/\.\w])[\w\-]+(?:/[\w\-\.]+)+\.(?:' + ext_pattern + r')(?![\w\-])'
    for match in re.finditer(rel_path_no_dot_pattern, text, re.IGNORECASE):
        path = match.group()
        if path not in paths: paths.append(path)
    # 4. Extract bare filenames: xxx.jpg (only when no other paths found)
    if not paths:
        filename_pattern = rf'[\w\-\.]+\.(?:' + ext_pattern + r')(?![\w\-])'
        for match in re.finditer(filename_pattern, text, re.IGNORECASE): paths.append(match.group())
    # Deduplicate while preserving order
    seen = set()
    unique_paths, useful_paths = [], []
    for p in paths:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            unique_paths.append(p)
            if os.path.isfile(p): useful_paths.append(p)
    return unique_paths, useful_paths

# Code prefix (prepended to every generated snippet before sandbox execution)
prefix_code="""import traceback, os, signal, contextlib, io, sys, math, random, re, collections
from PIL import Image
import numpy as np
from scipy import integrate, stats
import sympy
import cv2
from sklearn import preprocessing, decomposition
import pandas as pd
import pytesseract
"""

subagent_system_prompt = """You are evaluating whether a tool-called image is helpful for answering a visual question.

**Context:**
- The image is a processed version of an initial image, captured by a visual tool using operations such as: cropping, resizing, rotation, or contrast adjustment.
- Your job is to judge if this image successfully presents the information needed to answer the question.

**Guidelines for "is_helpful":**

Set to 'true' if:
- The object/attribute asked in the question is clearly visible in the image.
- You can reasonably identify the target object and its relevant attributes (color, position, shape, etc.).
- The image contains the key visual information needed.
- The tool's processing (crop, resize, rotate, contrast adjustment, etc.) helps reveal or highlight the relevant information.
- The question can be answered based on what is visible in the image.

Set to 'false' if:
- The target object is NOT in the image at all (e.g., tool cropped the wrong area).
- The image completely missed the relevant object/region.
- What's shown in the image is unrelated to what the question asks about.
- The key information needed is outside the region or obscured by the processing.

**CRITICAL: Your response MUST be ONLY a valid JSON object in this EXACT format:**
{"is_helpful": true or false, "reasons": "Your brief reason here"}

**Examples:**
{"is_helpful": true, "reasons": "The question asks about the tray's position. The image shows a tray on the right side of the frame. Though partially visible, its position is clear enough to answer."}

{"is_helpful": true, "reasons": "The question asks about shirt color. The image shows a person's torso with a white shirt clearly visible. The color can be determined from this image."}

{"is_helpful": false, "reasons": "The question asks about a red blanket, but the image shows only a kitchen counter with no blanket visible. The tool cropped the wrong area."}

{"is_helpful": false, "reasons": "The question asks about elephant's tail, but the image only shows the elephant's legs without the tail. The key information is missing from this image."}
"""

subagent_user_prompt = """[Question]: {question}

The following image is a processed version of an initial image (possibly cropped, resized, rotated, or contrast-adjusted). Does this image show the object/attribute the question is asking about? Reply with ONLY one line of JSON.
"""


def clean_text(text):
    text = re.sub(r"<\|im_start\|>|\<\|im_end\|>", "", text).strip()
    text = text.replace('\xa0', ' ')  # replace non-breaking spaces
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # strip control characters
    text = re.sub(r'\s+', ' ', text)  # collapse extra whitespace
    text = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)  # strip markdown code fences
    text = re.sub(r'```\s*', '', text)
    return text.strip()

# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class VisualExecutionWorker:
    """Worker for executing visual processing operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    logger.warning(f"Error when executing visual processing: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_visual_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize visual execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(VisualExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class VLMSubAgentTool(BaseTool):
    """A function critic that evaluates whether processed images from code execution are helpful.

    This tool acts as a "function critic" that inspects images generated by code execution
    (e.g., cropped regions, segmented objects) and determines whether they are helpful
    for answering the user's original question. It provides critical feedback to the
    main agent about the quality and relevance of the processed image.

    The tool follows a multi-agent approach where:
    1. Main agent calls a tool with a specific intent
    2. Tool processes the image (e.g., crops a region)
    3. This subagent (function critic) evaluates the result
    4. If helpful, the processed image is used; if not, critical feedback is provided

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the function critic evaluation
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    MIN_DIMENSION = 28

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 20)
        self.rate_limit = config.get("rate_limit", 50)
        self.timeout = config.get("timeout", 30)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_visual_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        logger.info(f"Initialized VLMSubAgentTool (function critic) with config: {config}")

    def _validate_bbox(self, left: float, top: float, right: float, bottom: float) -> bool:
        """Validate the bounding box dimensions and aspect ratio."""
        try:
            if not (left < right and top < bottom):
                logger.warning(f"Invalid bbox shape: left={left}, top={top}, right={right}, bottom={bottom}")
                return False

            height = bottom - top
            width = right - left

            # Prevent division by zero for zero-sized boxes
            if min(height, width) == 0:
                logger.warning(f"Bbox has zero width or height: left={left}, top={top}, right={right}, bottom={bottom}")
                return False

            if max(height, width) / min(height, width) > 100:
                logger.warning(f"Bbox aspect ratio > 100: left={left}, top={top}, right={right}, bottom={bottom}")
                return False

            return True
        except Exception as e:
            logger.warning(f"Bbox validation error: {e}")
            return False

    def _maybe_resize_bbox(self, bbox_2d: list[float], image_width: int, image_height: int) -> Optional[list[float]]:
        """
        Clamp, validate, and potentially resize a bounding box.

        This function ensures the final bounding box is within image bounds and meets the minimum
        dimension requirements. If the initial box is too small, it attempts to expand it
        from its center. It performs a final check to guarantee the output dimensions are valid.

        Returns:
            A valid bounding box as a list of coordinates, or None if validation fails.
        """
        left, top, right, bottom = bbox_2d

        # 1. Clamp the initial bounding box to the image dimensions.
        left = max(0.0, float(left))
        top = max(0.0, float(top))
        right = min(float(image_width), float(right))
        bottom = min(float(image_height), float(bottom))
        
        # 2.5 interpolation
        interpolation_factor: float = 0.05
        whole_bbox_2d = [0, 0, image_width, image_height]
        new_bbox_2d = [
            int(x * (1 - interpolation_factor) + y * interpolation_factor)
            for x, y in zip([left, top, right, bottom], whole_bbox_2d)
        ]
        current_bbox = new_bbox_2d
        left, top, right, bottom = current_bbox
        
        # 2. If clamped bbox is invalid, return immediately.
        if not self._validate_bbox(left, top, right, bottom):
            return None

        # current_bbox = [left, top, right, bottom]
        height = bottom - top
        width = right - left

        # 3. If the box is too small, attempt to resize it.
        if height < self.MIN_DIMENSION or width < self.MIN_DIMENSION:
            logger.info(f"Bbox {width}x{height} is smaller than {self.MIN_DIMENSION}, attempting resize.")
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0

            min_dim = min(height, width)
            if min_dim == 0:  # Safeguard for zero-area boxes
                return None

            # 1. Calculate the target dimensions to make the smallest side MIN_DIMENSION.
            ratio = self.MIN_DIMENSION / min_dim
            target_width = width * ratio
            target_height = height * ratio

            # 2. If the target size is larger than the image, scale it down to fit.
            #    This preserves the aspect ratio while respecting image boundaries.
            if target_width > image_width:
                scale_down = image_width / target_width
                target_width = image_width
                target_height *= scale_down

            if target_height > image_height:
                scale_down = image_height / target_height
                target_height = image_height
                target_width *= scale_down

            # 3. Determine the coordinates for the box centered on the original center.
            new_half_width = target_width / 2.0
            new_half_height = target_height / 2.0
            new_left = center_x - new_half_width
            new_top = center_y - new_half_height

            # 4. Shift the box if it extends beyond the image boundaries to keep its size.
            if new_left < 0:
                new_left = 0
            if new_top < 0:
                new_top = 0
            if new_left + target_width > image_width:
                new_left = image_width - target_width
            if new_top + target_height > image_height:
                new_top = image_height - target_height

            new_right = new_left + target_width
            new_bottom = new_top + target_height

            # Use floor and ceil for final integer coordinates.
            current_bbox = [floor(new_left), floor(new_top), ceil(new_right), ceil(new_bottom)]

        # 4. Final validation on the resulting bounding box (either original or resized).
        final_left, final_top, final_right, final_bottom = current_bbox
        if not self._validate_bbox(final_left, final_top, final_right, final_bottom):
            logger.warning(f"Final bbox is invalid after processing: {current_bbox}")
            return None

        final_height = floor(final_bottom) - floor(final_top)
        final_width = floor(final_right) - floor(final_left)

        if final_height < self.MIN_DIMENSION or final_width < self.MIN_DIMENSION:
            logger.warning(
                f"Final bbox size ({final_width}x{final_height}) are still smaller than minimum ({self.MIN_DIMENSION})."
                f"Original bbox: {bbox_2d}, original image size: {image_width}x{image_height}"
            )
            return None

        return current_bbox

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Creates a new instance for VLM subagent tool.

        This method initializes a new session for an image, which can then be used
        for operations like calling the VLM subagent. It fetches the image from various sources
        and stores it internally.

        Args:
            instance_id: An optional unique identifier for the instance. If not
                provided, a new UUID will be generated.
            **kwargs: Should contain 'image' key with image data, or 'create_kwargs'
                containing {'image': image_data}. Image can be one of the following:
                - A PIL.Image.Image object.
                - A string containing an HTTP or HTTPS URL.
                - A string containing a local file path.
                - A string containing a file URI (e.g., "file:///path/to/image.jpg").
                - A string containing a base64-encoded image in the format of "data:image/jpeg;base64,..."

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Get image from kwargs
        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing required 'image' parameter in kwargs")

        img = fetch_image({"image": image})
        self._instance_dict[instance_id] = {
            "image": img,
            "response": "",
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        llm_calling_toolkit = parameters.get("llm_calling_toolkit", None)
        processor = llm_calling_toolkit.get("processor")
        tokenizer = llm_calling_toolkit.get("tokenizer")
        apply_chat_template_kwargs = llm_calling_toolkit.get("apply_chat_template_kwargs")
        llm_server_manager = llm_calling_toolkit.get("llm_server_manager")
        sampling_params = llm_calling_toolkit.get("sampling_params")

        back_name = parameters.get("back_name", "vllm")
        original_query = parameters.get("original_query", "")
        original_query = original_query.replace("<image>", "").strip()
        instance_data = self._instance_dict[instance_id]
        orig_image = instance_data["image"]

        # excute Python code
        code_str = parameters.get("code_str", "")
        if code_str is None:
            error_msg = "[Code Error] The code content is empty. Try to write the executable python code again or focus on the initial image."
            logger.warning(f"Tool execution failed: {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}
        code_str = replace_temp_images_path(code_str)
        code_str = prefix_code + code_str

        # Execute code locally (faster than sandbox)
        success, result_dict = await execute_code_locally_async(instance_id, code_str)
        code_out = result_dict.get("stdout", "").strip()  # computation result or image path
        code_err = result_dict.get("stderr", "").strip()
        if not success:
            if len(code_err) == 0:
                error_msg = "[Sandbox Error] Sandbox execution error. Try to write the executable python code again or focus on the initial image."
                logger.warning(f"Tool execution failed: {result_dict.get('error', 'Sandbox execution error').strip()}.")
            else:
                error_msg = "[Code Error] Python code is not executable! Try to write the executable python code again or focus on the initial image."
                logger.warning(f"Tool execution failed: {error_msg}, {code_err.split('\n')[-1].strip()} code_str: {'<->'.join(code_str.split('\n'))}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}
        if len(code_out) == 0:  # no output at all
            error_msg = "[Code Error] Python code has no output! Try to write the executable python code again or focus on the initial image."
            logger.warning(f"Tool execution failed: {error_msg}, code_str: {'<->'.join(code_str.split('\n'))}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        unique_paths, useful_paths = extract_image_paths(code_out)
        if len(unique_paths) == 0: # executed successfully but no image path in output
            if len(code_out) > 100: code_out = code_out[-100:].strip()
            logger.warning(f"Tool execution results: {code_out.replace("\n", " ").replace("\b", " ")}")
            return ToolResponse(text=code_out), -0.05, {"success": True}
        if len(useful_paths) == 0: # image paths found but none exist on disk
            error_msg = "[Code Error] Python code outputs non correct image paths. Try to write the executable python code again or focus on the initial image."
            logger.warning(f"Tool execution failed: {error_msg}, code_str: {'<->'.join(code_str.split('\n'))}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Process all returned images
        processed_images = []
        for path in useful_paths:
            try:
                img_pil = Image.open(path)
                processed_img = resize_to_multiple_of_28(img_pil)
                processed_images.append(processed_img)
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
            finally:
                if os.path.isfile(path): os.remove(path)
        if len(processed_images) == 0:
            error_msg = "[Image Error] Failed to load any processed images. Try to write the executable python code again or focus on the initial image."
            logger.warning(error_msg)
            return ToolResponse(text=error_msg), -0.05, {"success": False}
        processed_images = processed_images[:1]

        # Prepare the function critic prompt and user_content
        judge_images = processed_images
        user_content = [{"type": "text", "text": subagent_user_prompt.format(question=original_query)}]
        for _ in processed_images: user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,"}})

        raw_messages = [
            {
                "role": "system",
                "content": subagent_system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        raw_prompt = processor.apply_chat_template(raw_messages, add_generation_prompt=True, tokenize=False, **apply_chat_template_kwargs)
        if "<|im_start|>system" not in raw_prompt: raw_prompt = f"<|im_start|>system\n{subagent_system_prompt}<|im_end|>\n" + raw_prompt  # guard against processors that render no system message
        model_inputs = processor(text=[raw_prompt], images=judge_images, return_tensors="pt")
        prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        request_id = uuid4().hex
        # Use a copy so we don't mutate the toolkit dict shared by the main agent / other concurrent subagents
        sampling_params = dict(sampling_params)
        sampling_params["temperature"] = 0.1
        sampling_params["max_tokens"] = 2048
        if back_name == "vllm":
            compress_prompt_ids = _compress_image_pad_tokens(prompt_ids) if judge_images else prompt_ids
            response_ids = await llm_server_manager.generate(request_id=request_id, prompt_ids=compress_prompt_ids, sampling_params=sampling_params, image_data=judge_images)
        else: response_ids = await llm_server_manager.generate(request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=judge_images)
        response_ids = [tid for tid in response_ids if tid not in _IMAGE_SPECIAL_TOKEN_IDS]  # the model may emit image placeholders, which would break generation

        response_text = processor.decode(response_ids)
        response_text = clean_text(response_text)
        # Try to parse subagent response
        return_images = processed_images
        try:
            json_match = re.search(r'[\[{].*?"is_helpful".*?[\]}]', response_text, re.DOTALL)
            if json_match:
                raw_json = json_match.group(0)
                is_helpful_match = re.search(r'"is_helpful"\s*:\s*(true|false)', raw_json, re.IGNORECASE)
                is_helpful = is_helpful_match.group(1).lower() == 'true' if is_helpful_match else True

                reasons_match = re.search(r'"reasons"\s*:\s*"([^"]*)"', raw_json, re.DOTALL)
                reasons = reasons_match.group(1) if reasons_match else ""

                if is_helpful: verdict_text = f"\n[Judgment: ACCEPT] {reasons}"
                else:
                    return_images = []
                    verdict_text = f"[Judgment: REJECT] {reasons}"

                return (ToolResponse(text=verdict_text, image=return_images), 0.0, {"success": True})
            else:
                logger.warning(f"Failed to parse JSON from function critic response:{re.sub(r'[\r\n]+', ' ', response_text)}")
                return (ToolResponse(text="\n[Judgment: ACCEPT] The image shows the information relevant to the question.", image=return_images), -0.05, {"success": True, "parse_error": True})
        except Exception as e:
            logger.warning(f"{e}, Error processing critic response:{re.sub(r'[\r\n]+', ' ', response_text)}")
            return (ToolResponse(text="\n[Judgment: ACCEPT] The image shows the information relevant to the question.", image=return_images), -0.05, {"success": False, "error": str(e)})

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
