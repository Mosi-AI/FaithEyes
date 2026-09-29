import os
import re
import json
import tempfile
import traceback
import subprocess
import time
import textwrap
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image
from io import BytesIO
import uuid

def extract_tool_call_contents(start_token, end_token, text):
    # pattern = r"<tool_call>(.*?)</tool_call>"
    pattern = re.escape(start_token) + r'(.*?)' + re.escape(end_token)
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


class PythonInterpreter(object):
    # Canonical registration name.  Using the short, generic label makes it
    # easier for dataset authors (`env_name="code"`).

    code_start = "<code>"
    code_end = "</code>"
    answer_start = "<answer>"
    answer_end = "</answer>"

    sandbox_prompt = """Analyze sandbox output to decide whether to answer the question with <answer> </answer> or run another round of python code with <code> </code>."""
    sandbox_prompt = """"""


    def __init__(self, _name, _desc, _params, aux_img_dir=None, **kwargs):
        self.session_id = str(uuid.uuid4())[:8]
        self.temp_dir = f"/tmp/codev_exec_{self.session_id}"
        self.multi_modal_data = None
        self.input_image_paths = []
        self.captured_outputs = []
        self.execution_count = 0
        self._preserve_dir = True
        self._input_img_sizes: List[Tuple[int, int]] = []
        self.image_filename = None  # Store the image filename from user prompt
        self.aux_img_dir = aux_img_dir  # Directory where extracted images are stored

        # Create temp directory
        os.makedirs(self.temp_dir, exist_ok=True)

        # Safe execution settings
        self.timeout = 10  # 30 second timeout
        self.max_memory = "512m"  # 512MB memory limit

        self.log_file_path = os.path.join(self.temp_dir, "execution_log.txt")


    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, extra_info=None, **kwargs):
        """Initialize tool with input images via symlinks"""
        self.multi_modal_data = origin_multi_modal_data
        self.execution_count = 0

        # Verify image data
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] No images in {origin_multi_modal_data.keys()}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'

        # Require extra_info with image filename
        assert extra_info is not None, f'[ERROR] extra_info is required'
        assert self.aux_img_dir is not None, f'[ERROR] aux_img_dir is required'
        assert isinstance(extra_info, dict) and 'image_file_name' in extra_info, f'[ERROR] extra_info must contain image_file_name'

        # Create symlinks with actual filenames
        self.image_filename = extra_info.get('image_file_name')
        self.create_image_symlinks()
        self._log("raw_prompt", raw_prompt)

    def create_image_symlinks(self):
        """Create symlinks in temp_dir pointing to actual images in aux_img_dir"""
        self.input_image_paths = []
        self._input_img_sizes = []

        if not self.image_filename:
            return

        # Source: actual image location
        source_path = os.path.join(self.aux_img_dir, self.image_filename)

        # Target: symlink in temp directory with the same filename
        target_path = os.path.join(self.temp_dir, self.image_filename)

        try:
            # Create symlink (protects original files from being modified)
            if os.path.exists(target_path):
                os.remove(target_path)
            os.symlink(source_path, target_path)

            self.input_image_paths.append(target_path)

            # Get image size for caching
            img = Image.open(source_path)
            self._input_img_sizes.append((img.width, img.height))
        except Exception as e:
            self._log("ERROR_create_symlink", f"Failed to create symlink: {e}")

    def _filter_image_filenames(self, output_text: str) -> str:
        """Remove lines that only contain image filenames from output.

        Training data shows sandbox_output should only contain <image> tags,
        not the printed filenames like "./cropped_1234.jpg"

        Keep other text output like calculation results, analysis text, etc.
        """
        import re

        lines = output_text.split('\n')
        filtered_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip lines that are just image filenames
            # Pattern: optional ./ followed by filename ending in .jpg/.png/.jpeg
            if re.match(r'^\.?/?[\w_-]+\.(?:jpe?g|png)$', stripped, re.IGNORECASE):
                continue
            # Skip [OUTPUT_TEXT] markers
            if stripped == '[OUTPUT_TEXT]':
                continue
            # Keep everything else
            filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def _fix_image_paths_in_code(self, code: str) -> str:
        """Replace incorrect image paths with just the filename.

        The model may hallucinate paths like:
        - /mnt/data/temp_processed_images/27.jpg
        - /path/to/image/27.jpg
        - /home/user/images/27.jpg

        We need to extract just the filename (27.jpg) and use it.
        Also fixes output directory paths to use current directory.
        """
        if not self.image_filename:
            return code

        import re

        # Fix 1: Replace input image paths with just the filename
        # Simple strategy: Find any quoted string ending with /filename.ext
        # Keep only the part after the last "/"
        # Examples:
        #   "/mnt/data/temp_processed_images/27.jpg" -> "27.jpg"
        #   "/mnt/data/data/temp_processed_images/1.jpg" -> "1.jpg"
        #   "./images/test.PNG" -> "test.PNG"

        # Pattern: quote + anything + "/" + (capture: filename.ext) + same quote
        # Match image files case-insensitively (jpg, JPG, png, PNG, etc.)
        pattern = r'(["\'])(?:[^"\']*/)([^/]+\.(?:jpe?g|png))\1'
        replacement = r'\1\2\1'  # Keep just the quotes and filename
        fixed_code = re.sub(pattern, replacement, code, flags=re.IGNORECASE)

        # Fix 2: Replace ALL absolute paths for image saves with just filename
        # The model hallucinates various patterns:
        #   path = '/mnt/data/temp_processed_images/crop1.png'
        #   img.save('/mnt/data/data/temp_processed_images/crop1.png')
        #   os.path.join("/mnt/data/whatever/", filename)
        #
        # Strategy: Find any absolute path (starting with /mnt, /tmp, etc.)
        # that ends with an image file, and replace with just the filename

        # Pattern: Any absolute path to an image file
        # Matches: "/mnt/anything/path/to/file.jpg" -> "file.jpg"
        #          "/tmp/whatever/image.png" -> "image.png"
        absolute_path_pattern = r'(["\'])(/(?:mnt|tmp|home|data|var)[^"\']*/)([^/"\' ]+\.(?:jpe?g|png))\1'

        def replace_absolute_with_filename(match):
            quote = match.group(1)  # " or '
            filename = match.group(3)  # just the filename
            return f'{quote}{filename}{quote}'

        fixed_code = re.sub(absolute_path_pattern, replace_absolute_with_filename, fixed_code, flags=re.IGNORECASE)

        # Fix 3: Also handle directory paths (ending with /) in os.path.join
        # Example: os.path.join("/mnt/data/whatever/", filename) -> os.path.join("./", filename)
        dir_only_pattern = r'(["\'])(/(?:mnt|tmp|home|data|var)[^"\']+/)\1'

        def replace_dir_with_current(match):
            quote = match.group(1)
            return f'{quote}./{quote}'

        fixed_code = re.sub(dir_only_pattern, replace_dir_with_current, fixed_code)

        # Log if we made changes
        if fixed_code != code:
            self._log("code_path_fix", {
                "original_snippet": code[:300],
                "fixed_snippet": fixed_code[:300],
                "image_filename": self.image_filename
            })

        return fixed_code

    def execute(self, action_string: str, **kwargs) -> Tuple[str, float, bool, Dict]:
        """Execute Python code or handle final answer"""
        self.execution_count += 1

        answers = extract_tool_call_contents(self.answer_start, self.answer_end, action_string)
        if answers:
            if self.execution_count == 1:
                self._preserve_dir = False
            return "", 0.0, True, {}

        codes = extract_tool_call_contents(self.code_start, self.code_end, action_string)
        codes = [code.replace("```python",'').replace('```','').replace("python",'') for code in codes]
        if not codes:
            self._preserve_dir = False
            return "", 0.0, False, {}
        code = "\n".join(code.strip() for code in codes)

        # Fix incorrect image paths in the code (model may hallucinate paths)
        code = self._fix_image_paths_in_code(code)

        # execute code
        success, output_text, captured_images, captured_image_paths = self.execute_code_safely(code)
        processed_images = []
        processed_image_paths = []
        for idx, img in enumerate(captured_images):
            proc = self.maybe_resize_image(img)
            if proc is not None:
                processed_images.append(proc)
                processed_image_paths.append(captured_image_paths[idx])
            else:
                print(
                    f"[PYTHON INTERPRETER DEBUG] Dropped malformed image with size: {img.size}"
                )

        if len(processed_images) > 5:
            processed_images = processed_images[:5]
            processed_image_paths = processed_image_paths[:5]
            output_text += "Only keep first five images"

        self._log(
            f"execution_result{self.execution_count}",
            {
                "success": success,
                "image_count": len(processed_images),
                "output_preview": output_text,
            },
        )

        if success:
            if len(processed_images) > 0:
                # Success with images: obs is a dict.
                image_placeholders = "".join("<image>" for _ in processed_images)

                # Filter out image filename printouts (e.g., "./cropped_1234.jpg")
                # but keep other text output like calculations, analysis, etc.
                filtered_output = self._filter_image_filenames(output_text)

                obs = {
                    "prompt": (
                        "<sandbox_output>"
                        + (f"{filtered_output}\n" if filtered_output.strip() else "")
                        + image_placeholders
                        + "</sandbox_output>"
                        + self.sandbox_prompt
                    ),
                    "multi_modal_data": {
                        "image": processed_images,
                        "image_paths": processed_image_paths
                    },
                }
            else:
                # Success without images – obs is a string
                obs = (
                    "<sandbox_output>"
                    + f"{output_text}"
                    + "</sandbox_output>"
                    + self.sandbox_prompt
                )

            reward = 0.0  # Reward for successful execution
            info = {"status": "success", "output": output_text}

        else:
            # -------------------- FAILURE BRANCH --------------------
            error_msg = self._summarise_error(output_text)
            if 'FileNotFoundError' in error_msg:
                # Provide specific filename guidance if we know it
                if self.image_filename:
                    obs = (
                        f"<sandbox_output>FileNotFoundError: To load the image, you must use Image.open('{self.image_filename}')</sandbox_output>"
                    )
                else:
                    obs = (
                        f"<sandbox_output>FileNotFoundError: To load the image, you must use Image.open('{self.image_filename}')</sandbox_output>"
                    )
            else:
                obs = (
                    f"<sandbox_output>Error: {error_msg}. You need to fix the bug and then answer</sandbox_output>"
                )

            reward = 0.0
            info = {"error": error_msg, "status": "failed"}

        self._log(f"exec_string_{self.execution_count}", obs)
        return obs, reward, False, info


    def execute_code_safely(self, user_code: str) -> Tuple[bool, str, List[Image.Image]]:
        """Run *user_code* in a sandboxed subprocess and capture its outputs.

        Returns ``(success, stdout_stderr, images)`` where ``success`` is
        ``True`` if – and only if – the subprocess terminates with exit code
        0.  The wrapper script now calls ``sys.exit(1)`` on any unhandled
        exception, making the exit status the single source of truth instead
        of parsing sentinel strings in the captured text output.
        """
        execution_script = self.create_safe_execution_environment(user_code)
        script_path = os.path.join(self.temp_dir, f"execution_{self.execution_count}.py")

        with open(script_path, 'w', encoding="utf-8") as f:
            f.write(execution_script)

        try:
            # Execute with resource limits
            result = subprocess.run([
                'python', script_path
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout + 5,  # Extra buffer for subprocess
            cwd=self.temp_dir
            )

            success = result.returncode == 0
            output_text = result.stdout + result.stderr

            # Load captured images from Image.show() calls
            captured_images = []
            captured_image_paths = []
            i = 0
            while True:
                output_path = os.path.join(
                    self.temp_dir, f"output_{self.execution_count}_{i}.png"
                )
                if os.path.exists(output_path) and i < 10:
                    captured_images.append(Image.open(output_path))
                    captured_image_paths.append(output_path)
                    i += 1
                else:
                    break

            # Additionally, load images that were explicitly saved by the model
            saved_images, saved_image_paths = self._load_saved_images(output_text)

            # Combine both types of images (Image.show() and saved files)
            all_images = captured_images + saved_images
            all_image_paths = captured_image_paths + saved_image_paths

            return success, output_text, all_images, all_image_paths

        except subprocess.TimeoutExpired:
            return False, "[EXECUTION_ERROR] Code execution timeout", [], []
        except Exception as e:
            return False, f"{str(e)}", [], []


    def create_safe_execution_environment(self, user_code: str) -> str:
        """Improved sandbox wrapper that captures stdout via
        ``contextlib.redirect_stdout`` and outputs consolidated segments:

        [OUTPUT_TEXT] – followed by captured stdout (if any)
        [OUTPUT_IMAGE] – emitted if images were captured via Image.show()
        [EXECUTION_SUCCESS] – final sentinel looked up by the caller

        A single '[EXECUTION_ERROR] …' line is printed on failure.
        """
        if user_code.strip():
            # Normalize indentation first (removes common leading whitespace)
            # This fixes model bugs like " sock_identified = ..." (extra leading space)
            normalized_code = textwrap.dedent(user_code)
            indented_code = "\n".join("        " + ln for ln in normalized_code.splitlines())

            # Handle comments-only case: if no executable statements exist,
            # add 'pass' to avoid IndentationError in the with block
            has_executable = any(
                line.strip() and not line.strip().startswith('#')
                for line in normalized_code.splitlines()
            )
            if not has_executable:
                indented_code += "\n        pass"
        else:
            indented_code = "        pass"  # placeholder to satisfy Python grammar

        return f'''\
import traceback, os, signal, contextlib, io, sys
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for sandboxed environment
import matplotlib.pyplot as plt
from scipy import integrate, stats
import sympy
import cv2  
from sklearn import preprocessing, decomposition
import pandas as pd

def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timeout")

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm({self.timeout})

captured_images = []

def _capture_show(self, *args, **kwargs):
    captured_images.append(self.copy())
    return None  # suppress Pillow repr

Image.Image.show = _capture_show

_stdout_buffer = io.StringIO()

# Make image filename available in the execution environment
image_filename = "{self.image_filename}" if "{self.image_filename}" != "None" else None

try:
    with contextlib.redirect_stdout(_stdout_buffer):
{indented_code}

    captured_text = _stdout_buffer.getvalue()

    for _i, _img in enumerate(captured_images):
        _out_path = os.path.join("{self.temp_dir}", f"output_{self.execution_count}_{{_i}}.png")
        _img.save(_out_path)

    if captured_text.strip():
        print('[OUTPUT_TEXT]')
        print(captured_text, end='')

    if captured_images:
        print('[OUTPUT_IMAGE]')

except BaseException as _e:
    print(f"[EXECUTION_ERROR] {{type(_e).__name__}}: {{_e}}")
    traceback.print_exc()
    sys.exit(1)
finally:
    signal.alarm(0)
'''
    
    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _summarise_error(raw_output: str) -> str:
        """Return a short, user-friendly error string extracted from the full
        subprocess stdout/stderr dump.  The strategy is:

        1.  Search for a line that starts with our custom `[EXECUTION_ERROR]`
            sentinel added in *create_safe_execution_environment*.
        2.  If not found, take the last non-empty line of the output (this is
            usually the exception message printed by *traceback.print_exc*).
        3.  As a final fallback, return the raw output unchanged.  This ensures
            we always provide *something* useful to the agent while avoiding
            excessive verbosity in the common path.
        """

        try:
            lines = [ln.strip() for ln in raw_output.strip().splitlines() if ln.strip()]
            if not lines:
                return "Code execution failed."

            # 1) Look for sentinel line.
            for ln in lines:
                if ln.startswith("[EXECUTION_ERROR]"):
                    return ln.replace("[EXECUTION_ERROR]", "").strip()

            # 2) Otherwise use the last meaningful line (usually 'ValueError: …').
            return lines[-1]
        except Exception:
            # Best-effort: fall back to the full raw output.
            return raw_output or "Code execution failed."

    @staticmethod
    def _validate_image_dims(width: int, height: int) -> bool:
        """Return False if the image is clearly malformed (extreme aspect ratio
        or one side too small)."""
        try:
            assert width > 0 and height > 0, "non-positive dimension"
            # Extremely long / thin images are suspicious.
            if max(width, height) / max(1, min(width, height)) > 100:
                raise ValueError("aspect ratio > 100")
            return True
        except Exception as e:
            print(f"[DEBUG] _validate_image_dims failed: {e}")
            return False

    def maybe_resize_image(self, img: Image.Image) -> Optional[Image.Image]:
        """Ensure the image meets minimal size/aspect-ratio requirements.

        • If the image has an extreme aspect ratio (>100) it is dropped (returns None).
        • If its smallest side is <28 px, it is up-scaled so that the minimum side
          becomes 28 px (keeping aspect ratio).
        """
        width, height = img.width, img.height

        # Basic validation (aspect ratio / positive dims)
        if not self._validate_image_dims(width, height):
            return None

        # Upscale very small images
        if min(width, height) < 28:
            import math
            ratio = 28 / float(min(width, height))
            new_w = max(1, int(math.ceil(width * ratio)))
            new_h = max(1, int(math.ceil(height * ratio)))
            try:
                img = img.resize((new_w, new_h), Image.BICUBIC)
            except Exception as e:
                print(f"[DEBUG] Resize failed: {e}")
                return None

        # After possible resize, double-check dimensions
        if not self._validate_image_dims(img.width, img.height):
            return None

        return img

    def _extract_saved_image_paths(self, output_text: str) -> List[str]:
        """Extract file paths of images explicitly saved by the model's code.

        Looks for patterns like:
        - "Saved to: result.png"
        - "Image saved as result.png"
        - "result.png saved successfully"
        - "./2_7112.jpg" (standalone path on a line)

        Returns list of filenames (not full paths).
        """
        patterns = [
            r"[Ss]aved (?:to|as):?\s+([^\s\n]+\.(?:png|jpg|jpeg))",
            r"([^\s\n]+\.(?:png|jpg|jpeg))\s+saved",
            r"[Oo]utput.*?:\s+([^\s\n]+\.(?:png|jpg|jpeg))",
            # Match standalone image paths (e.g., "./2_7112.jpg" or "result.png")
            r"^\s*([./]*[\w/_-]+\.(?:png|jpg|jpeg))\s*$",
        ]

        saved_files = []
        for pattern in patterns:
            matches = re.findall(pattern, output_text, re.MULTILINE)
            saved_files.extend(matches)

        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for f in saved_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)

        return unique_files

    def _load_saved_images(self, output_text: str) -> Tuple[List[Image.Image], List[str]]:
        """Load images that were explicitly saved by the model's code.

        Returns:
            Tuple of (loaded_images, image_paths)
        """
        saved_filenames = self._extract_saved_image_paths(output_text)
        loaded_images = []
        image_paths = []

        for filename in saved_filenames:
            full_path = os.path.join(self.temp_dir, filename)
            if os.path.exists(full_path):
                try:
                    img = Image.open(full_path)
                    # Validate before adding
                    if self._validate_image_dims(img.width, img.height):
                        loaded_images.append(img.copy())
                        image_paths.append(full_path)
                except Exception as e:
                    self._log("WARNING_load_saved_image", f"Failed to load {filename}: {e}")

        return loaded_images, image_paths

    def _log(self, tag: str, payload: Any):
        """Append an entry to *execution_log.txt*.

        The *payload* can be any serialisable object; to make sure we always
        succeed (and keep the implementation dependency-free), we fall back to
        ``str(payload)`` when JSON serialisation is not possible.  Lists and
        other containers are therefore explicitly converted to a string
        representation to honour the user's request ("be careful to convert
        list to str")."""

        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # Best-effort JSON serialisation – falls back to plain ``str`` on
            # failure so that *anything* can be logged without raising.
            try:
                if isinstance(payload, (dict, list, tuple)):
                    payload_str = json.dumps(payload, default=str, ensure_ascii=False)
                else:
                    payload_str = str(payload)
            except Exception:
                payload_str = str(payload)

            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] [{tag}] {payload_str}\n")
        except Exception as e:
            # The logger itself must never crash the interpreter – fallback to
            # stderr if anything goes wrong.
            print(f"[PYTHON INTERPRETER LOGGING ERROR] {e}")

    def cleanup(self):
        """Clean up temporary files"""
        # remark for better debugging.
        return
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"[WARNING] Failed to cleanup temp directory: {e}")

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.cleanup()
        except Exception:
            pass


if __name__ == '__main__':
    import requests
    from io import BytesIO
    
    def download_test_image(url):
        """Download test image from URL"""
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    
    def test_case_1_broken_code():
        """Test Case 1: Broken code that should fail"""
        print("=== Test Case 1: Broken Code ===")
        
        # Setup interpreter
        interpreter = PythonInterpreter("test", "test", {})
        test_img = download_test_image("https://m.media-amazon.com/images/I/71g1eZWQtNL.jpg")
        multi_modal_data = {'image': [test_img]}
        interpreter.reset("", multi_modal_data, multi_modal_data)
        
        # Execute broken code
        broken_code = """
<code>
# This should fail - undefined variable
print(undefined_variable)
img = Image.open(path_to_image)
img.show()
</code>
        """
        
        obs, reward, done, info = interpreter.execute(broken_code)
        print("=== EXECUTION RESULTS ===")
        print(f"obs: {obs}")
        print(f"reward: {reward}")
        print(f"done: {done}")
        print(f"info: {info}")
        interpreter.cleanup()
        print()
    
    def test_case_2_simple_crop():
        """Test Case 2: Simple crop operation"""
        print("=== Test Case 2: Simple Crop ===")
        
        # Setup interpreter
        interpreter = PythonInterpreter("test", "test", {})
        test_img = download_test_image("https://m.media-amazon.com/images/I/71g1eZWQtNL.jpg")
        multi_modal_data = {'image': [test_img]}
        interpreter.reset("", multi_modal_data, multi_modal_data)
        
        # Execute crop code
        crop_code = """
<code>
img = Image.open(path_to_image)
print(f"Original size: {img.size}")
cropped = img.crop((100, 200, 300, 400))
print(f"Cropped size: {cropped.size}")
cropped.show()
</code>
        """
        
        obs, reward, done, info = interpreter.execute(crop_code)
        print("=== EXECUTION RESULTS ===")
        print(f"obs: {obs}")
        print(f"reward: {reward}")
        print(f"done: {done}")
        print(f"info: {info}")
        interpreter.cleanup()
        print()
    
    def test_case_3_no_image_operation():
        """Test Case 3: No image operation, just print"""
        print("=== Test Case 3: No Image Operation ===")
        
        # Setup interpreter
        interpreter = PythonInterpreter("test", "test", {})
        test_img = download_test_image("https://m.media-amazon.com/images/I/71g1eZWQtNL.jpg")
        multi_modal_data = {'image': [test_img]}
        interpreter.reset("", multi_modal_data, multi_modal_data)
        
        # Execute simple math
        math_code = """
<code>
result = 10 + 10
print(f"Result: {result}")
print("Math calculation completed!")
</code>
        """
        
        obs, reward, done, info = interpreter.execute(math_code)
        print("=== EXECUTION RESULTS ===")
        print(f"obs: {obs}")
        print(f"reward: {reward}")
        print(f"done: {done}")
        print(f"info: {info}")
        interpreter.cleanup()
        print()
    
    def test_case_4_crop_failure():
        """Test Case 4: Crop operation that should fail"""
        print("=== Test Case 4: Crop Failure ===")
        
        # Setup interpreter
        interpreter = PythonInterpreter("test", "test", {})
        test_img = download_test_image("https://m.media-amazon.com/images/I/71g1eZWQtNL.jpg")
        multi_modal_data = {'image': [test_img]}
        interpreter.reset("", multi_modal_data, multi_modal_data)
        
        # Execute failing crop code (invalid coordinates)
        failing_crop_code = """
<code>
img = Image.open(path_to_image)
print(f"Original size: {img.size}")
# This should fail - crop coordinates outside image bounds
cropped = img.crop((1000, 2000, 3000, 4000))
cropped.show()
</code>
        """
        
        obs, reward, done, info = interpreter.execute(failing_crop_code)
        print("=== EXECUTION RESULTS ===")
        print(f"obs: {obs}")
        print(f"reward: {reward}")
        print(f"done: {done}")
        print(f"info: {info}")
        interpreter.cleanup()
        print()
    
    def test_case_5_answer_detection():
        """Test Case 5: Final answer detection"""
        print("=== Test Case 5: Answer Detection ===")
        
        # Setup interpreter
        interpreter = PythonInterpreter("test", "test", {})
        test_img = download_test_image("https://m.media-amazon.com/images/I/71g1eZWQtNL.jpg")
        multi_modal_data = {'image': [test_img]}
        interpreter.reset("", multi_modal_data, multi_modal_data)
        
        # Execute code with final answer
        answer_code = """
        <code>
        img = Image.open(path_to_image)
        print(f"Image size: {img.size}")
        </code>
        <answer>
        The image is a product photo with dimensions of the loaded image.
        </answer>
        """
        
        obs, reward, done, info = interpreter.execute(answer_code)
        print("=== EXECUTION RESULTS ===")
        print(f"obs: {obs}")
        print(f"reward: {reward}")
        print(f"done: {done}")
        print(f"info: {info}")
        interpreter.cleanup()
        print()
    
    # Run all test cases
    print("Running PythonInterpreter Test Cases...")
    print("=" * 50)
    
    try:
        test_case_1_broken_code()
        test_case_2_simple_crop()
        test_case_3_no_image_operation()
        test_case_4_crop_failure()
        test_case_5_answer_detection()
        print("All test cases completed!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
