# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import copy
import json
import logging
import os
import re
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_IMAGE_SPECIAL_TOKEN_IDS = {151652, 151653, 151655, 151656}  # vision_start, vision_end, image_pad, video_pad

_IMAGE_PAD_TOKEN_ID = 151655  # <|image_pad|>
_VISION_START_TOKEN_ID = 151652  # <|vision_start|>
_VISION_END_TOKEN_ID = 151653  # <|vision_end|>
_IM_START_TOKEN_ID = 151644  # <|im_start|>
# <|im_start|>user\n — start of a user turn, identical across Qwen2.5-VL / Qwen3-VL
_USER_TURN_HEADER_IDS = [151644, 872, 198]
_NEWLINE_TOKEN_ID = 198  # "\n"


def _find_sublist(haystack: list[int], needle: list[int]) -> int:
    """Return the index of the first occurrence of ``needle`` in ``haystack``, or -1."""
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i:i+n] == needle:
            return i
    return -1

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

def _find_image_placeholder_spans(token_ids: list[int]) -> list[tuple[int, int]]:
    """Locate every image placeholder span in ``token_ids`` (assumed compressed).

    A placeholder is a maximal run of ``<|image_pad|>`` tokens (after
    ``_compress_image_pad_tokens`` each run is a single token). Each placeholder
    is associated with the ``<|vision_start|> ... <|vision_end|>`` block that
    wraps it when one exists.

    Returns a list of ``(start, end)`` half-open spans covering the *entire*
    placeholder block:

    - Well-formed block:  ``<|vision_start|> [<|image_pad|>] <|vision_end|>``
      -> the span covers ``[vision_start, vision_end + 1)``.
    - Orphan pad (no matching ``<|vision_start|>``/``<|vision_end|>``): the span
      covers just the ``<|image_pad|>`` token. Orphan pads arise from prompt
      slicing that drops ``<|vision_start|>``; they still count as image
      positions for vLLM, so they must be aligned too.

    The spans are returned in order of appearance and are non-overlapping.
    """
    spans: list[tuple[int, int]] = []
    n = len(token_ids)
    i = 0
    while i < n:
        if token_ids[i] == _IMAGE_PAD_TOKEN_ID:
            # Found a (compressed) image_pad marker. Determine the enclosing
            # vision block boundaries, if any.
            start = i
            end = i + 1  # exclusive

            # Walk back to the nearest preceding <|vision_start|> with no
            # intervening <|vision_end|> (i.e. we are inside an open block).
            vs = -1
            j = i - 1
            while j >= 0:
                if token_ids[j] == _VISION_END_TOKEN_ID:
                    break  # a closed block ends before us -> we are orphan
                if token_ids[j] == _VISION_START_TOKEN_ID:
                    vs = j
                    break
                j -= 1
            # Walk forward to the nearest following <|vision_end|>.
            ve = -1
            j = i + 1
            while j < n:
                if token_ids[j] == _VISION_START_TOKEN_ID:
                    break  # a new block starts before we close -> orphan
                if token_ids[j] == _VISION_END_TOKEN_ID:
                    ve = j
                    break
                j += 1

            if vs != -1 and ve != -1:
                start = vs
                end = ve + 1
            # else: orphan pad, span is just the image_pad token itself.
            spans.append((start, end))
            i = end
        else:
            i += 1
    return spans

def _normalize_image_data(image_data) -> list:
    """Coerce ``image_data`` into a flat list of images (possibly empty)."""
    if image_data is None:
        return []
    if isinstance(image_data, list):
        # Flatten one level: nested lists of images can be produced by tool
        # responses that return multiple images.
        flat = []
        for item in image_data:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
    return [image_data]

def _align_images_to_placeholders(token_ids: list[int], image_data) -> tuple[list[int], list]:
    """Ensure the number of images matches the number of image placeholders.

    In multi-turn agent loops, tool responses may add images to ``image_data``
    without a matching placeholder in ``token_ids`` (or vice versa), which causes
    vLLM's ``masked_scatter`` to crash with a device-side assert.

    - If there are more images than placeholders: trim the image list.
    - If there are more placeholders than images: drop the excess placeholder
      blocks (or orphan pads) from ``token_ids``, keeping the first
      ``len(images)`` placeholders.

    This handles both well-formed ``<|vision_start|><|image_pad|><|vision_end|>``
    blocks and orphan ``<|image_pad|>`` tokens (produced by prompt slicing that
    drops ``<|vision_start|>``), so alignment is robust to malformed blocks.

    Returns the (possibly modified) ``(token_ids, image_data)``. ``image_data``
    is returned as a list when it holds more than one image, a single item when
    it holds exactly one, or ``None`` when empty.
    """
    images = _normalize_image_data(image_data)
    if not images:
        # No images: strip every placeholder block so vLLM is not asked to embed
        # images it was not given.
        spans = _find_image_placeholder_spans(token_ids)
        if not spans:
            return token_ids, None
        logger.warning(
            "Image/placeholder count mismatch: 0 images vs %d placeholders. Aligning.",
            len(spans),
        )
        result = []
        prev = 0
        for start, end in spans:
            result.extend(token_ids[prev:start])
            prev = end
        result.extend(token_ids[prev:])
        return result, None

    spans = _find_image_placeholder_spans(token_ids)
    n_placeholders = len(spans)
    n_images = len(images)

    if n_images == n_placeholders:
        return token_ids, (images if len(images) > 1 else images[0])

    logger.warning(
        "Image/placeholder count mismatch: %d images vs %d placeholders. Aligning.",
        n_images, n_placeholders,
    )

    if n_images > n_placeholders:
        # More images than placeholders: trim images to match.
        images = images[:n_placeholders]
        if n_placeholders == 0:
            return token_ids, None
        return token_ids, (images if len(images) > 1 else images[0])

    # More placeholders than images: drop the trailing excess placeholder
    # spans from token_ids, keeping the first ``n_images`` spans intact.
    keep = n_images
    drop_spans = spans[keep:]
    result = []
    prev = 0
    for start, end in drop_spans:
        result.extend(token_ids[prev:start])
        prev = end
    result.extend(token_ids[prev:])
    return result, (images if len(images) > 1 else images[0])

def _truncate_at_vision_boundary(token_ids: list[int], mask: list[int], max_length: int) -> tuple[list[int], list[int]]:
    """Truncate token_ids and mask to max_length, ensuring we don't cut inside a
    <|vision_start|>...<|image_pad|>...<|vision_end|> block.

    If the truncation point falls inside an unclosed vision block (i.e., after
    <|vision_start|> but before the matching <|vision_end|>), we roll the cut
    back to before the <|vision_start|> token so the block is either fully
    included or fully excluded.

    Returns:
        tuple of (truncated_token_ids, truncated_mask)
    """
    if len(token_ids) <= max_length: return token_ids, mask

    cut = max_length
    i = cut - 1
    while i >= 0:
        tid = token_ids[i]
        if tid == _VISION_END_TOKEN_ID: break
        if tid == _VISION_START_TOKEN_ID:
            cut = i
            break
        if tid == _IMAGE_PAD_TOKEN_ID:
            i -= 1
            continue
        break
    return token_ids[:cut], mask[:cut]

def extract_answer(text):
    pattern = r'<answer>((?:(?!<answer>).)*?)</answer>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = "recipe/vlm_subagent/configs/vlm_subagent_tool_config.yaml" #config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.max_model_len = config.actor_rollout_ref.rollout.max_model_len if config.actor_rollout_ref.rollout.max_model_len else cls.prompt_length + cls.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        cls.back_name = config.actor_rollout_ref.rollout.name

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        sampling_params = dict(sampling_params)
        sampling_params["max_tokens"] = 8192
        messages = list(kwargs["raw_prompt"])
        extra_info = dict(kwargs.get("extra_info", {"question": ""}))
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        response_mask = []
        tools_kwargs = kwargs.get("tools_kwargs", {})

        user_turns, assistant_turns = 0, 0
        tool_bboxs = []
        while True:
            # Check prompt length against the model's max length, reserving room for generation
            if len(prompt_ids) >= self.max_model_len - 1:
                break
            with simple_timer("generate_sequences", metrics):
                if self.back_name == "vllm":
                    compress_prompt_ids = _compress_image_pad_tokens(prompt_ids) if image_data else prompt_ids
                    if image_data:
                        compress_prompt_ids, aligned_image_data = _align_images_to_placeholders(compress_prompt_ids, image_data)
                    else:
                        aligned_image_data = image_data
                    response_ids = await self.server_manager.generate(request_id=request_id, prompt_ids=compress_prompt_ids, sampling_params=sampling_params, image_data=aligned_image_data)
                else: response_ids = await self.server_manager.generate(request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data)
                response_ids = [tid for tid in response_ids if tid not in _IMAGE_SPECIAL_TOKEN_IDS]  # the model may emit image placeholders, which would break generation

            # If already answered, skip tool-call parsing
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            answers = extract_answer(response_text)
            if answers:
                prompt_ids += response_ids
                response_mask += [1] * len(response_ids)
                assistant_turns += 1
                break

            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            assistant_turns += 1

            # reach max response length
            if len(response_mask) >= self.response_length:
                break

            # reach max assistant turns
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break

            # reach max user turns
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            # no tool calls
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            if not tool_calls: break
            # Inject original_query into arguments
            for i in range(len(tool_calls)):
                arguments = dict()
                orig_arguments = tool_calls[i].arguments
                attempts = 0
                while isinstance(orig_arguments, str) and attempts < 5:
                    try:
                        orig_arguments = json.loads(orig_arguments)
                        attempts += 1
                    except: break
                if isinstance(orig_arguments, dict): arguments = orig_arguments
                arguments["original_query"] = extra_info.get("question", "")
                tool_calls[i].arguments = json.dumps(arguments, ensure_ascii=False)

            # call tools
            tasks = []
            for tool_call in tool_calls[:self.max_parallel_calls]:
                # FIXME: add LLM calling toolkits into kwargs if you use subagent
                if "vlm_coding" in tool_call.name:
                    tools_kwargs["llm_calling_toolkit"] = {
                        "processor": self.processor,
                        "tokenizer": self.tokenizer,
                        "apply_chat_template_kwargs": self.apply_chat_template_kwargs,
                        "llm_server_manager": self.server_manager,
                        "sampling_params": dict(sampling_params),
                    }
                tasks.append(self._call_tool(tool_call, tools_kwargs))  # execute subagent call
            with simple_timer("tool_calls", metrics):
                tool_responses = await asyncio.gather(*tasks)

            if any(isinstance(item, Exception) for item in tool_responses):
                break

            # Extract messages and update multi_modal_data
            tool_messages = []
            new_images_this_turn = []
            for tool_response in tool_responses:
                if tool_response.bbox_2d: tool_bboxs.append(tool_response.bbox_2d)
                # Create message from tool response
                if tool_response.image or tool_response.video:
                    content = []
                    if tool_response.image:
                        if isinstance(tool_response.image, list): content.extend([{"type": "image"}]*len(tool_response.image))
                        else: content.append({"type": "image"})
                    if tool_response.video:
                        content.append({"type": "video"})
                    if tool_response.text:
                        content.append({"type": "text", "text": tool_response.text})
                    message = {"role": "user", "content": content}
                else:
                    message = {"role": "user", "content": tool_response.text or ""}

                tool_messages.append(message)

                # Handle image data
                if tool_response.image:
                    if image_data is None:
                        image_data = []
                    elif not isinstance(image_data, list):
                        image_data = [image_data]

                    # Add new image data
                    if isinstance(tool_response.image, list):
                        image_data.extend(tool_response.image)
                        new_images_this_turn.extend(tool_response.image)
                    else:
                        image_data.append(tool_response.image)
                        new_images_this_turn.append(tool_response.image)

                # Handle video data
                if tool_response.video:
                    # Currently not supported, raise informative error
                    logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                    raise NotImplementedError(
                        "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                    )

            # append tool_response_ids
            if self.processor is not None:
                raw_tool_response = await self.loop.run_in_executor(
                    None,
                    lambda messages=tool_messages: self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    ),
                )
                if isinstance(raw_tool_response, list):  # Ensure raw_tool_response is a string, not a list of tokens or string
                    if len(raw_tool_response) > 0 and isinstance(raw_tool_response[0], int): raw_tool_response = self.tokenizer.decode(raw_tool_response, skip_special_tokens=False)
                    elif len(raw_tool_response) > 0 and isinstance(raw_tool_response[0], str): raw_tool_response = "".join(raw_tool_response)

                current_images = new_images_this_turn if new_images_this_turn else None
                model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
                tool_response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
            else:
                tool_response_ids = await self.loop.run_in_executor(
                    None,
                    lambda messages=tool_messages: self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                    ),
                )
            
            _SANDBOX_OPEN = [44047, 31536, 7645, 29]  # <sandbox_output>
            _SANDBOX_CLOSE = [522, 76756, 7645, 29]  # </sandbox_output>
            user_start = _find_sublist(tool_response_ids, _USER_TURN_HEADER_IDS)
            if user_start == -1:
                raise ValueError(
                    f"Cannot find user-turn header {_USER_TURN_HEADER_IDS} in tool response; "
                    f"unexpected chat template output: {self.tokenizer.decode(tool_response_ids[:30])!r}"
                )
            content_start = user_start + len(_USER_TURN_HEADER_IDS)
            prefix_start = user_start - 1 if user_start > 0 and tool_response_ids[user_start-1] == _NEWLINE_TOKEN_ID else user_start
            tool_response_ids = tool_response_ids[prefix_start:content_start] + _SANDBOX_OPEN + tool_response_ids[content_start:-5] + _SANDBOX_CLOSE + tool_response_ids[-5:]

            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                break

            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)
            user_turns += 1

        response_ids = prompt_ids[-len(response_mask):]
        prompt_ids = prompt_ids[:len(prompt_ids)-len(response_mask)]

        multi_modal_data = {"image": image_data} if image_data is not None else {}

        response_ids, response_mask = _truncate_at_vision_boundary(
            response_ids, response_mask, self.response_length
        )

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            multi_modal_data=multi_modal_data,
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
            tool_bboxs=tool_bboxs,
        )
        return output

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]) -> ToolResponse:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            
            # FIXME: load LLM calling toolkit if you use subagent
            tool_args["llm_calling_toolkit"] = tools_kwargs.get("llm_calling_toolkit", None)
            tool_args["back_name"] = self.back_name
            
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, _, _ = await tool.execute(instance_id, tool_args, **kwargs.get("execute_kwargs", {}))
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return ToolResponse(
                text=f"Tool-Error: tool {e} is not supported",
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video", "bbox_2d"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs)
