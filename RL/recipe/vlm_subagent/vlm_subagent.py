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
import io
import logging
import os
import random
import re

import requests
from openai import OpenAI
from PIL import Image

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

openai_api_key = "EMPTY"
openai_api_base = os.environ.get("LLM_AS_A_JUDGE_BASE", "http://127.0.0.1:18901/v1")

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    timeout=float(os.environ.get("LLM_JUDGE_TIMEOUT", "120")),
    max_retries=2,
)

model_name = ""
if openai_api_base:
    try:
        response = requests.get(f"{openai_api_base}/models")
        response.raise_for_status()
        models = response.json()
        if models.get("data"):
            model_name = models["data"][0]["id"]
        else:
            logger.warning("No models found at the specified API base for reward scoring.")
    except (requests.exceptions.RequestException, KeyError, IndexError) as e:
        logger.warning(f"Failed to get model from {openai_api_base}: {e}. Reward scoring will be disabled.")

CONTEXT_LEARNING_EXAMPLES = """You are a helpful assistant.

Solve the following problem step by step, and optionally write Python code for image manipulation to enhance your reasoning process. The Python code will be executed by an external sandbox, and the processed image or result (wrapped in <sandbox_output></sandbox_output>) can be returned to aid your reasoning and help you arrive at the final answer.

**Reasoning & Image Manipulation (Optional but Encouraged):**
    * You have the capability to write executable Python code to perform image manipulations (e.g., cropping to a Region of Interest (ROI), resizing, rotation, adjusting contrast, draw auxiliary lines) or perform calculation for better reasoning.
    * The code will be executed in a secure sandbox, and its output will be provided back to you for further analysis.
    * All Python code snippets **must** be wrapped as follows:
    <code>
    ```python
    # your code.
    ```
    </code>
    * At the end of the code, print the path of the processed image (processed_path) or the result for further processing in a sandbox environment.

A critic subagent will help to judge whether the code execution result (i.e., processed image) is helpful for answering the user's original question and give the judgement and reasons in <sandbox_output></sandbox_output>.
You **must** carefully consider its judgment."""

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class CustomRLHFDataset(RLHFDataset):
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        row_dict[self.prompt_key] = [
            {
                "role": "system",
                "content": CONTEXT_LEARNING_EXAMPLES,
            },
            {
                "role": "user",
                "content": row_dict[self.prompt_key][1]["content"],
            },
        ]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                images = [Image.open(io.BytesIO(image["bytes"])) for image in row_dict.pop(self.image_key)]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205  # noqa: E501
                multi_modal_data["image"] = images

            model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = {
            "vlm_coding": {
                "create_kwargs": {"image": images[0]},
            }
        }
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["agent_name"] = "tool_agent"
        return row_dict



def extract_thinking(solution: str) -> str:
    thinking_match = re.search(r'<think>(.*?)</think>', solution, re.DOTALL)
    thinking = thinking_match.group(1) if thinking_match else ''
    if thinking == '': thinking = solution.split("</think>")[0].replace("<think>", "")
    thinking = thinking.replace("<image>", "")
    thinking = thinking.replace("</sandbox_output>", "").replace("<sandbox_output>", "")
    return thinking


def extract_answer(solution: str) -> str:
    answer_match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL)
    return answer_match.group(1).strip() if answer_match else ''


def extract_characters_regex(s: str) -> str:
    s = s.strip()
    for prefix in ("The best answer is", "The correct answer is", "The answer is",
                   "The answer: ", "Answer: ", "The best option is",
                   "The correct option is", "Best answer:", "Best option:"):
        s = s.replace(prefix, "")
    return s.strip()


def sympy_parse(expr: str):
    import sympy
    from sympy.parsing import sympy_parser
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations
                         + (sympy_parser.implicit_multiplication_application,)),
    )


def parse_latex(expr: str) -> str:
    from pylatexenc import latex2text
    expr = expr.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = expr.replace("√", "sqrt").replace("π", "pi").replace("∞", "inf")
    expr = expr.replace("∪", "U").replace("·", "*").replace("×", "*")
    return expr.strip()


def is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def rule_math_verify(student_answer: str, ground_truth: str) -> bool:
    """Symbolic equivalence check. Returns False (never raises) for non-math input,
    so non-math answers gracefully fall through to the LLM judge."""
    if not student_answer or not ground_truth:
        return False

    # Direct string equivalence after light normalization (handles "C"=="C", "Yes"=="Yes").
    if extract_characters_regex(student_answer) == extract_characters_regex(ground_truth):
        return True

    # Float equivalence (handles "18" vs "18.0").
    if is_float(student_answer) and is_float(ground_truth):
        try:
            if abs(float(student_answer) - float(ground_truth)) < 1e-6:
                return True
        except Exception:
            pass

    # sympy symbolic equivalence (handles "1/2" vs "0.5", "x**2" vs "x^2").
    import sympy
    for _ in range(2):  # try raw then latex-parsed
        try:
            a = sympy_parse(student_answer)
            b = sympy_parse(ground_truth)
            if sympy.simplify(a - b) == 0:
                return True
            break
        except Exception:
            try:
                a = sympy_parse(parse_latex(student_answer))
                b = sympy_parse(parse_latex(ground_truth))
                if sympy.simplify(a - b) == 0:
                    return True
                break
            except Exception:
                break
    return False


CST_TEMPLATE = '''You are an expert about question answering. I will provide a question, a generated solution (thinking process) and the final answer. Please evaluate the Consistency between the thinking process and the final answer: if the final answer is logically derived from the thinking process without contradiction, rate the Consistency as 1, else rate 0.
Here is the question, the generated thinking and the final answer for you to evaluate:\n

#### Question:
input_question\n

#### Generated Thinking (last part):
gen_thinking\n

#### Final Answer:
gen_answer\n

### Output Format (strictly follow)

Please provide an integer score to indicate the Consistency. Output the score in a JSON dictionary with nothing else for easy processing, in this form: {"Consistency": score}.

Your Evaluation Result:
'''

def llm_judge_json(prompt: str, max_try: int = 3) -> dict:
    import json
    if not client or not model_name: return {}
    messages = [{"role": "user", "content": prompt}]
    for _ in range(max_try):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                seed=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = resp.choices[0].message.content.strip()
            if "```" in text:
                matches = re.findall(r'```(.*?)```', text, re.DOTALL)
                if len(matches) == 1: text = matches[0]
            start = text.find('{')
            end = text.find('}', start + 1) if start != -1 else -1
            if start != -1 and end != -1:
                json_str = text[start:end + 1].replace('\\', '\\\\')
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f" [llm_judge_json] failed: {e}")
            continue
    return {}

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> dict:
    is_format_error = False
    count_think_1 = solution_str.count("<think>")
    count_think_2 = solution_str.count("</think>")
    if count_think_1 != count_think_2: is_format_error = True
    count_answer_1 = solution_str.count("<answer>")
    count_answer_2 = solution_str.count("</answer>")
    if count_answer_1 != count_answer_2: is_format_error = True
    code_call_1 = solution_str.count("<code>")
    code_call_2 = solution_str.count("</code>")
    if code_call_1 != code_call_2: is_format_error = True
    
    final_turn_ans = solution_str.split("assistant")[-1].strip()
    predict_no_think = final_turn_ans.split('</think>')[-1].strip()
    answer_text = extract_answer(final_turn_ans)
    if not answer_text:
        is_format_error = True
        answer_text = predict_no_think.strip()

    # Evaluate correctness using LLM judge
    question_text = extra_info.get("question", "") if extra_info else ""
    question_text = question_text.replace("<image>", "").strip()
    num_turns = extra_info.get("num_turns", 0) if extra_info else 0
    is_truncated = extra_info.get("is_truncated", False) if extra_info else False
    num_tools = num_turns // 2 - 1
    tool_error_num = solution_str.count("[Code Error]")
    subagent_rej_num = solution_str.count("[Judgment: REJECT]")
    subagent_acc_num = solution_str.count("[Judgment: ACCEPT]")
    acc_reward, format_reward, tool_reward, cst_reward, ratio = 0.0, 0.0, 0.0, 0.0, 0.0

    # Answer accuracy
    student_answer = extract_characters_regex(answer_text)
    if "$\boxed" in student_answer: student_answer = student_answer.replace("$\boxed", "$\\boxed")
    if rule_math_verify(student_answer, ground_truth): acc_reward = 1.0
    else: # LLM fallback
        if not client or not model_name:
            logger.warning("Reward function client not initialized or model name not found.")
            return {"score": 0.0, "acc_reward": acc_reward, "format_reward": format_reward, "cst_reward":cst_reward, "tool_reward": tool_reward, "ratio": ratio}
        system_prompt = (
            "You are an expert evaluator. Your task is to determine if a model's answer is semantically equivalent to a provided standard answer, given a specific question.\n"
            "Your evaluation must be strict. The model's answer is only correct if it fully matches the meaning of the standard answer.\n"
            'You **must** provide your final judgement as a **single word**: either "CORRECT" or "INCORRECT". Do not provide any explanation or other text.'
        )
        user_prompt = (
            f"I will provide a question, a standard answer, and a model's answer. You must evaluate if the model's answer is correct.\n\n"
            f"---\n"
            f"**Example 1:**\n"
            f"[Question]: Is the countertop tan or blue?\n"
            f"[Standard Answer]: The countertop is tan.\n"
            f"[Model's Answer]: tan\n"
            f"[Your Judgement]: CORRECT\n"
            f"---\n"
            f"**Example 2:**\n"
            f"[Question]: Is the man phone both blue and closed?\n"
            f"[Standard Answer]: Yes, the man phone is both blue and closed.\n"
            f"[Model's Answer]: No.\n"
            f"[Your Judgement]: INCORRECT\n"
            f"---\n"
            f"**Task:**\n"
            f"[Question]: {question_text}\n"
            f"[Standard Answer]: {ground_truth}\n"
            f"[Model's Answer]: {answer_text}\n"
            f"[Your Judgement]:"
        )
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                seed=0,
                temperature=0.1,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            response = chat_response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f" [WARNING] Chat completion request failed: {e}")
            return {"score": 0.0, "acc_reward": acc_reward, "format_reward": format_reward, "cst_reward":cst_reward, "tool_reward": tool_reward, "ratio": ratio}
        # Parse LLM judge response
        if re.search(r"\bCORRECT\b", response, re.IGNORECASE): acc_reward = 1.0
        elif re.search(r"\bINCORRECT\b", response, re.IGNORECASE): acc_reward = 0.0
        else:
            logger.warning(f" [WARNING] Judgement format error. Expected 'CORRECT' or 'INCORRECT', but get: {response.strip().replace("\n", " ").replace("\b", " ")}")
            return {"score": 0.0, "acc_reward": acc_reward, "format_reward": format_reward, "cst_reward":cst_reward, "tool_reward": tool_reward, "ratio": ratio}
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True
    # Logical consistency
    if acc_reward > 0.5:
        thinking = extract_thinking(final_turn_ans)
        prompt = CST_TEMPLATE.replace("input_question", question_text).replace("gen_thinking", thinking).replace("gen_answer", answer_text)
        result = llm_judge_json(prompt)
        cst_score = float(int(result.get("Consistency", 0)))
        cst_reward = 0.0 if cst_score > 0.0 else -1.0

    # Check tool usage - look for code_call/sandbox_output patterns instead of vision tokens
    has_code_usage = bool(re.search(r"<code>.*?</code>", solution_str, re.DOTALL) and re.search(r"<sandbox_output>.*?</sandbox_output>", solution_str, re.DOTALL))
    answer_match = re.search(r"<answer>(.*?)</answer>", final_turn_ans, re.DOTALL)
    if has_code_usage and answer_match:
        tool_usage_match = re.search(r"<code>.*?</code>", solution_str, re.DOTALL)
        sandbox_output_match = re.search(r"<sandbox_output>.*?</sandbox_output>", solution_str, re.DOTALL)
        answer_match_in_solution_str = re.search(r"<answer>.*?</answer>", solution_str, re.DOTALL)
        if bool(tool_usage_match): has_code_usage = (tool_usage_match.start() < answer_match_in_solution_str.start())
        if bool(sandbox_output_match): has_code_usage = has_code_usage and (sandbox_output_match.start() < answer_match_in_solution_str.start())

    # Final weighted score: acc_reward / format_reward / tool_reward / cst_reward
    format_reward = -1.0 if is_format_error else 0
    tool_reward = 1.0 if has_code_usage else 0.0
    if num_tools <= 0: ratio = 0.0
    else: ratio = max(0.0, 1.0-(tool_error_num+subagent_rej_num)/num_tools)
    final_score = acc_reward + 0.2 * format_reward + 0.2 * cst_reward + 0.2 * tool_reward * ratio

    if is_truncated: final_score = final_score - 0.2
    return {"score": final_score, "acc_reward": acc_reward, "format_reward": format_reward, "cst_reward":cst_reward, "tool_reward": tool_reward, "ratio": ratio}
