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

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **reward_kwargs) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            reward_kwargs: Additional keyword arguments. Currently supports `num_workers` (int) to control the
                concurrency of reward scoring (useful when compute_score issues blocking HTTP calls, e.g. an
                LLM-as-a-judge). Defaults to the env var `REWARD_NUM_WORKERS`, or 64 if unset.
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        # Concurrency for reward scoring. compute_score is typically I/O-bound (remote LLM judge
        # calls), so a thread pool can dramatically cut the per-step reward time. Overridable via
        # reward_kwargs.num_workers or the REWARD_NUM_WORKERS env var.
        self.num_workers = reward_kwargs.get(
            "num_workers", int(os.environ.get("REWARD_NUM_WORKERS", "32"))
        )

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        def _score_one(i):
            """Decode and score a single sample. Designed to run concurrently in a thread pool:
            `compute_score` typically issues blocking HTTP calls to a remote LLM judge, which is
            I/O-bound and thus benefits from concurrency. Decoding is cheap and done here to avoid
            sharing the per-sample tensors across threads."""
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            max_response_length = response_ids.shape[-1]
            is_truncated = bool(valid_response_length.item() >= max_response_length)

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            tool_bboxs = data_item.non_tensor_batch.get("__tool_bboxs__", None)
            extra_info["num_turns"] = num_turns
            extra_info["tool_bboxs"] = tool_bboxs
            extra_info["is_truncated"] = is_truncated
            extra_info["response_length"] = int(valid_response_length.item())
            extra_info["max_response_length"] = int(max_response_length)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            return i, prompt_str, response_str, ground_truth, data_source, int(valid_response_length.item()), score

        # Score all samples concurrently. Results are collected in input order so that the
        # reward_tensor / reward_extra_info layout is identical to the original serial loop.
        num_workers = max(1, int(self.num_workers))
        results = [None] * len(data)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_score_one, i) for i in range(len(data))]
            for future in futures:
                i, prompt_str, response_str, ground_truth, data_source, valid_response_length, score = future.result()
                results[i] = (prompt_str, response_str, ground_truth, data_source, valid_response_length, score)

        for i, (prompt_str, response_str, ground_truth, data_source, valid_response_length, score) in enumerate(results):
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
