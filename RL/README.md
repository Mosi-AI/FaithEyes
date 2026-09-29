<h1 align="center">FaithEyes: Towards Faithful Tool Use via<br/>Multi-Agent Process-Image Self-Verification</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2607.28225"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="arXiv"></a>
  <a href="https://hf.co/collections/Jackwang111/faitheyes"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffd21e.svg" alt="Hugging Face"></a>
  <a href="https://modelscope.cn/collections/whq1111/FaithEyes"><img src="https://img.shields.io/badge/ModelScope-Models-5f4bff.svg" alt="ModelScope"></a>
  <a href="https://github.com/Mosi-AI/FaithEyes"><img src="https://img.shields.io/badge/GitHub-Code-181717.svg?logo=github" alt="GitHub"></a>
</p>

<p align="center">
  <b>Haoqing Wang, Xingrun Xing, Ziheng Li, Jianyuan Guo, Yehui Tang<sup>✉</sup></b><br/>
  Samsung Research, Beijing &nbsp;&nbsp;·&nbsp;&nbsp; Peking University &nbsp;&nbsp;·&nbsp;&nbsp; City University of Hong Kong
</p>

<p align="center">This is the <b>reinforcement learning</b> component of <a href="../README.md">FaithEyes</a>.</p>

---

## 🔍 Overview

This directory contains the RL trainer of FaithEyes, built on a [verl](https://github.com/volcengine/verl) fork. After the cold-start [SFT](../SFT/README.md) stage, RL with **GRPO** reinforces generalization and robustness: the model rolls out multi-turn trajectories (main-agent tool calls + subagent judgements) in a sandboxed agent loop, and each trajectory is scored by a 4-component reward,

r(τ) = r_acc + 0.2·r_fmt + 0.2·r_cons + 0.2·r_tool,

so that answer correctness remains the dominant signal and the tool term acts as a faithfulness regularizer:

| Reward | Range | Description |
| :--- | :---: | :--- |
| **Accuracy** `r_acc` | {0, 1} | rule-based exact / programmatic matching of the final answer against the ground truth, with an LLM-as-judge fallback for semantic equivalence |
| **Format** `r_fmt` | {0, −1} | enforces the `<think>` / `<code>` / `<answer>` output structure so that each component (especially the executable code) can be reliably parsed |
| **Consistency** `r_cons` | {0, −1} | judges whether the final answer is logically entailed by the preceding reasoning, rather than appended as an unsupported guess |
| **Tool** `r_tool` | [0, 1] | the faithfulness-targeting term: scaled by the **helpful-tool ratio** `1 − (n_fail + n_unhelpful) / n_tool`, where `n_fail` counts calls that fail to execute and `n_unhelpful` counts calls judged unhelpful by the subagent — so decorative calls earn nothing |

Note that the tool reward is **decoupled from answer correctness**: gating it on a correct answer (as some prior works do) entangles the tool signal with the accuracy signal and leads to an up-to-18% code-failure spike and eventual collapse of tool use (see the paper appendix).

Tool observations (process images + judgements) are excluded from token counts and advantage estimation, since they are environmental feedback. All model-generated code runs in an isolated Python sandbox with read-only access to the input image (static scan of dangerous operations, wall-clock timeout, working-directory normalization, crop-coordinate clamping, pre-imported vision libraries).

## ⚙️ Installation

### Prerequisites

- Python 3.12 (a `cpython-3.11`+ environment works; pyc caches exist for 3.11–3.13)
- CUDA 12.6+, NVIDIA driver 570+
- The `tesseract` binary on PATH (used by the sandbox for OCR in generated code)

### Setup

```bash
# 1. Install PyTorch with the matching CUDA build first
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# 2. Install the verl package (this repo)
cd RL
pip install -e . --no-deps

# 3. Install pinned dependencies
pip install -r requirements.txt
```

Key pins: `vllm==0.11.0` (async server rollout), `transformers==4.57.1` (Qwen2.5-VL / Qwen3-VL support), `flash-attn==2.8.3`, `ray==2.53.0`.

## 🚀 Training

```bash
# set the llm-as-a-judge url
export LLM_AS_A_JUDGE_BASE="http://xxx:xxx/v1"

# set the concurrency level of reward calculations
export REWARD_NUM_WORKERS=32

# set the tensorboard dir
export TENSORBOARD_DIR="/path/to/tensorboard_dir"

PROJECT_NAME="VLM_Agent"
EXPERIMENT_NAME="FaithEyes"

# save experiment results
SAVE_CHECKPOINT_DIR="/path/to/verl_checkpoints/"
DATASET_TRAIN="/path/to/train.parquet"
DATASET_VAL="/path/to/eval.parquet"

REF_MODEL_PATH="/path/to/init_hf_model"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=./recipe/vlm_subagent/configs \
    --config-name='vlm_subagent_multiturn_grpo' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=[${DATASET_VAL}] \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=20480 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=12 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=null \
    trainer.critic_warmup=0 \
    trainer.logger=['tensorboard'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$WORLD_SIZE \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.max_critic_ckpt_to_keep=5 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto
```

After training, you can convert the FSDP checkpoint to HuggingFace format for evaluation:

```bash
python convert_fsdp_to_hf.py /path/to/fsdp_checkpoint /path/to/output_dir /path/to/ref_hf_checkpoint
```

## 🗂️ Repository Structure

```
RL/
├── requirements.txt                # pinned dependencies for the working environment
├── convert_fsdp_to_hf.py           # FSDP → HuggingFace checkpoint converter
├── recipe/vlm_subagent/
│   ├── vlm_subagent.py             # dataset + reward function (acc/format/consistency/tool)
│   └── configs/
│       ├── vlm_subagent_multiturn_grpo.yaml   # GRPO trainer config
│       └── vlm_subagent_tool_config.yaml      # vlm_coding tool schema & sandbox pool
├── verl/
│   ├── experimental/agent_loop/
│   │   ├── tool_agent_loop.py      # multi-turn agent loop (main agent + subagent calls)
│   │   └── tool_parser.py          # <code> block extraction
│   ├── tools/vlm_subagent_tool.py  # sandbox execution + subagent judgement injection
│   ├── trainer/ppo/                # GRPO trainer, core algorithms, reward pipeline
│   └── workers/                    # actor (FSDP), reward manager, vLLM rollout
└── scripts/                        # config generation, rollout viewer, model merger
```

## 📚 Citation

If you find this work useful, please consider citing:

```bibtex
@article{wang2026faitheyes,
  title={FaithEyes: Towards Faithful Tool Use via Multi-Agent Process-Image Verification},
  author={Wang, Haoqing and Xing, Xingrun and Xia, Wei and Li, Ziheng and Tang, Yehui},
  journal={arXiv preprint arXiv:2607.28225},
  year={2026}
}
```

## 🙏 Acknowledgements

This project is developed on top of [verl](https://github.com/volcengine/verl), the [DeepEyes](https://github.com/Visual-Agent/DeepEyes-RL) training recipe, and [Thyme](https://github.com/Kwai-Keye/Thyme). We thank their authors for open-sourcing these projects.
