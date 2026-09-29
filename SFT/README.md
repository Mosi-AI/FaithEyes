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

<p align="center">This is the <b>cold-start SFT</b> component of <a href="../README.md">FaithEyes</a>.</p>

---

## 🔍 Overview

This directory contains the cold-start supervised fine-tuning (SFT) trainer of FaithEyes, built on an [ms_swift](https://github.com/modelscope/ms-swift) fork adapted to `transformers 5.x`. The SFT stage cold-starts three capabilities prior to RL, so that the RL stage starts from a policy that already knows how to:

1. **Solve visual problems with code** — some base VLMs (e.g. Qwen2.5-VL-7B-Instruct) do not natively write code to solve visual problems, so this ability must be bootstrapped by SFT;
2. **Judge tool faithfulness** — emit the `{"is_helpful": ..., "reasons": ...}` verdict as JSON, with preferences aligned on clarity and crop concentration (e.g. neither judging a large lazy crop as helpful, nor judging a precise crop of a small object as unhelpful just because of its low native resolution);
3. **Reason from the judgement** — condition subsequent reasoning on the subagent's verdict rather than ignore it.

The main agent and subagent trajectories are trained **jointly in a single stage** in the same model, distinguished only by their system prompts. See [`data_scripts/`](../data_scripts/README.md) for how the training data is constructed.

## ⚙️ Installation

### Prerequisites

The verified working environment:

- Python 3.11
- CUDA 12.6 (driver for `torch 2.7.1+cu126`)

### Setup

```bash
# 1. Install PyTorch with the matching CUDA build first
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

# 2. Install the ms_swift package (this repo, a fork adapted to transformers 5.x)
cd SFT
pip install -e . --no-deps

# 3. Install pinned dependencies
pip install -r requirements.txt
```

Key pins: `transformers==5.16.1` (Qwen3-VL support), `flash-attn==2.8.3`, `deepspeed==0.16.9`, `peft==0.15.2`,
`qwen-vl-utils==0.0.10` + `decord==0.6.0` (Qwen3-VL vision preprocessing), `trl==0.20.0`.

## 🚀 Training

The SFT script is as follows:

```bash
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

export NPROC_PER_NODE=$NPROC_PER_NODE
export OMP_NUM_THREADS=8

output_dir="/path/to/output_dir"

FPS_MAX_FRAMES=10 \
MAX_PIXELS=3211264 \
swift sft \
    --model /path/to/init_hf_model \
    --dataset /path/to/train.jsonl \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --freeze_vit true \
    --freeze_parameters_regex 'model\.visual\.' \
    --gradient_accumulation_steps 4 \
    --save_strategy epoch \
    --max_length 10240 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --output_dir $output_dir \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    --report_to tensorboard \
    --ddp_backend nccl
```

Two masking rules are essential for a stable cold start (already handled by the trainer):

1. **Tool observations are masked out** from the loss (the returned process images, sandbox text output, and subagent judgement), so the model learns to produce code and answers rather than to predict environment feedback;
2. **For two-call problem-solving trajectories, the loss is computed only on the last-call response** and the preceding call is masked, which prevents the model from imitating the "deliberately-wrong-then-correct" pattern.

### Training data format

The dataset mixes **two kinds of trajectories**, both distilled into the same model (which plays both roles — main agent and subagent — distinguished only by the system prompt):

1. **Problem-solving trajectories** train the **main agent**: the `system` field holds the tool-use protocol with the critic-subagent instruction; `question` is the user turn; `response` is a multi-turn list `[assistant, user, assistant, user, ..., assistant]` of odd length (1, 3, 5 and so on) — even indices are assistant turns, odd indices are the tool observations returned as user turns. Single-turn records are direct answers without any tool call, teaching the model not to use the tool unnecessarily.
2. **Process-judging trajectories** train the **subagent**: the `system` field holds the judge prompt (is-helpful guidelines + JSON output format); `question` carries the original question plus the process image; `response` is always a single assistant turn containing only the verdict JSON. Every judging record has exactly one image (the process image to be judged).

In both kinds, `<image>` is the image placeholder: placeholders in `question` and in the elements of `response` are consumed **in order of first appearance** and mapped one-to-one onto the paths in `image`. In a problem-solving record the first placeholder (in `question`) is the user's original image and subsequent ones (in the tool-observation turns) are the process images produced by the sandbox, so a record with two images has exactly two placeholders in total.

## 🗂️ Repository Structure

```
SFT/
├── requirements.txt      # pinned dependencies for the working environment
├── setup.py              # installs the ms_swift package (entry point: `swift`)
└── swift/                # ms_swift fork, adapted to transformers 5.x
    ├── llm/
    │   ├── argument/     # swift CLI dataclasses (sft/lora/full args, train args)
    │   ├── dataset/      # dataset loading, preprocessing, media handling
    │   ├── model/        # model registry (Qwen3-VL meta, arch keys, patcher)
    │   ├── template/     # chat templates incl. Qwen3-VL (patch-aligned fetch_image)
    │   └── train/        # sft_main / pt_main pipelines
    ├── trainers/         # TrainerFactory, GRPO trainer (RL, sandbox), mixins
    └── tuners/           # LoRA / full / partial freeze logic
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

This project is developed on top of [ms_swift](https://github.com/modelscope/ms-swift) and [Thyme](https://github.com/Kwai-Keye/Thyme). We thank their authors for open-sourcing these projects.
