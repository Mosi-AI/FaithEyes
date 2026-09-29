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

<p align="center">This is the <b>evaluation</b> component of <a href="../README.md">FaithEyes</a>.</p>

---

## 🔍 Overview

This directory contains the evaluation harness of FaithEyes, built on a [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) fork. It evaluates agentic VLMs served on an OpenAI-compatible endpoint (vLLM), supporting multi-turn tool use: the model's `<code>` blocks are executed in a sandbox, and each returned process image is judged by the model itself under the subagent prompt, whose verdict is injected back into the tool observation — exactly the same interaction protocol as training and inference.

We evaluate six benchmarks covering visual perception and reasoning (Avg@10):

| Category | Benchmarks |
| :--- | :--- |
| **Perception** | [V<sup>*</sup> Bench](https://huggingface.co/datasets/penghao/VStarBench), [HR-Bench 4K](https://huggingface.co/datasets/lmms-lab/HRBench4K), [HR-Bench 8K](https://huggingface.co/datasets/lmms-lab/HRBench8K) |
| **Reasoning** | [MathVista](https://huggingface.co/datasets/AI4Math/MathVista), [MathVerse](https://huggingface.co/datasets/MathVerse-cuhk/MathVerse), [MathVision](https://huggingface.co/datasets/MathVision-cuhk/MathVision) |

## ⚙️ Installation

A full dependency list is provided in [`requirements.txt`](./requirements.txt) for reference.

## 🚀 Reproduce the results

First, start the vLLM service for the model to be tested:

```bash
MODEL_PATH=$1
PORT=$2

vllm serve $MODEL_PATH \
    --port $PORT \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 8 \
    --enforce-eager
```

Then run the test script:

```bash
export OPENAI_API_KEY=sk-xxx  # must start with "sk-"
export OPENAI_API_BASE=http://xxx:xxx/v1  # endpoint of the LLM judge (e.g. Qwen2.5-VL-72B-Instruct)
export LMUData=/path/to/LMUData/

python run.py --config faitheyes_eval.json --work-dir /path/to/workdir --port $PORT --reuse
```

If you want to evaluate the tool faithfulness ratio, you can run
```bash
python judge_faithfulness.py --process-dir /path/to/process_images/benchmark
```

### Configuration

[`faitheyes_eval.json`](./faitheyes_eval.json) defines the evaluated model and datasets:

- **Model** — the `faitheyes` entry uses the `LMDeployAPIWithToolUse` class with `use_tool: true`: `<code>...</code>` blocks are extracted (`tool_start_token` / `tool_end_token`), executed in the sandbox, and the subagent judgement is returned wrapped in `<sandbox_output>...</sandbox_output>`. The main-agent system prompt (matching training) is included in the config; `save_process_images` controls whether tool-produced images are kept for calculating tool faithfulness ratio.
- **Data** — the six benchmark entries (`VStarBench`, `HRBench4K`, `HRBench8K`, `MathVista_MINI`, `MathVerse_MINI`, `MathVision_MINI`) are resolved by the underlying VLMEvalKit dataset classes; downloaded data is stored under `LMUData`.

## 🗂️ Repository Structure

```
VLMEvalKit/
├── run.py                  # entry point: build model & datasets from config, run inference & scoring
├── faitheyes_eval.json     # model (LMDeployAPIWithToolUse) + benchmark configuration
├── requirements.txt        # pinned dependencies for the working environment
└── vlmeval/                # VLMEvalKit fork
    ├── api/lmdeploy.py     # LMDeployAPIWithToolUse: multi-turn tool-use inference loop
    ├── dataset/            # benchmark dataset classes (V*, HR-Bench, MathVista, ...)
    └── ...                 # inference, scoring, and utility modules
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

This project is developed on top of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). We thank their authors for open-sourcing these projects.
