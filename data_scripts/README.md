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

<p align="center">This is the <b>data construction</b> component of <a href="../README.md">FaithEyes</a>.</p>

---

## 🔍 Overview

The SFT stage of FaithEyes cold-starts three capabilities before RL: (i) **code-based problem solving**, (ii) **faithfulness judging** (emitting the `{"is_helpful": ..., "reasons": ...}` verdict as JSON), and (iii) **feedback-driven reasoning** that conditions on the subagent's judgement rather than ignoring it.

This directory contains the pipeline that builds the corresponding SFT data. It adapts the open-source SFT trajectories from [Thyme](https://github.com/Kwai-Keye/Thyme), which include single- and two-tool-call trajectories. Their structure provides natural ground-truth judgement: the sole call in a single-tool-call trajectory is labeled `is_helpful=true`, and in a two-tool-call trajectory the first call is labeled `is_helpful=false` and the second `is_helpful=true`. On top of these labels, the scripts in this directory:

1. **Make every code block standalone** (`rewrite_turn2.py`) so that each block executes correctly in isolation, matching the RL-time sandbox where no state carries over between calls;
2. **Write the judgement rationale** for each ground-truth label (`judge_force.py`) with a large VLM (Qwen3-VL-235B-A22B-Instruct), producing the subagent supervision;
3. **Rewrite the reasoning** that follows each judgement (`rewrite_thinking.py`) so the main agent explicitly reacts to the verdict — helpful images are returned together with their judgement, unhelpful ones are discarded and replaced by their judgement only.

The final training set interleaves two supervision sources: problem-solving trajectories for the main agent and judgement trajectories for the subagent (both played by the same model, distinguished only by the system prompt), which are trained jointly in the [SFT](../SFT/README.md) trainer.

## 🧪 Scripts

### A. Standalone code rewriting: `rewrite_turn2.py`

In `len(response)==5` trajectories (`[assistant, user(judgement), assistant, user(judgement), assistant]`), the second code block often depends on the first one: it loads the first call's `processed_path` (making crop coordinates *relative* to the first crop), inherits variables (`image`, `temp_dir`, `filename`, ...), and inherits imports (`os`, `cv2`, `uuid4`, ...). During RL every code block runs independently in the sandbox, so such code would silently break.

The script classifies each second code block and rewrites it to be fully standalone:

| Category | Diagnosis | Rewrite |
|---|---|---|
| **A** | loads from `processed_path` → coords are relative to the first crop | replace with `image_path` loading; convert crop coords to absolute (`first-crop offset + relative`) |
| **B** | already loads from `image_path` with absolute coords | keep as-is; only add missing imports/variables |
| **C** | uses an inherited `image` variable (coords already absolute) | prepend standalone image loading (`cv2.imread(image_path)` / `Image.open(image_path)`) |
| **D** | other / unparseable | skip |

Missing imports and variable definitions are carried over from the first code block via AST-based dependency analysis (handling f-strings, tuple unpacking, and chained dependencies) and topologically sorted so definitions precede uses. Records whose image files are missing on disk are filtered out; non-`len=5` records pass through unchanged.

Usage — paths are configured at the top of the file:

```python
INPUT_FILE  = '/path/to/input.jsonl'
OUTPUT_FILE = '/path/to/output.jsonl'
STATS_FILE  = '/path/to/rewrite_stats.txt'
```
```bash
python3 rewrite_turn2.py
```

A statistics report (category counts, rewrite coverage of `len=5` records, skip reasons) is written to `STATS_FILE`.

### B. Judgement rationale generation: `judge_force.py`

Given the ground-truth helpful/unhelpful label of each tool-produced image, this script prompts a VLM served on an OpenAI-compatible endpoint (vLLM, e.g. Qwen3-VL-235B-A22B-Instruct) to write the matching rationale, using the subagent judging prompt (system prompt + JSON output format). To keep the generated rationale consistent with the ground-truth label, a hint is injected into the user prompt — `hint_accept` for helpful images, `hint_reject` for unhelpful ones (the reason must not mention the hint; for images that do contain the target object, the reason should focus on imprecise cropping). Which hint to use is selected by whether `"accept"` appears in the input filename, so split the data into `..._accept.jsonl` / `..._reject.jsonl` files first.

Input is a JSONL where each row has:

```json
{"question": "...", "images": ["/path/to/original.jpg", "/path/to/tool_image.jpg"], "label": true}
```

Output rows append the LLM judgement:

```json
{"index": 0, "question": "...", "original_image": "...", "tool_image": "...",
 "human_label": true, "llm_is_helpful": true, "llm_reasons": "..."}
```

Usage:

```bash
# serve the judge model first, then:
python3 judge_force.py \
    --input  /path/to/judge_input_accept.jsonl \
    --output /path/to/judge_output_accept.jsonl \
    --model default --concurrency 64
```

The judge endpoint URL (`JUDGE_URL`) is configured at the top of the file. The script preloads and base64-encodes all tool images in batches, runs requests asynchronously with bounded concurrency and retries, and parses the JSON verdict defensively (markdown fences, escape fixing, regex fallback).

### C. Reasoning rewriting: `rewrite_thinking.py`

This step makes the main agent's reasoning *reference* the preceding subagent judgement, so that subsequent reasoning demonstrably conditions on the verdict. For every assistant turn that directly follows a `<sandbox_output>` judgement turn (in `len(response)==3` and `len(response)==5` rows; `len==1` rows pass through unchanged), a short opening sentence is **prepended in front of the original thinking, which is kept verbatim** — the agent's own visual evidence and decision reasoning are never rewritten.

Two modes:

- `--mode llm` (default): a VLM writes one natural reference sentence. When the judgement was helpful (`<sandbox_output>` contains an `<image>` placeholder), the processed image (the 2nd entry of the row's `image` field) is sent as a multimodal input so the sentence is grounded in the actual image; otherwise the sentence reacts to the rejection. The raw reply is sanitized to a single clean sentence (code fences / think tags / quotes stripped, truncated at the first sentence end, length-capped).
- `--mode prepend`: deterministic, no LLM calls — a short fixed reference to the verdict/reason (e.g. *"The judgement indicates the previous result was not helpful: ..."*) is prepended.

Usage:

```bash
python3 rewrite_thinking.py \
    --input       /path/to/input.jsonl \
    --output      /path/to/output.jsonl \
    --mode        llm \
    --base-url    http://localhost:8000/v1 \
    --concurrency 32
```

The model id is auto-detected from the endpoint if `--model` is empty. The script supports **resume** (completed source-line indices are recorded in the output and skipped on rerun), writes results strictly in input order through a single writer coroutine, flushes incrementally, and shuts down gracefully on SIGINT/SIGTERM after draining in-flight rows — so an interrupted run never leaves gaps in the output file.

## 📦 Output format

The resulting SFT data follows the format consumed by the [SFT](../SFT/README.md) trainer: `system` / `question` / `response` (a multi-turn list where even indices are assistant turns and odd indices are the tool observations returned as user turns) / `image` (paths mapped one-to-one onto the `<image>` placeholders in order of first appearance). Problem-solving trajectories (main agent system prompt) and judgement trajectories (subagent system prompt, single assistant turn containing only the verdict JSON) are mixed into one file.

## 🗂️ Repository Structure

```
data_scripts/
├── rewrite_turn2.py     # make len=5 second code blocks standalone (imports, vars, absolute coords)
├── judge_force.py       # LLM-generated judgement rationales for ground-truth helpful/unhelpful labels
└── rewrite_thinking.py  # prepend judgement-reference sentences to the agent's thinking blocks
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

The source SFT trajectories are adapted from [Thyme](https://github.com/Kwai-Keye/Thyme). We thank their authors for open-sourcing these projects.
