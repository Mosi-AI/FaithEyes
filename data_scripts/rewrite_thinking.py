#!/usr/bin/env python3
"""
Rewrite agent thinking blocks to reference the preceding subagent judgement.

For rows with len(response) == 3:
    response = [assistant1, user1(judgement), assistant2]
    -> rewrite assistant2's <think> using user1's judgement.
For rows with len(response) == 5:
    response = [assistant1, user1(judgement), assistant2, user2(judgement), assistant3]
    -> rewrite assistant2's <think> using user1's judgement,
       and assistant3's <think> using user2's judgement.

len(response) == 1 rows are copied through unchanged.

The external LLM is an OpenAI-compatible endpoint (vLLM serving
Qwen3-VL-235B). Only text is sent (no images), so it works as a text model.

Features:
  - async + connection pool with bounded concurrency
  - per-turn retry with exponential backoff
  - resume: a progress file records completed source-line indices; on rerun
    those lines are skipped (read from the output file if present).
  - incremental writes (flush every N rows).
"""

import argparse
import asyncio
import aiohttp
import base64
import json
import mimetypes
import os
import random
import re
import sys
import time
from collections import deque

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Judgement verdict markers that the RL-time tool emits inside <sandbox_output>.
# We strip the machine tag ([Judgment: ACCEPT]/[Judgment: REJECT] ...) and keep
# the human-readable reason, so the reference sentence stays natural.
JUDGE_TAG_RE = re.compile(r"\[Judgment:\s*(ACCEPT|REJECT)\]", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clean_judge_reason(judgement):
    """Extract (verdict, reason) from a <sandbox_output> judgement block.

    verdict: "ACCEPT" | "REJECT" | "" (unknown)
    reason : the free-text reason with the machine tag and sandbox tags removed.
    """
    m = JUDGE_TAG_RE.search(judgement)
    verdict = m.group(1).upper() if m else ""
    # strip <sandbox_output>/</sandbox_output> and <image> placeholders
    body = judgement.replace("<sandbox_output>", "").replace("</sandbox_output>", "")
    body = JUDGE_TAG_RE.sub("", body)
    body = body.replace("<image>", " ")
    # collapse whitespace
    body = " ".join(body.split())
    # drop leading punctuation/dashes left after removing the tag
    body = body.lstrip(" :-–—").strip()
    # keep it short: the reference sentence only needs the gist
    if len(body) > 220:
        body = body[:220].rsplit(" ", 1)[0].rstrip(" ,;") + "..."
    return verdict, body


def build_prepend_reference(verdict, reason):
    """A short, natural one-sentence reference to the judgement.

    This is prepended in front of the ORIGINAL thinking, which is kept intact,
    so the model keeps its own visual evidence and decision reasoning.
    """
    reason = reason.strip()
    if verdict == "REJECT":
        if reason:
            return f"The judgement indicates the previous result was not helpful: {reason} "
        return "The judgement indicates the previous result was not helpful. "
    # ACCEPT / unknown: reference lightly, without overclaiming
    if reason:
        return f"As the judgement noted, {reason[0].lower() + reason[1:] if reason else reason} "
    return "As the judgement noted. "
def split_thinking(text):
    """Return (thinking, rest) for an assistant turn.

    The turn is expected to start with `<think>...thinking...</think>`
    followed by the code/answer body. If the tag structure is unexpected we
    return (None, text) so the caller can skip rewriting safely.
    """
    if not text.startswith(THINK_OPEN):
        return None, text
    end = text.find(THINK_CLOSE)
    if end == -1:
        return None, text
    thinking = text[len(THINK_OPEN):end]
    rest = text[end + len(THINK_CLOSE):]
    return thinking, rest


def build_messages(question, system, judgement, original_thinking, rest, helpful_image_path=None):
    """Build the chat messages asking the LLM to rewrite the thinking.

    helpful_image_path: if not None, the preceding sandbox_output contained an
        <image> (i.e. is_helpful=True), and this path is the actual processed
        image (the 2nd entry of the row's `image` field). In that case the
        image is sent as a multimodal input and the prompt instructs the model
        to ground the rewritten thinking in the image content rather than in
        the judgement text. If None (is_helpful=False, no <image>), the model
        falls back to referencing the judgement text as before.
    """
    judge_clean = judgement.strip()
    has_image = helpful_image_path is not None

    if has_image:
        grounding_section = f"""- The processed image produced by the previous code operation (provided to you below as an actual image). The agent *should* be looking at this image.
<image: {helpful_image_path}>

- The subagent's judgement on this operation (it ACCEPTed the image as helpful):
<sandbox_output>
{judge_clean}
</sandbox_output>"""
        grounding_instruction = """1. The sentence should briefly connect the judgement's conclusion to the processed image (e.g. "The cropped image, as the judgement noted, now clearly shows ...")."""
    else:
        grounding_section = f"""- The subagent's judgement that was returned right before this reasoning block (there is NO processed image this time — the operation was judged unhelpful and its image discarded — so this is what the agent *should* be reacting to):
<sandbox_output>
{judge_clean}
</sandbox_output>"""
        grounding_instruction = """1. The sentence should briefly note the judgement's conclusion and that the previous result was unhelpful (e.g. "As the judgement noted, the previous crop did not isolate the target.")."""

    text_content = f"""You write ONE short opening sentence that references the result of the preceding code operation. This sentence will be placed in front of an existing reasoning block by the caller — you do NOT rewrite or repeat that block.

# Context
- System prompt (the agent's role):
{system}

- Original user question (image placeholders <image> refer to figures in the trajectory):
{question}

# Preceding operation result
{grounding_section}

# The agent's original reasoning block (for tone/language reference ONLY — do NOT include it in your output):
<thinking>
{original_thinking}
</thinking>

# Task
Write a SINGLE natural opening sentence that reacts to the judgement above, to be prepended before the agent's own reasoning. Concretely:
{grounding_instruction}
2. Keep it to exactly ONE sentence (occasionally two short clauses joined by a comma/semicolon are fine), and make the wording natural and varied — avoid the same canned opener every time.
3. Do NOT invent facts beyond the judgement{", image, and the agent's own reasoning" if True else ""}; do NOT restate the whole reasoning, just bridge from the judgement to it.
4. Match the language and tone of the agent's original reasoning block (e.g. first-person analytical voice).

Output ONLY the single reference sentence, with no `<thinking>`/`</thinking>` tags, no quotes, no preamble, no explanation."""

    # Build multimodal user content: text + (optional) image.
    user_content = [{"type": "text", "text": text_content}]
    if has_image:
        # vLLM does not allow loading local files by path; embed the image as
        # a base64 data URL instead.
        img_path = helpful_image_path[len("file://"):] if helpful_image_path.startswith("file://") else helpful_image_path
        mime, _ = mimetypes.guess_type(img_path)
        mime = mime or "image/jpeg"
        with open(img_path, "rb") as _f:
            b64 = base64.b64encode(_f.read()).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})

    messages = [
        {"role": "system", "content": "You are a careful editor that rewrites reasoning blocks to integrate the preceding operation's result (image or judgement) while preserving meaning. You output only the rewritten text."},
        {"role": "user", "content": user_content},
    ]
    return messages


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
class LLMClient:
    def __init__(self, base_url, model, timeout=180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)  # bounded by semaphore
        self.session = aiohttp.ClientSession(timeout=self.timeout, connector=connector)
        return self

    async def __aexit__(self, *exc):
        if self.session:
            await self.session.close()

    async def chat(self, messages, temperature=0.3, max_tokens=2048, max_retries=6):
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        backoff = 2.0
        last_err = None
        for attempt in range(max_retries):
            try:
                async with self.session.post(url, json=payload) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        # transient
                        text = await resp.text()
                        last_err = f"HTTP {resp.status}: {text[:200]}"
                    elif resp.status != 200:
                        text = await resp.text()
                        last_err = f"HTTP {resp.status}: {text[:300]}"
                        # non-transient: still retry a couple times then give up
                    else:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = repr(e)
            # backoff before next attempt
            await asyncio.sleep(backoff + random.uniform(0, 1))
            backoff = min(backoff * 2, 30)
        raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


# --------------------------------------------------------------------------- #
# Per-row processing
# --------------------------------------------------------------------------- #
def rewrite_targets(response):
    """Yield (assistant_index, preceding_user_judgement) for turns to rewrite."""
    n = len(response)
    if n == 1:
        return
    # user turns are at odd indices 1, 3, ...; the assistant right after them
    # is at even indices 2, 4, ...
    for u_idx in range(1, n, 2):
        a_idx = u_idx + 1
        if a_idx >= n:
            break
        judgement = response[u_idx]
        if "<sandbox_output>" not in judgement:
            continue
        yield a_idx, judgement


async def process_row(client, sem, row, max_tokens, mode="llm"):
    """Rewrite all eligible thinking blocks in one row. Returns (row, n_rewritten, n_failed).

    mode == "llm": (default) ask an LLM to prepend a short reference to the
        judgement while preserving the original thinking verbatim (the prompt
        forbids rewriting the body). Needs a running LLM endpoint.
    mode == "prepend": (no LLM call) keep the original thinking completely
        intact and only prepend a short, deterministic reference to the
        judgement's verdict/reason in front of it. The agent's own visual
        evidence and decision reasoning are never touched.
    """
    response = row["response"]
    images = row.get("image", [])
    n_rewritten = 0
    n_failed = 0
    targets = list(rewrite_targets(response))
    # Rewrite sequentially within a row (each rewrite is independent, but we
    # keep them ordered). Concurrency across rows is handled by the semaphore.
    for a_idx, judgement in targets:
        text = response[a_idx]
        thinking, rest = split_thinking(text)
        if thinking is None:
            continue

        verdict, reason = _clean_judge_reason(judgement)

        if mode == "prepend":
            # Deterministic, no LLM: original thinking preserved verbatim.
            reference = build_prepend_reference(verdict, reason)
            original = thinking.strip()
            new_thinking = reference + original
            response[a_idx] = f"{THINK_OPEN}{new_thinking}{THINK_CLOSE}{rest}"
            n_rewritten += 1
            continue

        # ---- mode == "llm": LLM prepends a reference, body left untouched ----
        # is_helpful=True  <=>  sandbox_output contains an <image> placeholder.
        # In that case the processed image is the 2nd entry of the row's
        # `image` field; send it as a multimodal input so the reference sentence
        # is grounded in the actual image rather than judgement text alone.
        helpful_image_path = None
        if "<image>" in judgement and len(images) >= 2:
            helpful_image_path = "file://" + images[1]
        messages = build_messages(
            row.get("question", ""),
            row.get("system", ""),
            judgement,
            thinking,
            rest,
            helpful_image_path=helpful_image_path,
        )
        async with sem:
            try:
                raw_reference = await client.chat(messages, max_tokens=max_tokens)
            except Exception as e:
                sys.stderr.write(f"[warn] rewrite failed (a_idx={a_idx}): {e}\n")
                n_failed += 1
                continue
        reference = _sanitize_reference(raw_reference)
        if not reference:
            sys.stderr.write(f"[warn] empty/invalid reference (a_idx={a_idx}); skipping\n")
            n_failed += 1
            continue
        # Program-side assembly: reference sentence + original thinking VERBATIM.
        new_thinking = reference + " " + thinking.strip()
        response[a_idx] = f"{THINK_OPEN}{new_thinking}{THINK_CLOSE}{rest}"
        n_rewritten += 1
    return row, n_rewritten, n_failed


# --------------------------------------------------------------------------- #
# Reference-sentence sanitisation (llm mode)
# --------------------------------------------------------------------------- #
def _sanitize_reference(raw):
    """Reduce an LLM reply to a single clean reference sentence.

    The LLM is asked to output ONLY the opening reference sentence, but it may
    occasionally echo extra text (the original thinking, quotes, markdown, or
    several sentences). We trim all of that so the caller can safely prepend it
    to the original thinking verbatim. Returns "" if nothing usable remains.
    """
    if not raw:
        return ""
    text = raw.strip()
    # strip code fences / think tags / surrounding quotes the model might add
    text = re.sub(r"```(?:[a-zA-Z]+)?", "", text)
    text = text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    text = text.strip().strip('"').strip("'").strip()
    # take only the first line if the model echoed multiple lines
    text = text.split("\n", 1)[0].strip()
    if not text:
        return ""
    # cut at the end of the FIRST sentence so we never drag in echoed body text
    m = re.search(r"(.+?[.!?。！？])", text)
    if m:
        text = m.group(1).strip()
    # safety cap: a reference sentence should be short
    if len(text) > 300:
        text = text[:300].rsplit(" ", 1)[0].rstrip(" ,;") + "..."
    return text


# --------------------------------------------------------------------------- #
# Resume / progress
# --------------------------------------------------------------------------- #
def load_done_indices(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                idx = obj.get("_src_idx")
                if idx is not None:
                    done.add(idx)
            except json.JSONDecodeError:
                continue
    return done


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def run(args):
    total = count_lines(args.input)
    done = load_done_indices(args.output)
    if done:
        print(f"[resume] {len(done)} rows already written to {args.output}; skipping them.")

    # Discover model name if not given (only needed for --mode llm).
    model = args.model
    if args.mode == "llm" and not model:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{args.base_url.rstrip('/')}/models") as r:
                data = await r.json()
                models = [m["id"] for m in data.get("data", [])]
        if not models:
            sys.exit("No models found at endpoint and --model not given.")
        model = models[0]
        print(f"[model] using {model}")

    client = LLMClient(args.base_url, model, timeout=args.timeout)
    sem = asyncio.Semaphore(args.concurrency)
    if args.mode == "prepend":
        print("[mode] prepend: original thinking preserved; only a judgement reference is prepended (no LLM calls).")
    else:
        print(f"[mode] llm: LLM ({model}) prepends a judgement reference; original thinking kept verbatim.")

    # Rolling latency tracker for ETA.
    latencies = deque(maxlen=500)
    t0 = time.time()
    rewritten = 0
    failed = 0
    skipped = 0

    # ---- Ordered producer/consumer with graceful shutdown ----------------- #
    # All rows (len 1/3/5 alike) go through the same async path. Results are
    # buffered and written to disk in strict input order by a single writer
    # coroutine, so the output file is always a correct, contiguous prefix of
    # the run — even if the process is interrupted mid-flight, no rows are
    # silently dropped (the previous version wrote len==1 rows synchronously
    # and len>=3 rows asynchronously, so an interruption left only the len==1
    # rows on disk).
    results = {}            # idx -> processed row dict, awaiting ordered write
    next_to_write = 0       # next idx that should be flushed to file
    write_event = asyncio.Event()
    write_event.set()
    stop_event = asyncio.Event()

    loop = asyncio.get_event_loop()

    def _signal_handler(signum, *_):
        if not stop_event.is_set():
            sys.stderr.write(
                f"\n[signal] {signum} received; finishing in-flight rows then exiting...\n"
            )
            stop_event.set()
            write_event.set()

    import signal
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler, sig)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, _signal_handler)

    async def writer():
        """Single consumer that writes results to disk strictly in input order."""
        nonlocal next_to_write, skipped
        out_f = open(args.output, "a", encoding="utf-8")
        try:
            while True:
                # flush every contiguous completed prefix
                while next_to_write in results:
                    row = results.pop(next_to_write)
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    next_to_write += 1
                write_event.clear()
                # stop only after everything ready has been flushed
                if stop_event.is_set() and not results:
                    break
                try:
                    await asyncio.wait_for(write_event.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
        finally:
            out_f.flush()
            out_f.close()

    async def worker(idx, row):
        nonlocal rewritten, failed, skipped
        n = len(row["response"])
        if n == 1:
            # nothing to rewrite; pass through unchanged
            row["_src_idx"] = idx
            skipped += 1
            results[idx] = row
            write_event.set()
            return
        rt = time.time()
        try:
            new_row, nr, nf = await process_row(client, sem, row, args.max_tokens, mode=args.mode)
        except Exception as e:
            sys.stderr.write(f"[error] row {idx} crashed: {e}\n")
            new_row, nr, nf = row, 0, 0
        latencies.append(time.time() - rt)
        rewritten += nr
        failed += nf
        new_row["_src_idx"] = idx
        results[idx] = new_row
        write_event.set()

    async with client:
        writer_task = asyncio.create_task(writer())
        in_flight = set()
        idx = 0
        with open(args.input, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if stop_event.is_set():
                    break
                row = json.loads(line)
                if idx in done:
                    # already written in a previous run; advance the ordered
                    # write cursor past it so the writer stays in sync.
                    if idx == next_to_write:
                        next_to_write = idx + 1
                    done.discard(idx)
                else:
                    task = asyncio.create_task(worker(idx, row))
                    in_flight.add(task)
                    task.add_done_callback(in_flight.discard)
                    # Bound in-flight rows to keep memory bounded.
                    while len(in_flight) >= args.concurrency * 4:
                        await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                        write_event.set()
                idx += 1
                if idx % 500 == 0:
                    elapsed = time.time() - t0
                    rate = idx / elapsed if elapsed else 0
                    avg_lat = (sum(latencies) / len(latencies)) if latencies else 0
                    eta = (total - idx) / rate if rate else 0
                    print(
                        f"[progress] {idx}/{total} rows read "
                        f"({100*idx/total:.1f}%) | written={next_to_write} | "
                        f"rate={rate:.1f} rows/s | avg_row_lat={avg_lat:.2f}s | "
                        f"ETA={eta/3600:.2f}h | rewritten={rewritten} failed={failed} skipped={skipped}",
                        flush=True,
                    )

        # Input exhausted (or stop requested): drain in-flight workers.
        while in_flight:
            await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            write_event.set()
            if stop_event.is_set():
                # on signal, let remaining tasks finish one last drain
                await asyncio.gather(*in_flight, return_exceptions=True)
                break
        # tell writer to flush & exit
        stop_event.set()
        write_event.set()
        await writer_task

    print(
        f"[done] read {idx} rows | written={next_to_write} | rewritten={rewritten} | "
        f"failed={failed} | skipped(pass-through)={skipped} | output={args.output}"
    )


def main():
    p = argparse.ArgumentParser(description="Rewrite agent thinking blocks to reference subagent judgements.")
    p.add_argument("--input", default="/path/to/input.jsonl")
    p.add_argument("--output", default="/path/to/output.jsonl")
    p.add_argument("--mode", choices=["llm", "prepend"], default="llm",
                   help="llm: (default) ask an LLM to prepend a short reference to the judgement "
                        "while preserving the original thinking verbatim (more natural wording, "
                        "needs a running LLM endpoint). prepend: keep the original thinking intact "
                        "and only prepend a short deterministic reference (no LLM call).")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="", help="Model id; auto-detected if empty. Only used in --mode llm.")
    p.add_argument("--concurrency", type=int, default=32, help="Max concurrent LLM calls (only --mode llm).")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args()

    # If output exists but input changed, warn.
    if os.path.exists(args.output):
        print(f"[resume] appending to existing {args.output}")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
