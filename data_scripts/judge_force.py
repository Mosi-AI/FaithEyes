import json
import re
import asyncio
import aiohttp
import aiofiles
import argparse
import base64
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import functools

JUDGE_URL = "http://localhost:8000/v1"

critic_system_prompt = """You are evaluating whether a tool-called image is helpful for answering a visual question.

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
hint_reject = "Hint: This test case should be judged as **is_helpful=false**. You need to provide a suitable reason, and the reason **must not** mention judging based on the hint. For the image containing target objects, you **must** prioritize explaining the reasons from the perspective of imprecise cropping, such as not focus enough or the inclusion of other irrelevant targets."
hint_accept = "Hint: This test case should be judged as **is_helpful=true**. You need to provide a suitable reason, and the reason **must not** mention judging based on the hint."

# Thread pool for CPU-bound tasks (image encoding)
_executor = ThreadPoolExecutor(max_workers=8)

def get_image_media_type(image_path: str) -> str:
    """Get the media type based on file extension."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(ext, "image/jpeg")


def encode_image_sync(image_path: str) -> Optional[str]:
    """Synchronously encode an image file to base64 (runs in thread pool)."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to encode image {image_path}: {e}")
        return None


async def encode_image_async(image_path: str) -> Optional[str]:
    """Asynchronously encode an image using thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, encode_image_sync, image_path)


def parse_llm_response(response_text: str) -> dict:
    """Parse the LLM response to extract is_helpful and reasons."""
    text = response_text.strip()
    if "reasones" in text: text = text.replace("reasones", "reasons")

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) > 1:
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

    # Try to parse as JSON
    try:
        result = json.loads(text)
        if isinstance(result, dict) and "is_helpful" in result and "reasons" in result: return result
    except json.JSONDecodeError:
        pass

    # Try parsing after fixing escapes
    def fix_json_escapes(s: str) -> str:
        result = []
        i = 0
        in_string = False
        while i < len(s):
            char = s[i]

            # Track whether we're inside a JSON string
            if char == '"' and (i == 0 or s[i - 1] != '\\'):
                in_string = not in_string
                result.append(char)
                i += 1
            # Handle backslash inside string
            elif char == '\\' and in_string and i + 1 < len(s):
                next_char = s[i + 1]
                # Valid JSON escape sequences
                if next_char in {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}:
                    # Keep valid escape as-is
                    result.append(char)
                    result.append(next_char)
                    i += 2  # Skip both \ and the next char
                else:
                    # Invalid escape: convert \X to \\X
                    # This makes \( become \\( which represents \( in the final string
                    result.append('\\\\')
                    i += 1  # Only skip \, keep the next char for next iteration
            else:
                result.append(char)
                i += 1
        return ''.join(result)
    try:
        fixed_text = fix_json_escapes(text)
        result = json.loads(fixed_text)
        if isinstance(result, dict) and "is_helpful" in result and "reasons" in result: return result
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from the text using regex
    try:
        # More flexible pattern to find JSON object
        json_match = re.search(r'\{[^{}]*"is_helpful"[^{}]*\}', text, re.DOTALL)
        if json_match:
            raw_json = json_match.group(0)

            # Fix escapes before parsing
            fixed_json = fix_json_escapes(raw_json)

            # Try parsing the fixed JSON
            try:
                result = json.loads(fixed_json)
                if isinstance(result, dict) and "is_helpful" in result and "reasons" in result:
                    return result
            except json.JSONDecodeError:
                pass

            # Fallback: extract fields manually with more robust regex
            is_helpful_match = re.search(
                r'"is_helpful"\s*:\s*(true|false)', raw_json, re.IGNORECASE
            )
            is_helpful = (
                is_helpful_match.group(1).lower() == "true"
                if is_helpful_match
                else True
            )

            # Try to extract reasons - handle both string and array formats
            reasons_match = re.search(
                r'"reasons"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_json, re.DOTALL
            )
            reasons = reasons_match.group(1) if reasons_match else ""

            if reasons:
                # Unescape the reasons string
                reasons = reasons.replace('\\(', '(').replace('\\)', ')')
                reasons = reasons.replace('\\\\', '\\')
                return {"is_helpful": is_helpful, "reasons": reasons.strip()}
    except Exception:
        pass

    return {
        "is_helpful": None,
        "reasons": f"Failed to parse LLM response: {response_text.strip()}",
    }


async def judge_one(
    session: aiohttp.ClientSession,
    question: str,
    base64_image: str,
    media_type: str,
    model: str,
    task_flag: bool,
    semaphore: asyncio.Semaphore
) -> dict:
    user_prompt = f"""[Question]: {question}\n\nThe following image is a processed version of an initial image (possibly cropped, resized, rotated, or contrast-adjusted). Does this image show the object/attribute the question is asking about? Reply with ONLY one line of JSON."""
    if task_flag: user_prompt = hint_accept + "\n\n" + user_prompt
    else: user_prompt = hint_reject + "\n\n" + user_prompt

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": critic_system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{base64_image}"}}
                ]
            }
        ],
        "temperature": 0,
        "max_tokens": 512,
    }

    async with semaphore:
        for attempt in range(5):
            try:
                async with session.post(f"{JUDGE_URL}/chat/completions", json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    response_text = data["choices"][0]["message"]["content"].strip()
                    return parse_llm_response(response_text)
            except Exception as e:
                if attempt == 4:
                    print(f"[ERROR] judge failed after 5 retries: {e}")
                    return {
                        "is_helpful": None,
                        "reasons": f"API request failed: {str(e)[:100]}"
                    }
                await asyncio.sleep(2 * (attempt + 1))

    return {
        "is_helpful": None,
        "reasons": "Unknown error"
    }


async def preload_images(items: list, batch_size: int = 1000) -> list:
    """Preload and encode all images asynchronously."""
    print(f"Preloading {len(items)} images...")

    # Collect unique image paths (only tool images at index 1)
    image_paths = set()
    for item in items:
        if len(item.get("images", [])) > 1:
            image_paths.add(item["images"][1])

    print(f"Found {len(image_paths)} unique tool images to encode")

    # Encode all unique images
    image_cache = {}
    semaphore = asyncio.Semaphore(64)  # Limit concurrent file reads

    async def encode_with_semaphore(path):
        async with semaphore:
            result = await encode_image_async(path)
            return path, result

    # Process in batches to avoid memory issues
    paths_list = list(image_paths)
    for i in range(0, len(paths_list), batch_size):
        batch = paths_list[i:i+batch_size]
        tasks = [encode_with_semaphore(p) for p in batch]
        results = await asyncio.gather(*tasks)
        for path, encoded in results:
            image_cache[path] = encoded
        if (i + batch_size) % 10000 == 0 or i + batch_size >= len(paths_list):
            print(f"  Encoded {min(i + batch_size, len(paths_list))}/{len(paths_list)} images")

    print(f"Image encoding complete. Cache size: {len(image_cache)}")
    return image_cache


async def process_item(
    session: aiohttp.ClientSession,
    item: dict,
    model: str,
    semaphore: asyncio.Semaphore,
    idx: int,
    task_flag: bool,
    image_cache: dict
) -> dict:
    """Process a single item and return the result with LLM judgment."""

    images = item.get("images", [])
    if len(images) < 2:
        return {
            "index": idx,
            "question": item["question"],
            "original_image": images[0] if len(images) > 0 else None,
            "tool_image": None,
            "human_label": item["label"],
            "llm_is_helpful": None,
            "llm_reasons": "No tool-called image provided"
        }

    tool_image_path = images[1]
    base64_image = image_cache.get(tool_image_path)

    if base64_image is None:
        return {
            "index": idx,
            "question": item["question"],
            "original_image": images[0],
            "tool_image": tool_image_path,
            "human_label": item["label"],
            "llm_is_helpful": None,
            "llm_reasons": f"Image not in cache: {tool_image_path}"
        }

    media_type = get_image_media_type(tool_image_path)
    llm_result = await judge_one(
        session,
        item["question"],
        base64_image,
        media_type,
        model,
        task_flag,
        semaphore
    )

    return {
        "index": idx,
        "question": item["question"],
        "original_image": images[0],
        "tool_image": tool_image_path,
        "human_label": item["label"],
        "llm_is_helpful": llm_result["is_helpful"],
        "llm_reasons": llm_result["reasons"]
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--model", type=str, default="default", help="Model name to use")
    parser.add_argument("--concurrency", type=int, default=64, help="API concurrency limit")
    parser.add_argument("--encode-workers", type=int, default=8, help="Image encoding thread workers")
    parser.add_argument("--skip-preload", action="store_true", help="Skip image preloading (encode on-demand)")
    args = parser.parse_args()

    global _executor
    _executor = ThreadPoolExecutor(max_workers=args.encode_workers)

    items = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"Loaded {len(items)} items")
    task_flag = "accept" in args.input

    # Preload all images
    image_cache = {}
    if not args.skip_preload:
        image_cache = await preload_images(items)

    connector = aiohttp.TCPConnector(
        limit=args.concurrency,
        limit_per_host=args.concurrency,
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    timeout = aiohttp.ClientTimeout(total=120, connect=30)

    fout = open(args.output, "w")
    completed = 0
    lock = asyncio.Lock()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Try to get available models
        try:
            async with session.get(f"{JUDGE_URL}/models") as resp:
                if resp.status == 200:
                    models_data = await resp.json()
                    available = [m["id"] for m in models_data.get("data", [])]
                    print(f"Available models: {available}")
                    if args.model == "default" and available:
                        args.model = available[0]
                        print(f"Using model: {args.model}")
        except Exception as e:
            print(f"Could not list models: {e}, using provided model name")

        semaphore = asyncio.Semaphore(args.concurrency)
        pbar_total = len(items)

        async def run_and_write(idx, item):
            nonlocal completed
            res = await process_item(session, item, args.model, semaphore, idx, task_flag, image_cache)
            async with lock:
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()
                completed += 1
                if completed % 500 == 0 or completed == pbar_total:
                    print(f"Progress: {completed}/{pbar_total} ({100*completed/pbar_total:.1f}%)")

        tasks = [run_and_write(i, item) for i, item in enumerate(items)]
        await asyncio.gather(*tasks)

    fout.close()
    _executor.shutdown(wait=True)
    print(f"Done! Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
