import argparse
import base64
import json
import os
import os.path as osp
import string

JUDGE_SYSTEM_PROMPT = """You are evaluating whether a tool-processed (cropped/zoomed) image is helpful for answering a visual question.

You will see the **Original Image** and the **Processed Image**. Compare them: identify the question-relevant target in the original image, then check whether the processed image contains that target and stays relatively focused on it.

**Rule:**
- Reply "true" if the processed image CONTAINS the question-relevant target AND is relatively focused on it. Blurriness and low resolution are all acceptable since the target itself might be small.
- Reply "false" only if the target is absent (wrong area / irrelevant object) or the crop is so unfocused it barely concentrates on the target (e.g., near full-frame copy with no real zoom/crop).

Reply with ONLY a single token: "true" or "false". No explanation, no punctuation, no other text.

Examples:
Question: "What does the distant sign say?"
- true (the processed image zooms to it; blurry due to small object, but contains the target and is focused on it)
Question: "What color is the blanket?"
- false (the processed image shows only a kitchen counter; the target is absent)
"""

JUDGE_USER_PROMPT = """[Question]: {question}

Compare the two images: does the processed image contain the question-relevant target? Blurry and low-resolution still count as helpful. Reply with ONLY "true" or "false"."""


def img_to_data_uri(path):
    try:
        with open(path, 'rb') as f:
            data = base64.b64encode(f.read()).decode()
        ext = osp.splitext(path)[1].lower().lstrip('.')
        mime = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png',
                'gif': 'gif', 'bmp': 'bmp', 'webp': 'webp'}.get(ext, 'png')
        return f'data:image/{mime};base64,{data}'
    except Exception:
        return None

def load_traj(archive_dir):
    traj_path = osp.join(archive_dir, 'traj.jsonl')
    if not osp.exists(traj_path):
        return []
    entries = []
    for line in open(traj_path, encoding='utf-8'):
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries

def build_judge_model(args):
    """Return a judge client using the lenient JUDGE_SYSTEM_PROMPT / JUDGE_USER_PROMPT.

    Unlike the sandbox self_judge (which fed only the processed image), this
    judge receives BOTH the original and processed images so it can judge
    (a) whether the crop is on the correct target and (b) whether it is focused
    enough. We still send the request the same way the sandbox did: raw
    requests.post, system message as a separate role, user content = text then
    image_url blocks with NO detail field, max_tokens=2048, temperature=0.0.
    """
    api_base = args.api_base or os.environ.get('OPENAI_API_BASE', None)
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY', '')
    # Normalize to the full chat completions URL (env may be just .../v1).
    if api_base and not api_base.rstrip('/').endswith('/chat/completions'):
        api_base = api_base.rstrip('/') + '/chat/completions'
    model = args.judge_model
    if model is None:
        from vlmeval.dataset.utils.judge_util import get_model_from_api_base
        model = get_model_from_api_base()
        if model:
            print(f'[judge] auto-detected model: {model}')
    if model is None:
        raise RuntimeError('No judge model: set --judge-model or OPENAI_API_BASE')
    return {
        'model': model,
        'api_base': api_base,
        'key': api_key,
        'system_prompt': JUDGE_SYSTEM_PROMPT,
        'user_prompt': JUDGE_USER_PROMPT,
    }

def _judge_text(judge, prompt):
    """Text-only completion through the same judge endpoint (for correctness
    matching). Returns the assistant content string."""
    import requests
    payload = dict(model=judge['model'], messages=[
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ], max_tokens=2048, n=1, temperature=0.0)
    headers = {'Content-Type': 'application/json'}
    if judge.get('key'):
        headers['Authorization'] = f'Bearer {judge["key"]}'
    resp = requests.post(judge['api_base'], headers=headers, data=json.dumps(payload), timeout=600)
    resp_struct = json.loads(resp.text)
    return resp_struct['choices'][0]['message']['content'].strip()

def judge_correctness(judge, question, options, gt, prediction):
    """Map prediction to an option letter via the project's build_prompt, then
    compare with gt. Returns (is_correct: bool|None, matched_letter: str, log)."""
    from vlmeval.dataset.utils.multiple_choice import build_prompt
    from vlmeval.utils.matching_util import can_infer
    from vlmeval.smp import cn_string

    # Build a fake item for build_choices / can_infer.
    choices = {k: v for k, v in options.items()} if options else {}
    choice_labels = list(choices.keys())

    # Fast path: try to infer directly from the prediction text.
    ret = can_infer(prediction, choices) if choices else None
    if ret is None and judge is not None:
        # Use the project's prompt to ask the judge for the matched letter.
        option_str = ' '.join(f'{k}. {v}' for k, v in choices.items()) if choices else ''
        prompt = build_prompt(question, option_str, prediction)
        try:
            ans = _judge_text(judge, prompt)
            ret = can_infer(ans, choices)
        except Exception as e:
            return None, None, f'judge error: {e}'
    if ret is None:
        ret = 'Z'
    is_correct = (ret == gt) if gt else None
    return is_correct, ret, f'matched={ret} gt={gt}'


def build_judge_question(entry):
    """Reconstruct the exact text the model saw at test time.

    At test time self_judge used `original_query`, which is the text message
    from `inputs` -- i.e. the dataset's `build_prompt` output, NOT the bare
    `question` field. For ImageMCQ datasets (e.g. VStarBench) build_prompt
    wraps the question with its Options and a "Please select..." line (see
    vlmeval/dataset/image_mcq.py build_prompt). Judging with the bare question
    instead of this full prompt changes the judge's helpfulness threshold
    (options reveal the answer space, making the judge more lenient), which
    shows up as a one-sided ACCEPT->REJECT drift vs the recorded judgments.
    Rebuild that prompt here from the question/options stored in the traj.
    """
    question = entry.get('question', '')
    options = entry.get('options', {}) or {}
    prompt = f'Question: {question}\n'
    if options:
        prompt += 'Options:\n'
        for key, item in options.items():
            prompt += f'{key}. {item}\n'
        prompt += 'Please select the correct answer from the options above. \n'
    return prompt

def judge_turn_helpful(judge, question, image_paths, original_image_path=None):
    """Replay the sandbox self_judge for ONE tool-use turn.

    Faithfully mirrors python_tool.py's self_judge request path, but feeds BOTH
    the original and processed images (the lenient judge needs the original to
    judge crop correctness + focus):
      - system = JUDGE_SYSTEM_PROMPT (separate role message)
      - user   = text(JUDGE_USER_PROMPT.format(question=...)), then an
                 "[Original Image]:" label + the original image, then a
                 "[Processed Image]:" label + each processed image
      - image_url blocks carry ONLY a url (no `detail` field), base64 via img2base64
      - max_tokens=2048, temperature=0.0
      - parse the bare "true"/"false" token (parse failure => NOT helpful, 0)
      - remote service unavailable / timeout => raise (do NOT silently default)
    Returns (flag: 1/0, reasons: str) for the whole turn (one judgment, as the
    sandbox produced one ACCEPT/REJECT per turn regardless of image count).
    """
    if judge is None:
        return None, 'no judge'
    import requests
    from PIL import Image
    from vlmeval.utils.python_tool import img2base64

    def _img_block(label_text, img_or_path):
        blocks = []
        if label_text:
            blocks.append({"type": "text", "text": label_text})
        if isinstance(img_or_path, str):
            img = Image.open(img_or_path).convert('RGB')
        else:
            img = img_or_path
        blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2base64(img)}"}})
        return blocks

    user_content = []
    # Send the original image first (if available) so the judge can compare.
    if original_image_path and osp.exists(original_image_path):
        user_content += _img_block("[Original Image]:", original_image_path)
    user_content.append({"type": "text", "text": "[Processed Image]:"})
    for img in image_paths:
        user_content += _img_block(None, img)
    user_content += [{"type": "text", "text": judge['user_prompt'].format(question=question)}]

    raw_messages = [
        {"role": "system", "content": judge['system_prompt']},
        {"role": "user", "content": user_content},
    ]
    payload = dict(model=judge['model'], messages=raw_messages, max_tokens=8, n=1, temperature=0.0)
    headers = {'Content-Type': 'application/json'}
    if judge.get('key'):
        headers['Authorization'] = f'Bearer {judge["key"]}'

    # Network errors (timeout, connection refused, DNS failure, non-2xx HTTP,
    # missing choices, etc.) propagate as exceptions -- the caller must see the
    # failure rather than silently defaulting.
    resp = requests.post(judge['api_base'], headers=headers, data=json.dumps(payload), timeout=300)
    resp_struct = json.loads(resp.text)
    judge_response = resp_struct['choices'][0]['message']['content'].strip().lower()
    # Expect a bare "true"/"false"; fall back to first token if the model
    # added extra text despite the prompt.
    first_token = judge_response.split()[0] if judge_response else ''
    if 'true' in first_token:
        return 1, judge_response
    elif 'false' in first_token:
        return 0, judge_response
    # Request succeeded but the response is unparseable -> default to NOT helpful.
    return 0, f'unparseable response, defaulting to 0: {judge_response!r}'


def find_original_image(sdir):
    """Locate the original image stored alongside the trajectory (original.jpg/png/jpeg)."""
    for e in ('original.jpg', 'original.png', 'original.jpeg'):
        p = osp.join(sdir, e)
        if osp.exists(p):
            return p
    return None


def render_sample(idx, entry, turns, correct, img_judgements):
    html = [f'<div class="sample">', f'<h2>Sample index={idx}'
            f' {"[✓ 正确]" if correct else "[✗ 错误]" if correct is False else "[?]"}'
            f'</h2>']
    html.append(f'<p><b>问题:</b> {entry.get("question","")}</p>')
    opts = entry.get('options', {})
    if opts:
        html.append('<p><b>选项:</b><br>' + '<br>'.join(f'{k}: {v}' for k, v in opts.items()) + '</p>')
    html.append(f'<p><b>正确答案:</b> {entry.get("gt_answer","")} &nbsp; <b>模型预测:</b> {entry.get("answer","")}</p>')

    sdir = entry['_dir']
    orig_path = find_original_image(sdir)
    if orig_path:
        uri = img_to_data_uri(orig_path)
        if uri:
            html.append(f'<p><b>原图:</b><br><img src="{uri}" style="max-width:400px"></p>')

    for t in turns:
        turn_flag = img_judgements.get(t.get('turn'))
        flag_str = '🟢' if turn_flag == 1 else '🔴' if turn_flag == 0 else '❔'
        html.append(f'<hr><h3>Turn {t.get("turn")}'
                    f'{" · 过程图帮助: " + flag_str if turn_flag is not None else ""}'
                    f'{" · code-judge=" + t.get("judge") if t.get("judge") else ""}</h3>')
        code = t.get('code', '')
        if code:
            html.append(f'<pre>{code}</pre>')
        sb = t.get('sandbox_output', '')
        if sb:
            html.append(f'<p><b>sandbox_output:</b> {sb[:500]}</p>')
        raw_ips = t.get('image_paths') or []
        if not raw_ips and t.get('image_path'):
            raw_ips = [t['image_path']]
        for ip in raw_ips:
            if not ip or not osp.exists(ip):
                continue
            uri = img_to_data_uri(ip)
            if uri:
                tag = '✓有帮助' if turn_flag == 1 else '✗无帮助' if turn_flag == 0 else '?未判定'
                html.append(f'<p><img src="{uri}" style="max-width:400px" title="{osp.basename(ip)}"> [{tag}]</p>')
    html.append('</div>')
    return '\n'.join(html)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--process-dir', required=True, help='process_images/<dataset> directory')
    ap.add_argument('--out', default='review.html')
    ap.add_argument('--stats', default='/tmp/stats.json', help='write helpfulness statistics here')
    ap.add_argument('--judge-model', default=None, help='judge model id (else auto-detect from OPENAI_API_BASE)')
    ap.add_argument('--api-base', default=None, help='override OPENAI_API_BASE')
    ap.add_argument('--api-key', default=None, help='override OPENAI_API_KEY')
    ap.add_argument('--no-judge', action='store_true', help='skip LLM judging, only render samples with images')
    ap.add_argument('--include-wrong', action='store_true', help='also render wrong samples')
    ap.add_argument('--nproc', type=int, default=8, help='parallel judge calls')
    args = ap.parse_args()

    from vlmeval.smp import load_env
    load_env()

    # Collect all samples.
    samples = []
    for name in sorted(os.listdir(args.process_dir), key=lambda x: int(x) if x.isdigit() else x):
        sdir = osp.join(args.process_dir, name)
        if not osp.isdir(sdir): continue
        traj = load_traj(sdir)
        if not traj: continue
        final = next((e for e in traj if e.get('final')), None)
        if final is None: continue
        turns = [e for e in traj if not e.get('final')]
        final['_dir'] = sdir
        final['_idx'] = name
        samples.append((name, final, turns))
    print(f'found {len(samples)} samples')

    judge = None if args.no_judge else build_judge_model(args)
    def judge_sample(sample):
        name, entry, turns = sample
        sdir = entry['_dir']
        # Judge with the FULL build_prompt text (question + options), matching
        # the original_query the sandbox self_judge saw at test time.
        question = build_judge_question(entry)
        options = entry.get('options', {})
        gt = entry.get('gt_answer', '')
        pred = entry.get('answer', '')
        correct, matched, c_log = judge_correctness(judge, entry.get('question', ''), options, gt, pred)

        # Per-turn helpfulness flag: ONE judgment per turn (matching the
        # sandbox, which judged the whole turn's images in a single request).
        img_judgements = {}
        # Per-turn old (sandbox self_judge) ACCEPT/REJECT flag, normalised to
        # 1=helpful / 0=not-helpful. None when the traj row carries no judge.
        old_judgements = {}
        for t in turns:
            turn_no = t.get('turn')
            old = t.get('judge')
            if old in ('ACCEPT', 'REJECT'):
                old_judgements[turn_no] = 1 if old == 'ACCEPT' else 0
        if judge is not None:
            orig_path = find_original_image(sdir)
            for t in turns:
                turn_no = t.get('turn')
                # Support both the plural `image_paths` (older traj) and the
                # singular `image_path` (e.g. deepeyes-crop) field names.
                raw_ips = t.get('image_paths') or []
                if not raw_ips and t.get('image_path'):
                    raw_ips = [t['image_path']]
                ips = [ip for ip in raw_ips if ip and osp.exists(ip)]
                if ips:
                    flag, _ = judge_turn_helpful(judge, question, ips, original_image_path=orig_path)
                    img_judgements[turn_no] = flag
        print(f'  [done] sample {name}: correct={correct} img_judgements={img_judgements}')
        return (name, entry, turns, correct, img_judgements, old_judgements)

    from vlmeval.utils.mp_util import track_progress_rich
    if judge is not None:
        judged = track_progress_rich(judge_sample, [(s,) for s in samples], nproc=args.nproc)
        judged = [j for j in judged if j is not None]
    else:
        judged = []
        for name, entry, turns in samples:
            old_j = {t.get('turn'): (1 if t.get('judge') == 'ACCEPT' else 0) for t in turns if t.get('judge') in ('ACCEPT', 'REJECT')}
            judged.append((name, entry, turns, None, {}, old_j))

    group_stats = {}
    n_correct = 0
    n_wrong = 0
    n_correct_with_imgs = 0
    agree_stats = {'agree': 0, 'disagree': 0, 'total': 0, 'both_helpful': 0, 'both_not': 0, 'old_yes_new_no': 0, 'old_no_new_yes': 0}
    for name, entry, turns, correct, img_judgements, old_judgements in judged:
        if correct is True: n_correct += 1
        elif correct is False: n_wrong += 1
        # Per-turn agreement: one new judgment vs the turn's old ACCEPT/REJECT.
        for turn_no, old_val in old_judgements.items():
            f = img_judgements.get(turn_no)
            if f is None: continue
            agree_stats['total'] += 1
            if f == old_val:
                agree_stats['agree'] += 1
                if f == 1: agree_stats['both_helpful'] += 1
                else: agree_stats['both_not'] += 1
            else:
                agree_stats['disagree'] += 1
                if old_val == 1 and f == 0: agree_stats['old_yes_new_no'] += 1
                else: agree_stats['old_no_new_yes'] += 1
        if correct is not True: continue
        # Total number of turns = number of turn rows in the trajectory.
        n_turns = len(turns)
        # All judged turn flags for this sample (across turns).
        all_flags = [f for f in img_judgements.values() if f is not None]
        if not all_flags: continue
        n_correct_with_imgs += 1
        sample_helpful = 1 if any(f == 1 for f in all_flags) else 0
        h, t = group_stats.get(n_turns, (0, 0))
        group_stats[n_turns] = (h + sample_helpful, t + 1)

    def pct(x):
        return f'{100.0 * x[0] / x[1]:.2f}%' if x[1] else 'N/A'

    # Per-group breakdown (sorted by number of turns).
    per_group = {int(k): {'helpful': v[0], 'total': v[1], 'rate': (v[0] / v[1] if v[1] else None)} for k, v in group_stats.items()}
    total_h = sum(v[0] for v in group_stats.values())
    total_t = sum(v[1] for v in group_stats.values())
    overall = {'helpful': total_h, 'total': total_t, 'rate': (total_h / total_t if total_t else None)}
    a = agree_stats
    agreement = {
        'total_compared_turns': a['total'],
        'agree': a['agree'],
        'disagree': a['disagree'],
        'agreement_rate': (a['agree'] / a['total'] if a['total'] else None),
        'both_helpful': a['both_helpful'],
        'both_not_helpful': a['both_not'],
        'old_helpful_new_not': a['old_yes_new_no'],
        'old_not_new_helpful': a['old_no_new_yes'],
    }
    stats = {
        'n_samples': len(judged),
        'n_correct': n_correct,
        'n_wrong': n_wrong,
        'n_correct_with_process_images': n_correct_with_imgs,
        'per_total_turns_group': per_group,
        'overall': overall,
        'judge_agreement': agreement,
    }
    with open(args.stats, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print('\n===== Helpfulness among correct samples (grouped by total #turns) =====')
    print(f'correct samples: {n_correct} / {len(judged)} (with process images: {n_correct_with_imgs})')
    print('  a sample is helpful if ANY of its process images is judged helpful')
    for n_turns in sorted(per_group):
        s = per_group[n_turns]
        print(f'  {n_turns} turns: helpful {s["helpful"]}/{s["total"]} = {pct((s["helpful"], s["total"]))}')
    print(f'  overall:   helpful {overall["helpful"]}/{overall["total"]} = {pct((overall["helpful"], overall["total"]))}')

    print('\n===== Old (sandbox self_judge ACCEPT/REJECT) vs new (helpfulness) agreement =====')
    print(f'  compared images: {a["total"]}  (per process-image, paired with its turn\'s old judge)')
    if a['total']:
        print(f'  agreement: {a["agree"]}/{a["total"]} = {100.0*a["agree"]/a["total"]:.2f}%   '
              f'disagreement: {a["disagree"]}/{a["total"]} = {100.0*a["disagree"]/a["total"]:.2f}%')
        print(f'  both helpful:      {a["both_helpful"]}')
        print(f'  both not helpful:  {a["both_not"]}')
        print(f'  old=helpful new=not (old over-trusted): {a["old_yes_new_no"]}')
        print(f'  old=not new=helpful (old under-trusted): {a["old_no_new_yes"]}')
    else:
        print('  no images with both old and new judgements available')
    print(f'stats written to {args.stats}')

if __name__ == '__main__':
    main()
