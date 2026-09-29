# from http import HTTPStatus
import os
import requests
from ..dataset import DATASET_TYPE, DATASET_MODALITY
from vlmeval.api.base import BaseAPI
from vlmeval.smp import *
from ..utils import PythonInterpreter
from ..utils.python_tool import (
    extract_tool_call_contents,
    parse_critic_response,
    subagent_system_prompt,
    subagent_user_prompt,
)
import base64
from io import BytesIO
import threading
from typing import List, Any, Optional
from PIL import Image
import re
import numpy as np

def encode_pil_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


class ThreadSafeAppendOnlyArray:
    """Thread-safe append-only array implementation using threading.Lock"""
    
    def __init__(self):
        self._data = []
        self._lock = threading.Lock()
    
    def append(self, item):
        """Thread-safe append operation"""
        with self._lock:
            self._data.append(item)
    
    def extend(self, items):
        """Thread-safe extend operation"""
        with self._lock:
            self._data.extend(items)
    
    def get_copy(self):
        """Get a copy of the current data (thread-safe)"""
        with self._lock:
            return self._data.copy()
    
    def __len__(self):
        """Thread-safe length operation"""
        with self._lock:
            return len(self._data)
    
    def __getitem__(self, index):
        """Thread-safe item access"""
        with self._lock:
            return self._data[index]
    
    def __iter__(self):
        """Thread-safe iteration (returns copy to avoid modification during iteration)"""
        with self._lock:
            return iter(self._data.copy())

    def log_records_append_only(self, f):
        with self._lock:
            for i, item in enumerate(self._data.copy()):
                f.write(json.dumps(item) + '\n')
            # clear after writing to prevent duplicates
            self._data.clear()


class InternVL2_PromptUtil:

    def __init__(self, use_mpo_prompt=False):
        self.use_mpo_prompt = use_mpo_prompt

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        assert DATASET_MODALITY(dataset) != 'VIDEO', 'not supported'
        if dataset in [
            'atomic_dataset', 'electro_dataset', 'mechanics_dataset',
            'optics_dataset', 'quantum_dataset', 'statistics_dataset'
        ]:
            return False
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT', 'MMAlignBench'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def build_prompt(self, line, dataset=None):
        use_cot = (os.getenv('USE_COT') == '1')
        use_mpo_prompt = self.use_mpo_prompt and (use_cot or dataset in ['MMStar', 'HallusionBench', 'OCRBench'])

        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        from ..vlm.internvl.utils import (build_multi_choice_prompt,
                                          build_mcq_cot_prompt,
                                          build_qa_cot_prompt,
                                          build_mpo_prompt,
                                          reorganize_prompt)

        tgt_path = self.dump_image(line, dataset)
        max_num = self.get_max_num(dataset)
        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath',
                            'QSpatial', 'WeMath', 'LogicVista'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt)

        message = [dict(type='text', value=prompt)]
        image_num = len(tgt_path)
        max_num = max(1, min(max_num, 64 // image_num))
        # TODO：support upscale_flag
        message.extend([dict(type='image', value=s, max_dynamic_patch=max_num) for s in tgt_path])

        if use_mpo_prompt:
            message = build_mpo_prompt(message, line, dataset)

        # reorganize_prompt
        prompt = reorganize_prompt(message, image_num, dataset=dataset)
        prompt.replace('<image>', '<IMAGE_TOKEN>')
        message[0] = dict(type='text', value=prompt)
        return message

    def get_max_num(self, dataset):
        self.total_max_num = 64
        if dataset is None:
            self.max_num = 6
            return None
        res_1_datasets = ['MMBench-Video', 'Video-MME', 'MVBench', 'Video', 'WorldSense']
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld', 'VCR_EN', 'VCR_ZH', 'OCRVQA', 'BMMR']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST', 'DUDE', 'MMLongBench_DOC', 'SLIDEVQA']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if DATASET_MODALITY(dataset) == 'VIDEO':
            self.max_num = 1
        elif listinstr(res_12_datasets, dataset):
            return 12
        elif listinstr(res_18_datasets, dataset):
            return 18
        elif listinstr(res_24_datasets, dataset):
            return 24
        else:
            return 6


class CogVLM2_PromptUtil:

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) in 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            question = line['question']
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            if hint is not None:
                question = hint + '\n' + question

            option_candidate = string.ascii_uppercase
            options = {
                cand: line[cand]
                for cand in option_candidate
                if cand in line and not pd.isna(line[cand])
            }
            for key, item in options.items():
                question += f'\n{key}. {item}'
            prompt = question

            if not cn_string(prompt):
                prompt = prompt + '\n' + "Answer with the option's letter from the given choices directly."
            else:
                prompt = prompt + '\n' + '请直接回答选项字母。'
        else:
            prompt = line['question']
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=p) for p in tgt_path])
        return message


class LMDeployWrapper(BaseAPI):

    is_api: bool = True

    custom_prompt: str = None
    prompt_map = {
        'cogvlm2': CogVLM2_PromptUtil(),
        'internvl2': InternVL2_PromptUtil(),
        'internvl2-mpo-cot': InternVL2_PromptUtil(use_mpo_prompt=True),
    }

    def __init__(self,
                 model: str = None,
                 retry: int = 5,
                 key: str = 'sk-123456',
                 verbose: bool = True,
                 temperature: float = 0.0,
                 timeout: int = 60,
                 api_base: str = None,
                 system_prompt: str = None,
                 max_tokens: int = 1024,
                 use_tool: bool = False,
                 **kwargs):
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.timeout = timeout

        key = os.environ.get('LMDEPLOY_API_KEY', key)
        api_base = os.environ.get('LMDEPLOY_API_BASE', api_base)
        assert key is not None, 'Please set the environment variable LMDEPLOY_API_KEY.'
        assert api_base is not None, 'Please set the environment variable LMDEPLOY_API_BASE.'
        self.key = key
        self.api_base = api_base
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        model_url = ''.join([api_base.split('v1')[0], 'v1/models'])
        resp = requests.get(model_url)
        model_id_list = [str(data['id']) for data in resp.json()['data']]
        self.model = model if model in model_id_list else model_id_list[0]
        self.logger.info(f'lmdeploy evaluate model: {self.model}')
        self.set_prompt_pattern(self.model)
        if hasattr(self, 'custom_prompt'):
            self.logger.info(f'using custom prompt {self.custom_prompt}')
        self.temperature = temperature
        self.logger.info(f'Init temperature: {self.temperature}')
        self.safe_append_array = ThreadSafeAppendOnlyArray()
        self.save_file = kwargs.get('save_file', 'saved_results.jsonl')

    def set_dump_image(self, dump_image_func):
        if self.custom_prompt in self.prompt_map:
            self.prompt_map[self.custom_prompt].dump_image_func = dump_image_func
        self.dump_image_func = dump_image_func

    def use_custom_prompt(self, dataset):
        if self.custom_prompt in self.prompt_map:
            return self.prompt_map[self.custom_prompt].use_custom_prompt(dataset)
        return False

    def build_prompt(self, line, dataset=None):
        if self.custom_prompt in self.prompt_map:
            return self.prompt_map[self.custom_prompt].build_prompt(line, dataset)
        raise NotImplementedError

    def set_prompt_pattern(self, model_name):
        if 'Phi-3.5-Vision'.lower() in model_name.lower():
            self.max_tokens = 1000
            self.temperature = 0.0
        if 'cogvlm2-llama3-chat-19B'.lower() in model_name.lower():
            self.max_tokens = 2048
            self.temperature = 0.0
            self.custom_prompt = 'cogvlm2'
        if 'internvl2' in model_name.lower() or 'internvl3' in model_name.lower():
            self.max_tokens = 1024
            self.temperature = 0.0
            if 'mpo' in model_name.lower():
                self.max_tokens = 4096
                self.logger.info('Use custom prompt internvl2-mpo-cot')
                self.custom_prompt = 'internvl2-mpo-cot'
            else:
                self.logger.info('Use custom prompt internvl2')
                self.custom_prompt = 'internvl2'
        if 'internvl2-8b-mpo-cot'.lower() in model_name.lower():
            self.use_mpo_prompt = True
            self.max_tokens = 1024
            self.temperature = 0.0
            self.logger.info('Use custom prompt internvl2-mpo-cot')
            self.custom_prompt = 'internvl2-mpo-cot'
        if 'qvq'.lower() in model_name.lower():
            self.max_tokens = 4096
            self.temperature = 0.0
            self.logger.info('QVQ model detected, do not use custom prompt')

    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    # Use Qwen's custom image preprocessing
                    from ..vlm.qwen2_vl.model import encode_image
                    b64, mime_type = encode_image(msg['value'])
                    extra_args = msg.copy()
                    extra_args.pop('type')
                    extra_args.pop('value')
                    img_struct = dict(url=f'data:{mime_type};base64,{b64}', **extra_args)
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)

        temperature = kwargs.pop('temperature', self.temperature)
        self.logger.info(f'Generate temperature: {temperature}')
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        dataset = kwargs.pop('dataset', None)
        if dataset is not None and listinstr(['BMMR'], dataset):
            # BMMR dataset has a very long prompt, so we need to increase max_tokens
            max_tokens = 8196
            self.logger.info('BMMR dataset detected, set max_tokens to 8196')

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            **kwargs)
        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()

            # for internvl2-8b-mpo-cot
            if getattr(self, 'use_mpo_prompt', False):
                from ..vlm.internvl.utils import mpo_post_processing
                answer = mpo_post_processing(answer, kwargs.get('dataset'))
        except:
            pass
        
        logging_msgs = []
        logging_msgs.append({"role": "system", "content": self.system_prompt})
        logging_msgs.append({"role": "user", "content": inputs})
        logging_msgs.append({"role": "assistant", "content": answer})
        self.safe_append_array.append(logging_msgs)
        return ret_code, answer, response


class LMDeployAPI(LMDeployWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, message, dataset=None):
        ret = super(LMDeployAPI, self).generate(message, dataset=dataset)
        with open(self.save_file, 'a') as f:
            self.safe_append_array.log_records_append_only(f)
        return ret

    def redact_images(self, inputs, placeholder='<REDACTED_IMAGE>'):
        """Replace image base64 data with file paths for logging

        For sandbox-generated images, use the actual file path from _log_path metadata.
        For original input images, use the placeholder (original image path).
        """
        for msg in inputs:
            if "content" in msg and isinstance(msg['content'], list):
                for c in msg['content']:
                    if 'image' in c['type']:
                        # Check if this has _log_path metadata (sandbox-generated image)
                        if '_log_path' in c and c['_log_path']:
                            # Use the actual file path from sandbox
                            c['image_url'] = c['_log_path']
                            # Remove metadata
                            del c['_log_path']
                        else:
                            # Original input image - use placeholder
                            c['image_url'] = placeholder
            else:
                pass
        return inputs

filename_hint = """
### User Image Path:** "{image_filename}"
### User Image Size:** "{image_size[0]}x{image_size[1]}"

### **Output Format (strict adherence required):**

<think>Your detailed reasoning process should go here.</think>
<code>Your compliant executable code should go here.</code>
<answer>Your final answer to the user's question goes here.</answer>"""

class LMDeployAPIWithToolUse(LMDeployAPI):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_tool = kwargs.get('use_tool', False)
        self.tool_start_token = kwargs.get('tool_start_token', None)
        self.tool_end_token = kwargs.get('tool_end_token', None)
        self.verbose = kwargs.get('verbose', False)
        self.max_turns = kwargs.get('max_turns', 10)
        if self.use_tool:
            assert self.tool_start_token and self.tool_end_token, "Both tool_start_token and tool_end_token must be provided when use_tool is True"
        # Process-image archival (for human review of code-agent trajectories).
        # When save_process_images is True, each tool-use turn's sandbox images
        # and code are persisted under {process_image_dir}/{sample_index}/.
        self.save_process_images = kwargs.get('save_process_images', False)
        self.process_image_dir = kwargs.get('process_image_dir', None)

    def _archive_turn(self, archive_dir, turn, code_response, obs, image_paths):
        """Persist one tool-use turn: code, sandbox output text, judge, and images.

        Images live in the interpreter's temp dir which is removed on interpreter
        destruction, so they must be copied here, before generate_inner returns.
        """
        import shutil
        # Extract the code block(s) the model produced this turn.
        code_blocks = extract_tool_call_contents(self.tool_start_token, self.tool_end_token, code_response)
        code_text = '\n\n'.join(c.replace('```python', '').replace('```', '').replace('python', '', 1).strip()
                                for c in code_blocks) if code_blocks else ''

        # Sandbox textual output + judgment (ACCEPT/REJECT) from obs prompt.
        sandbox_text = ''
        judge = None
        if isinstance(obs, dict):
            sandbox_text = obs.get('prompt', '')
            m = re.search(r'\[Judgment:\s*(ACCEPT|REJECT)\]', sandbox_text)
            if m:
                judge = m.group(1)
        elif isinstance(obs, str):
            sandbox_text = obs

        # Copy process images into the archive (rename to turnN_imgK.ext).
        archived_img_paths = []
        for idx, p in enumerate(image_paths or []):
            if not p or not os.path.exists(p):
                continue
            ext = os.path.splitext(p)[1] or '.png'
            dst = os.path.join(archive_dir, f'turn{turn}_img{idx}{ext}')
            try:
                shutil.copy2(p, dst)
                archived_img_paths.append(dst)
            except Exception as e:
                self.logger.warning(f'[archive] failed to copy {p}: {e}')

        # Save the code to a .py file for easy inspection.
        if code_text:
            try:
                with open(os.path.join(archive_dir, f'turn{turn}_code.py'), 'w', encoding='utf-8') as f:
                    f.write(code_text)
            except Exception:
                pass

        # Append a trajectory entry.
        traj_entry = {
            'turn': turn,
            'code': code_text,
            'sandbox_output': sandbox_text,
            'judge': judge,
            'image_paths': archived_img_paths,
        }
        try:
            with open(os.path.join(archive_dir, 'traj.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(traj_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.warning(f'[archive] failed to write traj: {e}')

    def _archive_final(self, archive_dir, sample_index, original_image_path, answer, sample_meta=None):
        """Append the final answer and copy the original input image for reference.

        The question, options and ground-truth answer from sample_meta are stored
        alongside the prediction so traj.jsonl is self-contained for later review.
        """
        entry = {'final': True, 'sample_index': sample_index, 'answer': answer}
        if isinstance(sample_meta, dict):
            for k in ('question', 'options', 'gt_answer', 'category'):
                if k in sample_meta:
                    entry[k] = sample_meta[k]
        try:
            with open(os.path.join(archive_dir, 'traj.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception:
            pass
        if original_image_path and os.path.exists(original_image_path):
            import shutil
            ext = os.path.splitext(original_image_path)[1] or '.jpg'
            dst = os.path.join(archive_dir, f'original{ext}')
            try:
                if not os.path.exists(dst):
                    shutil.copy2(original_image_path, dst)
            except Exception:
                pass

    def _archive_full_history(self, archive_dir, input_msgs, final_response, sample_meta=None):
        """Persist the COMPLETE conversation history for a sample.

        ``input_msgs`` is the full multi-turn message list the model saw across
        the entire interaction (system prompt, user query, every assistant
        response, and every tool/observation turn), already image-redacted by
        ``redact_images`` (base64 replaced with file paths). Together with the
        final response this is a self-contained record that reproduces the whole
        trajectory — beyond the per-turn code/sandbox slices written by
        ``_archive_turn``/``_archive_final``.

        The final response is taken from the last assistant message in
        ``input_msgs`` (which always holds the model's most recent output,
        regardless of which return path was taken); ``final_response`` is only a
        fallback when no assistant message is present.
        """
        import copy
        last_assistant = None
        for msg in reversed(input_msgs):
            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                last_assistant = msg.get('content')
                break
        entry = {
            'full_history': True,
            'input_msgs': copy.deepcopy(input_msgs),
            'final_response': last_assistant if last_assistant is not None else final_response,
        }
        if isinstance(sample_meta, dict):
            for k in ('sample_index', 'question', 'options', 'gt_answer', 'category'):
                if k in sample_meta:
                    entry[k] = sample_meta[k]
        try:
            with open(os.path.join(archive_dir, 'traj.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.warning(f'[archive] failed to write full_history: {e}')

    def generate(self, **kwargs):
        ret = super().generate(**kwargs)

        with open(self.save_file, 'a') as f:
            for item in self.safe_append_array.get_copy():
                f.write(json.dumps(item) + '\n')
        self.safe_append_array._data.clear()  # Clear after writing to prevent duplicates
        return ret

    def setup_interpreter_with_images(self, inputs):
        """Setup interpreter with input images"""
        if not self.use_tool:
            return None

        # Extract image path and filename from inputs
        image_path = None
        image_filename = None
        aux_img_dir = None
        image_size = None

        for msg in inputs:
            if msg['type'] == 'image':
                image_path = msg['value']
                image_filename = os.path.basename(image_path)  # '1.jpg'
                aux_img_dir = os.path.dirname(image_path)
                break  # Use first image for now

        # Create fresh interpreter instance with aux_img_dir
        interpreter = PythonInterpreter("python", "Python code execution", {}, aux_img_dir=aux_img_dir, api_base=self.api_base, key=self.key, model=self.model)

        # Extract PIL Images from inputs (for multi_modal_data)
        from PIL import Image
        images = []
        for msg in inputs:
            if msg['type'] == 'image':
                img = Image.open(msg['value'])
                image_size = [img.width, img.height]  # Store size as [width, height]
                images.append(img)

        if images:
            # Create extra_info with image filename, path, and size
            extra_info = {
                'image_file_name': image_filename,
                'image_file_path': image_path,
                'image_size': image_size,
            }

            # Reset interpreter with PIL Images and extra_info
            multi_modal_data = {'image': images}
            interpreter.reset(inputs, multi_modal_data, multi_modal_data, extra_info=extra_info)
        
        return interpreter

    def generate_inner(self, inputs, **kwargs) -> str:
        if not self.use_tool:
            return super().generate_inner(inputs, **kwargs)

        # Process-image archival context (thread-local, set by BaseAPI.generate).
        sample_meta = getattr(self._tls, 'sample_meta', None) or {}
        sample_index = sample_meta.get('sample_index', None) if isinstance(sample_meta, dict) else None
        original_image_path = sample_meta.get('image_path', None) if isinstance(sample_meta, dict) else None
        do_archive = bool(self.save_process_images and self.process_image_dir and sample_index is not None)
        archive_dir = None
        if do_archive:
            archive_dir = os.path.join(self.process_image_dir, str(sample_index))
            # A fresh attempt: clear any leftover from a previous (failed) retry so
            # traj.jsonl does not interleave two attempts' turns. Each sample_index
            # is handled by exactly one task in track_progress_rich, so there is no
            # cross-thread race here; retries are serial within this thread.
            import shutil
            if os.path.isdir(archive_dir):
                shutil.rmtree(archive_dir, ignore_errors=True)
            os.makedirs(archive_dir, exist_ok=True)

        # Extract image filename and size to append to user prompt
        image_filename = None
        image_size = None
        original_query = ""
        for msg in inputs:
            if msg["type"] == "text": original_query = msg['value'].strip()
            if msg['type'] == 'image':
                from PIL import Image
                image_filename = os.path.basename(msg['value'])
                img = Image.open(msg['value'])
                image_size = (img.width, img.height)
                if original_image_path is None:
                    original_image_path = msg['value']

        # Append image filename and size info to the last text input (matches training format)
        if image_filename and image_size:
            for msg in reversed(inputs):
                if msg['type'] == 'text':
                    msg['value'] += filename_hint.format(image_filename="/cpfs01/haoqingwang/huggingface_datasets/CodeV-RL-Data/images/"+image_filename, image_size=image_size)  # consistant with training
                    break
        
        # Setup interpreter with input images
        interpreter = self.setup_interpreter_with_images(inputs)
        input_msgs = self.prepare_inputs(inputs)  # Full input

        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        self.logger.info(f'Generate temperature:{temperature} max_tokens:{max_tokens}')
        dataset = kwargs.pop('dataset', None)

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}

        response_message = ""
        try_count = 0
        ret = (500, self.fail_msg, None)

        # Track all sandbox image paths for logging
        sandbox_image_paths = []
        try:
            while try_count < self.max_turns:  # Limit number of rounds
                # Prepare payload with tool stop token
                payload = dict(
                    model=self.model,
                    messages=input_msgs,
                    max_tokens=max_tokens,
                    n=1,
                    temperature=temperature,
                    stop=[self.tool_end_token, "</answer>"],
                    include_stop_str_in_output=True,
                    **kwargs)

                response = requests.post(
                    self.api_base,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout * 1.1)

                ret_code = response.status_code
                ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
                if ret_code != 0:
                    print(f"Request Error! ret_code={ret_code}, response text={response.text}")
                    return ret_code, self.fail_msg, response

                try:
                    resp_struct = json.loads(response.text)
                    response_message = resp_struct['choices'][0]['message']['content'].strip()
                except:
                    return ret_code, self.fail_msg, response

                # Add assistant response to message history
                input_msgs.append({"role": "assistant", "content": response_message})
                interpreter._log(f"Response_{interpreter.execution_count}", response_message)

                answers = extract_tool_call_contents("<answer>", "</answer>", response_message)
                if answers:
                    if do_archive:
                        self._archive_final(archive_dir, sample_index, original_image_path, answers[0].strip(), sample_meta)
                    return ret_code, answers[0].strip(), response

                # Check for tool usage - add missing end token if needed (model may stop before generating it)
                if self.tool_start_token in response_message and self.tool_end_token not in response_message:
                    response_message = response_message + self.tool_end_token

                if self.tool_start_token in response_message and self.tool_end_token in response_message:
                    obs, reward, done, info = interpreter.execute(response_message, original_query=original_query)

                    content_f = []
                    if isinstance(obs, dict):
                        images = obs.get('multi_modal_data', {}).get('image', [])
                        image_paths = obs.get('multi_modal_data', {}).get('image_paths', [])

                        # Track image paths for logging
                        sandbox_image_paths.extend(image_paths)

                        # Persist process images + trajectory for this turn (human review).
                        if do_archive and archive_dir is not None:
                            self._archive_turn(
                                archive_dir=archive_dir,
                                turn=interpreter.execution_count,
                                code_response=response_message,
                                obs=obs,
                                image_paths=image_paths,
                            )

                        # Embed execution textual output (strip control tokens)
                        execution_text = obs['prompt']
                        # Remove system specific tokens for readability
                        execution_text = execution_text.replace("\n<|im_start|>user\n", "").replace("<|im_end|>\n<|im_start|>assistant\n", "")

                        # Split on "<image>" placeholders and interleave image_url
                        # blocks at each placeholder position, so images land inside
                        # <sandbox_output>...</sandbox_output> instead of trailing it.
                        # <image> is plain text (not a Qwen special token), so it does
                        # not double-count image_pad vs. the image_url blocks.
                        if images:
                            parts = execution_text.split("<image>")
                            for part_idx, part in enumerate(parts):
                                if part:
                                    content_f.append({"type": "text", "text": part})
                                if part_idx < len(images):
                                    im = images[part_idx]
                                    try:
                                        im_b64 = encode_pil_image_to_base64(im)
                                        img_path = image_paths[part_idx] if part_idx < len(image_paths) else None
                                        content_f.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{im_b64}"}, "_log_path": img_path})
                                    except Exception: pass
                        else:
                            content_f.append({"type": "text", "text": execution_text.replace("<image>", "")})
                    elif isinstance(obs, str):
                        content_f.append({"type": "text", "text": obs})
                    input_msgs.append({"role": "user", "content": content_f})
                    if done:
                        if do_archive:
                            self._archive_final(archive_dir, sample_index, original_image_path, response_message, sample_meta)
                        return ret_code, response_message, response

                    try_count += 1
                else:
                    break
            ret = (ret_code, response_message, response)
        except Exception as e:
            self.logger.error(f"Error in tool use generation: {e}")
        finally:
            # Redact images from input messages
            placeholder = '<REDACTED_IMAGE>'
            i = 0
            while i < len(inputs) and inputs[i]['type'] == 'image':
                placeholder = inputs[i]['value']
                i += 1
            self.safe_append_array.append(self.redact_images(input_msgs, placeholder=placeholder))
            # Persist the complete (image-redacted) conversation history + final
            # response, so the whole multi-turn trajectory is recoverable.
            if do_archive and archive_dir is not None:
                self._archive_full_history(archive_dir, input_msgs, ret[1], sample_meta)

        if do_archive and ret[0] == 0:
            self._archive_final(archive_dir, sample_index, original_image_path, ret[1], sample_meta)
        return ret


if __name__ == '__main__':
    from unittest.mock import patch, MagicMock
    
    # Mock environment variables and create instance
    with patch.dict(os.environ, {'LMDEPLOY_API_KEY': 'test-key', 'LMDEPLOY_API_BASE': 'http://0.0.0.0:8000/v1/chat/completions'}):
        with patch('requests.get') as mock_get:
            # Mock the model list response
            mock_response = MagicMock()
            mock_response.json.return_value = {'data': [{'id': 'test-model'}]}
            mock_get.return_value = mock_response
            
            # Create instance
            api = LMDeployAPIWithToolUse(
                model='test-model',
                use_tool=True,
                tool_start_token='<code>',
                tool_end_token='</code>',
                verbose=True
            )
    
    # Test 1: Text-only input
    print("Test 1: Text-only input")
    inputs = [{'type': 'text', 'value': 'What is 2+2?'}]
    result = api.prepare_inputs(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {result}")
    print()
    
    # Test 2: Multi-turn conversation
    print("Test 2: Multi-turn conversation")
    inputs = [
        {'role': 'user', 'content': [{'type': 'text', 'value': 'Hello'}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'value': 'Hi there!'}]},
        {'role': 'user', 'content': [{'type': 'text', 'value': 'How are you?'}]}
    ]
    result = api.prepare_inputs(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {result}")
    print()
    
    # Test 3: Text with image (simulated)
    print("Test 3: Text with image")
    inputs = [
        {'type': 'text', 'value': 'Describe this image'},
        {'type': 'image', 'value': '/path/to/image.jpg'}
    ]
    result = api.prepare_inputs(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {result}")
    print()
    
    # Test 4: generate_inner with mocked response
    print("Test 4: generate_inner test")
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"choices": [{"message": {"content": "The answer is 4"}}]}'
        mock_post.return_value = mock_response
        
        inputs = [{'type': 'text', 'value': 'What is 2+2?'}]
        ret_code, answer, _ = api.generate_inner(inputs)
        print(f"Input: {inputs}")
        print(f"Return code: {ret_code}, Answer: {answer}")
    
    print("All tests completed!")


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

import math
class LMDeployAPIWithCrop(LMDeployAPI):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_tool = kwargs.get('use_tool', False)
        self.tool_start_token = kwargs.get('tool_start_token', None)
        self.tool_end_token = kwargs.get('tool_end_token', None)
        self.verbose = kwargs.get('verbose', False)
        self.max_turns = kwargs.get('max_turns', 10)
        assert self.tool_start_token == '<tool_call>'
        if self.use_tool:
            assert self.tool_start_token and self.tool_end_token, "Both tool_start_token and tool_end_token must be provided when use_tool is True"
        # Process-image archival (mirrors LMDeployAPIWithToolUse). When
        # save_process_images is True, each crop turn's cropped image, bbox and
        # model reasoning are persisted under {process_image_dir}/{sample_index}/.
        self.save_process_images = kwargs.get('save_process_images', False)
        self.process_image_dir = kwargs.get('process_image_dir', None)

    def generate(self, **kwargs):
        return super().generate(**kwargs)

    def _archive_crop_turn(self, archive_dir, turn, model_response, action_list, bbox,
                           normalized_bbox, cropped_image):
        """Persist one crop tool-use turn: the cropped image, bbox and model reasoning.

        Mirrors LMDeployAPIWithToolUse._archive_turn but for the image-crop flow:
        the 'process image' is the cropped PIL image produced this turn.
        """
        archived_img_path = None
        if cropped_image is not None:
            try:
                dst = os.path.join(archive_dir, f'turn{turn}_crop.png')
                cropped_image.save(dst)
                archived_img_path = dst
            except Exception as e:
                self.logger.warning(f'[archive] failed to save crop: {e}')

        tool_name = action_list.get('name', '') if isinstance(action_list, dict) else ''
        traj_entry = {
            'turn': turn,
            'tool': tool_name,
            'bbox': list(bbox) if bbox is not None else None,
            'normalized_bbox': list(normalized_bbox) if normalized_bbox is not None else None,
            'model_response': model_response,
            'image_path': archived_img_path,
        }
        try:
            with open(os.path.join(archive_dir, 'traj.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(traj_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.warning(f'[archive] failed to write traj: {e}')

    def _archive_crop_final(self, archive_dir, sample_index, original_image_path, answer, sample_meta=None):
        """Append the final answer and copy the original input image for reference.

        Same layout as LMDeployAPIWithToolUse._archive_final so traj.jsonl is
        self-contained for later review.
        """
        entry = {'final': True, 'sample_index': sample_index, 'answer': answer}
        if isinstance(sample_meta, dict):
            for k in ('question', 'options', 'gt_answer', 'category'):
                if k in sample_meta:
                    entry[k] = sample_meta[k]
        try:
            with open(os.path.join(archive_dir, 'traj.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception:
            pass
        if original_image_path and os.path.exists(original_image_path):
            import shutil
            ext = os.path.splitext(original_image_path)[1] or '.jpg'
            dst = os.path.join(archive_dir, f'original{ext}')
            try:
                if not os.path.exists(dst):
                    shutil.copy2(original_image_path, dst)
            except Exception:
                pass

    def _archive_full_history(self, archive_dir, input_msgs, final_response, sample_meta=None):
        """Persist the COMPLETE conversation history for a sample.

        ``input_msgs`` is the full multi-turn message list the model saw across
        the entire interaction (system prompt, user query, every assistant
        response, and every crop tool turn), already image-redacted by
        ``redact_images`` (base64 replaced with file paths / placeholder).
        Together with the final response this reproduces the whole trajectory
        — beyond the per-turn crop slices written by
        ``_archive_crop_turn``/``_archive_crop_final``.

        The final response is taken from the last assistant message in
        ``input_msgs`` (which always holds the model's most recent output,
        regardless of which return path was taken); ``final_response`` is only a
        fallback when no assistant message is present.
        """
        import copy
        last_assistant = None
        for msg in reversed(input_msgs):
            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                last_assistant = msg.get('content')
                break
        entry = {
            'full_history': True,
            'input_msgs': copy.deepcopy(input_msgs),
            'final_response': last_assistant if last_assistant is not None else final_response,
        }
        if isinstance(sample_meta, dict):
            for k in ('sample_index', 'question', 'options', 'gt_answer', 'category'):
                if k in sample_meta:
                    entry[k] = sample_meta[k]
        try:
            with open(os.path.join(archive_dir, 'traj.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.warning(f'[archive] failed to write full_history: {e}')

    def generate_inner(self, inputs, **kwargs) -> str:
        # Create unique crop temp directory for this sample (thread-safe)
        import uuid
        session_id = str(uuid.uuid4())[:8]
        crop_temp_dir = f"/tmp/crop_{session_id}"
        os.makedirs(crop_temp_dir, exist_ok=True)

        if "DeepEyes" in self.model:
            user_msg = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "
            for item in inputs[::-1]:
                if item['type'] == 'text':
                    item['value'] += user_msg
                    break
        elif "PixelReasoner" in self.model:
            user_msg = "\n\nGuidelines: Understand the given visual information and the user query. Determine if it is beneficial to employ the given visual operations (tools). For an image, we can look closer by `crop_image_normalized`. Reason with the visual information step by step, and put your final answer within \\boxed{}."
            for item in inputs[::-1]:
                if item['type'] == 'text':
                    item['value'] += user_msg
                    break

        if not self.use_tool:
            return super().generate_inner(inputs, **kwargs)

        # Process-image archival context (thread-local, set by BaseAPI.generate).
        sample_meta = getattr(self._tls, 'sample_meta', None) or {}
        sample_index = sample_meta.get('sample_index', None) if isinstance(sample_meta, dict) else None
        original_image_path = sample_meta.get('image_path', None) if isinstance(sample_meta, dict) else None
        do_archive = bool(self.save_process_images and self.process_image_dir and sample_index is not None)
        archive_dir = None
        if do_archive:
            archive_dir = os.path.join(self.process_image_dir, str(sample_index))
            # Clear any leftover from a previous (failed) retry so traj.jsonl does
            # not interleave two attempts' turns. Each sample_index is handled by
            # exactly one task in track_progress_rich, so no cross-thread race;
            # retries are serial within this thread.
            import shutil
            if os.path.isdir(archive_dir):
                shutil.rmtree(archive_dir, ignore_errors=True)
            os.makedirs(archive_dir, exist_ok=True)

        # Setup interpreter with input images
        input_msgs = self.prepare_inputs(inputs)
        for msg in inputs:
            if msg['type'] == 'image':
                # msg['value'] is the image path
                pil_img = Image.open(msg['value'])
                if original_image_path is None:
                    original_image_path = msg['value']

        temperature = kwargs.pop('temperature', self.temperature)
        self.logger.info(f'Generate temperature: {temperature}')
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        dataset = kwargs.pop('dataset', None)

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}

        response_message = ""
        try_count = 0
        crop_turn_idx = 0  # Track crop turn index for saving images
        ret = (500, self.fail_msg, None)
        try:
            while try_count < self.max_turns:  # Limit number of rounds
                crop_turn_idx += 1
                # Prepare payload with tool stop token
                payload = dict(
                    model=self.model,
                    messages=input_msgs,
                    max_tokens=max_tokens,
                    n=1,
                    temperature=temperature,
                    stop=[self.tool_end_token],
                    include_stop_str_in_output=True,
                    **kwargs)

                response = requests.post(
                    self.api_base,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout * 1.1)

                ret_code = response.status_code
                ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code

                if ret_code != 0:
                    return ret_code, self.fail_msg, response

                try:
                    resp_struct = json.loads(response.text)
                    response_message = resp_struct['choices'][0]['message']['content'].strip()
                except:
                    return ret_code, self.fail_msg, response

                # Add assistant response to message history
                input_msgs.append({"role": "assistant", "content": response_message})

                # Cascade answer extraction: try \boxed{} first, then <answer> tags
                # Extract \boxed{} format (PixelReasoner style)
                boxed_answers = extract_tool_call_contents("\\boxed{", "}", response_message)
                if boxed_answers:
                    if do_archive:
                        self._archive_crop_final(archive_dir, sample_index, original_image_path, boxed_answers[0], sample_meta)
                    ret = (ret_code, boxed_answers[0], response)
                    return ret

                # Extract <answer> tags (DeepEyes style)
                answer_tag_answers = extract_tool_call_contents("<answer>", "</answer>", response_message)
                if answer_tag_answers:
                    if do_archive:
                        self._archive_crop_final(archive_dir, sample_index, original_image_path, answer_tag_answers[0], sample_meta)
                    ret = (ret_code, answer_tag_answers[0], response)
                    return ret

                # Check for tool usage - add missing end token if needed (model may stop before generating it)
                if self.tool_start_token in response_message and self.tool_end_token not in response_message:
                    response_message = response_message + self.tool_end_token

                if self.tool_start_token in response_message and self.tool_end_token in response_message:
                    action_str = response_message.split(self.tool_start_token)[1].split(self.tool_end_token)[0].strip()
                    # Try to parse as JSON first, fall back to eval if needed
                    try:
                        action_list = json.loads(action_str)
                    except:
                        # Replace single quotes with double quotes for JSON compatibility
                        action_str = action_str.replace("'", '"')
                        try:
                            action_list = json.loads(action_str)
                        except:
                            # Last resort: use eval
                            action_list = eval(action_str)

                    bbox_list = []
                    cropped_pil_image_content_list = []

                    # Check if this is a supported image crop tool (not select_frames or other tools)
                    tool_name = action_list.get('name', '')
                    supported_tools = ['crop_image_normalized', 'image_zoom_in_tool']
                    if tool_name not in supported_tools:
                        self.logger.warning(f"Unsupported tool call: {tool_name}. Skipping tool execution.")
                        break

                    if 'bbox_2d' not in action_list.get('arguments', {}):
                        self.logger.error(f"Tool call missing bbox_2d argument. Action list: {action_list}")
                        break

                    bbox_str = action_list['arguments']['bbox_2d']
                    bbox = bbox_str
                    left, top, right, bottom = bbox

                    # Both crop_image_normalized and image_zoom_in_tool use the same logic
                    if tool_name in ['crop_image_normalized', 'image_zoom_in_tool']:
                        # Normalized coordinates (0-1) with adaptive padding
                        img_x, img_y = pil_img.size
                        # Adaptive padding: cap at 600 pixels to avoid excessive padding on high-res images
                        padding_x = min(0.1, 600.0/img_x)
                        padding_y = min(0.1, 600.0/img_y)

                        # Check if already normalized or need to normalize
                        if bbox[0] < 1 and bbox[1] < 1 and bbox[2] < 1 and bbox[3] < 1:
                            normalized_bbox_2d = (float(bbox[0])-padding_x, float(bbox[1])-padding_y, float(bbox[2])+padding_x, float(bbox[3])+padding_y)
                        else:
                            normalized_bbox_2d = (float(bbox[0])/img_x-padding_x, float(bbox[1])/img_y-padding_y, float(bbox[2])/img_x+padding_x, float(bbox[3])/img_y+padding_y)

                        # Clamp to [0, 1]
                        normalized_x1 = min(max(0, normalized_bbox_2d[0]), 1)
                        normalized_y1 = min(max(0, normalized_bbox_2d[1]), 1)
                        normalized_x2 = min(max(0, normalized_bbox_2d[2]), 1)
                        normalized_y2 = min(max(0, normalized_bbox_2d[3]), 1)

                    # Crop and resize (same for both tools)
                    cropped_image = pil_img.crop((left, top, right, bottom))
                    new_w, new_h = smart_resize((right - left), (bottom - top), factor=IMAGE_FACTOR)
                    cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
                    cropped_pil_image = encode_pil_image_to_base64(cropped_image)
                    bbox_list.append(bbox)
                    cropped_pil_image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_pil_image}"}}
                    cropped_pil_image_content_list.append(cropped_pil_image_content)

                    if len(bbox_list) == 1:
                        bbox_list = bbox_list[0]

                    # Persist process image + trajectory for this crop turn.
                    if do_archive and archive_dir is not None:
                        self._archive_crop_turn(
                            archive_dir=archive_dir,
                            turn=crop_turn_idx,
                            model_response=response_message,
                            action_list=action_list,
                            bbox=bbox,
                            normalized_bbox=normalized_bbox_2d,
                            cropped_image=cropped_image,
                        )

                    content_f = []
                    if "DeepEyes" in self.model:
                        content_f.append({"type": "text", "text": "<tool_response>"})
                    else:
                        content_f.append({"type": "text", "text": f"Here is the cropped image (Image Size: {cropped_image.size[0]}x{cropped_image.size[1]}):"})
                    for cropped_pil_image_content in cropped_pil_image_content_list:
                        content_f.append(cropped_pil_image_content)
                    if "DeepEyes" in self.model:
                        content_f.append({"type": "text", "text": user_msg})
                        content_f.append({"type": "text", "text": "</tool_response>"})

                    input_msgs.append({"role": "user", "content": content_f})

                    try_count += 1
                else:
                    # No tool usage detected, return the response
                    break

            ret = (ret_code, response_message, response)
        except Exception as e:
            self.logger.error(f"Error in tool use generation: {e}")
        finally:
            # Redact images from input messages
            placeholder = '<REDACTED_IMAGE>'
            i = 0
            while i < len(inputs) and inputs[i]['type'] == 'image':
                placeholder = inputs[i]['value']
                i += 1
            self.safe_append_array.append(self.redact_images(input_msgs, placeholder=placeholder))
            # Persist the complete (image-redacted) conversation history + final
            # response, so the whole multi-turn trajectory is recoverable.
            if do_archive and archive_dir is not None:
                self._archive_full_history(archive_dir, input_msgs, ret[1], sample_meta)

        # Archive the final answer for the no-tool / loop-exhausted path.
        if do_archive and ret[0] == 0:
            self._archive_crop_final(archive_dir, sample_index, original_image_path, ret[1], sample_meta)
        return ret


# the following code is copied from qwen-vl-utils
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


if __name__ == '__main__':
    from unittest.mock import patch, MagicMock
    
    # Mock environment variables and create instance
    with patch.dict(os.environ, {'LMDEPLOY_API_KEY': 'test-key', 'LMDEPLOY_API_BASE': 'http://0.0.0.0:8000/v1/chat/completions'}):
        with patch('requests.get') as mock_get:
            # Mock the model list response
            mock_response = MagicMock()
            mock_response.json.return_value = {'data': [{'id': 'test-model'}]}
            mock_get.return_value = mock_response
            
            # Create instance
            api = LMDeployAPIWithToolUse(
                model='test-model',
                use_tool=True,
                tool_start_token='<code>',
                tool_end_token='</code>',
                verbose=True
            )
    
    # Test 1: Text-only input
    print("Test 1: Text-only input")
    inputs = [{'type': 'text', 'value': 'What is 2+2?'}]
    result = api.prepare_inputs(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {result}")
    print()
    
    # Test 2: Multi-turn conversation
    print("Test 2: Multi-turn conversation")
    inputs = [
        {'role': 'user', 'content': [{'type': 'text', 'value': 'Hello'}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'value': 'Hi there!'}]},
        {'role': 'user', 'content': [{'type': 'text', 'value': 'How are you?'}]}
    ]
    result = api.prepare_inputs(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {result}")
    print()
    
    # Test 3: Text with image (simulated)
    print("Test 3: Text with image")
    inputs = [
        {'type': 'text', 'value': 'Describe this image'},
        {'type': 'image', 'value': '/path/to/image.jpg'}
    ]
    result = api.prepare_inputs(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {result}")
    print()
    
    # Test 4: generate_inner with mocked response
    print("Test 4: generate_inner test")
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"choices": [{"message": {"content": "The answer is 4"}}]}'
        mock_post.return_value = mock_response
        
        inputs = [{'type': 'text', 'value': 'What is 2+2?'}]
        ret_code, answer, _ = api.generate_inner(inputs)
        print(f"Input: {inputs}")
        print(f"Return code: {ret_code}, Answer: {answer}")
    
    print("All tests completed!")
