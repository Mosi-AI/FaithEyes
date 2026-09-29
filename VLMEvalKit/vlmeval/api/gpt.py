from ..smp import *
import os
import sys
import copy as cp
from .base import BaseAPI

APIBASES = {
    'OFFICIAL': 'https://api.openai.com/v1/chat/completions',
}


def _derive_responses_api_base(api_base: str) -> str:
    """Return a responses endpoint for the given chat completions base."""
    if api_base is None:
        return None
    if '/responses' in api_base:
        return api_base
    if '/chat/completions' in api_base:
        return api_base.replace('/chat/completions', '/responses')
    return api_base.rstrip('/') + '/responses'


def _strip_endpoint_from_base(api_base: str) -> str:
    """Strip trailing endpoint segment to get the root base url."""
    if api_base is None:
        return None
    for suffix in ('/responses', '/chat/completions'):
        if suffix in api_base:
            return api_base.split(suffix)[0]
    return api_base


def _build_response_messages(raw_msgs, strip_log_path=False):
    """Convert chat-completions style messages to Responses API schema."""
    messages = cp.deepcopy(raw_msgs)
    if strip_log_path:
        for msg in messages:
            content = msg.get('content')
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        part.pop('_log_path', None)
    response_msgs = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if isinstance(content, str):
            content_list = [dict(type='text', text=content)]
        elif isinstance(content, dict):
            content_list = [content]
        else:
            content_list = content
        if content_list is None:
            content_list = []
        converted_parts = []
        for part in content_list:
            if isinstance(part, str):
                part = dict(type='text', text=part)
            if not isinstance(part, dict):
                continue
            p_type = part.get('type', 'text')
            if p_type in ['text', 'input_text', 'output_text']:
                text_value = part.get('text') or part.get('value') or ''
                if not text_value:
                    continue
                if p_type not in ['input_text', 'output_text']:
                    p_type = 'output_text' if role == 'assistant' else 'input_text'
                converted_parts.append(dict(type=p_type, text=text_value))
            elif p_type in ['image_url', 'input_image']:
                image_payload = part.get('image_url') or part.get('image')
                if image_payload is None:
                    continue
                # Extract URL string if image_payload is a dict
                if isinstance(image_payload, dict):
                    image_url = image_payload.get('url')
                else:
                    image_url = image_payload
                if not image_url:
                    continue
                converted_parts.append(dict(type='input_image', image_url=image_url))
            else:
                # Pass through other content types (e.g., tool_calls) unchanged
                converted_parts.append(part)
        if not converted_parts and isinstance(content, str):
            converted_parts.append(dict(
                type='output_text' if role == 'assistant' else 'input_text', text=content))
        response_msgs.append(dict(role=role, content=converted_parts))
    return response_msgs


def _extract_text_from_responses(resp_obj):
    """Extract concatenated assistant text from Responses API outputs."""
    outputs = getattr(resp_obj, 'output', None)
    if outputs is None and isinstance(resp_obj, dict):
        outputs = resp_obj.get('output', [])
    text_chunks = []
    if outputs is None:
        return ''
    for item in outputs:
        content = None
        if isinstance(item, dict):
            content = item.get('content', [])
        else:
            content = getattr(item, 'content', None)
        if content is None:
            continue
        for part in content:
            if isinstance(part, dict):
                p_type = part.get('type')
                if p_type in ('output_text', 'text'):
                    text_chunks.append(part.get('text', ''))
            else:
                p_type = getattr(part, 'type', None)
                if p_type in ('output_text', 'text'):
                    text_chunks.append(getattr(part, 'text', ''))
    return '\n'.join([t for t in text_chunks if t]).strip()


def GPT_context_window(model):
    length_map = {
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-turbo-preview': 128000,
        'gpt-4-1106-preview': 128000,
        'gpt-4-0125-preview': 128000,
        'gpt-4-vision-preview': 128000,
        'gpt-4-turbo': 128000,
        'gpt-4-turbo-2024-04-09': 128000,
        'gpt-3.5-turbo': 16385,
        'gpt-3.5-turbo-0125': 16385,
        'gpt-3.5-turbo-1106': 16385,
        'gpt-3.5-turbo-instruct': 4096,
    }
    if model in length_map:
        return length_map[model]
    else:
        return 128000


class OpenAIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-3.5-turbo-0613',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 300,
                 api_base: str = None,
                 max_tokens: int = 2048,
                 img_size: int = 512,
                 img_detail: str = 'low',
                 use_azure: bool = False,
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_azure = use_azure

        if 'step' in model:
            env_key = os.environ.get('STEPAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'yi-vision' in model:
            env_key = os.environ.get('YI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'internvl2-pro' in model:
            env_key = os.environ.get('InternVL2_PRO_KEY', '')
            if key is None:
                key = env_key
        elif 'abab' in model:
            env_key = os.environ.get('MiniMax_API_KEY', '')
            if key is None:
                key = env_key
        elif 'moonshot' in model:
            env_key = os.environ.get('MOONSHOT_API_KEY', '')
            if key is None:
                key = env_key
        elif 'grok' in model:
            env_key = os.environ.get('XAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'gemini' in model and 'preview' in model:
            # Will only handle preview models
            env_key = os.environ.get('GOOGLE_API_KEY', '')
            if key is None:
                key = env_key
            api_base = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        elif 'ernie' in model:
            env_key = os.environ.get('BAIDU_API_KEY', '')
            if key is None:
                key = env_key
            api_base = 'https://qianfan.baidubce.com/v2/chat/completions'
            self.baidu_appid = os.environ.get('BAIDU_APP_ID', None)
        else:
            if use_azure:
                env_key = os.environ.get('AZURE_OPENAI_API_KEY', None)
                assert env_key is not None, 'Please set the environment variable AZURE_OPENAI_API_KEY. '

                if key is None:
                    key = env_key
                assert isinstance(key, str), (
                    'Please set the environment variable AZURE_OPENAI_API_KEY to your openai key. '
                )
            else:
                env_key = os.environ.get('OPENAI_API_KEY', '')
                if key is None:
                    key = env_key
                assert isinstance(key, str) and key.startswith('sk-'), (
                    f'Illegal openai_key {key}. '
                    'Please set the environment variable OPENAI_API_KEY to your openai key. '
                )

        self.key = key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail
        self.timeout = timeout
        # Reasoning models: use max_completion_tokens and don't support temperature
        self.o1_model = ('o1' in model) or ('o3' in model) or ('o4' in model) or ('gpt-5' in model)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if use_azure:
            api_base_template = (
                '{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}'
            )
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', None)
            assert endpoint is not None, 'Please set the environment variable AZURE_OPENAI_ENDPOINT. '
            deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', None)
            assert deployment_name is not None, 'Please set the environment variable AZURE_OPENAI_DEPLOYMENT_NAME. '
            api_version = os.getenv('OPENAI_API_VERSION', None)
            assert api_version is not None, 'Please set the environment variable OPENAI_API_VERSION. '

            self.api_base = api_base_template.format(
                endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                api_version=os.getenv('OPENAI_API_VERSION')
            )
        else:
            if api_base is None:
                if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] != '':
                    self.logger.info('Environment variable OPENAI_API_BASE is set. Will use it as api_base. ')
                    api_base = os.environ['OPENAI_API_BASE']
                else:
                    api_base = 'OFFICIAL'

            assert api_base is not None

            if api_base in APIBASES:
                self.api_base = APIBASES[api_base]
            elif api_base.startswith('http'):
                # Automatically append the /chat/completions path (if not already included)
                if '/chat/completions' not in api_base:
                    api_base = api_base.rstrip('/') + '/chat/completions'
                self.api_base = api_base
            else:
                self.logger.error('Unknown API Base. ')
                raise NotImplementedError
            if os.environ.get('BOYUE', None):
                self.api_base = os.environ.get('BOYUE_API_BASE')
                self.key = os.environ.get('BOYUE_API_KEY')

        self.logger.info(f'Using API Base: {self.api_base}; API Key: {self.key}')

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
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
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        # Will send request if use Azure, dk how to use openai client for it
        if self.use_azure:
            headers = {'Content-Type': 'application/json', 'api-key': self.key}
        elif 'internvl2-pro' in self.model:
            headers = {'Content-Type': 'application/json', 'Authorization': self.key}
        else:
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        if hasattr(self, 'baidu_appid'):
            headers['appid'] = self.baidu_appid

        payload = dict(
            model=self.model,
            messages=input_msgs,
            n=1,
            temperature=temperature,
            **kwargs)

        if self.o1_model:
            payload['max_completion_tokens'] = max_tokens
            payload.pop('temperature')
        else:
            payload['max_tokens'] = max_tokens

        if 'gemini' in self.model:
            payload.pop('max_tokens')
            payload.pop('n')
            payload['reasoning_effort'] = 'high'

        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            message = resp_struct['choices'][0]['message']
            # Support the reasoning_content field of GLM-style models
            if message.get('content') is not None:
                answer = message['content'].strip()
            elif message.get('reasoning_content') is not None:
                answer = message['reasoning_content'].strip()
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(response.text if hasattr(response, 'text') else response)

        return ret_code, answer, response

    def get_image_token_len(self, img_path, detail='low'):
        import math
        if detail == 'low':
            return 85

        im = Image.open(img_path)
        height, width = im.size
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024

        h = math.ceil(height / 512)
        w = math.ceil(width / 512)
        total = 85 + 170 * h * w
        return total

    def get_token_len(self, inputs) -> int:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except Exception as err:
            if 'gpt' in self.model.lower():
                if self.verbose:
                    self.logger.warning(f'{type(err)}: {err}')
                enc = tiktoken.encoding_for_model('gpt-4')
            else:
                return 0
        assert isinstance(inputs, list)
        tot = 0
        for item in inputs:
            if 'role' in item:
                tot += self.get_token_len(item['content'])
            elif item['type'] == 'text':
                tot += len(enc.encode(item['value']))
            elif item['type'] == 'image':
                tot += self.get_image_token_len(item['value'], detail=self.img_detail)
        return tot


class GPT4V(OpenAIWrapper):

    def generate(self, message, dataset=None):
        return super(GPT4V, self).generate(message)


class GPTWithToolUseSDK(OpenAIWrapper):
    """OpenAI wrapper with Python code execution using official OpenAI SDK."""

    def __init__(self,
                 use_tool: bool = False,
                 tool_start_token: str = '<code>',
                 tool_end_token: str = '</code>',
                 save_file: str = 'saved_results.jsonl',
                 **kwargs):
        # Remove tool-specific kwargs before passing to parent
        self.use_tool = use_tool
        self.tool_start_token = tool_start_token
        self.tool_end_token = tool_end_token
        self.save_file = save_file

        super().__init__(**kwargs)
        self.responses_api_base = _derive_responses_api_base(self.api_base)

        # Thread-safe logging array
        from vlmeval.api.lmdeploy import ThreadSafeAppendOnlyArray
        self.safe_append_array = ThreadSafeAppendOnlyArray()

        # Initialize OpenAI client
        from openai import OpenAI
        base_url = _strip_endpoint_from_base(self.responses_api_base or self.api_base)
        self.client = OpenAI(api_key=self.key, base_url=base_url)

    def generate(self, message, dataset=None):
        """Generate with tool use support and logging."""
        ret = super().generate(message, dataset=dataset)

        # Save logs
        with open(self.save_file, 'a') as f:
            self.safe_append_array.log_records_append_only(f)

        return ret

    def setup_interpreter_with_images(self, inputs):
        """Setup Python interpreter with input images."""
        if not self.use_tool:
            return None

        from vlmeval.utils import PythonInterpreter
        from PIL import Image

        # Extract image path and filename from inputs
        image_path = None
        image_filename = None
        aux_img_dir = None
        image_size = None

        for msg in inputs:
            if msg['type'] == 'image':
                image_path = msg['value']
                image_filename = os.path.basename(image_path)
                aux_img_dir = os.path.dirname(image_path)
                break

        # Create interpreter instance with aux_img_dir
        interpreter = PythonInterpreter("python", "Python code execution", {}, aux_img_dir=aux_img_dir)

        # Extract PIL Images from inputs
        images = []
        for msg in inputs:
            if msg['type'] == 'image':
                img = Image.open(msg['value'])
                image_size = [img.width, img.height]
                images.append(img)

        if images:
            # Create extra_info with image metadata
            extra_info = {
                'image_file_name': image_filename,
                'image_file_path': image_path,
                'image_size': image_size,
            }

            # Reset interpreter with PIL Images
            multi_modal_data = {'image': images}
            interpreter.reset(inputs, multi_modal_data, multi_modal_data, extra_info=extra_info)

        return interpreter

    def generate_inner(self, inputs, **kwargs):
        """Generate with tool execution loop using OpenAI SDK."""
        if not self.use_tool:
            # Log without tool use
            ret_code, answer, response = super().generate_inner(inputs, **kwargs)

            logging_msgs = []
            logging_msgs.append({"role": "system", "content": self.system_prompt})
            logging_msgs.append({"role": "user", "content": inputs})
            logging_msgs.append({"role": "assistant", "content": answer})
            self.safe_append_array.append(logging_msgs)

            return ret_code, answer, response

        # Setup interpreter with input images
        interpreter = self.setup_interpreter_with_images(inputs)

        # Append image filename hint to inputs so model knows what file to open
        from PIL import Image
        image_filename = None
        image_size = None
        for msg in inputs:
            if msg['type'] == 'image':
                image_filename = os.path.basename(msg['value'])
                img = Image.open(msg['value'])
                image_size = (img.width, img.height)
                break

        # Append image filename and size info to the last text input
        if image_filename and image_size:
            filename_hint = f"""

### User Image Path:** \"{image_filename}\"
### User Image Size:** \"{image_size[0]}x{image_size[1]}\"

### **Output Format (strict adherence required):**


<think>Your detailed reasoning process, including any code, should go here.</think>
<answer>Your final answer to the user's question goes here.</answer>
"""
            for msg in reversed(inputs):
                if msg['type'] == 'text':
                    msg['value'] += filename_hint
                    break

        input_msgs = self.prepare_inputs(inputs)

        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        kwargs.pop('dataset', None)  # Remove dataset parameter - OpenAI doesn't accept it

        response_message = ""
        try_count = 0
        ret = (500, self.fail_msg, None)
        response = None

        from vlmeval.utils.python_tool import extract_tool_call_contents
        import base64
        from io import BytesIO

        def encode_pil_image_to_base64(image):
            """Convert PIL Image to base64 string."""
            buffer = BytesIO()
            image.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return img_str

        try:
            while try_count < 10:  # Limit number of rounds
                # Prepare kwargs for Responses API
                api_msgs = _build_response_messages(input_msgs, strip_log_path=True)
                completion_kwargs = {
                    'model': self.model,
                    'input': api_msgs,
                    **kwargs
                }

                completion_kwargs['max_output_tokens'] = max_tokens
                if not self.o1_model:
                    completion_kwargs['temperature'] = temperature

                # Call OpenAI SDK Responses API
                try:
                    completion = self.client.responses.create(**completion_kwargs)
                    response_message = _extract_text_from_responses(completion) or ''
                    ret_code = 0
                    response = completion
                except Exception as e:
                    if self.verbose:
                        self.logger.error(f"OpenAI SDK error: {e}")
                    return 500, self.fail_msg, None

                # Add assistant response to message history
                input_msgs.append({"role": "assistant", "content": response_message})
                interpreter._log(f"Response_{interpreter.execution_count}", response_message)

                # Check for final answer
                answers = extract_tool_call_contents("<answer>", "</answer>", response_message)
                if answers:
                    ret = (ret_code, answers[0], response)
                    break

                # Check for tool usage - add missing end token if needed (OpenAI excludes stop string)
                if self.tool_start_token in response_message and self.tool_end_token not in response_message:
                    response_message = response_message + self.tool_end_token

                if self.tool_start_token in response_message and self.tool_end_token in response_message:
                    obs, reward, done, info = interpreter.execute(response_message)

                    content_f = []
                    if isinstance(obs, dict):
                        images = obs.get('multi_modal_data', {}).get('image', [])
                        image_paths = obs.get('multi_modal_data', {}).get('image_paths', [])

                        # Embed execution textual output
                        execution_text = obs['prompt']
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
                                        content_f.append({
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/jpeg;base64,{im_b64}"}
                                        })
                                    except Exception:
                                        pass
                        else:
                            content_f.append({"type": "text", "text": execution_text.replace("<image>", "")})

                    elif isinstance(obs, str):
                        content_f.append({"type": "text", "text": obs})

                    input_msgs.append({"role": "user", "content": content_f})

                    # If interpreter signals completion, return
                    if done:
                        ret = (ret_code, response_message, response)
                        break

                    try_count += 1
                else:
                    # No tool usage detected, return the response
                    ret = (ret_code, response_message, response)
                    break

        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error in tool use generation: {e}")
        finally:
            # Log the conversation
            placeholder = '<REDACTED_IMAGE>'
            i = 0
            while i < len(inputs) and inputs[i]['type'] == 'image':
                placeholder = inputs[i]['value']
                i += 1

            # Redact images for logging
            logged_msgs = []
            for msg in input_msgs:
                msg_copy = msg.copy()
                if isinstance(msg_copy.get('content'), list):
                    content_copy = []
                    for c in msg_copy['content']:
                        c_copy = c.copy()
                        if c_copy.get('type') == 'image_url':
                            c_copy['image_url'] = placeholder
                        content_copy.append(c_copy)
                    msg_copy['content'] = content_copy
                logged_msgs.append(msg_copy)

            self.safe_append_array.append(logged_msgs)

        return ret


class GPTWithToolUse(OpenAIWrapper):
    """OpenAI wrapper with Python code execution capabilities."""

    def __init__(self,
                 use_tool: bool = False,
                 tool_start_token: str = '<code>',
                 tool_end_token: str = '</code>',
                 save_file: str = 'saved_results.jsonl',
                 **kwargs):
        # Remove tool-specific kwargs before passing to parent
        self.use_tool = use_tool
        self.tool_start_token = tool_start_token
        self.tool_end_token = tool_end_token
        self.save_file = save_file

        super().__init__(**kwargs)
        self.responses_api_base = _derive_responses_api_base(self.api_base)

        # Thread-safe logging array
        from vlmeval.api.lmdeploy import ThreadSafeAppendOnlyArray
        self.safe_append_array = ThreadSafeAppendOnlyArray()

    def generate(self, message, dataset=None):
        """Generate with tool use support and logging."""
        ret = super().generate(message, dataset=dataset)

        # Save logs
        with open(self.save_file, 'a') as f:
            self.safe_append_array.log_records_append_only(f)

        return ret

    def redact_images(self, input_msgs, placeholder='<REDACTED_IMAGE>'):
        """Replace image base64 data with file paths for logging.

        For sandbox-generated images, use the actual file path from _log_path metadata.
        For original input images, use the placeholder (original image path).
        """
        for msg in input_msgs:
            if "content" in msg and isinstance(msg['content'], list):
                for c in msg['content']:
                    if c.get('type') == 'image_url':
                        # Check if this has _log_path metadata (sandbox-generated image)
                        if '_log_path' in c and c['_log_path']:
                            # Use the actual file path from sandbox
                            c['image_url'] = c['_log_path']
                            # Remove metadata
                            del c['_log_path']
                        else:
                            # Original input image - use placeholder
                            c['image_url'] = placeholder
        return input_msgs

    def setup_interpreter_with_images(self, inputs):
        """Setup Python interpreter with input images."""
        if not self.use_tool:
            return None

        from vlmeval.utils import PythonInterpreter
        from PIL import Image

        # Extract image path and filename from inputs
        image_path = None
        image_filename = None
        aux_img_dir = None
        image_size = None

        for msg in inputs:
            if msg['type'] == 'image':
                image_path = msg['value']
                image_filename = os.path.basename(image_path)
                aux_img_dir = os.path.dirname(image_path)
                break

        # Create interpreter instance with aux_img_dir
        interpreter = PythonInterpreter("python", "Python code execution", {}, aux_img_dir=aux_img_dir)

        # Extract PIL Images from inputs
        images = []
        for msg in inputs:
            if msg['type'] == 'image':
                img = Image.open(msg['value'])
                image_size = [img.width, img.height]
                images.append(img)

        if images:
            # Create extra_info with image metadata
            extra_info = {
                'image_file_name': image_filename,
                'image_file_path': image_path,
                'image_size': image_size,
            }

            # Reset interpreter with PIL Images
            multi_modal_data = {'image': images}
            interpreter.reset(inputs, multi_modal_data, multi_modal_data, extra_info=extra_info)

        return interpreter

    def generate_inner(self, inputs, **kwargs):
        """Generate with tool execution loop."""
        if not self.use_tool:
            # Log without tool use
            ret_code, answer, response = super().generate_inner(inputs, **kwargs)

            logging_msgs = []
            logging_msgs.append({"role": "system", "content": self.system_prompt})
            logging_msgs.append({"role": "user", "content": inputs})
            logging_msgs.append({"role": "assistant", "content": answer})
            self.safe_append_array.append(logging_msgs)

            return ret_code, answer, response

        # Extract image filename and size to append to user prompt
        image_filename = None
        image_size = None
        for msg in inputs:
            if msg['type'] == 'image':
                from PIL import Image
                image_filename = os.path.basename(msg['value'])
                img = Image.open(msg['value'])
                image_size = (img.width, img.height)
                break

        # Append image filename and size info to the last text input (matches training format)
        if image_filename and image_size:
            filename_hint = f"""

### User Image Path:** \"{image_filename}\"
### User Image Size:** \"{image_size[0]}x{image_size[1]}\"

### **Output Format (strict adherence required):**


<think>Your detailed reasoning process, including any code, should go here.</think>
<answer>Your final answer to the user's question goes here.</answer>
"""
            for msg in reversed(inputs):
                if msg['type'] == 'text':
                    msg['value'] += filename_hint
                    break

        # Setup interpreter with input images
        interpreter = self.setup_interpreter_with_images(inputs)
        input_msgs = self.prepare_inputs(inputs)

        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        kwargs.pop('dataset', None)  # Remove dataset parameter - OpenAI doesn't accept it

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}

        response_message = ""
        try_count = 0
        ret = (500, self.fail_msg, None)

        from vlmeval.utils.python_tool import extract_tool_call_contents
        import base64
        from io import BytesIO

        def encode_pil_image_to_base64(image):
            """Convert PIL Image to base64 string."""
            buffer = BytesIO()
            image.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return img_str

        try:
            while try_count < 10:  # Limit number of rounds
                api_msgs = _build_response_messages(input_msgs, strip_log_path=True)

                # Prepare payload for Responses API
                payload = dict(
                    model=self.model,
                    input=api_msgs,
                    **kwargs)

                payload['max_output_tokens'] = max_tokens
                if not self.o1_model:
                    payload['temperature'] = temperature

                response = requests.post(
                    self.responses_api_base or self.api_base,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout * 1.1)

                ret_code = response.status_code
                ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code

                if ret_code != 0:
                    return ret_code, self.fail_msg, response

                try:
                    resp_struct = json.loads(response.text)
                    response_message = _extract_text_from_responses(resp_struct) or ''
                except:
                    return ret_code, self.fail_msg, response

                # Add assistant response to message history
                input_msgs.append({"role": "assistant", "content": response_message})
                interpreter._log(f"Response_{interpreter.execution_count}", response_message)

                # Check for final answer
                answers = extract_tool_call_contents("<answer>", "</answer>", response_message)
                if answers:
                    ret = (ret_code, answers[0], response)
                    break

                # Check for tool usage - add missing end token if needed (OpenAI excludes stop string)
                if self.tool_start_token in response_message and self.tool_end_token not in response_message:
                    response_message = response_message + self.tool_end_token

                if self.tool_start_token in response_message and self.tool_end_token in response_message:
                    obs, reward, done, info = interpreter.execute(response_message)

                    content_f = []
                    if isinstance(obs, dict):
                        images = obs.get('multi_modal_data', {}).get('image', [])
                        image_paths = obs.get('multi_modal_data', {}).get('image_paths', [])

                        # Embed execution textual output
                        execution_text = obs['prompt']
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
                                        content_f.append({
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/jpeg;base64,{im_b64}"},
                                            "_log_path": img_path  # For logging - removed before API call
                                        })
                                    except Exception:
                                        pass
                        else:
                            content_f.append({"type": "text", "text": execution_text.replace("<image>", "")})

                    elif isinstance(obs, str):
                        content_f.append({"type": "text", "text": obs})

                    input_msgs.append({"role": "user", "content": content_f})

                    # If interpreter signals completion, return
                    if done:
                        ret = (ret_code, response_message, response)
                        break

                    try_count += 1
                else:
                    # No tool usage detected, return the response
                    ret = (ret_code, response_message, response)
                    break

            # If loop exits naturally (max iterations reached), return last response
            ret = (ret_code, response_message, response)
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error in tool use generation: {e}")
        finally:
            # Log the conversation with images redacted to file paths
            placeholder = '<REDACTED_IMAGE>'
            i = 0
            while i < len(inputs) and inputs[i]['type'] == 'image':
                placeholder = inputs[i]['value']
                i += 1

            # Use redact_images to replace base64 with file paths
            self.safe_append_array.append(self.redact_images(input_msgs, placeholder=placeholder))

        return ret
