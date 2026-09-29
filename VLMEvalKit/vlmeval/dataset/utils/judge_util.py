import os
import requests
from ...smp import load_env

INTERNAL = os.environ.get('INTERNAL', 0)


def get_model_from_api_base():
    """Auto-detect model name from OPENAI_API_BASE by querying /v1/models endpoint."""
    api_base = os.environ.get('OPENAI_API_BASE', None)
    api_key = os.environ.get('OPENAI_API_KEY', '')

    if not api_base: return None

    # Extract the base URL from api_base (strip paths like /v1/chat/completions)
    # e.g.: http://host:port/v1/chat/completions -> http://host:port/v1
    if '/v1/' in api_base:
        base_url = api_base.split('/v1/')[0] + '/v1'
    else:
        base_url = api_base.rstrip('/')
    models_url = base_url + '/models'

    try:
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        response = requests.get(models_url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                # Return the first available model
                model_id = data['data'][0].get('id', None)
                if model_id:
                    # Return the full model ID (a vLLM server needs the full path as the model name)
                    # If a simplified name is needed for file names, handle it elsewhere
                    return model_id
    except Exception as e:
        print(f"[Judge] Failed to auto-detect model from API: {e}")

    return None


def get_short_model_name(model_name):
    """Get a filesystem-safe short name from a model name/path.

    For model paths like '/cpfs01/user/models/Qwen3-VL-32B-Instruct',
    returns 'Qwen3-VL-32B-Instruct' for use in file names.
    """
    if model_name is None:
        return None
    # If it is in path format, take only the last part
    if '/' in model_name:
        return model_name.rstrip('/').split('/')[-1]
    return model_name


def build_judge(**kwargs):
    from ...api import OpenAIWrapper, SiliconFlowAPI, HFChatModel
    model = kwargs.pop('model', None)
    kwargs.pop('nproc', None)
    load_env()

    # Auto-detect model from OPENAI_API_BASE if not specified
    if model is None:
        model = get_model_from_api_base()
        if model: print(f"[Judge] Auto-detected model from OPENAI_API_BASE: {model}")

    LOCAL_LLM = os.environ.get('LOCAL_LLM', None)
    if LOCAL_LLM is None:
        model_map = {
            'gpt-4-turbo': 'gpt-4-1106-preview',
            'gpt-4-0613': 'gpt-4-0613',
            'gpt-4-0125': 'gpt-4-0125-preview',
            'gpt-4-0409': 'gpt-4-turbo-2024-04-09',
            'chatgpt-1106': 'gpt-3.5-turbo-1106',
            'chatgpt-0125': 'gpt-3.5-turbo-0125',
            'gpt-4o': 'gpt-4o-2024-05-13',
            'gpt-4o-0806': 'gpt-4o-2024-08-06',
            'gpt-4o-1120': 'gpt-4o-2024-11-20',
            'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
            'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
            'qwen-72b': 'Qwen/Qwen2.5-72B-Instruct',
            'deepseek': 'deepseek-ai/DeepSeek-V3',
            'llama31-8b': 'meta-llama/Llama-3.1-8B-Instruct',
        }
        model_version = model_map[model] if model in model_map else model
    else:
        model_version = LOCAL_LLM

    if model in ['qwen-7b', 'qwen-72b', 'deepseek']:
        model = SiliconFlowAPI(model_version, **kwargs)
    elif model == 'llama31-8b':
        model = HFChatModel(model_version, **kwargs)
    else:
        model = OpenAIWrapper(model_version, **kwargs)
    return model


DEBUG_MESSAGE = """
To debug the OpenAI API, you can try the following scripts in python:
```python
from vlmeval.api import OpenAIWrapper
model = OpenAIWrapper('gpt-4o', verbose=True)
msgs = [dict(type='text', value='Hello!')]
code, answer, resp = model.generate_inner(msgs)
print(code, answer, resp)
```
You cam see the specific error if the API call fails.
"""
