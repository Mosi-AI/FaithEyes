import json
import os
import subprocess
from functools import partial


# GET the number of GPUs on the node without importing libs like torch
def get_gpu_list():
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if CUDA_VISIBLE_DEVICES != '':
        gpu_list = [int(x) for x in CUDA_VISIBLE_DEVICES.split(',')]
        return gpu_list
    try:
        ps = subprocess.Popen(('nvidia-smi', '--list-gpus'), stdout=subprocess.PIPE)
        output = subprocess.check_output(('wc', '-l'), stdin=ps.stdout)
        return list(range(int(output)))
    except:
        return []


RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE",1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK",1))

GPU_LIST = get_gpu_list()
if LOCAL_WORLD_SIZE > 1 and len(GPU_LIST):
    NGPU = len(GPU_LIST)
    assert NGPU >= LOCAL_WORLD_SIZE, "The number of processes should be less than or equal to the number of GPUs"
    GPU_PER_PROC = NGPU // LOCAL_WORLD_SIZE
    DEVICE_START_IDX = GPU_PER_PROC * LOCAL_RANK
    CUDA_VISIBLE_DEVICES = [str(i) for i in GPU_LIST[DEVICE_START_IDX: DEVICE_START_IDX + GPU_PER_PROC]]
    CUDA_VISIBLE_DEVICES = ','.join(CUDA_VISIBLE_DEVICES)
    # Set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    print(
        f'RANK: {RANK}, LOCAL_RANK: {LOCAL_RANK}, WORLD_SIZE: {WORLD_SIZE},'
        f'LOCAL_WORLD_SIZE: {LOCAL_WORLD_SIZE}, CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}'
    )


from vlmeval.config import supported_VLM
from vlmeval.dataset.video_dataset_config import supported_video_datasets
from vlmeval.dataset import build_dataset
from vlmeval.dataset.utils.judge_util import get_model_from_api_base
from vlmeval.inference import infer_data_job
from vlmeval.inference_video import infer_data_job_video
from vlmeval.inference_mt import infer_data_job_mt
from vlmeval.smp import *
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer


# Make WORLD_SIZE invisible when build models
def build_model_from_config(cfg, model_name, use_vllm=False, port=None, max_turns=None):
    import vlmeval.api
    import vlmeval.vlm
    ws_bak = os.environ.pop('WORLD_SIZE', None)

    config = cp.deepcopy(cfg[model_name])
    if use_vllm:
        config['use_vllm'] = use_vllm
    # Override api_base port if port is specified
    if port is not None and 'api_base' in config:
        import re
        config['api_base'] = config['api_base'].replace("8000", str(port))
    # Override max_turns if specified
    if max_turns is not None:
        config['max_turns'] = max_turns
    if 'class' not in config:
        return supported_VLM[model_name](**config)
    cls_name = config.pop('class')
    if hasattr(vlmeval.api, cls_name):
        model = getattr(vlmeval.api, cls_name)(**config)
    elif hasattr(vlmeval.vlm, cls_name):
        model = getattr(vlmeval.vlm, cls_name)(**config)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.api` or `vlmeval.vlm`')

    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak
    return model

def build_dataset_from_config(cfg, dataset_name):
    import vlmeval.dataset
    import inspect
    config = cp.deepcopy(cfg[dataset_name])
    if config == {}:
        return supported_video_datasets[dataset_name]()
    assert 'class' in config
    cls_name = config.pop('class')
    if hasattr(vlmeval.dataset, cls_name):
        cls = getattr(vlmeval.dataset, cls_name)
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in config.items() if k in sig.parameters}
        if cls.MODALITY == 'VIDEO':
            if valid_params.get('fps', 0) > 0 and valid_params.get('nframe', 0) > 0:
                raise ValueError('fps and nframe should not be set at the same time')
            if valid_params.get('fps', 0) <= 0 and valid_params.get('nframe', 0) <= 0:
                raise ValueError('fps and nframe should be set at least one valid value')
        return cls(**valid_params)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.dataset`')

def parse_args():
    parser = argparse.ArgumentParser(description="help_msg", formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')
    parser.add_argument('--model', type=str, nargs='+', help='Names of Models')
    parser.add_argument('--config', type=str, help='Path to the Config Json File')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer'])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--api-nproc', type=int, default=8, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    parser.add_argument('--judge-args', type=str, default=None, help='Judge arguments in JSON format')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
    # Reuse: will reuse the existing prediction files
    parser.add_argument('--reuse', action='store_true')
    # Reuse-aux: if set, when reuse is True, will also reuse the auxiliary evaluation files
    parser.add_argument('--reuse-aux', type=bool, default=False, help='reuse auxiliary evaluation files')
    parser.add_argument('--use-vllm', action='store_true', help='use vllm to generate, the flag is only supported in Llama4 for now')
    # Max Samples
    parser.add_argument('--max-samples', type=int, default=None, help='Limit the dataset to the first n records.')
    # API Port
    parser.add_argument('--port', type=int, default=None, help='API server port for model inference')
    # Max turns for tool use
    parser.add_argument('--max-turns', type=int, default=10, help='Maximum number of tool use rounds')
    args = parser.parse_args()
    return args


def main():
    logger = get_logger('RUN')
    args = parse_args()
    use_config, cfg = False, None
    if args.config is not None:
        assert args.data is None and args.model is None, '--data and --model should not be set when using --config'
        use_config, cfg = True, load(args.config)
        args.model = list(cfg['model'].keys())
        args.data = list(cfg['data'].keys())
    else:
        assert len(args.data), '--data should be a list of data files'

    if RANK == 0:
        if not args.reuse:
            logger.warning('--reuse is not set, will not reuse previous (before one day) temporary files')
        else:
            logger.warning('--reuse is set, will reuse the latest prediction & temporary pickle files')

    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    if not use_config:
        for k, v in supported_VLM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords and args.retry is not None:
                v.keywords['retry'] = args.retry
                supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords and args.verbose is not None:
                v.keywords['verbose'] = args.verbose
                supported_VLM[k] = v

        # If FWD_API is set, will use class `GPT4V` for all API models in the config
        if os.environ.get('FWD_API', None) == '1':
            from vlmeval.config import api_models as supported_APIs
            from vlmeval.api import GPT4V
            for m in args.model:
                if m in supported_APIs:
                    kws = supported_VLM[m].keywords
                    supported_VLM[m] = partial(GPT4V, **kws)
                    logger.warning(f'FWD_API is set, will use class `GPT4V` for {m}')

    if WORLD_SIZE > 1:
        import torch.distributed as dist
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
        )

    for _, model_name in enumerate(args.model):
        model = None
        date, commit_id = timestr('day'), githash(digits=8)
        eval_id = f"T{date}_G{commit_id}"

        pred_root = osp.join(args.work_dir, model_name, eval_id)
        pred_root_meta = osp.join(args.work_dir, model_name)
        os.makedirs(pred_root_meta, exist_ok=True)

        prev_pred_roots = ls(osp.join(args.work_dir, model_name), mode='dir')
        if len(prev_pred_roots) and args.reuse:
            prev_pred_roots.sort()

        if not osp.exists(pred_root):
            os.makedirs(pred_root, exist_ok=True)

        if use_config:
            model = build_model_from_config(cfg['model'], model_name, args.use_vllm, args.port, args.max_turns)

        for _, dataset_name in enumerate(args.data):
            os.makedirs(f'{args.work_dir}/{model_name}', exist_ok=True)
            model.save_file = f'{args.work_dir}/' + model_name + '/' + dataset_name + "_" + timestr() + '.jsonl'
            # Configure process-image archival directory per dataset (only effective
            # when the model opts in via save_process_images=True in the config).
            if use_config and hasattr(model, 'save_process_images') and getattr(model, 'save_process_images', False):
                _process_dir = osp.join(pred_root, 'process_images', dataset_name)
                os.makedirs(_process_dir, exist_ok=True)
                model.process_image_dir = _process_dir
                if RANK == 0:
                    logger.info(f'[Process Images] archiving to {_process_dir}')
            if WORLD_SIZE > 1:
                dist.barrier()

            try:
                result_file_base = f'{model_name}_{dataset_name}.xlsx'

                if use_config:
                    if WORLD_SIZE > 1:
                        if RANK == 0:
                            dataset = build_dataset_from_config(cfg['data'], dataset_name)
                        dist.barrier()
                    dataset = build_dataset_from_config(cfg['data'], dataset_name)
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        continue
                else:
                    dataset_kwargs = {}
                    if dataset_name in ['MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI']:
                        dataset_kwargs['model'] = model_name

                    # If distributed, first build the dataset on the main process for doing preparation works
                    if WORLD_SIZE > 1:
                        if RANK == 0:
                            dataset = build_dataset(dataset_name, **dataset_kwargs)
                        dist.barrier()

                    dataset = build_dataset(dataset_name, **dataset_kwargs)
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        continue

                max_samples = args.max_samples if hasattr(args, 'max_samples') else None
                if max_samples is not None:
                    dataset = dataset.limit(max_samples)
                    logger.info(f'Limiting dataset {dataset_name} to the first {max_samples} samples.')

                # Handling Multi-Turn Dataset
                if dataset.TYPE == 'MT':
                    result_file_base = result_file_base.replace('.xlsx', '.tsv')

                result_file = osp.join(pred_root, result_file_base)
                # Reuse the previous prediction file if exists
                if RANK == 0 and len(prev_pred_roots):
                    prepare_reuse_files(
                        pred_root_meta=pred_root_meta, eval_id=eval_id, model_name=model_name,
                        dataset_name=dataset_name, reuse=args.reuse, reuse_aux=args.reuse_aux
                    )

                if WORLD_SIZE > 1:
                    dist.barrier()

                if model is None:
                    model = model_name  # which is only a name

                # Perform the Inference
                if dataset.MODALITY == 'VIDEO':
                    model = infer_data_job_video(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        result_file_name=result_file_base,
                        verbose=args.verbose,
                        api_nproc=args.api_nproc,
                        use_vllm=args.use_vllm)
                elif dataset.TYPE == 'MT':
                    model = infer_data_job_mt(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        verbose=args.verbose,
                        api_nproc=args.api_nproc,
                        ignore_failed=args.ignore,
                        use_vllm=args.use_vllm)
                else:
                    model = infer_data_job(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        verbose=args.verbose,
                        api_nproc=args.api_nproc,
                        ignore_failed=args.ignore,
                        use_vllm=args.use_vllm)

                # Set the judge kwargs first before evaluation or dumping
                judge_kwargs = {
                    'nproc': args.api_nproc,
                    'verbose': args.verbose,
                    'retry': args.retry if args.retry is not None else 3,
                    **(json.loads(args.judge_args) if args.judge_args else {}),
                }

                if args.retry is not None:
                    judge_kwargs['retry'] = args.retry
                if args.judge is not None:
                    judge_kwargs['model'] = args.judge
                else:
                    # Auto-detect the model name from OPENAI_API_BASE
                    detected_model = get_model_from_api_base()
                    if detected_model:
                        judge_kwargs['model'] = detected_model
                        if RANK == 0:
                            logger.info(f"[Judge] Auto-detected model from OPENAI_API_BASE: {detected_model}")

                if RANK == 0:
                    logger.info(judge_kwargs)

                if WORLD_SIZE > 1:
                    dist.barrier()

                # Only RANK 0 handles the evaluation part
                if RANK == 0:
                    # Prepare Submission Files for MMMU_TEST AND MMT-Bench_ALL
                    if dataset_name in ['MMMU_TEST']:
                        result_json = MMMU_result_transfer(result_file)
                        logger.info(f'Transfer MMMU_TEST result to json for official evaluation, '
                                    f'json file saved in {result_json}')
                        continue
                    elif 'MMT-Bench_ALL' in dataset_name:
                        submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
                        logger.info(f'Extract options from prediction of MMT-Bench FULL split for official evaluation '
                                    f'(https://eval.ai/web/challenges/challenge-page/2328/overview), '
                                    f'submission file saved in {submission_file}')
                        continue

                    # Skip the evaluation part if only infer
                    if args.mode == 'infer':
                        continue

                    # Skip the evaluation part if the dataset evaluation is not supported or annotations are missing
                    if 'MLLMGuard_DS' in dataset_name:
                        logger.info('The evaluation of MLLMGuard_DS is not supported yet. ')
                        continue
                    elif 'AesBench_TEST' == dataset_name:
                        logger.info(f'The results are saved in {result_file}. '
                                    f'Please send it to the AesBench Team via huangyipo@hotmail.com.')
                        continue
                    elif dataset_name in ['DocVQA_TEST', 'InfoVQA_TEST', 'Q-Bench1_TEST', 'A-Bench_TEST']:
                        logger.info(f'{dataset_name} is a test split without ground-truth. '
                                    'Thus only the inference part is supported for those datasets. ')
                        continue
                    elif dataset_name in [
                        'MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN',
                        'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11'
                    ] and not MMBenchOfficialServer(dataset_name):
                        logger.error(
                            f'Can not evaluate {dataset_name} on non-official servers, will skip the evaluation.')
                        continue

                    # Setup the proxy for the evaluation
                    eval_proxy = os.environ.get('EVAL_PROXY', None)
                    old_proxy = os.environ.get('HTTP_PROXY', '')
                    if eval_proxy is not None:
                        proxy_set(eval_proxy)

                    # Perform the Evaluation
                    eval_results = dataset.evaluate(result_file, **judge_kwargs)
                    # Display Evaluation Results in Terminal
                    if eval_results is not None:
                        assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                        print(f'The evaluation of model {model_name} x dataset {dataset_name} has finished! ')
                        print('Evaluation Results:')
                        if isinstance(eval_results, dict):
                            print('\n' + json.dumps(eval_results, indent=4))
                        elif isinstance(eval_results, pd.DataFrame):
                            if len(eval_results) < len(eval_results.columns):
                                eval_results = eval_results.T
                            print('\n' + tabulate(eval_results))

                    # Restore the proxy
                    if eval_proxy is not None:
                        proxy_set(old_proxy)

                    # Create the symbolic links for the prediction files
                    files = os.listdir(pred_root)
                    files = [x for x in files if (f'{model_name}_{dataset_name}' in x or "status.json" in x)]
                    for f in files:
                        cwd = os.getcwd()
                        file_addr = osp.join(cwd, pred_root, f)
                        link_addr = osp.join(cwd, pred_root_meta, f)
                        if osp.exists(link_addr) or osp.islink(link_addr):
                            os.remove(link_addr)
                        os.symlink(file_addr, link_addr)

            except Exception as e:
                import traceback
                print(f'Model {model_name} x Dataset {dataset_name} combination failed: {e}, skipping this combination.')
                traceback.print_exc()
                continue

    if WORLD_SIZE > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    load_env()
    main()
