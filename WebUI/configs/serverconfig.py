
import sys
from .modelconfig import LLM_DEVICE

HTTPX_DEFAULT_TIMEOUT = 300.0

HTTPX_LOAD_TIMEOUT = 180.0
HTTPX_RELEASE_TIMEOUT = 120

HTTPX_LOAD_VOICE_TIMEOUT = 60.0
HTTPX_RELEASE_VOICE_TIMEOUT = 40

OPEN_CROSS_DOMAIN = False

# The server will listen on all available network interfaces.
DEFAULT_BIND_HOST = "0.0.0.0"

#webui server
WEBUI_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 8818,
}

# api server
API_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 8819,
}

# fastchat openai_api server
FSCHAT_OPENAI_API = {
    "host": DEFAULT_BIND_HOST,
    "port": 20000,
}

FSCHAT_MODEL_WORKERS = {
    "default": {
        "host": DEFAULT_BIND_HOST,
        "port": 20002,
        "device": LLM_DEVICE,
        "vllm_enable": False,

        #"gpus": None,
        #"num_gpus": 1,
        #"max_gpu_memory": 20

        # "load_8bit": False, # enable 8-bit quantization
        # "cpu_offloading": None,
        # "gptq_ckpt": None,
        # "gptq_wbits": 16,
        # "gptq_groupsize": -1,
        # "gptq_act_order": False,
        # "awq_ckpt": None,
        # "awq_wbits": 16,
        # "awq_groupsize": -1,
        # "model_names": LLM_MODELS,
        # "conv_template": None,
        # "limit_worker_concurrency": 5,
        # "stream_interval": 2,
        # "no_register": False,
        # "embed_in_truncate": False,

        # tokenizer = model_path # If the tokenizer is inconsistent with the model_path, add it here
        # 'tokenizer_mode':'auto',
        # 'trust_remote_code':True,
        # 'download_dir':None,
        # 'load_format':'auto',
        # 'dtype':'auto',
        # 'seed':0,
        # 'worker_use_ray':False,
        # 'pipeline_parallel_size':1,
        # 'tensor_parallel_size':1,
        # 'block_size':16,
        # 'swap_space':4 , # GiB
        # 'gpu_memory_utilization':0.90,
        # 'max_num_batched_tokens':2560,
        # 'max_num_seqs':256,
        # 'disable_log_stats':False,
        # 'conv_template':None,
        # 'limit_worker_concurrency':5,
        # 'no_register':False,
        # 'num_gpus': 1
        # 'engine_use_ray': False,
        # 'disable_log_requests': False
    },
}

# fastchat multi model worker server
FSCHAT_MULTI_MODEL_WORKERS = {
    # TODO:
}

# fastchat controller server
FSCHAT_CONTROLLER = {
    "host": DEFAULT_BIND_HOST,
    "port": 20001,
    "dispatch_method": "shortest_queue",
}