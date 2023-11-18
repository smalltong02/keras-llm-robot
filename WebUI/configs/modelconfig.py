
EMBEDDING_MODEL = "m3e-base"

LLM_MODELS = [""]

Agent_MODEL = None

KERAS_LLM_VERSION = "1.0.0"

MODEL_ROOT_PATH = ""
# auto, cuda, mps, cpu
LLM_DEVICE = "auto"

HISTORY_LEN = 5

MAX_TOKENS = None

TEMPERATURE = 0.7
# TOP_K = 50
# TOP_P = 0.95

ONLINE_LLM_MODEL = {
    "openai-api": {
        "model_list" : ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-1106-preview", "gpt-4-vision-preview"],
        "base_url": "https://api.openai.com/v1",
        "api_version": "",  # API version
        "api_key": "",
        "api_proxy": "",
    },

    # Azure API
    "azure-api": {
        "model_list": [""],
        "resource_name": "",  # https://{resource_name}.openai.azure.com/openai/
        "api_version": "",  # API version
        "api_key": "",
        "api_proxy": "",
        "provider": "AzureWorker",
    },
}

MODEL_PATH = {
    "embedding_model": {
        "jina-embeddings-v2-small-en": "jinaai/jina-embeddings-v2-small-en",
        "jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
        "m3e-small": "moka-ai/m3e-small",
        "m3e-base": "moka-ai/m3e-base",
        "m3e-large": "moka-ai/m3e-large",
        "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
        "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
        "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
        "text-embedding-ada-002": "OPENAI_API_KEY",
    },

    "llm_model": {
        "llama-2-7b-hf": "meta-llama/llama-2-7b-hf",
        "llama-2-13b-hf": "meta-llama/llama-2-13b-hf",
        "llama-2-70b-hf": "meta-llama/llama-2-70b-hf",
        "llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
        "llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
        "llama-2-70b-chat-hf": "meta-llama/Llama-2-70b-chat-hf",
        "chatglm2-6b": "THUDM/chatglm2-6b",
        "chatglm2-6b-32k": "THUDM/chatglm2-6b-32k",
    },
}