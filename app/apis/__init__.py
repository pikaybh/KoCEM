from llms import llm_models

from .standalone import GPUFreeAPI
from .from_langserve import RemoteAPI
from .from_vllm import VLLMOpenAIAPI


module = {
    "anthropic": GPUFreeAPI,
    "google": GPUFreeAPI,
    "openai": GPUFreeAPI,
}

for kprovider, vapi in module.items():
    for llm in llm_models:
        if kprovider == llm.provider:
            module[kprovider] = vapi
            break


__all__ = ["module"]