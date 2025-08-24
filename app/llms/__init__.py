from .anthropic import Anthropic
from .google import GeminiFamily
from .lgai import LGAI
from .llava import LlavaModels
from .meta import MetaLlama
from .openai import OpenAI
from .qwen import QwenModels

llm_models = [
    Anthropic(),
    GeminiFamily(),
    LGAI(),
    LlavaModels(),
    MetaLlama(),
    OpenAI(),
    QwenModels()
]

__all__ = [
    'llm_models',
    # ------- Individual LLM schemas -------
    'Anthropic',
    'GeminiFamily',
    'LGAI',
    'LlavaModels',
    'MetaLlama',
    'OpenAI',
    'QwenModels'
]