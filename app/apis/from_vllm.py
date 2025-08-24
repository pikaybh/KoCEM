import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from models import APIBase


load_dotenv()
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT")



class VLLMOpenAIAPI(APIBase):
    def __init__(self, 
        model_id: str, 
        base_url: str = VLLM_ENDPOINT, 
        # api_key: str = "EMPTY",
        locale: str = "en", 
        task: str = "mcqa",
        prompt_type: str = "mcqa", 
        prompt_version: str = "latest",
        **kwargs
    ):
        super().__init__(locale=locale, task=task, prompt_type=prompt_type, prompt_version=prompt_version)
        self.model_id = model_id
        self.model = ChatOpenAI(
            base_url=base_url,           # 예: "http://localhost:8000/v1"
            # api_key=api_key,             # vLLM은 기본적으로 아무 문자열이나 허용
            model=model_id,              # 예: "meta-llama/Meta-Llama-3-8B-Instruct"
            **kwargs                     # temperature, top_p, max_tokens 등
        )



__all__ = ["VLLMOpenAIAPI"]