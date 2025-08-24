from schemas import (
    LanguageModel,
    ModelSize,
    ModelVersion,
    Modality,
)
from models import LLMBase



class MetaLlama(LLMBase):
    def __init__(self, 
        provider: str = "meta", 
        docs: str = "https://llama.developer.meta.com/docs/models"
    ):
        super().__init__(provider, docs)

    models = [
        LanguageModel(
            name="Llama 4 Maverick",
            description="Industry-leading natively multimodal model for image and text understanding with groundbreaking intelligence and fast responses at a low cost.",
            version=ModelVersion(
                releases=["Llama-4-Maverick-17B-128E-Instruct-FP8"],
                stable="Llama-4-Maverick-17B-128E-Instruct-FP8"
            ),
            size=ModelSize(parameters=109_000_000, aunounced=True),
            modality=Modality(
                input_type=["text", "image"],
                output_type=["text"]
            ),
            features={
                "context_window": 128_000,
                "mixture_of_experts": 128,
            },
            pricing="open-source"
        ),
        LanguageModel(
            name="Llama 4 Scout",
            description="A class-leading natively multimodal model that offers superior text and visual intelligence.",
            version=ModelVersion(
                releases=["Llama-4-Scout-17B-16E-Instruct-FP8"],
                stable="Llama-4-Scout-17B-16E-Instruct-FP8"
            ),
            size=ModelSize(parameters=400_000_000, aunounced=True),
            modality=Modality(
                input_type=["text", "image"],
                output_type=["text"]
            ),
            features={
                "context_window": 128_000,
                "mixture_of_experts": 16,
            },
            pricing="open-source"
        )
    ]



__all__ = ['MetaLlama']