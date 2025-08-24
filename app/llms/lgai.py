from schemas import (
    LanguageModel,
    ModelSize,
    ModelVersion,
    Modality,
)
from models import LLMBase



class LGAI(LLMBase):
    def __init__(self, 
        provider: str = "lgai", 
        docs: str = "https://github.com/LG-AI-EXAONE/EXAONE-4.0"
    ):
        super().__init__(provider, docs)

    models = [
        LanguageModel(
            name="EXAONE-4.0-1.2B",
            description="The EXAONE 4.0 model series consists of two sizes: a mid-size 32B model optimized for high performance, and a small-size 1.2B model designed for on-device applications.",
            version=ModelVersion(
                releases=["LGAI-EXAONE/EXAONE-4.0-1.2B"],
                stable="LGAI-EXAONE/EXAONE-4.0-1.2B"
            ),
            size=ModelSize(parameters=1_200_000, aunounced=True),
            modality=Modality(
                input_type=["text"],
                output_type=["text"]
            ),
            features={
                "reasoning": True,
                "non-reasoning": True,
                "function_calling": True
            },
            pricing="open-source"
        ),
        LanguageModel(
            name="EXAONE-4.0-32B",
            description="The EXAONE 4.0 model series consists of two sizes: a mid-size 32B model optimized for high performance, and a small-size 1.2B model designed for on-device applications.",
            version=ModelVersion(
                releases=["LGAI-EXAONE/EXAONE-4.0-32B"],
                stable="LGAI-EXAONE/EXAONE-4.0-32B"
            ),
            size=ModelSize(parameters=32_000_000, aunounced=True),
            modality=Modality(
                input_type=["text"],
                output_type=["text"]
            ),
            features={
                "reasoning": True,
                "non-reasoning": True,
                "function_calling": True
            },
            pricing="open-source"
        )
    ]



__all__ = ['LGAI']