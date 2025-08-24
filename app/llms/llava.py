from schemas import (
    LanguageModel,
    ModelSize,
    ModelVersion,
    Modality,
)
from models import LLMBase



class LlavaModels(LLMBase):
    def __init__(self, 
        provider: str = "liuhaotian", 
        docs: str = "https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/MODEL_ZOO.md"
    ):
        super().__init__(provider, docs)

    models = [
        LanguageModel(
            name="LLaVA-1.6",
            description="Visual instruction tuning towards large language and vision models with GPT-4 level capabilities.",
            version=ModelVersion(
                releases=[
                    "liuhaotian/llava-v1.6-vicuna-7b",
                    "liuhaotian/llava-v1.6-vicuna-13b",
                    "liuhaotian/llava-v1.6-mistral-7b",
                    "liuhaotian/llava-v1.6-34b"
                ],
                stable="liuhaotian/llava-v1.6-34b"
            ),
            size=ModelSize(
                parameters={
                    "liuhaotian/llava-v1.6-vicuna-7b": 7_000_000,
                    "liuhaotian/llava-v1.6-vicuna-13b": 13_000_000,
                    "liuhaotian/llava-v1.6-mistral-7b": 7_000_000,
                    "liuhaotian/llava-v1.6-34b": 34_000_000
                }, 
                aunounced=False
            ),
            modality=Modality(
                input_type=["text", "image"],
                output_type=["text"]
            ),
            features={},
            pricing="open-source"
        )
    ]



__all__ = ['LlavaModels']