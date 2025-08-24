from schemas import (
    LanguageModel,
    ModelSize,
    ModelVersion,
    Modality,
    Pricing
)
from models import LLMBase



class OpenAI(LLMBase):
    def __init__(self, 
        provider: str = "openai", 
        docs: str = "https://platform.openai.com/docs/models"
    ):
        super().__init__(provider, docs)

    models = [
        LanguageModel(
            name="GPT-5",
            description="The best model for coding and agentic tasks across domains",
            size=ModelSize(parameters="unknown", aunounced=False),
            version=ModelVersion(
                releases=["gpt-5-2025-08-07"],
                stable="gpt-5"
            ),
            modality=Modality(
                input_type=["text", "image"],
                output_type=["text"]
            ),
            features={
                "context_window": 400_000,
                "max_output_tokens": 128_000,
                "reasoning": True,
                "function_calling": True,
                "structured_output": True,
            },
            pricing=Pricing(
                text_input=1.25,
                text_cached_input=0.25,
                text_output=10.00
            )
        ),
        LanguageModel(
            name="GPT-5 mini",
            description="A faster, cost-efficient version of GPT-5 for well-defined tasks",
            version=ModelVersion(
                releases=["gpt-5-mini-2025-08-07"],
                stable="gpt-5-mini"
            ),
            size=ModelSize(parameters="unknown", aunounced=False),
            modality=Modality(
                input_type=["text", "image"],
                output_type=["text"]
            ),
            features={
                "context_window": 400_000,
                "max_output_tokens": 128_000,
                "reasoning": True,
                "function_calling": True,
                "structured_output": True,
            },
            pricing=Pricing(
                text_input=0.25,
                text_cached_input=0.0025,
                text_output=2.00
            )
        ),
        LanguageModel(
            name="GPT-5 nano",
            description="Fastest, most cost-efficient version of GPT-5",
            version=ModelVersion(
                releases=["gpt-5-nano-2025-08-07"],
                stable="gpt-5-nano"
            ),
            size=ModelSize(parameters="unknown", aunounced=False),
            modality=Modality(
                input_type=["text", "image"],
                output_type=["text"]
            ),
            features={
                "context_window": 400_000,
                "max_output_tokens": 128_000,
                "reasoning": True,
                "function_calling": True,
                "structured_output": True,
            },
            pricing=Pricing(
                text_input=0.05,
                text_cached_input=0.005,
                text_output=0.40
            )
        ),
        LanguageModel(
            name="GPT-4.1",
            description="Smartest non-reasoning model",
            version=ModelVersion(
                releases=["gpt-4.1-2025-04-14"],
                stable="gpt-4.1"
            ),
            size=ModelSize(parameters="unknown", aunounced=False),
            modality=Modality(
                input_type=["text", "image"],
                output_type=["text"]
            ),
            features={
                "context_window": 1_047_576,
                "max_output_tokens": 32_768,
                "reasoning": False,
                "function_calling": True,
                "structured_output": True,
            },
            pricing=Pricing(
                text_input=0.50,
                text_cached_input=0.05,
                text_output=4.00
            )
        ),
        LanguageModel(
            name="gpt-oss-120b",
            description="Most powerful open-weight model, fits into an H100 GPU",
            version=ModelVersion(
                releases=["gpt-oss-120b"],
                stable="gpt-oss-120b"
            ),
            size=ModelSize(parameters=117_000_000, aunounced=True),
            modality=Modality(
                input_type=["text"],
                output_type=["text"]
            ),
            features={
                "context_window": 131_072,
                "max_output_tokens": 131_072,
                "reasoning": True,
                "function_calling": True,
                "structured_output": True,
            },
            pricing="open-source"
        ),
        LanguageModel(
            name="gpt-oss-20b",
            description="Medium-sized open-weight model for low latency",
            version=ModelVersion(
                releases=["gpt-oss-20b"],
                stable="gpt-oss-20b"
            ),
            size=ModelSize(parameters=21_000_000, aunounced=True),
            modality=Modality(
                input_type=["text"],
                output_type=["text"]
            ),
            features={
                "context_window": 131_072,
                "max_output_tokens": 131_072,
                "reasoning": True,
                "function_calling": True,
                "structured_output": True,
            },
            pricing="open-source"
        )
    ]



__all__ = ['OpenAI']