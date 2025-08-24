from schemas import (
    LanguageModel,
    ModelSize,
    ModelVersion,
    Modality,
    Pricing
)
from models import LLMBase



class Anthropic(LLMBase):
    def __init__(self, 
        provider: str = "anthropic", 
        docs: str = "https://docs.anthropic.com/en/docs/about-claude/models/overview#model-comparison-table"
    ):
        super().__init__(provider, docs)

    models = [
        LanguageModel(
            name="Claude Opus 4.1",
            description="Anthropic's most powerful and capable model",
            version=ModelVersion(
                releases=["claude-opus-4-1-20250805"],
                stable="claude-opus-4-1"
            ),
            size=ModelSize(parameters="unknown", aunounced=False),
            modality=Modality(
                input_type=["text", "image"],
                output_type=["text"]
            ),
            features={
                "context_window": 200_000,
                "max_output_tokens": 32_000,
                "reasoning": True,
            },
            pricing=Pricing(
                text_input=15.0,
                text_cached_input=30.0,
                text_output=75.0
            )
        ),
        LanguageModel(
            name="Claude Sonnet 4",
            description="High-performance model",
            version=ModelVersion(
                releases=["claude-sonnet-4-20250514"],
                stable="claude-sonnet-4"
            ),
            size=ModelSize(parameters="unknown", aunounced=False),
            modality=Modality(
                input_type=["text", "image"],
                output_type=["text"]
            ),
            features={
                "context_window": 200_000,
                "max_output_tokens": 64_000,
                "reasoning": True,
            },
            pricing=Pricing(
                text_input=3.0,
                text_cached_input=6.0,
                text_output=15.0
            )
        )
    ]



__all__ = ['Anthropic']