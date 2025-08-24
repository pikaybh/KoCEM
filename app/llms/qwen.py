from schemas import (
    LanguageModel,
    ModelSize,
    ModelVersion,
    Modality,
    Pricing
)
from models import LLMBase



class QwenModels(LLMBase):
    def __init__(self, 
        provider: str = "alibaba", 
        docs: str = "https://bailian.console.alibabacloud.com/?tab=doc#/doc/?type=model&url=2840914"
    ):
        super().__init__(provider, docs)

    models = [
        LanguageModel(
            name="Qwen-Max",
            description="Flagship Qwen model for complex reasoning and broad multimodal understanding.",
            version=ModelVersion(
                releases=["qwen-max"],
                stable="qwen-max"
            ),
            size=ModelSize(parameters="unknown", aunounced=False),
            modality=Modality(
                input_type=["text", "image"],
                output_type=["text"]
            ),
            features={
                "reasoning": True,
                "function_calling": True,
                "structured_output": True,
            },
            pricing=Pricing(
                text_input="unknown",
                text_cached_input=None,
                text_output="unknown"
            )
        ),
        LanguageModel(
            name="Qwen-Plus",
            description="Balanced Qwen model offering strong quality with lower latency and cost than Qwen-Max.",
            version=ModelVersion(
                releases=["qwen-plus"],
                stable="qwen-plus"
            ),
            size=ModelSize(parameters="unknown", aunounced=False),
            modality=Modality(
                input_type=["text", "image"],
                output_type=["text"]
            ),
            features={
                "reasoning": True,
                "function_calling": True,
                "structured_output": True,
            },
            pricing=Pricing(
                text_input="unknown",
                text_cached_input=None,
                text_output="unknown"
            )
        ),
        LanguageModel(
            name="Qwen-Turbo",
            description="Latency-optimized Qwen model suitable for high-throughput applications.",
            version=ModelVersion(
                releases=["qwen-turbo"],
                stable="qwen-turbo"
            ),
            size=ModelSize(parameters="unknown", aunounced=False),
            modality=Modality(
                input_type=["text"],
                output_type=["text"]
            ),
            features={
                "reasoning": False,
                "function_calling": True,
                "structured_output": True,
            },
            pricing=Pricing(
                text_input="unknown",
                text_cached_input=None,
                text_output="unknown"
            )
        ),
        LanguageModel(
            name="Qwen3-30B-A3B-Instruct-2507",
            description="Qwen3 30B Instruct (A3B 2507) instruction-tuned open weight model (approx. 30B params).",
            version=ModelVersion(
                releases=["Qwen/Qwen3-30B-A3B-Instruct-2507"],
                stable="Qwen/Qwen3-30B-A3B-Instruct-2507"
            ),
            size=ModelSize(parameters=30_000_000, aunounced=True),
            modality=Modality(
                input_type=["text"],
                output_type=["text"]
            ),
            features={
                "reasoning": True,
                "function_calling": True,
                "structured_output": True,
            },
            pricing="open-source"
        ),
        LanguageModel(
            name="Qwen3-30B-A3B-Thinking-2507",
            description="Qwen3 30B Thinking (A3B 2507) model optimized for extended reasoning / chain-of-thought generation.",
            version=ModelVersion(
                releases=["Qwen/Qwen3-30B-A3B-Thinking-2507"],
                stable="Qwen/Qwen3-30B-A3B-Thinking-2507"
            ),
            size=ModelSize(parameters=30_000_000, aunounced=True),
            modality=Modality(
                input_type=["text"],
                output_type=["text"]
            ),
            features={
                "reasoning": True,
                "function_calling": True,
                "structured_output": True,
            },
            pricing="open-source"
        )
    ]



__all__ = ['QwenModels']