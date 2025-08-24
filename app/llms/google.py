from schemas import (
    LanguageModel,
    ModelSize,
    ModelVersion,
    Modality,
    Pricing
)
from models import LLMBase



class GeminiFamily(LLMBase):
    def __init__(self, 
        provider: str = "google", 
        docs: str = "https://platform.openai.com/docs/models"
    ):
        super().__init__(provider, docs)

    models = [
        LanguageModel(
            name="Gemini 2.5 Pro",
            description="Gemini 2.5 Pro는 최첨단 사고 모델로, 코드, 수학, STEM의 복잡한 문제를 추론할 수 있을 뿐만 아니라 긴 컨텍스트를 사용하여 대규모 데이터 세트, 코드베이스, 문서를 분석할 수 있습니다.",
            version=ModelVersion(
                releases=["gemini-2.5-pro"],
                stable="gemini-2.5-pro"
            ),
            size=ModelSize(parameters="unknown", aunounced=False),
            modality=Modality(
                input_type=["text", "image", "video", "audio", "pdf"],
                output_type=["text"]
            ),
            features={
                "context_window": 1_048_576,
                "max_output_tokens": 65_536,
                "reasoning": True,
                "function_calling": True,
                "structured_output": True,
            },
            pricing=Pricing(
                text_input="unknown",
                text_cached_input="unknown",
                text_output="unknown"
            )
        ),
        LanguageModel(
            name="Gemini 2.5 Flash",
            description="최고의 가격 대비 성능을 갖추었으며 다양한 기능을 제공하는 모델 2.5 Flash는 대규모 처리, 짧은 지연 시간, 사고력이 필요한 대량 작업, 에이전트 사용 사례에 가장 적합합니다.",
            version=ModelVersion(
                releases=["gemini-2.5-flash", "gemini-2.5-flash-preview-05-20"],
                stable="gemini-2.5-flash",
                preview="gemini-2.5-flash-preview-05-20"
            ),
            size=ModelSize(parameters="unknown", aunounced=False),
            modality=Modality(
                input_type=["text", "image", "video", "audio"],
                output_type=["text"]
            ),
            features={
                "max_input_tokens": 1_048_576,
                "max_output_tokens": 65_536,
                "reasoning": True,
                "function_calling": True,
                "structured_output": True,
            },
            pricing=Pricing(
                text_input="unknown",
                text_cached_input="unknown",
                text_output="unknown"
            )
        ),
        LanguageModel(
            name="Gemini 2.5 Flash-Lite",
            description="비용 효율성과 높은 처리량에 최적화된 Gemini 2.5 Flash 모델입니다.",
            version=ModelVersion(
                releases=["gemini-2.5-flash-lite", "gemini-2.5-flash-lite-06-17"],
                stable="gemini-2.5-flash-lite",
                preview="gemini-2.5-flash-lite-06-17"
            ),
            size=ModelSize(parameters="unknown", aunounced=False),
            modality=Modality(
                input_type=["text", "image", "video", "audio", "pdf"],
                output_type=["text"]
            ),
            features={
                "max_input_tokens": 1_048_576,
                "max_output_tokens": 65_536,
                "reasoning": True,
                "function_calling": True,
                "structured_output": True,
            },
            pricing=Pricing(
                text_input="unknown",
                text_cached_input="unknown",
                text_output="unknown"
            )
        )
    ]



__all__ = ['GeminiFamily']