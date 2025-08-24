from typing import List, Literal, Optional
from pydantic import BaseModel, Field


supported_data_types = Literal["text", "image", "audio", "video", "code", "json", "csv", "html", "markdown", "xml", "pdf", "binary"]



class Modality(BaseModel):
    input_type: List[supported_data_types] = Field(..., description="Type of input (e.g., text, image)")
    output_type: List[supported_data_types] = Field(..., description="Type of output (e.g., text, image)")



class Pricing(BaseModel):
    text_input: Optional[float] | Literal["unknown"] = Field("unknown", description="Cost per 1k tokens for text input")
    text_cached_input: Optional[float] | Literal["unknown"] | None = Field(None, description="Cost per 1k tokens for cached text input")
    text_output: Optional[float] | Literal["unknown"] = Field("unknown", description="Cost per 1k tokens for text output")



class ModelSize(BaseModel):
    parameters: dict | int | Literal["unknown"] = Field("unknown", description="Size of the model in billions of parameters or 'unknown'")
    aunounced: bool = Field(False, description="Whether the size is announced by the provider")



class ModelVersion(BaseModel):
    releases: List[str] = Field([], description="Version of the model")
    stable: str = Field(..., description="Stable version of the model")
    preview: Optional[str] = Field(None, description="Preview version of the model, if available")



class LanguageModel(BaseModel):
    name: str = Field(..., description="Name of the LLM")
    description: str = Field(..., description="Description of the LLM")
    size: Optional[ModelSize] = Field(..., description="Size of the model in billions of parameters")
    version: ModelVersion = Field(..., description="Version information of the LLM")
    modality: Modality = Field(..., description="Modality of the LLM (input/output types)")
    features: Optional[dict] = Field({}, description="Additional features of the LLM")
    pricing: Optional[Pricing] | Literal["open-source"] = Field("open-source", description="Pricing information for the LLM")



class LLMGroup(BaseModel):
    provider: str = Field(..., description="Provider of the LLM (e.g., OpenAI, Google, Anthropic)")
    docs: Optional[str] = Field(None, description="Documentation URL for the provider")
    models: list[LanguageModel] = Field(..., description="List of LLMs under this provider")


__all__ = ['Modality', 'Pricing', 'ModelSize', 'ModelVersion', 'LanguageModel', 'LLMGroup']