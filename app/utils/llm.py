from llms import llm_models


def get_provider(model_id: str) -> str:
    for llm_model in llm_models:
        for language_model in llm_model.models:
            # Get provider by name
            if model_id == language_model.name:
                return llm_model.provider
            # Get provider by stable version
            if model_id == language_model.version.stable:
                return llm_model.provider
            # Get provider by specific version
            for release in language_model.version.releases:
                if model_id == release:
                    return llm_model.provider
    raise ValueError(f"Model ID '{model_id}' not found in any provider.")


__all__ = ["get_provider"]