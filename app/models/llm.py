from schemas import (
    LanguageModel,
    ModelSize,
    ModelVersion,
    Modality,
    Pricing
)



class LLMBase:
    def __init__(self,
        provider: str,
        docs: str | None = None
    ):
        self.provider = provider
        self.docs = docs
        self._models: list[LanguageModel] = []

    @property
    def provider(self) -> str:
        """Provider name."""
        return self._provider
    
    @provider.setter
    def provider(self, value: str):
        if not isinstance(value, str) or not value:
            raise ValueError("provider must be a non-empty string")
        self._provider = value

    @property
    def models(self) -> list[LanguageModel]:
        """List of LanguageModel instances."""
        return self._models

    @models.setter
    def models(self, value: list[LanguageModel]):
        if not isinstance(value, list):
            raise ValueError("models must be a list of LanguageModel")
        self._models = value

    def __iter__(self):
        return iter(self._models)
    
    def append(self, value: LanguageModel):
        """
        Append a LanguageModel to the list of models.
        Args:
            value (LanguageModel): The LanguageModel instance to append.
        """
        self._models.append(value)
    
    def add(self,
        name: str,
        description: str,
        stable: str,
        parameters: dict | int | None = None,
        aunounced: bool = False,
        releases: list[str] | None = None,
        input_type: list[str] | None = None,
        output_type: list[str] | None = None,
        **features,
    ):
        """
        Update the LanguageModel with a dictionary of data.
        Args:
            name (str): Name of the LLM.
            description (str): Description of the LLM.
            stable (str): Stable version of the model.
            parameters (dict | int | None): Size of the model in billions of parameters or None.
            aunounced (bool): Whether the size is announced by the provider.
            releases (list[str]): List of releases for the model.
            input_type (list[str]): List of input types for the model.
            output_type (list[str]): List of output types for the model.
            **features: Additional features of the model.
        """
        releases = releases or []
        input_type = input_type or ["text"]
        output_type = output_type or ["text"]
        resolved_parameters = parameters if parameters is not None else "unknown"
        self._models.append(LanguageModel(
            name=name,
            description=description,
            size=ModelSize(parameters=resolved_parameters, aunounced=aunounced),
            version=ModelVersion(releases=releases, stable=stable),
            modality=Modality(input_type=input_type, output_type=output_type),
            features=features,
            pricing=Pricing(text_input="unknown", text_cached_input=None, text_output="unknown")
        ))

    def configure(self, values: list[dict | LanguageModel]):
        """
        Configure the LLMBase with a list of LanguageModel dictionaries.
        Args:
            values (list[dict]): List of dictionaries containing model configurations.
        """
        for value in values:
            if isinstance(value, LanguageModel):
                self.append(value)
            else:
                self.add(**value)

    def __len__(self):
        return len(self._models)

    def __call__(self,
        name: str | None = None,
        verbose: bool = False
    ) -> LanguageModel:
        """
        Retrieve a LanguageModel by its name.
        Args:
            name (str | None): Name of the model to retrieve. If None, returns all models.
            verbose (bool): If True, returns the full LanguageModel instance; otherwise, returns the stable version.
        Returns:
            LanguageModel: The LanguageModel instance with the specified name.
        Raises:
            ValueError: If no model with the specified name exists.
        """
        if not name:
            if verbose:
                return self.models
            else:
                return [model.version.stable for model in self.models]

        def _conditions(model_name: str, query: str) -> bool:
            return any([
                model_name.lower() == query.lower(),
                model_name.replace(' ', '_') == query.replace(' ', '_'),
                model_name.replace(' ', '-') == query.replace(' ', '-'),
            ])
        
        for model in self.models:
            if _conditions(model.name, name):
                return model if verbose else model.version.stable
        
        raise ValueError(f"No model with name '{name}' found in {self.provider} models.")



__all__ = ["LLMBase"]