from models import APIBase
from langchain.chat_models import init_chat_model



class GPUFreeAPI(APIBase):
    """
    Standard API class for initializing and managing chat models.
    Inherits from APIBase and provides additional functionality.
    """
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self._model = init_chat_model(model_id.replace("/", ":"))



__all__ = ["GPUFreeAPI"]