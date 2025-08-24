from models import APIBase
from langserve import RemoteRunnable



class RemoteAPI(APIBase):
    """
    Remote API class for initializing and managing chat models via a remote endpoint.
    Inherits from APIBase and provides additional functionality.
    """
    def __init__(self, 
            model_id: str, 
            locale: str, 
            prompt_type: str, 
            prompt_version: str, 
            endpoint: str,
            **kwargs
        ):
        super().__init__(locale=locale, prompt_type=prompt_type, prompt_version=prompt_version)
        self.model_id = model_id
        self.model = RemoteRunnable(url=endpoint, **kwargs)



__all__ = ["RemoteAPI"]