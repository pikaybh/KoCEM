import os
import base64
import logging
from io import BytesIO
from random import choice
from typing import Any, Dict, List, Union

from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from functools import cached_property

from .basemodel import Template
# from ..api.registry import register_model


# Root 
logger_name = 'models.google'
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)
# File Handler
file_handler = logging.FileHandler(f'logs/{logger_name}.log', encoding='utf-8-sig')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(r'%(asctime)s [%(name)s, line %(lineno)d] %(levelname)s: %(message)s'))
logger.addHandler(file_handler)
# Stream Handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter(r'%(message)s'))
logger.addHandler(stream_handler)


# Convert image_data to bytes
buffered = BytesIO()

load_dotenv()

API_KEY = "GOOGLE_API_KEY"

# @register_model("openai-chat-completions")
class GoogleAIGenerativeModel(Template):
    def __init__(self,
                 model: str,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model

    @cached_property
    def api_key(self) -> str:
        """Override this property to return the API key for the API request."""
        try:
            key: str = os.environ.get(API_KEY, None)
        except:
            key: str = os.getenv(API_KEY)
        finally:
            if key is None:
                raise ValueError(f"API key not found. Please set the `{API_KEY}` environment variable.")
        return key

    def run(self, message: str, **kwargs) -> Union[Dict, List[Dict[str, Any]]]:
        self.client = genai.GenerativeModel(model_name=self.model)

        if not message:
            raise ValueError(f"No message. ({message = })")
        if "image" in kwargs:
            response = self.client.generate_content([message, kwargs["image"]])
        else:
            response = self.client.generate_content([message])

        return response
        
    @staticmethod
    def image_processor(image_data: Dict[str, Union[bytes, str]]) -> Dict[str, str]:
        """
        이미지를 base64로 인코딩하여 data URL 형태로 반환합니다.
        """
        # Check if image_data is a bytes-like object or already an image file
        if isinstance(image_data, bytes):
            return Image.open(BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            return image_data
        else:
            raise TypeError("Unsupported image data type")

    def __call__(self, message: str, **kwargs) -> str:
        response = self.run(message, **kwargs)
        try:
            result: str = response.text
        except(ValueError):
            result: str = choice([chr(ord("A") + i) for i in range(5)])
            logger.error(f"ValueError occurred: {result = }")
        logger.debug(result)

        return result