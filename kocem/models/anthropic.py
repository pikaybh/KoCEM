import os
import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Union

import anthropic
from PIL import Image
from dotenv import load_dotenv
from functools import cached_property

from .basemodel import Template

# Root 
logger_name = 'models.anthropic'
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

load_dotenv()

API_KEY = "ANTHROPIC_API_KEY"

class AnthropicGenerativeModel(Template):
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
        self.client = anthropic.Anthropic()
        if not message:
            raise ValueError(f"No message. ({message = })")
        
        # Initialize content list
        content = []
        
        # Add system prompt if exists
        if self.system_prompt:
            content.append({
                "type": "text",
                "text": self.system_prompt
            })

        # Handle image if present
        if "image" in kwargs and kwargs["image"] is not None:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": kwargs["image"]["url"].split(",")[1]  # Remove data URL prefix
                }
            })

        # Add the message
        content.append({
            "type": "text",
            "text": message
        })

        # Prepare the payload
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": content
            }],
            "max_tokens": kwargs.pop("max_tokens", 1_024)
        }

        return self.client.messages.create(**payload)
        
    @staticmethod
    def image_processor(image_data: Union[Image.Image, Dict[str, Union[bytes, str]]]) -> Dict[str, str]:
        """
        Process image data and return base64 encoded data URL and PIL Image object.
        
        Args:
            image_data: Either a PIL Image object or a dictionary containing image bytes
            
        Returns:
            Dictionary containing the base64 data URL and PIL Image object
        """
        try:
            if isinstance(image_data, Image.Image):
                image = image_data
            elif isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(BytesIO(image_data['bytes']))
            else:
                raise ValueError("Invalid image data format")

            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            # Save to buffer and encode
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "image": image
            }
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    def __call__(self, message: str, **kwargs) -> str:
        result: str = str(self.run(message, **kwargs))
        logger.debug(result)
        return result