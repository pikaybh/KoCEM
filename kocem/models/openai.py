import os
import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Union

from openai import OpenAI
from dotenv import load_dotenv
from functools import cached_property

from .basemodel import Template
# from ..api.registry import register_model


# Root 
logger_name = 'models.openai'
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

API_KEY = "OPENAI_API_KEY"

# @register_model("openai-chat-completions")
class OpenAIChatCompletions(Template):
    def __init__(self,
                 model: str,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = OpenAI()

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
        if not message:
            raise ValueError(f"No message. ({message = })")
        message_template: List[Dict[str, Any]] = [{"system": self.system_prompt}] if self.system_prompt else []
        if "image" in kwargs:
            message_template.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": kwargs["image"]["url"]
                        },
                    }
                ]
            })
        elif "image_1" in kwargs:
            message_template.append({"type": "text", "text": message})
            message_template.append("""image_1, image_2, image_3 ... 넣는 코드""")
        else:
            message_template.append({
                "role": "user",
                "content": message
            })

        payload = {
            "model": self.model,
            "messages": message_template,
            "logprobs": kwargs.pop("logprobs", False)
        }

        return self.client.chat.completions.create(**payload)
        
    @staticmethod
    def image_processor(image_data: Dict[str, Union[bytes, str]]) -> Dict[str, str]:
        """
        이미지를 base64로 인코딩하여 data URL 형태로 반환합니다.
        """
        # 이미지 데이터를 base64로 인코딩
        # print(f"{image_data = }")
        # base64_image = base64.b64encode(image_data['bytes']).decode('utf-8')
        image_data.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # data URL 형식으로 반환
        return {"url": f"data:image/jpeg;base64,{base64_image}"}

    def __call__(self, message: str, **kwargs) -> str:
        response = self.run(message, **kwargs)
        result: str = str(response.choices[0].message.content)
        logger.debug(result)

        return result