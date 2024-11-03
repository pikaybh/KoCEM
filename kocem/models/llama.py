import os
import logging
from typing import Any, Dict, List, Union

import torch
import base64
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from functools import cached_property
from transformers import MllamaForConditionalGeneration, AutoProcessor


# Logger 설정
logger_name = 'models.llama'
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



class LlamaImageCaptioner:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.model = self._load_model()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _load_model(self) -> MllamaForConditionalGeneration:
        """Load the model with specified configurations."""
        logger.info(f"Loading model: {self.model_id}")
        return MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def process_image(self, image_url: str) -> Image:
        """Download and process image from a URL."""
        logger.debug(f"Processing image from URL: {image_url}")
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        return Image.open(response.raw)

    def generate_caption(self, image: Image, prompt_text: str) -> str:
        """Generate a caption for the given image with the prompt text."""
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        logger.info("Generating caption...")
        output = self.model.generate(**inputs, max_new_tokens=30)
        caption = self.processor.decode(output[0])
        
        logger.debug(f"Generated caption: {caption}")
        return caption

    @staticmethod
    def image_to_base64(image: Image) -> Dict[str, str]:
        """Encode the image to base64 format."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return {"url": f"data:image/png;base64,{base64_image}"}

    def __call__(self, image_url: str, prompt_text: str) -> str:
        """Main callable interface to process image and generate caption."""
        image = self.process_image(image_url)
        caption = self.generate_caption(image, prompt_text)
        return caption
