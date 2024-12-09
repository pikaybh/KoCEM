from random import random
from typing import Dict, Union
import logging

import torch


# Root 
logger_name = 'utils.model'
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


CHECK_MODEL = {
    "machine" : ["canny"],
    "openai" : [
        "gpt-4o",                       # 100t ?
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",                        # 1t ?
        "gpt-3.5-turbo"                 # 175b
    ],
    "google" : [
        # "gemini-1.0-ultra-latest",      # Undisclosed (2024-11-04)
        "gemini-1.5-pro-latest",          # 600b
        "gemini-1.0-pro-latest",          # None Multimodal
        "gemini-1.5-flash-latest",        # >= 8b
        # "gemini-1.5-flash-8b-latest",   # 8b
        # "gemini-1.0-nano-latest",       # 1.8b
    ],
    "anthropic" : ["claude-3-5-sonnet-20241022"],
    "hf" : []
}


def call_engine_df(sample, model):
    prompt = sample['final_input_prompt']
    image = sample['image']
    if image:
        response = model(prompt, image=image)
        # logger.debug(f"{sample['id']}: vlm.")
    elif prompt:
        response = model(prompt)
        # logger.debug(f"{sample['id']}: lm.")
    else:  # multiple images actually
        if sample['question_type'] == 'multiple-choice':
            response = random.choice(sample['all_choices'])
            # logger.debug(f"{sample['id']}: random choice.")
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'

    return response