import os
import random
from typing import Dict, List, Optional, Union, T

import io
import fire
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

from models.basemodel import Gacha
from models.llama import LlamaImageCaptioner
from models.openai import OpenAIChatCompletions
from models.google import GoogleAIGenerativeModel
from models.anthropic import AnthropicGenerativeModel

from utils.os import check_model_output_path
from utils.model import call_engine_df, CHECK_MODEL
from utils.eval import parse_multi_choice_response, parse_open_response
from utils.data import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG


SampleType = Dict[str, Union[str, List[str], Dict[str, str]]]

del CAT_SHORT2LONG["di"]

def completion(sample: SampleType, model) -> SampleType:
    outkey_list = ["id", "question_type", "answer", "all_choices", "index2ans", "explanation"]
    if "difficulty" in sample:
        outkey_list.append("difficulty")

    out_dict: SampleType = {key: sample[key] for key in outkey_list if key in sample}
    
    if type(model).__name__ == "Gacha":
        response = model(sample["all_choices"])
    else:
        response =  call_engine_df(sample, model)  # call_model_engine_fn(args, sample, model, tokenizer, processor)
    
    out_dict["full_completions"] = response

    if sample['question_type'] == 'multiple-choice':
        pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
    else:  # open question
        pred_ans = response
    out_dict["response"] = pred_ans
    
    return out_dict


def run(samples, model) -> List[SampleType]:  # @pikaybh
    _field_id = samples[0]["id"].split('_')
    if len(_field_id) > 3:
        _field = f"{_field_id[1]} {_field_id[2]}"
    else:
        _field = str(_field_id[1])
    return [completion(sample, model) for sample in tqdm(samples, postfix=f"LLM running {_field}.")]


def main(llm: Optional[str] = "openai", 
        model: Optional[str] = "gpt-4o", 
        data_path: Optional[str] = "pikaybh/KoCEM-genuine", 
        split: Optional[str] = 'dev', 
        subjects: Optional[List[str]] = ["ALL"],
        seed: Optional[int] = 42,
        output_path: Optional[str] = "output_genuine"):
    
    # subjects 문자열을 쉼표로 분리하여 리스트로 변환
    subject_list = list(subjects)
    
    if subject_list[0] == 'ALL':
        subject_list = list(CAT_SHORT2LONG.keys())

    for cat_short in subject_list:
        subject = CAT_SHORT2LONG[cat_short]
        result_output_path = check_model_output_path(output_path, model, split, subject)
        
        # load config and process to one value
        config = load_yaml(f"configs/{llm}.yaml")
        for key, value in config.items():
            if key != 'eval_params' and isinstance(value, list):
                assert len(value) == 1, 'key {} has more than one value'.format(key)
                config[key] = value[0]

        # merge all dataset
        dataset = load_dataset(data_path, subject, split=split)  # concatenate_datasets(sub_dataset_list)

        # check llm
        if not llm in CHECK_MODEL.keys():
            raise ValueError(f"Unknown llm ({llm = }).")
        # check model
        if not model in CHECK_MODEL[llm]:
            raise ValueError(f"Wrong LLM dependencies ({model = }).")

        # load model
        if llm == "machine":
            model_obj = Gacha(model)
        elif llm == "openai":
            model_obj = OpenAIChatCompletions(model=model)
        elif llm == "google":
            model_obj = GoogleAIGenerativeModel(model=model)
        elif llm == "anthropic":
            model_obj = AnthropicGenerativeModel(model=model)
        elif llm == "hf":
            model_obj = LlamaImageCaptioner(model_id=model)
        else:
            raise ValueError(f"Unknown value error ({llm = }).")

        samples = []
        for sample in dataset:
            sample = process_single_sample(sample)
            sample = construct_prompt(sample, config)
            # sample['image'] = model_obj.image_processor(sample['image']) if sample['image'] else None
            if sample['image']:
                # 딕셔너리 형식의 'image' 데이터를 PIL 이미지로 변환
                image_data = sample['image']
                if isinstance(image_data, dict) and "bytes" in image_data:
                    sample['image'] = Image.open(io.BytesIO(image_data["bytes"]))
                
            # 이후에 model_obj.image_processor(sample['image']) 호출
            sample['image'] = model_obj.image_processor(sample['image']) if sample['image'] else None
            
            samples.append(sample)

        # run ex
        out_samples = run(samples, model_obj)  # run_model(samples, model)  # , call_model_engine, tokenizer, processor)
        save_json(result_output_path, out_samples)


if __name__ == '__main__':
    fire.Fire(main)
