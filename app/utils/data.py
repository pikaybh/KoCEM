"""Utils for data load, save, and process (e.g., prompt construction)"""

import os, re, json, yaml
from random import choice

from .logs import set_logger


logger = set_logger(__name__)


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    """
    start_chr = 'A'
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))
    return index2ans, all_choices


def load_yaml(file_path) -> dict:
    with open(file_path, 'r', encoding='utf-8') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_dict


def load_json(file_path) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def parse_img_path(text):
    matches = re.findall("<img='(.*?)'>", text)
    return matches


def process_single_sample(data):
    question = data['question']
    o_imgs_paths = []
    for option in data['options']:
        current_o_imgs_paths = parse_img_path(option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)

    if len(o_imgs_paths) > 1:  # multiple images in options, used for random selection
        return {'id': data['id'], 
                'question': question, 
                'options': data['options'], 
                'answer': data['answer_key'],
                'explanation': data['explanation'],
                'image': None, 
                'question_type': data['question_type']}
    else:
        return {'id': data['id'], 
                'question': question, 
                'options': data['options'], 
                'answer': data['answer_key'],
                'explanation': data['explanation'],
                'image': data['image'], 
                'question_type': data['question_type']}


# -------------- Data Saving -------------
def save_json(filename, ds):
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(ds, f, ensure_ascii=False, indent=4)


def save_jsonl(filename, data):
    """
    Save a dictionary of data to a JSON Lines file with the filename as key and caption as value.

    Args:
        filename (str): The path to the file where the data should be saved.
        data (dict): The dictionary containing the data to save where key is the image path and value is the caption.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for img_path, caption in data.items():
            # Extract the base filename without the extension
            base_filename = os.path.basename(img_path)
            # Create a JSON object with the filename as the key and caption as the value
            json_record = json.dumps({base_filename: caption}, ensure_ascii=False)
            # Write the JSON object to the file, one per line
            f.write(json_record + '\n')


def save_args(args, path_dir):
    argsDict = args.__dict__
    with open(path_dir + 'setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


# -------------- Data Processing --------------
def construct_prompt(sample, config) -> dict:
    logger.debug(f"{sample = }")
    question = sample['question']
    options = str(sample['options'])
    logger.debug(f"1{options = }")
    options = eval(options)  # eval(sample['options'])
    logger.debug(f"2{options = }")
    example = ""

    if sample['question_type'] == 'multiple-choice':
        start_chr = 'A'
        prediction_range = []
        index2ans = {}

        for option in options:
            prediction_range.append(start_chr)
            example += f"({start_chr}) {option}\n"
            index2ans[start_chr] = option
            start_chr = chr(ord(start_chr) + 1)

        empty_prompt_sample_structure = config['multi_choice_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question, example)
        res_dict = {}
        res_dict['index2ans'] = index2ans
        res_dict['correct_choice'] = sample['answer']
        res_dict['all_choices'] = prediction_range
        res_dict['empty_prompt'] = empty_prompt

        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt

        try:
            res_dict['gt_content'] = options[ord(sample['answer'].upper()) - ord('A')]
        except:
            res_dict['gt_content'] = choice(["A", "B", "C", "D", "E"])
            logger.error(f"{res_dict['gt_content'] = }, {options = }")

    else:
        empty_prompt_sample_structure = config['short_ans_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question)
        res_dict = {}
        res_dict['empty_prompt'] = empty_prompt

        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt
        
        res_dict['gt_content'] = sample['answer']

    res_dict.update(sample)
    logger.debug(f"{res_dict = }")

    return res_dict


__all__ = [
    "get_multi_choice_info",
    "load_yaml",
    "load_json",
    "process_single_sample",
    "save_json",
    "save_jsonl",
    "save_args",
    "construct_prompt"
]