"""Utils for data load, save, and process (e.g., prompt construction)"""

import os
import re

import json
import yaml


DOMAIN_CAT2SUB_CAT = {
    'Domain Knowledge': ['Materials', 'Structural_Engineering']
}
{
    'Construction Terminology': ['Industry_Jargon', 'Standard_Nomenclature'],
    'Domain Knowledge': ['Interior', 'Materials', 'Safety_Management', 'Architectural_Planning', 'Construction_Management', 'Structural_Engineering', 'Building_System'],
    'Domain Reasoning': ['Cost_Estimating', 'Quality_Control', 'Structural_Engineering', 'Project_Management'],
    'Drawing Interpretation': ['위치', '치수', '설계'],
    'Comprehensive Understanding': ['나열', '분류', '산출', '오류'],
}


CAT_SHORT2LONG = {
    "ap" : "Architectural_Planning",
    "bs" : "Building_System",
    "ce" : "Cost_Estimating",
    "cm" : "Construction_Management",
    "ij" : "Industry_Jargon",
    "int": "Interior",
    "mtl": "Materials",
    "pm" : "Project_Management",
    "qc" : "Quality_Control",
    "se" : "Structural_Engineering",
    "sm" : "Safety_Management",
    "sn" : "Standard_Nomenclature"
}


"""
# DATA SAVING
def save_json(filename, ds):
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(ds, f, ensure_ascii=False, indent=4)
"""

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

def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict


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


# DATA SAVING
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



# DATA PROCESSING
def construct_prompt(sample, config):
    question = sample['question']
    options = eval(str(sample['options']))
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

        res_dict['gt_content'] = options[ord(sample['answer'].upper()) - ord('A')]
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
    return res_dict

