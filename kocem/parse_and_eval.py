"""Parse and Evaluate"""

import os
import logging
from typing import List, Optional

import fire
import json

from utils.data import save_json, CAT_SHORT2LONG
from utils.eval import evaluate, parse_multi_choice_response, parse_open_response


# Root 
logger_name = 'parse_and_eval'
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


def main(path: Optional[str] = "output/llava1.5_13b", 
         subject: Optional[List[str]] = None, 
         split: Optional[str] = 'dev') -> None:
    """Evaluates model output for specified categories and saves parsed results.

    Args:
        path (str): The path to the model output directory. Default is "./example_outputs/llava1.5_13b".
        subject (list): A list of category short names to evaluate. Use 'ALL' to evaluate all available categories.
            Available values are the keys in CAT_SHORT2LONG.

    Example:
        To evaluate specific categories:
            python script_name.py --path="./output" --subject "cat1" "cat2"
        To evaluate all categories:
            python script_name.py --path="./output" --subject "ALL"
    
    This function performs the following steps:
        1. Checks and sets categories to evaluate.
        2. Iterates through specified categories, loading JSON outputs.
        3. Parses and evaluates each sample based on question type (multiple-choice or open).
        4. Calculates evaluation metrics and saves parsed results and metrics as JSON files.

    Raises:
        FileNotFoundError: If the specified category's output file is not found in the output directory.
    """

    if subject is None or subject[0] == 'ALL':
        subject = list(CAT_SHORT2LONG.keys())

    if "genuine" in path.split('/')[0]:
        subject.remove("di")

    ex_output_path = os.path.join(path, split)

    all_results = {}
    for cat_short in subject:
        category = CAT_SHORT2LONG[cat_short]
        logger.info("Evaluating: {}".format(category))
        if category not in os.listdir(ex_output_path):
            logger.info("Skipping {} for not found".format(category))
        else:
            cat_folder_path = os.path.join(ex_output_path, category)
            cat_outputs = json.load(open(os.path.join(cat_folder_path, 'output.json'), encoding='utf-8'))
            # Evaluation
            eval_samples = []
            for cat_output in cat_outputs:
                response = cat_output['response']
                if cat_output['question_type'] == 'multiple-choice':         
                    all_choices = cat_output['all_choices']
                    index2ans = cat_output['index2ans']
                    parsed_pred = parse_multi_choice_response(response, all_choices, index2ans)
                    eval_samples.append(
                        {
                            'id': cat_output['id'],
                            'question_type': cat_output['question_type'],
                            'answer': cat_output['answer'],  # the content in option, not answer index.
                            'response': response,
                            'parsed_pred': parsed_pred,
                            'index2ans': index2ans,
                        }
                    )
                else:  # open
                    parsed_pred = parse_open_response(response)
                    eval_samples.append(
                        {
                            'id': cat_output['id'],
                            'question_type': cat_output['question_type'],
                            'answer': cat_output['answer'],
                            'response': response,
                            'parsed_pred': parsed_pred,
                        }
                    )

            logger.info("Num of valid samples: {}, Expected Num: {}".format(len(eval_samples), len(cat_outputs)))
            
            judge_dict, metric_dict = evaluate(eval_samples)
            metric_dict.update({"num_example": len(eval_samples)})
            for eval_sample in eval_samples:
                eval_sample.update({"judge": judge_dict[eval_sample['id']]})

            save_json(os.path.join(cat_folder_path, 'parsed_output.json'), eval_samples)
            save_json(os.path.join(cat_folder_path, 'result.json'), metric_dict)

if __name__ == '__main__':
    fire.Fire(main)
