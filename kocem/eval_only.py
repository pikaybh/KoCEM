"""Parse and Evaluate Model Output by Category and Domain."""
import os
from typing import Optional

import json
import fire

from utils.data import save_json, CAT_SHORT2LONG, DOMAIN_CAT2SUB_CAT
from utils.eval import evaluate, parse_multi_choice_response, parse_open_response, calculate_ins_level_acc


def main(output_path: Optional[str] = "output/qwen_vl/total_val_output.json", 
         answer_path: str = "answer_dict_test.json") -> None:
    """Evaluates model output by grouping data by category and domain, and calculates accuracy metrics.

    Args:
        output_path (str): The path to the JSON file containing the model's output.
        answer_path (str): The path to the JSON file containing the ground-truth answers.

    This function performs the following steps:
        1. Loads model outputs and answers from specified files.
        2. Groups both outputs and answers by category.
        3. Evaluates each category's accuracy by comparing predictions to ground-truth answers.
        4. Aggregates results by domain and calculates overall accuracy metrics.
        5. Prints the evaluation summary with metrics for each category and domain.

    Example:
        Run the evaluation with default paths:
            python script_name.py
        Run the evaluation with specific paths:
            python script_name.py --output_path="./path/to/output.json" --answer_path="./path/to/answer.json"
    
    Raises:
        FileNotFoundError: If either the output file or the answer file is not found.
    """
    
    output_dict = json.load(open(output_path))
    answer_dict = json.load(open(answer_path))

    # Group output by category
    output_dict_w_cat = {}
    for data_id, parsed_pred in output_dict.items():
        category = "_".join(data_id.split("_")[1:-1])
        if category not in output_dict_w_cat:
            output_dict_w_cat.update({category: {}})
        output_dict_w_cat[category].update({data_id: parsed_pred})

    # Group answer by category
    answer_dict_w_cat = {}
    for data_id, parsed_pred in answer_dict.items():
        category = "_".join(data_id.split("_")[1:-1])
        if category not in answer_dict_w_cat:
            answer_dict_w_cat.update({category: {}})
        answer_dict_w_cat[category].update({data_id: parsed_pred})

    evaluation_result = {}

    # Evaluate each category
    for category in CAT_SHORT2LONG.values():
        print("Evaluating: {}".format(category))
        try:
            cat_outputs = output_dict_w_cat[category]
            cat_answers = answer_dict_w_cat[category]
        except KeyError:
            print("Skipping {} as it is not found.".format(category))
            continue

        examples_to_eval = []
        for data_id, parsed_pred in cat_outputs.items():
            question_type = cat_answers[data_id]['question_type']
            if question_type != 'multiple-choice':
                parsed_pred = parse_open_response(parsed_pred)
            
            examples_to_eval.append({
                "id": data_id,
                "question_type": question_type,
                "answer": cat_answers[data_id]['ground_truth'],
                "parsed_pred": parsed_pred
            })

        judge_dict, metric_dict = evaluate(examples_to_eval)
        metric_dict.update({"num_example": len(examples_to_eval)})

        evaluation_result[category] = metric_dict

    # Aggregate results by domain and calculate overall metrics
    printable_results = {}
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
        
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum([cat_results['num_example'] for cat_results in in_domain_cat_results.values()])
        printable_results['Overall-' + domain] = {"num": int(in_domain_data_num),
                                                  "acc": round(in_domain_ins_acc, 3)}
        # Add each subcategory
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {"num": int(cat_results['num_example']),
                                           "acc": round(cat_results['acc'], 3)}
    
    # Calculate overall accuracy
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results['Overall'] = {"num": sum([cat_results['num_example'] for cat_results in evaluation_result.values()]),
                                    "acc": round(all_ins_acc, 3)}

    print(printable_results)

if __name__ == '__main__':
    fire.Fire(main)
