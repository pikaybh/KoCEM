from typing import List, Optional

import fire
import json
import logging
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

from utils.os import check_model_output_path
from utils.data import CAT_SHORT2LONG


# Root 
logger_name = 'digger'
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


def main(data_path: Optional[str] = "pikaybh/KoCEM", 
        splits: Optional[List[str]] = ["ALL"], 
        subjects: Optional[List[str]] = ["ALL"]):

    """
    Load a dataset from Hugging Face and save specific information for each split
    into JSON files in the format: "gold/answer_dict_{split}.json".

    Args:
        data_path (str): The Hugging Face dataset path or identifier.
        splits (list): A list of dataset splits to process (e.g., ["dev", "extra", "test", "val"]). If the list contains only ["ALL"], all available splits ("dev", "extra", "test", "val") are used.

    Process:
        1. Load the dataset from Hugging Face using the specified path.
        2. If `splits` is ["ALL"], replace with all available splits in the dataset.
        3. For each split:
            - Retrieve all data samples for the split.
            - For each sample, extract `id`, `question_type`, and `answer` fields.
            - Organize the extracted data into a dictionary where each key is an `id`, and the value is a dictionary with `question_type` and `ground_truth` fields.
        4. Save the resulting dictionary to "gold/answer_dict_{split}.json" for each split.

    dataset:
        >>> save_gold_answers("hf_dataset_path", ["dev", "test"])
        This will save JSON files for "dev" and "test" splits.

        >>> save_gold_answers("hf_dataset_path", ["ALL"])
        This will save JSON files for "dev", "extra", "test", and "val" splits.

    Raises:
        KeyError: If the dataset does not contain the required fields (`id`, `question_type`, `answer`).
    """

    if splits[0] == 'ALL':
        splits = ['dev', 'test', 'val', 'extra']

    for split in tqdm(splits):
        tqdm.write(f"Processing {split} split data.")

        answer_dict = {}

        if subjects[0] == 'ALL':
            subjects = list(CAT_SHORT2LONG.keys())

        for cat_short in subjects:
            subject = CAT_SHORT2LONG[cat_short]

            # run for each subject
            # sub_dataset_list = []
            for subject in CAT_SHORT2LONG.values():
                dataset = load_dataset(data_path, subject, split=split)
                # sub_dataset_list.append(sub_dataset)

                # merge all dataset
                # dataset = concatenate_datasets(sub_dataset_list)
                
                # Iterate over each example in the dataset
                for example in dataset:
                    answer_dict[example["id"]] = {
                        "question_type": example["question_type"],
                        "ground_truth": example["answer_key"]
                    }

        # Step 5: Save each split's data to JSON files
        file_path = f"gold/answer_dict_{split}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(answer_dict, f, ensure_ascii=False, indent=4)
        logger.info(f"\nSaved {file_path}")

if __name__ == "__main__":
    fire.Fire(main)
