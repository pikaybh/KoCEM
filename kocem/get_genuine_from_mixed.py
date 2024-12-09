import os
from typing import Optional, List
import fire
import json
from tqdm import tqdm
from datasets import load_dataset

from utils.data import CAT_SHORT2LONG  # Subject에 대한 매핑이 포함된 모듈
from utils.os import check_model_output_path  # 출력 경로 생성 함수

# 필요 없는 "di" 항목 제거
del CAT_SHORT2LONG["di"]

def main(model: Optional[str] = "gpt-4o", 
         data_path: Optional[str] = "pikaybh/KoCEM-genuine", 
         split: Optional[str] = 'val', 
         subjects: Optional[List[str]] = ["ALL"],
         output_path: Optional[str] = "output"):
    
    # "ALL" 설정일 경우 모든 subject 리스트 사용
    subject_list = list(subjects)
    if subject_list[0] == 'ALL':
        subject_list = list(CAT_SHORT2LONG.keys())
    
    # 각 subject에 대해 데이터 필터링 및 저장
    for cat_short in tqdm(subject_list, desc="Processing Subjects"):
        subject = CAT_SHORT2LONG[cat_short]

        # 원본 parsed_output.json 파일 경로
        parsed_output_path = os.path.join(output_path, model, split, subject, "output.json")
        
        # parsed_output.json 파일 읽기
        try:
            with open(parsed_output_path, 'r', encoding="utf-8") as f:
                parsed_data = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {parsed_output_path}")
            continue

        # Huggingface 데이터셋에서 subject별로 데이터 로드
        try:
            dataset = load_dataset(data_path, subject, split=split)
            dataset_ids = {item["id"] for item in dataset}  # Huggingface 데이터의 id 목록만 추출
        except Exception as e:
            print(f"Error loading dataset for subject {subject}: {e}")
            continue

        # parsed_output.json의 항목 중 Huggingface 데이터셋에 없는 ID만 필터링
        filtered_data = [item for item in parsed_data if item["id"] in dataset_ids]
        
        # _genuine 경로 정의 및 파일 저장 경로 설정
        output_file_path = check_model_output_path(f"{output_path}_genuine", model, split, subject)
        
        # 필터링된 데이터를 _genuine 경로에 저장
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w', encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)

        print(f"Filtered data for {subject} saved to {output_file_path}")

if __name__ == "__main__":
    fire.Fire(main)
