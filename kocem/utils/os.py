import os


def check_model_output_path(*args) -> str:
    output_path = os.path.join(*args) 
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    return os.path.join(output_path, f"output.json")