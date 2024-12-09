import os


def check_model_output_path(*args, **kwargs) -> str:
    output_path = os.path.join(*args) 
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if "json_name" in kwargs.keys():
        return os.path.join(output_path, kwargs["json_name"])
    
    return os.path.join(output_path, f"output.json")