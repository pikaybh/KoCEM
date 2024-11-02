from typing import Any, Optional



class Template:
    def __init__(self, system_prompt: Optional[str] = "") -> None:
        self.system_prompt = system_prompt

    def run(self, message: str) -> Any:
        """
        Model 별 돌리는 방법에 맞게 편집
        """
        ...
        return

    def __call__(self, message: str) -> str:
        """
        Strip from `self.run()` result
        """
        ...
        return 