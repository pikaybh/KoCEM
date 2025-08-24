import os, yaml
from dotenv import load_dotenv

load_dotenv()



class PromptManager:
    def __init__(self,
        name: str,
        locale: str = "en",
        version: str = "latest"
    ):
        # Validate version: allow 'latest' or date-like string with two dashes
        if not (version == "latest" or version.count("-") == 2):
            raise ValueError("Version must be 'latest' or in the format 'YYYY-MM-DD'.")

        self.name = name
        self.locale = locale
        self.prompt_dir = os.path.join(os.getenv("PROMPT_PATH", "prompts"), locale, name)
        if not os.path.exists(self.prompt_dir):
            raise FileNotFoundError(f"Prompt directory '{self.prompt_dir}' does not exist.")

        # Collect YAML releases (strip extension) and sort
        self.releases = sorted([
            file_.rsplit(".", 1)[0]
            for file_ in os.listdir(self.prompt_dir)
            if file_.lower().endswith('.yaml')
        ])
        if not self.releases:
            raise FileNotFoundError(f"No .yaml prompt files found in '{self.prompt_dir}'.")

        if version == "latest":
            self.version = self.releases[-1]
        else:
            if version not in self.releases:
                raise ValueError(f"Version '{version}' not in releases: {self.releases}")
            self.version = version

        with open(os.path.join(self.prompt_dir, f"{self.version}.yaml"), 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f) or {}
        self.prompt = loaded  # uses setter
        self.system = self.prompt.get("system", "")
        self.human = self.prompt.get("human", "")

    # ---- prompt property ----
    @property
    def prompt(self) -> dict:
        return self._prompt

    @prompt.setter
    def prompt(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError("Prompt must be a dictionary.")
        self._prompt = value

    # ---- system property ----
    @property
    def system(self) -> str:
        return self._system

    @system.setter
    def system(self, value: str):
        if not isinstance(value, str):
            raise ValueError("System prompt must be a string.")
        self._system = value

    # ---- human property ----
    @property
    def human(self) -> str:
        return self._human

    @human.setter
    def human(self, value: str):
        if not isinstance(value, str):
            raise ValueError("Human prompt must be a string.")
        self._human = value



__all__ = ['PromptManager']