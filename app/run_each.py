from apis import module
from schemas.kocem import LocaleType, SplitType, Subject, KoCEM
from utils.llm import get_provider
from utils.logs import set_logger


logger = set_logger(__name__)
SubjectsDict = {name: val for name, val in KoCEM.__dict__.items() if isinstance(val, Subject)}


def run_each(
    model: str = "gpt-4.1",
    locale: LocaleType = "en",
    subjects: str | list[str] = list(SubjectsDict),
    splits: SplitType | list[SplitType] = ["dev", "test", "val"],
    task: str = "mcqa",
    retries: int = 3,
    timeout: int = 30,
    **kwargs
):
    """
    Main function to run the API for specified model, locale, subject, and split.
    
    Args:
        model (str): The model to use (default: "gpt-4.1").
        locale (LocaleType): The locale to use (default: "en").
        subject (str | list[str]): The subject(s) to use (default: all subjects).
        split (SplitType | list[SplitType]): The split(s) to use (default: ["dev", "test", "val"]).
        task (str): The task type, e.g., "mcqa" or "open" (default: "mcqa").
        retries (int): Number of retries for API calls (default: 3).
        timeout (int): Timeout in seconds for each API call (default: 30).
        **kwargs: Additional keyword arguments for the API.
    """
    provider = get_provider(model)
    api = module[provider](model_id=f"{provider}/{model}", locale=locale, task=task, **kwargs)

    subjects = subjects if isinstance(subjects, list) else [subjects]
    splits = splits if isinstance(splits, list) else [splits]
    for subset in subjects:
        for split in splits:
            if SubjectsDict[subset].split[split] is None:
                logger.info(f"Skipping {subset} - {split} as it does not have data.")
                continue

            logger.debug(f"Subject: {subset}, Split: {split}")
            api(subset=subset, split=split, max_retries=retries, max_timeout=timeout)


if __name__ == "__main__":
    from fire import Fire
    Fire(run_each)

__all__ = [
    "run_each"
]