import os

from schemas.kocem import LocaleType, SplitType, Subject, KoCEM
from utils.data import load_json, save_json
from utils.eval import evaluate_difficulties
from utils.logs import set_logger


logger = set_logger(__name__)
SubjectsDict = {name: val for name, val in KoCEM.__dict__.items() if isinstance(val, Subject)}
OUTPUT_DIR = os.getenv("OUTPUT_PATH", "output")


def eval_difficulties(
    model: str = "gpt-4.1",
    prompt: str = "mcqa",
    locale: LocaleType = "en",
    subjects: str | list[str] = list(SubjectsDict),
    splits: SplitType | list[SplitType] = ["dev", "test", "val"],
):
    """
    Evaluate the difficulty levels of the model's performance on specified subjects and splits.
    
    Args:
        locale (LocaleType): The locale to use (default: "en").
        subject (str | list[str]): The subject(s) to evaluate (default: all subjects).
        split (SplitType | list[SplitType]): The split(s) to evaluate (default: ["dev", "test", "val"]).
        prompt (str): The prompt type, e.g., "mcqa" or "open" (default: "mcqa").
        **kwargs: Additional keyword arguments for evaluation.
    """

    subjects = subjects if isinstance(subjects, list) else [subjects]
    splits = splits if isinstance(splits, list) else [splits]
    for subset in subjects:
        if not SubjectsDict[subset].has_difficulty:
            logger.info(f"Skipping {subset} as it does not have difficulty levels.")
            continue

        for split in splits:
            logger.debug(f"Subject: {subset}, Split: {split}")
            output_dir = os.path.join(OUTPUT_DIR, prompt, locale, model, split, subset)
            
            samples = load_json(os.path.join(output_dir, "output.json"))
            results = load_json(os.path.join(output_dir, "result.json"))
            results.update(evaluate_difficulties(samples))

            save_json(results, os.path.join(output_dir, "result.json"))
            logger.debug(f"Saved evaluation results to {os.path.join(output_dir, 'result.json')}")


if __name__ == "__main__":
    from fire import Fire
    Fire(eval_difficulties)

__all__ = [
    "eval_difficulties"
]