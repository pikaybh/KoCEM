from fire import Fire

from run_each import run_each
from eval_difficulties import eval_difficulties
from eval_total import evaluate_total
from update_readme import update_readme


def run_sequentially(**kwargs):
    """
    Run both the run_each and eval_difficulties functions sequentially.
    """
    run_each(**kwargs)
    eval_difficulties(**kwargs)
    for split in ["dev", "val", "test"]:
        evaluate_total(split=split, **kwargs)
    update_readme()


if __name__ == '__main__':
    Fire({
        "diff": eval_difficulties,
        "per": evaluate_total,
        "auto": run_sequentially,
        "each": run_each,
        "readme": update_readme
    })