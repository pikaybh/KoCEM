"""
Update README.md Leaderboard from aggregated metrics in output/<prompt>/<locale>/<model>/<split>/<subject>/result.json.

- Locales come from prompts/ (e.g., prompts/en, prompts/ko)
- Base path: prefer output/mcqa; fallback to output/test
- For each locale, aggregate per model & per subject across all splits (weighted by num_example)
- Render per-locale HTML tables with columns: Rank, Model, and each subject's accuracy
- Rank within each locale by overall weighted accuracy across subjects

Usage:
	uv run python -m app.update_readme
	# or
	uv run python app/update_readme.py
"""

from __future__ import annotations

import json, os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
from schemas.kocem import KoCEM, Subject


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
README_PATH = os.path.join(REPO_ROOT, "README.md")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", os.path.join(REPO_ROOT, "output"))
PROMPTS_DIR = os.path.join(REPO_ROOT, "prompts")

KNOWN_LOCALES = {"en", "ko", "zh", "ja", "es", "fr", "de", "it", "pt", "ru"}

# Static README template. The {LEADERBOARD} placeholder will be replaced with generated HTML tables.
README_TEMPLATE = """# KoCEM: A Multimodal Knowledge and Reasoning Benchmark for Korean Construction Engineering & Management

{LEADERBOARD}

## Evaluation Guidelines
This repo provides a simple CLI to run evaluation over KoCEM subjects, produce outputs, and update the README leaderboard.

### Output layout
Results are saved under `output/<prompt>/<locale>/<model>/<split>/<subject>/` as:

```
output/
	<prompt>/            # e.g., mcqa (preferred) or test (fallback)
		<locale>/          # e.g., en, ko
			<model>/         # e.g., gpt-4.1
				<split>/       # dev, test, val, extra
					<subject>/   # e.g., Architectural_Planning
						output.json        # per-sample entries with predictions and judges
						evaluation.json    # per-sample judge details
						result.json        # aggregated metrics (acc, std_dev, num_example, ...)
```

### Run inference
Use the `run_each` command to generate predictions and metrics.

```pwsh
uv run python -m app run_each --model gpt-4.1 --locale en --subjects Architectural_Planning --splits dev --prompt test

# Multiple subjects
uv run python -m app run_each --model gpt-4.1 --locale en --subjects '["Architectural_Planning","Materials"]' --splits '["dev","val"]' --prompt test
```

### Evaluate difficulties (optional)
If a subject provides difficulty labels, compute difficulty-wise metrics from saved outputs:

```pwsh
uv run python -m app eval_difficulties --model gpt-4.1 --prompt test --locale en
```

### Update README leaderboard
Generate per-locale leaderboards and inject them into this README:

```pwsh
uv run python -m app.update_readme
```

The updater prefers `output/mcqa`; if absent, it falls back to `output/test`.

### Prompts
Prompts live under `prompts/<locale>/...`. Pick a prompt by name with `--prompt` (e.g., `test`, `mcqa`).

### Configuration
Set the following environment variables (via `.env` or shell) as needed:

- `DS_PATH`: Hugging Face dataset path (e.g., `pikaybh/KoCEM`)
- `DS_CACHE_PATH`: HF cache directory
- `OUTPUT_PATH`: Root directory for outputs (defaults to `./output`)
- Provider credentials (e.g., OpenAI) according to your model choice
"""


@dataclass
class SubjectAgg:
	acc_sum: float = 0.0
	n: int = 0

	@property
	def acc(self) -> float:
		return 0.0 if self.n <= 0 else self.acc_sum / self.n


@dataclass
class ModelAgg:
	model: str
	locale: str
	subjects: Dict[str, SubjectAgg] = field(default_factory=dict)

	@property
	def overall_acc(self) -> float:
		total_sum = sum(sa.acc_sum for sa in self.subjects.values())
		total_n = sum(sa.n for sa in self.subjects.values())
		return 0.0 if total_n == 0 else total_sum / total_n


def iter_result_files(base_dir: str):
	for root, _dirs, files in os.walk(base_dir):
		if "result.json" in files:
			yield os.path.join(root, "result.json")


def parse_path_components(base_prompt_dir: str, result_path: str) -> Optional[Tuple[str, str, str, str]]:
	"""Return (locale, model, split, subject) for a result.json path under base_prompt_dir.

	Expected pattern: <base_prompt_dir>/<locale>/<model>/<split>/<subject>/result.json
	If the locale segment is missing, skip (we only aggregate per-locale).
	"""
	rel = os.path.relpath(result_path, base_prompt_dir)
	parts = rel.replace("\\", "/").split("/")
	if len(parts) < 5:
		return None
	# Require the standard with-locale pattern
	locale = parts[0]
	model = parts[1]
	split = parts[2]
	subject = parts[3]
	return locale, model, split, subject


def load_result(path: str) -> Optional[Dict]:
	try:
		with open(path, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return None


def _build_subject_to_dimension() -> Dict[str, str]:
	"""Create mapping from subject folder name to its dimension using KoCEM schema."""
	mapping: Dict[str, str] = {}
	for name, val in KoCEM.__dict__.items():
		if isinstance(val, Subject):
			mapping[name] = val.dimension
	return mapping


def aggregate_per_locale_split_and_overall(base_prompt_dir: str, locales: List[str], eval_by: Literal["subject", "dimension"]) -> Tuple[Dict[str, Dict[str, Dict[str, ModelAgg]]], Dict[str, Dict[str, ModelAgg]]]:
	"""Aggregate metrics into two views:
	- per_split: locale -> split -> model -> ModelAgg
	- overall:   locale -> model -> ModelAgg (all splits combined, weighted by num_example)
	"""
	per_split: Dict[str, Dict[str, Dict[str, ModelAgg]]] = {}
	overall: Dict[str, Dict[str, ModelAgg]] = {}
	subject_to_dimension = _build_subject_to_dimension() if eval_by == "dimension" else {}

	for res_file in iter_result_files(base_prompt_dir):
		parsed = parse_path_components(base_prompt_dir, res_file)
		if not parsed:
			continue
		locale, model, split, subject = parsed
		if locale not in locales:
			continue
		data = load_result(res_file)
		if not data:
			continue
		acc = float(data.get("acc", 0.0))
		n = int(data.get("num_example", 0))

		key = subject if eval_by == "subject" else subject_to_dimension.get(subject)
		if not key:
			continue

		# overall aggregation
		loc_bucket = overall.setdefault(locale, {})
		mod_agg = loc_bucket.setdefault(model, ModelAgg(model=model, locale=locale))
		subj_agg = mod_agg.subjects.setdefault(key, SubjectAgg())
		subj_agg.acc_sum += acc * n
		subj_agg.n += n

		# per-split aggregation
		split_bucket = per_split.setdefault(locale, {}).setdefault(split, {})
		mod_agg_s = split_bucket.setdefault(model, ModelAgg(model=model, locale=locale))
		subj_agg_s = mod_agg_s.subjects.setdefault(key, SubjectAgg())
		subj_agg_s.acc_sum += acc * n
		subj_agg_s.n += n

	return per_split, overall


def _format_table_for_locale(models: Dict[str, ModelAgg], title: str, label: str) -> List[str]:
	out: List[str] = []
	if not models:
		return out
	# Collect union of groups for this slice
	subjects = sorted({s for m in models.values() for s in m.subjects.keys()})
	out.append(f"### {title}\n\n")
	if not subjects:
		out.append("<p><em>No results found.</em></p>\n\n")
		return out
	ranked = sorted(models.values(), key=lambda m: m.overall_acc, reverse=True)
	out.append("<table>\n<thead>\n<tr>")
	out.append("<th>Rank</th><th>Model</th><th>Total</th>")
	for s in subjects:
		out.append(f"<th>{s}</th>")
	out.append("</tr>\n</thead>\n<tbody>\n")
	for i, m in enumerate(ranked, start=1):
		out.append("<tr>")
		out.append(f"<td>{i}</td><td>{m.model}</td>")
		# overall weighted accuracy across groups (for this slice)
		out.append(f"<td>{m.overall_acc*100:.2f}%</td>")
		for subj in subjects:
			sa = m.subjects.get(subj)
			cell = f"{sa.acc*100:.2f}%" if sa and sa.n > 0 else "-"
			out.append(f"<td>{cell}</td>")
		out.append("</tr>\n")
	out.append("</tbody>\n</table>\n\n")
	return out


def format_leaderboard_with_splits(per_split: Dict[str, Dict[str, Dict[str, ModelAgg]]], overall: Dict[str, Dict[str, ModelAgg]], eval_by: Literal["subject", "dimension"]) -> str:
	out: List[str] = []
	out.append("## Leaderboard\n\n")
	label = "subject" if eval_by == "subject" else "dimension"
	out.append(f"<p>Per-locale rankings. First, per split tables; then an overall weighted table. Columns show {label} accuracies. Higher is better.</p>\n\n")

	# Preferred split display order
	split_order = ["dev", "test", "val", "extra"]

	for locale in sorted(set(per_split.keys()) | set(overall.keys())):
		out.append(f"## Locale: {locale}\n\n")
		# per-split
		splits = per_split.get(locale, {})
		# Keep present splits; sort by preferred order then name
		present = list(splits.keys())
		present.sort(key=lambda s: (split_order.index(s) if s in split_order else len(split_order), s))
		for split in present:
			title = f"Locale: {locale} — Split: {split}"
			out.extend(_format_table_for_locale(splits[split], title, label))
		# overall
		if locale in overall:
			title = f"Locale: {locale} — All splits (weighted)"
			out.extend(_format_table_for_locale(overall[locale], title, label))

	return "".join(out)
def replace_leaderboard_section(readme_text: str, new_section_md: str) -> str:
	# Deprecated: kept for backward compatibility; now we overwrite via template.
	return new_section_md


def ensure_project_header(readme_text: str) -> str:
	"""If the header looks like another project's template, replace it with KoCEM header."""
	return readme_text


def get_locales_from_prompts(prompts_dir: str) -> List[str]:
	if not os.path.isdir(prompts_dir):
		return []
	return [d for d in os.listdir(prompts_dir) if os.path.isdir(os.path.join(prompts_dir, d))]


def update_readme(eval_by: Literal["subject", "dimension"] = "subject") -> int:
	if not os.path.isdir(OUTPUT_PATH):
		print(f"No output directory found at {OUTPUT_PATH}")
		return 1

	# Choose base prompt folder: mcqa preferred, else test
	mcqa_dir = os.path.join(OUTPUT_PATH, "mcqa")
	test_dir = os.path.join(OUTPUT_PATH, "test")
	if os.path.isdir(mcqa_dir):
		base_prompt_dir = mcqa_dir
	elif os.path.isdir(test_dir):
		base_prompt_dir = test_dir
	else:
		print("Neither output/mcqa nor output/test found; nothing to update.")
		return 1

	locales = get_locales_from_prompts(PROMPTS_DIR)
	if not locales:
		print("No locales found under prompts/.")
		return 1

	per_split, overall = aggregate_per_locale_split_and_overall(base_prompt_dir, locales, eval_by)
	if not (any(per_split.values()) or any(overall.values())):
		print("No result.json files found for detected locales; nothing to update.")
		return 1

	leaderboard_md = format_leaderboard_with_splits(per_split, overall, eval_by)

	# Build full README content from template and overwrite
	readme_full = README_TEMPLATE.format(LEADERBOARD=leaderboard_md)
	with open(README_PATH, "w", encoding="utf-8") as f:
		f.write(readme_full)

	print("README.md Leaderboard updated.")
	return 0


if __name__ == "__main__":
	from fire import Fire
	Fire(update_readme)
	
__all__ = [
    "update_readme"
]