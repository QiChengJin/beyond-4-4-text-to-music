import os
import csv
import re
import time
from pathlib import Path
from google import genai
from google.genai import types

# ── Configuration ──────────────────────────────────────────────────────────────
# How many runs to generate per prompt this session.
# Start at 1 to gauge credit usage, increase toward 5 as budget allows.
RUNS_PER_PROMPT = 5

MAX_RUNS = 5  # hard cap — never generate more than 5 per prompt

API_KEY = os.environ.get("GOOGLE_AI_API_KEY", "AIzaSyDjhcQQFNDvko4EVTtHP2FuvBQViyXCSZ0")

PROMPTS_CSV = Path(__file__).parent.parent / "prompts.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "audio" / "Lyria"
# ──────────────────────────────────────────────────────────────────────────────

client = genai.Client(api_key=API_KEY)


def _file_stem(time_sig: str, prompt_type: str, prompt_num: int) -> str:
    """e.g. '6/8', 'formal', 5  →  '6_8_formal_5'"""
    prefix = time_sig.replace("/", "_")
    return f"{prefix}_{prompt_type}_{prompt_num}"


def _existing_run_nums(output_dir: Path, stem: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+)\.wav$")
    nums = []
    for f in output_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def _generate_one(prompt: str, output_path: Path) -> None:
    response = client.models.generate_content(
        model="lyria-3-clip-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
        ),
    )
    if not response.parts:
        candidate = response.candidates[0] if response.candidates else None
        raise ValueError(
            f"Empty response — finish_reason: {candidate.finish_reason if candidate else 'no candidates'}"
        )
    for part in response.parts:
        if part.inline_data is not None:
            output_path.write_bytes(part.inline_data.data)
            return
    raise ValueError(f"No audio in response: {response}")


def run_pipeline(runs_per_prompt: int = RUNS_PER_PROMPT) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROMPTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    target = min(runs_per_prompt, MAX_RUNS)
    generated = 0
    skipped = 0

    for row in rows:
        time_sig = row["time_sig"]
        prompt_num = int(row["prompt_num"])
        prompt_type = row["prompt_type"]
        prompt_text = row["prompt"]

        stem = _file_stem(time_sig, prompt_type, prompt_num)
        existing = _existing_run_nums(OUTPUT_DIR, stem)
        existing_count = len(existing)

        if existing_count >= MAX_RUNS:
            print(f"[skip] {stem} — already at max {MAX_RUNS}")
            skipped += 1
            continue

        to_generate = target - existing_count
        if to_generate <= 0:
            print(f"[skip] {stem} — already has {existing_count} run(s), target is {target}")
            skipped += 1
            continue

        for _ in range(to_generate):
            # re-scan each iteration so run_num is always accurate
            current = _existing_run_nums(OUTPUT_DIR, stem)
            run_num = (max(current) + 1) if current else 1
            output_path = OUTPUT_DIR / f"{stem}_{run_num}.wav"

            print(f"[gen]  {output_path.name}")
            print(f"       prompt: {prompt_text!r}")
            _generate_one(prompt_text, output_path)
            generated += 1
            time.sleep(0.5)  # brief pause between requests

    print(f"\nDone — generated {generated} new file(s), skipped {skipped}/{len(rows)} prompts.")


if __name__ == "__main__":
    run_pipeline()
