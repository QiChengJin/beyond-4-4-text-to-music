#!/usr/bin/env python3
"""Generate Suno audio for all prompts in prompts.csv.

Naming convention: {time_sig}_{prompt_type}_{prompt_num}_{sample_num}.mp3
e.g. 2_4_cultural_1_1.mp3  (time_sig 2/4, cultural, prompt 1, sample 1)

Each API call to /api/generate returns 2 clips; we make ceil(needed/2) calls
per row and assign clips to sequential sample numbers. Already-existing files
are skipped so the script is safe to re-run.
"""

import csv
import os
import signal
import subprocess
import sys
import time

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUNO_API_DIR = os.path.join(PROJECT_ROOT, "suno-api")
AUDIO_DIR = os.path.join(PROJECT_ROOT, "audio", "Suno")
CSV_PATH = os.path.join(PROJECT_ROOT, "prompts.csv")
BASE_URL = "http://localhost:3000"

SAMPLES_PER_PROMPT = 5
POLL_INTERVAL = 10   # seconds between status checks
MAX_WAIT = 600       # seconds before giving up on a generation batch
BATCH_DELAY = 8      # seconds between successive API calls (rate-limit buffer)

server_proc = None


# ── helpers ──────────────────────────────────────────────────────────────────

def ts_prefix(time_sig: str) -> str:
    """'3/4' → '3_4',  '6/8' → '6_8'"""
    return time_sig.strip().replace("/", "_")


def audio_filename(time_sig: str, prompt_type: str, prompt_num: int, sample: int) -> str:
    return f"{ts_prefix(time_sig)}_{prompt_type}_{prompt_num}_{sample}.mp3"


def existing_samples(time_sig: str, prompt_type: str, prompt_num: int) -> set[int]:
    existing = set()
    for s in range(1, SAMPLES_PER_PROMPT + 1):
        fname = audio_filename(time_sig, prompt_type, prompt_num, s)
        if os.path.exists(os.path.join(AUDIO_DIR, fname)):
            existing.add(s)
    return existing


# ── server management ────────────────────────────────────────────────────────

def start_server() -> subprocess.Popen:
    print("Starting suno-api server (npm run dev)…")
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=SUNO_API_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def wait_for_server(timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/api/get_limit", timeout=5)
            if r.status_code == 200:
                info = r.json()
                print(f"Server ready — quota: {info}")
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


# ── API calls ────────────────────────────────────────────────────────────────

def generate_batch(prompt: str) -> list[dict]:
    """Submit one generation request; returns the initial clip list (2 clips)."""
    r = requests.post(
        f"{BASE_URL}/api/generate",
        json={"prompt": prompt, "make_instrumental": True, "wait_audio": False},
        headers={"Content-Type": "application/json"},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def poll_until_ready(ids: list[str], max_wait: int = MAX_WAIT) -> list[dict]:
    ids_str = ",".join(ids)
    deadline = time.time() + max_wait
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        try:
            r = requests.get(f"{BASE_URL}/api/get?ids={ids_str}", timeout=30)
            r.raise_for_status()
            clips = r.json()
            statuses = [c.get("status") for c in clips]
            print(f"    polling… {statuses}")
            if all(
                c.get("status") in ("streaming", "complete") and c.get("audio_url")
                for c in clips
            ):
                return clips
            if all(c.get("status") == "error" for c in clips):
                raise TimeoutError(f"All clips errored: {ids_str}")
        except Exception as exc:
            print(f"    poll error: {exc}")
    raise TimeoutError(f"Timed out waiting for clips: {ids_str}")


def download_mp3(url: str, dest: str) -> None:
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=16_384):
            fh.write(chunk)


# ── main loop ────────────────────────────────────────────────────────────────

def process_row(row: dict) -> None:
    time_sig = row["time_sig"].strip()
    prompt_num = int(row["prompt_num"].strip())
    prompt_type = row["prompt_type"].strip()
    prompt = row["prompt"].strip()

    done = existing_samples(time_sig, prompt_type, prompt_num)
    needed = [s for s in range(1, SAMPLES_PER_PROMPT + 1) if s not in done]

    label = f"{time_sig} {prompt_type} #{prompt_num}"
    if not needed:
        print(f"[skip] {label} — all 5 samples present")
        return

    print(f"\n[gen]  {label} — need samples {needed}")
    print(f"       prompt: {prompt}")

    assigned = 0  # how many of `needed` we've saved so far

    while assigned < len(needed):
        try:
            clips = generate_batch(prompt)
        except Exception as exc:
            print(f"  ERROR submitting generation: {exc}. Retrying in 30s…")
            time.sleep(30)
            continue

        ids = [c["id"] for c in clips]
        print(f"  submitted IDs: {', '.join(ids)}")

        try:
            ready = poll_until_ready(ids)
        except TimeoutError as exc:
            print(f"  {exc}. Skipping this batch.")
            break

        for clip in ready:
            if assigned >= len(needed):
                break
            sample_num = needed[assigned]
            fname = audio_filename(time_sig, prompt_type, prompt_num, sample_num)
            fpath = os.path.join(AUDIO_DIR, fname)
            audio_url = clip.get("audio_url")
            if not audio_url:
                print(f"  WARNING: no audio_url for clip {clip.get('id')}; skipping")
                assigned += 1
                continue
            try:
                print(f"  downloading → {fname}")
                download_mp3(audio_url, fpath)
                print(f"  saved: {fname}  ({os.path.getsize(fpath):,} bytes)")
            except Exception as exc:
                print(f"  ERROR downloading {fname}: {exc}")
            assigned += 1

        if assigned < len(needed):
            time.sleep(BATCH_DELAY)


def main() -> None:
    global server_proc

    os.makedirs(AUDIO_DIR, exist_ok=True)

    env_path = os.path.join(SUNO_API_DIR, ".env")
    if not os.path.exists(env_path):
        sys.exit("ERROR: suno-api/.env not found. Please configure it first.")

    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loaded {len(rows)} rows from {CSV_PATH}")

    # Check if server already running
    try:
        requests.get(f"{BASE_URL}/api/get_limit", timeout=3)
        print("Detected running suno-api server — skipping startup.")
    except Exception:
        server_proc = start_server()
        if not wait_for_server(timeout=120):
            sys.exit("ERROR: suno-api server failed to start within 120 s.")

    for row in rows:
        process_row(row)

    print("\nAll done!")


def _shutdown(sig=None, frame=None) -> None:
    if server_proc:
        print("\nShutting down suno-api server…")
        server_proc.terminate()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    try:
        main()
    finally:
        if server_proc:
            server_proc.terminate()
