import os
import sys
import csv
import re
import warnings
import numpy as np
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

warnings.filterwarnings("ignore", message="dtype().*align")

# RNN weights are expensive to load — do it once
_rnn = RNNDownBeatProcessor()
# Two DBN processors: constrained for common sigs, full for 5/4 and 6/8
_dbn_234 = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=100)
_dbn_full = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4, 5, 6], fps=100)


def detect_time_signature(audio_path: str, desired_numerator: int) -> dict:
    dbn = _dbn_full if desired_numerator in (5, 6) else _dbn_234
    activations = _rnn(audio_path)
    beats = dbn(activations)

    numerator = int(np.max(beats[:, 1]))
    denominator = 8 if numerator == 6 else 4

    # Tempo from median beat interval
    beat_times = beats[:, 0]
    tempo = round(60.0 / float(np.median(np.diff(beat_times))), 1) if len(beat_times) > 1 else 0.0

    # Confidence: peak downbeat activation strength
    confidence = round(float(np.max(activations[:, 1])), 4)

    return {
        "actual_time_sig": f"{numerator}/{denominator}",
        "tempo": round(tempo, 1),
        "confidence": round(confidence, 4),
    }


def parse_filename(filename: str) -> dict | None:
    """Extract metadata from filename like 3_4_cultural_1_1.wav"""
    stem = os.path.splitext(filename)[0]
    m = re.fullmatch(r"(\d+)_(\d+)_(cultural|formal)_(\d+)_(\d+)", stem)
    if not m:
        return None
    num, den, prompt_type, prompt_num, run = m.groups()
    return {
        "desired_time_sig": f"{num}/{den}",
        "prompt_type": prompt_type,
        "prompt_num": int(prompt_num),
        "run": int(run),
    }


def analyze_folder(folder: str, output_csv: str) -> None:
    model = os.path.basename(folder.rstrip("/"))
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg"}

    files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in audio_extensions
    )

    if not files:
        print(f"No audio files found in {folder}")
        return

    fieldnames = [
        "filename", "model", "desired_time_sig", "actual_time_sig",
        "correct", "prompt_type", "prompt_num", "run", "tempo", "confidence",
    ]

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, fname in enumerate(files, start=1):
            path = os.path.join(folder, fname)
            print(f"[{i}/{len(files)}] {fname}")

            meta = parse_filename(fname)
            if meta is None:
                print(f"  skipping — filename doesn't match expected pattern")
                continue

            desired_num = int(meta["desired_time_sig"].split("/")[0])
            try:
                result = detect_time_signature(path, desired_num)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            writer.writerow({
                "filename": fname,
                "model": model,
                "desired_time_sig": meta["desired_time_sig"],
                "actual_time_sig": result["actual_time_sig"],
                "correct": meta["desired_time_sig"] == result["actual_time_sig"],
                "prompt_type": meta["prompt_type"],
                "prompt_num": meta["prompt_num"],
                "run": meta["run"],
                "tempo": result["tempo"],
                "confidence": result["confidence"],
            })

    print(f"\nSaved {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_folder.py <audio_folder> [output.csv]")
        sys.exit(1)

    folder = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(folder, "results.csv")
    analyze_folder(folder, output_csv)
