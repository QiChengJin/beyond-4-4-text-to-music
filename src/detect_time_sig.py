import sys
import numpy as np
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor


def detect_time_signature(audio_path: str) -> tuple[int, int]:
    activations = RNNDownBeatProcessor()(audio_path)

    # beats_per_bar covers all time sigs in the experiment: 2/4, 3/4, 4/4, 5/4, 6/8
    beats = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4, 5, 6], fps=100)(activations)

    # beats[:, 1] = beat position within bar (1-indexed); max = numerator
    numerator = int(np.max(beats[:, 1]))
    denominator = 8 if numerator == 6 else 4

    return numerator, denominator


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "audio/MusicGen/test.wav"
    num, den = detect_time_signature(path)
    print(f"{path}: {num}/{den}")
