import os
import modal

app = modal.App("musicgen-pipeline")

# Persistent volume — model weights (~1.5GB) downloaded once, reused across runs
volume = modal.Volume.from_name("musicgen-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchaudio", "transformers", "scipy", "accelerate")
)

PROMPTS = {
    "2_4": {
        "cultural": [
            "an energetic march with brass and snare drums",
            "a cheerful polka with accordion and woodwinds",
            "a festive can-can with strings and light percussion",
            "a military march with trumpets, tuba, and bass drum",
            "a bouncy polka with piano and clarinet",
        ],
        "formal": [
            "an energetic piece in 2/4 time with brass and snare drums",
            "a cheerful dance in 2/4 time with accordion and woodwinds",
            "a festive dance in 2/4 time with strings and light percussion",
            "a military-style piece in 2/4 time with trumpets, tuba, and bass drum",
            "a bouncy dance in 2/4 time with piano and clarinet",
        ],
    },
    "3_4": {
        "cultural": [
            "an elegant Viennese waltz with piano and strings",
            "a melancholic mazurka with accordion and violin",
            "a stately minuet with harpsichord and flute",
            "a rustic country waltz with guitar and fiddle",
            "a dramatic waltz with full orchestra and sweeping strings",
        ],
        "formal": [
            "an elegant song in 3/4 time with piano and strings",
            "a melancholic dance in 3/4 time with accordion and violin",
            "a stately piece in 3/4 time with harpsichord and flute",
            "a rustic song in 3/4 time with guitar and fiddle",
            "a dramatic piece in 3/4 time with full orchestra and sweeping strings",
        ],
    },
    "4_4": {
        "cultural": [
            "a deep house track with four-on-the-floor kick and warm synth bass",
            "a driving techno track with pounding kick drum and hypnotic synthesizers",
            "a jersey club track with rapid triplet hi-hats and bouncy bass",
            "a glittery disco groove with lush strings, bass guitar, and steady kick drum",
            "a roots reggae song with one-drop rhythm, guitar skank on the offbeats, and deep bass",
        ],
        "formal": [
            "an electronic dance track in 4/4 time with steady kick on every beat and warm synth bass",
            "a driving electronic track in 4/4 time with pounding kick drum and hypnotic synthesizers",
            "a club track in 4/4 time with rapid triplet hi-hats and bouncy bass",
            "a dance track in 4/4 time with lush strings, bass guitar, and steady kick drum",
            "a song in 4/4 time with guitar emphasis on the offbeats and deep bass",
        ],
    },
    "5_4": {
        "cultural": [
            "a jazz piece with the same groove and rhythmic feel as Take Five, with piano and saxophone",
            "a progressive rock track with an asymmetric, five-beat groove, synth and electric guitar",
            "a tense spy theme in the style and rhythmic feel as Mission Impossible with strings",
            "a Bulgarian folk-inspired dance with woodwinds and hand percussion",
            "a moody jazz fusion piece with that quintuplet swing feel",
        ],
        "formal": [
            "a jazz piece in 5/4 time with piano and saxophone",
            "a progressive rock track in 5/4 time with synth and electric guitar",
            "a tense spy theme piece in 5/4 time with strings",
            "a folk-style dance in 5/4 time with woodwinds and hand percussion",
            "a moody jazz fusion piece in 5/4 time",
        ],
    },
    "6_8": {
        "cultural": [
            "a lively Irish jig with fiddle and bodhrán",
            "a spirited tarantella with guitar and tambourine",
            "a gentle barcarolle with piano in a flowing gondola style",
            "a furlana with uilleann pipes and frame drum",
            "a playful saltarello with flute and harp",
        ],
        "formal": [
            "a lively dance in 6/8 time with fiddle and bodhrán",
            "a spirited Italian folk dance in 6/8 time with guitar and tambourine",
            "a gentle piece in 6/8 time with piano in a flowing style",
            "a Celtic-style dance in 6/8 time with pipes and frame drum",
            "a playful dance in 6/8 time with flute and harp",
        ],
    },
}

RUNS = 5
OUTPUT_DIR = "audio/MusicGen"


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/model-cache": volume},
    timeout=600,
)
class MusicGenerator:
    @modal.enter()
    def load_model(self):
        import os
        from transformers import pipeline as hf_pipeline
        os.environ["HF_HOME"] = "/model-cache"
        self.pipe = hf_pipeline(
            "text-to-audio",
            "facebook/musicgen-medium",
            device="cuda",
        )

    @modal.method()
    def generate(self, prompt: str) -> bytes:
        import io
        import scipy.io.wavfile
        result = self.pipe(prompt, forward_params={"do_sample": True})
        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, rate=result["sampling_rate"], data=result["audio"])
        return buf.getvalue()


def build_task_list() -> list[tuple[str, str]]:
    """Return (prompt, output_path) pairs, skipping already-generated files."""
    tasks = []
    for time_sig, prompt_types in PROMPTS.items():
        for prompt_type, prompts in prompt_types.items():
            for i, prompt in enumerate(prompts, start=1):
                for run in range(1, RUNS + 1):
                    path = os.path.join(OUTPUT_DIR, f"{time_sig}_{prompt_type}_{i}_{run}.wav")
                    if not os.path.exists(path):
                        tasks.append((prompt, path))
    return tasks


@app.local_entrypoint()
def test():
    """Generate a single song to verify setup."""
    gen = MusicGenerator()
    audio_bytes = gen.generate.remote("a house music with a solid piano")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "test.wav")
    with open(out, "wb") as f:
        f.write(audio_bytes)
    print(f"Saved {out}")


@app.local_entrypoint()
def run_batch():
    """Generate all 250 songs, skipping already-completed files."""
    tasks = build_task_list()
    if not tasks:
        print("All files already generated.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating {len(tasks)} songs...")

    gen = MusicGenerator()
    prompts = [t[0] for t in tasks]
    paths = [t[1] for t in tasks]

    for i, (audio_bytes, path) in enumerate(zip(gen.generate.map(prompts), paths), start=1):
        with open(path, "wb") as f:
            f.write(audio_bytes)
        print(f"[{i}/{len(tasks)}] {os.path.basename(path)}")
