import argparse
import os
import queue
import sys
from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf

# Permet d'utiliser le moteur VAD partagé : python test_vad_chunks.py --engine silero|rms
_LIBS = os.path.join(os.path.dirname(__file__), "libs")
if _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)
_VAD_SERVICE = os.path.join(os.path.dirname(__file__), "services", "vad")
if _VAD_SERVICE not in sys.path:
    sys.path.insert(0, _VAD_SERVICE)


SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 30

MIN_AUDIO_MS = 1500
MAX_AUDIO_MS = 9000

SILENCE_LIMIT_MS = 300
ENERGY_THRESHOLD = 0.006

PRE_ROLL_MS = 100
POST_ROLL_MS = 200

MIN_SPEECH_RATIO = 0.20
MIN_AVG_ENERGY = 0.0015

OUTPUT_DIR = "call_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

block_size = int(SAMPLE_RATE * CHUNK_MS / 1000)

pre_roll_blocks = PRE_ROLL_MS // CHUNK_MS
post_roll_blocks = POST_ROLL_MS // CHUNK_MS

audio_queue = queue.Queue()
pre_buffer = deque(maxlen=pre_roll_blocks)


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    audio_queue.put(indata.copy())


def rms_energy(audio_block):
    return float(np.sqrt(np.mean(audio_block ** 2)))


def compute_metrics(blocks, speech_flags):
    audio = np.concatenate(blocks, axis=0)
    duration_sec = len(audio) / SAMPLE_RATE
    duration_ms = int(duration_sec * 1000)

    avg_energy = rms_energy(audio)
    speech_ratio = sum(speech_flags) / max(len(speech_flags), 1)
    peak = float(np.max(np.abs(audio)))

    return duration_ms, duration_sec, avg_energy, speech_ratio, peak


def save_chunk(blocks, speech_flags, chunk_id):
    duration_ms, duration_sec, avg_energy, speech_ratio, peak = compute_metrics(
        blocks, speech_flags
    )

    if duration_ms < MIN_AUDIO_MS:
        print(f"Chunk ignoré: durée trop courte {duration_ms} ms")
        return False

    if avg_energy < MIN_AVG_ENERGY:
        print(f"Chunk ignoré: énergie faible {avg_energy:.5f}")
        return False

    if speech_ratio < MIN_SPEECH_RATIO:
        print(f"Chunk ignoré: speech_ratio faible {speech_ratio:.2f}")
        return False

    audio = np.concatenate(blocks, axis=0)

    path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_id:03d}.wav")
    sf.write(path, audio, SAMPLE_RATE)

    print(
        f"Chunk sauvegardé: {path} | "
        f"durée={duration_ms} ms ({duration_sec:.2f}s) | "
        f"energy={avg_energy:.5f} | "
        f"speech_ratio={speech_ratio:.2f} | "
        f"peak={peak:.3f}"
    )

    return True


parser = argparse.ArgumentParser(description="Capture micro + VAD")
parser.add_argument(
    "--engine",
    choices=["rms", "silero"],
    default=os.getenv("VAD_ENGINE", "rms"),
    help="Moteur VAD (rms=local, silero=via services/vad)",
)
args, _ = parser.parse_known_args()
if args.engine == "silero":
    os.environ["VAD_ENGINE"] = "silero"

print("Micro ouvert pendant tout l'appel.")
print(f"Moteur VAD: {args.engine}")
print("Segments voix en millisecondes.")
print("CTRL+C = fin appel.\n")

chunk_id = 1
recording = False
current_blocks = []
speech_flags = []

silence_duration_ms = 0
segment_duration_ms = 0
post_roll_count = 0

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=block_size,
        callback=audio_callback,
    ):
        while True:
            block = audio_queue.get()
            energy = rms_energy(block)
            is_speech = energy > ENERGY_THRESHOLD

            if not recording:
                pre_buffer.append(block)

            if is_speech and not recording:
                recording = True
                current_blocks = list(pre_buffer)
                speech_flags = [False] * len(current_blocks)

                silence_duration_ms = 0
                segment_duration_ms = len(current_blocks) * CHUNK_MS
                post_roll_count = 0

                print("Voix détectée → début segment")

            if recording:
                current_blocks.append(block)
                speech_flags.append(is_speech)
                segment_duration_ms += CHUNK_MS

                if is_speech:
                    silence_duration_ms = 0
                    post_roll_count = 0
                else:
                    silence_duration_ms += CHUNK_MS
                    post_roll_count += 1

                cut_by_silence = (
                    segment_duration_ms >= MIN_AUDIO_MS
                    and silence_duration_ms >= SILENCE_LIMIT_MS
                    and post_roll_count >= post_roll_blocks
                )

                cut_by_max = segment_duration_ms >= MAX_AUDIO_MS

                if cut_by_silence or cut_by_max:
                    ok = save_chunk(current_blocks, speech_flags, chunk_id)

                    if ok:
                        chunk_id += 1

                    recording = False
                    current_blocks = []
                    speech_flags = []
                    silence_duration_ms = 0
                    segment_duration_ms = 0
                    post_roll_count = 0

except KeyboardInterrupt:
    print("\nFin appel demandée.")

    if recording and current_blocks:
        save_chunk(current_blocks, speech_flags, chunk_id)

    print("Fin test VAD.")