import argparse
import csv
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch
from transformers import pipeline


DEFAULT_INPUT_CSV = "dataset/data.csv"
DEFAULT_OUTPUT_CSV = "dataset/processed/data_aligned_segments.csv"
DEFAULT_OUTPUT_AUDIO_DIR = "dataset/audio/aligned_segments"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("align_and_segment")


@dataclass
class WordToken:
    raw: str
    norm: str
    start: Optional[float] = None
    end: Optional[float] = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aligner un texte de reference avec des timestamps mots Whisper, "
            "puis decouper l'audio et le texte en segments."
        )
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="CSV source avec colonnes audio,text")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="CSV de sortie segmente")
    parser.add_argument(
        "--output-audio-dir",
        default=DEFAULT_OUTPUT_AUDIO_DIR,
        help="Dossier de sortie des segments audio",
    )
    parser.add_argument(
        "--model",
        default="whisper-finetuned",
        help="Modele ASR (chemin local ou modele HF)",
    )
    parser.add_argument("--language", default="fr", help="Langue pour la transcription")
    parser.add_argument("--max-duration", type=float, default=10.0, help="Duree max d'un segment (s)")
    parser.add_argument("--min-duration", type=float, default=1.5, help="Duree min cible d'un segment (s)")
    parser.add_argument(
        "--max-gap",
        type=float,
        default=0.8,
        help="Si ecart entre 2 mots > max-gap, on coupe le segment",
    )
    parser.add_argument("--padding", type=float, default=0.08, help="Padding ajoute aux bornes audio (s)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Frequence des segments exportes")
    parser.add_argument("--max-rows", type=int, default=0, help="Traiter seulement les N premieres lignes (0=tout)")
    parser.add_argument(
        "--force-silence-align",
        action="store_true",
        help="N'utilise pas l'ASR timestamps, segmente l'audio via silences et aligne le texte par segments",
    )
    parser.add_argument(
        "--append-data-csv",
        default="dataset/data.csv",
        help="CSV principal a enrichir automatiquement avec les segments (colonnes audio,text)",
    )
    parser.add_argument(
        "--skip-append",
        action="store_true",
        help="Ne pas ajouter automatiquement les segments dans --append-data-csv",
    )
    return parser.parse_args()


def normalize_word(token: str) -> str:
    token = token.lower().strip()
    token = re.sub(r"[\"“”'`´’]", "", token)
    token = re.sub(r"[^a-z0-9A-ZÀ-ÿ]+", "", token)
    return token


def text_to_tokens(text: str) -> List[WordToken]:
    words = text.strip().split()
    tokens = []
    for w in words:
        norm = normalize_word(w)
        if norm:
            tokens.append(WordToken(raw=w, norm=norm))
    return tokens


def split_chunk_text(chunk_text: str) -> List[str]:
    return [w for w in chunk_text.strip().split() if normalize_word(w)]


def flatten_asr_chunks(chunks: List[dict]) -> List[WordToken]:
    words: List[WordToken] = []
    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        ts = chunk.get("timestamp", (None, None))
        start, end = ts
        split_words = split_chunk_text(chunk_text)
        if not split_words:
            continue

        if start is None or end is None or end <= start:
            for w in split_words:
                words.append(WordToken(raw=w, norm=normalize_word(w), start=None, end=None))
            continue

        step = (end - start) / len(split_words)
        for i, w in enumerate(split_words):
            w_start = start + i * step
            w_end = start + (i + 1) * step
            words.append(WordToken(raw=w, norm=normalize_word(w), start=w_start, end=w_end))
    return words


def align_reference_tokens(ref_tokens: List[WordToken], hyp_tokens: List[WordToken]) -> None:
    ref_norm = [t.norm for t in ref_tokens]
    hyp_norm = [t.norm for t in hyp_tokens]
    matcher = SequenceMatcher(a=ref_norm, b=hyp_norm, autojunk=False)

    aligned: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    for block in matcher.get_matching_blocks():
        if block.size == 0:
            continue
        for offset in range(block.size):
            r_idx = block.a + offset
            h_idx = block.b + offset
            aligned[r_idx] = (hyp_tokens[h_idx].start, hyp_tokens[h_idx].end)

    for i, token in enumerate(ref_tokens):
        if i in aligned:
            token.start, token.end = aligned[i]

    fill_missing_timestamps(ref_tokens)


def fill_missing_timestamps(tokens: List[WordToken]) -> None:
    known_indices = [i for i, t in enumerate(tokens) if t.start is not None and t.end is not None]
    if not known_indices:
        return

    for i, token in enumerate(tokens):
        if token.start is not None and token.end is not None:
            continue

        left_idx = max((k for k in known_indices if k < i), default=None)
        right_idx = min((k for k in known_indices if k > i), default=None)

        if left_idx is not None and right_idx is not None:
            left = tokens[left_idx]
            right = tokens[right_idx]
            gap = max(0.0, (right.start or 0.0) - (left.end or 0.0))
            span = right_idx - left_idx
            step = gap / max(1, span)
            rel = i - left_idx
            est_start = (left.end or 0.0) + (rel - 1) * step
            est_end = est_start + step
            token.start = max(0.0, est_start)
            token.end = max(token.start + 0.02, est_end)
        elif left_idx is not None:
            left = tokens[left_idx]
            base = left.end or left.start or 0.0
            token.start = base
            token.end = base + 0.08
        elif right_idx is not None:
            right = tokens[right_idx]
            base = max(0.0, (right.start or 0.0) - 0.08)
            token.start = base
            token.end = (right.start or 0.0)


def safe_audio_path(audio_value: str) -> str:
    audio_path = audio_value.strip()
    if not audio_path.startswith("dataset/"):
        audio_path = os.path.join("dataset", audio_path)
    return audio_path


def cut_audio_ffmpeg(source_audio: str, dest_audio: str, start: float, end: float, sample_rate: int) -> bool:
    os.makedirs(os.path.dirname(dest_audio), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        source_audio,
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        dest_audio,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def get_audio_duration(source_audio: str) -> Optional[float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        source_audio,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def detect_silence_boundaries(source_audio: str) -> List[float]:
    cmd = [
        "ffmpeg",
        "-i",
        source_audio,
        "-af",
        "silencedetect=noise=-35dB:d=0.35",
        "-f",
        "null",
        "NUL",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr or ""

    boundaries = []
    for line in stderr.splitlines():
        match = re.search(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)", line)
        if match:
            boundaries.append(float(match.group(1)))
    return boundaries


def split_by_max_duration(intervals: List[Tuple[float, float]], max_duration: float) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for start, end in intervals:
        if end <= start:
            continue
        curr = start
        while curr < end:
            nxt = min(end, curr + max_duration)
            out.append((curr, nxt))
            curr = nxt
    return out


def merge_short_intervals(intervals: List[Tuple[float, float]], min_duration: float) -> List[Tuple[float, float]]:
    if not intervals:
        return []

    merged: List[Tuple[float, float]] = []
    i = 0
    while i < len(intervals):
        start, end = intervals[i]
        dur = end - start

        if dur >= min_duration or i == len(intervals) - 1:
            merged.append((start, end))
            i += 1
            continue

        nstart, nend = intervals[i + 1]
        merged.append((start, nend))
        i += 2

    # Passe finale: corriger chevauchements potentiels
    cleaned: List[Tuple[float, float]] = []
    for start, end in merged:
        if not cleaned:
            cleaned.append((start, end))
            continue
        pstart, pend = cleaned[-1]
        if start < pend:
            cleaned[-1] = (pstart, max(pend, end))
        else:
            cleaned.append((start, end))
    return cleaned


def split_text_for_n_segments(text: str, n: int) -> List[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if n <= 1:
        return [clean] if clean else []

    sentences = [s.strip() for s in re.split(r"(?<=[\.!\?])\s+", clean) if s.strip()]
    if not sentences:
        return [clean] if clean else []

    if len(sentences) == n:
        return sentences

    if len(sentences) > n:
        groups = [""] * n
        for i, sent in enumerate(sentences):
            idx = min(n - 1, int(i * n / len(sentences)))
            groups[idx] = (groups[idx] + " " + sent).strip()
        return [g for g in groups if g]

    # Pas assez de phrases: completer par repartition des mots.
    words = clean.split()
    if not words:
        return []
    target = max(1, len(words) // n)
    chunks = []
    i = 0
    for seg_i in range(n):
        if seg_i == n - 1:
            chunk = " ".join(words[i:]).strip()
        else:
            chunk = " ".join(words[i : i + target]).strip()
        i += target
        if chunk:
            chunks.append(chunk)
    return chunks


def build_silence_aligned_segments(source_audio: str, text: str, args) -> List[Tuple[float, float, str]]:
    duration = get_audio_duration(source_audio)
    if duration is None or duration <= 0:
        return []

    silence_bounds = detect_silence_boundaries(source_audio)
    points = [0.0] + [b for b in silence_bounds if 0.0 < b < duration] + [duration]
    points = sorted(set(points))

    intervals: List[Tuple[float, float]] = []
    for i in range(len(points) - 1):
        s = points[i]
        e = points[i + 1]
        if e - s >= 0.2:
            intervals.append((s, e))

    if not intervals:
        intervals = [(0.0, duration)]

    intervals = split_by_max_duration(intervals, args.max_duration)
    intervals = merge_short_intervals(intervals, args.min_duration)
    if not intervals:
        intervals = [(0.0, duration)]

    text_chunks = split_text_for_n_segments(text, len(intervals))
    if not text_chunks:
        return []

    # Harmoniser tailles en cas d'ecart.
    n = min(len(intervals), len(text_chunks))
    intervals = intervals[:n]
    text_chunks = text_chunks[:n]

    out = []
    for (start, end), chunk in zip(intervals, text_chunks):
        if chunk.strip():
            out.append((start, end, chunk.strip()))
    return out


def is_segment_break(current_word: str, duration: float, next_gap: Optional[float], args) -> bool:
    if duration >= args.max_duration:
        return True
    if duration >= args.min_duration and re.search(r"[\.!\?;:,]$", current_word):
        return True
    if next_gap is not None and next_gap > args.max_gap and duration >= args.min_duration:
        return True
    return False


def build_segments(tokens: List[WordToken], args) -> List[Tuple[float, float, str]]:
    valid = [t for t in tokens if t.start is not None and t.end is not None and t.end > t.start]
    if not valid:
        return []

    segments: List[Tuple[float, float, str]] = []
    start_idx = 0

    while start_idx < len(valid):
        seg_words = [valid[start_idx].raw]
        seg_start = valid[start_idx].start or 0.0
        seg_end = valid[start_idx].end or seg_start

        j = start_idx
        while j + 1 < len(valid):
            next_token = valid[j + 1]
            curr_duration = (next_token.end or seg_end) - seg_start
            next_gap = (next_token.start or seg_end) - (valid[j].end or seg_end)

            seg_words.append(next_token.raw)
            seg_end = next_token.end or seg_end

            if is_segment_break(next_token.raw, curr_duration, next_gap, args):
                j += 1
                break
            j += 1

        text = " ".join(seg_words).strip()
        if text:
            segments.append((seg_start, seg_end, text))
        start_idx = max(j, start_idx + 1)

    return segments


def relative_audio_for_csv(audio_path: str) -> str:
    norm = audio_path.replace("\\", "/")
    if norm.startswith("dataset/"):
        return norm[len("dataset/") :]
    return norm


def transcribe_with_timestamps(asr_pipe, audio_path: str, language: str) -> List[WordToken]:
    speech_array, sampling_rate = sf.read(audio_path)
    if len(speech_array.shape) > 1:
        speech_array = speech_array.mean(axis=1)

    result = asr_pipe(
        {"array": speech_array, "sampling_rate": sampling_rate},
        return_timestamps="word",
        generate_kwargs={"language": language, "task": "transcribe"},
    )
    chunks = result.get("chunks", [])
    return flatten_asr_chunks(chunks)


def append_segments_to_data_csv(data_csv_path: str, rows: List[Dict[str, str]]) -> int:
    if not rows:
        return 0

    os.makedirs(os.path.dirname(data_csv_path), exist_ok=True)
    data_exists = os.path.exists(data_csv_path)

    existing_audio = set()
    if data_exists:
        with open(data_csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_value = (row.get("audio") or row.get("\ufeffaudio") or "").strip()
                if audio_value:
                    existing_audio.add(audio_value)

    to_append = []
    for row in rows:
        audio_value = row["audio"].strip()
        if audio_value in existing_audio:
            continue
        to_append.append({"audio": audio_value, "text": row["text"].strip()})
        existing_audio.add(audio_value)

    if not to_append:
        return 0

    with open(data_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text"])
        if not data_exists or os.path.getsize(data_csv_path) == 0:
            writer.writeheader()
        writer.writerows(to_append)

    return len(to_append)


def main():
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg introuvable dans PATH")
    if shutil.which("ffprobe") is None:
        raise EnvironmentError("ffprobe introuvable dans PATH")

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"CSV introuvable: {args.input_csv}")

    output_csv_dir = os.path.dirname(args.output_csv)
    if output_csv_dir:
        os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(args.output_audio_dir, exist_ok=True)

    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model=args.model,
        chunk_length_s=30,
        stride_length_s=(5, 2),
        device=device,
        torch_dtype=torch_dtype,
    )

    with open(args.input_csv, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    output_rows = []
    global_segment_count = 0
    fallback_count = 0

    for idx, row in enumerate(rows, start=1):
        audio_value = row.get("audio") or row.get("\ufeffaudio")
        text_value = row.get("text")
        if not audio_value or not text_value:
            continue

        source_audio = safe_audio_path(audio_value)
        if not os.path.exists(source_audio):
            LOGGER.warning("[%s] Audio introuvable: %s", idx, source_audio)
            continue

        segments = []
        if args.force_silence_align:
            segments = build_silence_aligned_segments(source_audio, text_value, args)
            if segments:
                fallback_count += 1
        else:
            try:
                hyp_tokens = transcribe_with_timestamps(asr_pipe, source_audio, args.language)
                ref_tokens = text_to_tokens(text_value)
                if ref_tokens and hyp_tokens:
                    align_reference_tokens(ref_tokens, hyp_tokens)
                    segments = build_segments(ref_tokens, args)
            except Exception as exc:
                LOGGER.warning("[%s] Echec transcription timestamps: %s | %s", idx, source_audio, exc)

            if not segments:
                segments = build_silence_aligned_segments(source_audio, text_value, args)
                if segments:
                    fallback_count += 1

        if not segments:
            LOGGER.warning("[%s] Aucun segment genere: %s", idx, source_audio)
            continue

        base = os.path.splitext(os.path.basename(source_audio))[0]
        local_count = 0

        for seg_id, (start, end, seg_text) in enumerate(segments, start=1):
            start = max(0.0, start - args.padding)
            end = max(start + 0.05, end + args.padding)

            out_name = f"{base}_seg{seg_id:04d}.wav"
            out_audio = os.path.join(args.output_audio_dir, out_name)

            ok = cut_audio_ffmpeg(source_audio, out_audio, start, end, args.sample_rate)
            if not ok:
                LOGGER.error("[%s] Echec decoupage ffmpeg: %s", idx, out_audio)
                continue

            output_rows.append(
                {
                    "audio": relative_audio_for_csv(out_audio),
                    "text": seg_text,
                    "source_audio": source_audio.replace("\\", "/"),
                    "start": f"{start:.3f}",
                    "end": f"{end:.3f}",
                }
            )
            local_count += 1
            global_segment_count += 1

        LOGGER.info("[%s/%s] %s -> %s segments", idx, len(rows), source_audio, local_count)

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text", "source_audio", "start", "end"])
        writer.writeheader()
        writer.writerows(output_rows)

    appended_count = 0
    if not args.skip_append:
        appended_count = append_segments_to_data_csv(args.append_data_csv, output_rows)

    LOGGER.info("=== Termine ===")
    LOGGER.info("Segments crees: %s", global_segment_count)
    LOGGER.info("Fichiers traites en mode silence-align: %s", fallback_count)
    LOGGER.info("CSV sortie: %s", args.output_csv)
    LOGGER.info("Audio sortie: %s", args.output_audio_dir)
    if not args.skip_append:
        LOGGER.info("Ajoutes dans %s: %s", args.append_data_csv, appended_count)


if __name__ == "__main__":
    main()
