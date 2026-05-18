import argparse
import logging
import os
import shutil
import subprocess

SOURCE_DIR = "."

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

LOGGER = logging.getLogger("convert_audio")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convertit les fichiers .m4a en .wav 16kHz mono"
    )

    parser.add_argument(
        "--source-dir",
        default=SOURCE_DIR,
        help="Dossier contenant les fichiers .m4a"
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Frequence d'echantillonnage cible"
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Parcourir les sous-dossiers"
    )

    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Conserver les fichiers .m4a originaux"
    )

    return parser.parse_args()


def iter_m4a_files(source_dir: str, recursive: bool):

    if recursive:
        for root, _, files in os.walk(source_dir):
            for filename in files:
                if filename.lower().endswith(".m4a"):
                    yield os.path.join(root, filename)

    else:
        for filename in os.listdir(source_dir):
            if filename.lower().endswith(".m4a"):
                yield os.path.join(source_dir, filename)


def convert_and_cleanup_m4a(
    source_dir: str,
    sample_rate: int,
    recursive: bool,
    keep_original: bool
):

    if not os.path.exists(source_dir):
        raise FileNotFoundError(
            f"Dossier introuvable: {source_dir}"
        )

    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg introuvable dans le PATH"
        )

    converted = 0
    failed = 0
    skipped = 0

    for source_path in iter_m4a_files(source_dir, recursive):

        base = os.path.splitext(source_path)[0]
        target_path = f"{base}.wav"

        if os.path.exists(target_path):
            skipped += 1
            LOGGER.info("Deja converti: %s", target_path)
            continue

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            source_path,
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            target_path,
        ]

        LOGGER.info(
            "Conversion: %s -> %s",
            source_path,
            target_path
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            failed += 1

            LOGGER.error(
                "Erreur conversion: %s",
                source_path
            )

            LOGGER.error(result.stderr)
            continue

        converted += 1

        if not keep_original:
            os.remove(source_path)
            LOGGER.info("Supprime: %s", source_path)

    LOGGER.info("===== TERMINE =====")
    LOGGER.info("Convertis: %d", converted)
    LOGGER.info("Ignores: %d", skipped)
    LOGGER.info("Echecs: %d", failed)


if __name__ == "__main__":

    args = parse_args()

    convert_and_cleanup_m4a(
        source_dir=args.source_dir,
        sample_rate=args.sample_rate,
        recursive=args.recursive,
        keep_original=args.keep_original,
    )