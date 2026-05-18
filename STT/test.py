import torch
from transformers import pipeline, AutoProcessor
import librosa
import os
import csv

model_path = "output_darja/checkpoint-2576"
processor_path = "output_darja"
output_file = "transcription_darja.txt"

liste_audios= []
with open("Segments/segmentation_log.csv", "r", encoding="utf-8-sig") as f:

    reader = csv.DictReader(f)

    for row in reader:
        liste_audios.append(row["audio_path"])

print("Chargement du modèle...")
processor = AutoProcessor.from_pretrained(processor_path)

# Pipeline avec CHUNK_LENGTH pour éviter les coupures à 9s
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_path,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=10,   
    stride_length_s=2,    # Évite de couper les mots entre deux blocs
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

def transcrire_audio(file_path):
    if not os.path.exists(file_path):
        return f"Erreur : Le fichier {file_path} est introuvable."
    
    # 1. Charger et forcer 16kHz (Règle le problème de vitesse/durée)
    # librosa gère le m4a si ffmpeg est installé
    audio_array, _ = librosa.load(file_path, sr=16000)
    
    # 2. Transcription avec paramètres de génération
    result = pipe(
        audio_array, 
        generate_kwargs={
            "language": "ar", 
            "task": "transcribe",
            "max_new_tokens" : 256,
            "num_beams" : 1,
            "do_sample" : False
        }
    )
    return result["text"]

print(liste_audios)

# --- EXECUTION ---
print(f"Début du traitement de {len(liste_audios)} fichiers...")

with open(output_file, "a", encoding="utf-8-sig") as f:
    f.write("--- RESULTATS DES TESTS WHISPER ARABE ---\n\n")

for audio in liste_audios:
    print(f"Transcription de : {audio} ...")
    try:
        texte = transcrire_audio(audio)
        
        with open(output_file, "a", encoding="utf-8-sig") as f:
            f.write(f"NOM DU FICHIER : {os.path.basename(audio)}\n")
            f.write(f"TRANSCRIPTION : {texte}\n")
            f.write("-" * 30 + "\n")
        print("Ok !")
    except Exception as e:
        print(f"Erreur sur {audio}: {e}")

print(f"\nTerminé ! Vérifie le fichier : {output_file}")