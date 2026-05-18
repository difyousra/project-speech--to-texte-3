import pandas as pd
import os

def split_validation_set(input_path, output_path):
    print("Chargement du dataset...")
    if not os.path.exists(input_path):
        print(f"Erreur : Le fichier {input_path} est introuvable !")
        return

    df = pd.read_csv(input_path)
    
    # On mélange tout le dataset avant de splitter
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calcul des 15% pour la validation
    num_val = int(len(df) * 0.1)
    print(f"Séparation : 10% ({num_val} lignes) pour la validation et 90% pour l'entraînement...")
    
    df_val = df.iloc[:num_val]
    df_train = df.iloc[num_val:] # On garde tout le reste pour le train
    
    # Gestion du dossier FINAL
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # 1. Sauvegarde du fichier de validation
    df_val.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # Je l'écrase pour que fine-tuning utilise le fichier propre
    df_train.to_csv(input_path, index=False, encoding='utf-8-sig')
    
    print(f"Terminé !")
    print(f"-> Validation : {len(df_val)} lignes dans {output_path}")
    print(f"-> Entraînement : {len(df_train)} lignes restantes dans {input_path}")

if __name__ == "__main__":
    input_file = r"laststation/dataset_augmented.csv" 
    validation_file = r"laststation/validation.csv"
    
    split_validation_set(input_file, validation_file)