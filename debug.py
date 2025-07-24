import pandas as pd
import os

def debug_dataset():
    print("[DEBUG] Analyse du dataset...")
    
    # Vérification fichier
    if not os.path.exists("data/output.csv"):
        print("[ERROR] Fichier data/output.csv non trouvé")
        return
    
    # Charger et analyser
    try:
        df = pd.read_csv("data/output.csv")
        print(f"[OK] Dataset chargé: {df.shape}")
        
        # Les Colonnes
        print(f"[INFO] Colonnes: {list(df.columns)}")
        
        # Vérifier colonnes requises
        required = ['date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 
                   'sqft_lot', 'floors', 'waterfront', 'view', 'condition']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            print(f"[ERROR] Colonnes manquantes: {missing}")
        else:
            print("[OK] Toutes les colonnes requises présentes")
        
        # Stats de base
        print(f"[INFO] Prix: min={df['price'].min():,.0f}, max={df['price'].max():,.0f}, moyenne={df['price'].mean():,.0f}")
        
        # Valeurs manquantes
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"[WARNING] Valeurs manquantes:")
            for col, count in missing_values[missing_values > 0].items():
                print(f"  {col}: {count}")
        else:
            print("[OK] Pas de valeurs manquantes")
            
    except Exception as e:
        print(f"[ERROR] Erreur lecture dataset: {e}")

if __name__ == "__main__":
    debug_dataset()