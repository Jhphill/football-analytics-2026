import pandas as pd

df = pd.read_csv('data/processed/matches_features_v2.csv', low_memory=False)
print("Shape:", df.shape)
print("\nNulos en fav_form_scored:", df['fav_form_scored'].isna().sum())
print("\nDistribución target_1x2_fav_dog:")
print(df['target_1x2_fav_dog'].value_counts())

# Verificar específicamente los 481 partidos que antes estaban pendientes
pending = df[df['pending_feature_engineering'] == 1]
print(f"\nPartidos antes 'pending_feature_engineering=1': {len(pending)}")
print("De esos, con fav_form_scored NO nulo:", pending['fav_form_scored'].notna().sum())