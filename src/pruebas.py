import pandas as pd
df = pd.read_csv('data/processed/matches_features_v2.csv', low_memory=False, nrows=3)
print(list(df.columns))
print(df['_date'].max())
print(df[['fav_team','dog_team','fav_elo','dog_elo','fav_form_scored','fav_form_win_rate']].head(3))