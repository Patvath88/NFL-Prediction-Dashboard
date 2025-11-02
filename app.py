import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import nfl_data_py as nfl

st.set_page_config(page_title="NFL Active Player Predictor", layout="wide")

st.title("🏈 NFL Active Player Predictor — 2025 Season")
st.caption("View stats and predict next-game performance for active players who have played this season.")

# ============ LOAD DATA ============
@st.cache_data(ttl=1800)
def load_current_season_data(year):
    df = nfl.import_weekly_data([year])
    df = df[df["season_type"] == "REG"]
    df = df[df["fantasy_points"].notna()]  # filter to players who have played
    df = df[df["player_display_name"].notna()]
    return df

CURRENT_YEAR = 2025
df = load_current_season_data(CURRENT_YEAR)

if df.empty:
    st.error("No player data available for the current season yet.")
    st.stop()

# Get active players (played at least one game)
active_players = df.groupby("player_display_name")["week"].nunique()
active_players = active_players[active_players > 0].index.tolist()
df = df[df["player_display_name"].isin(active_players)]

teams = sorted(df["recent_team"].dropna().unique())
selected_team = st.selectbox("Select a team", teams)

team_df = df[df["recent_team"] == selected_team].copy()
players = sorted(team_df["player_display_name"].unique())
selected_player = st.selectbox("Select a player", players)

player_df = team_df[team_df["player_display_name"] == selected_player].copy()

st.subheader(f"📊 {selected_player} — 2025 Season Stats")
st.dataframe(player_df[[
    "week","opponent_team","passing_yards","passing_tds",
    "rushing_yards","rushing_tds","receiving_yards","receptions","fantasy_points"
]].fillna(0), use_container_width=True, height=400)

# ============ PREDICTION ENGINE ============
numeric_cols = player_df.select_dtypes(include=[np.number]).columns.tolist()
target_cols = [
    "passing_yards","passing_tds","rushing_yards",
    "rushing_tds","receiving_yards","receptions","fantasy_points"
]
targets = [t for t in target_cols if t in numeric_cols]
if not targets:
    st.warning("No numeric stats found for this player.")
    st.stop()

selected_target = st.selectbox("Stat to predict for next game", targets)

feature_cols = [c for c in numeric_cols if c not in ["week","season","game_key",selected_target]]
data = player_df[feature_cols + [selected_target]].dropna()

# Ensure enough data
if len(data) < 3:
    st.warning("Not enough weekly data for this player this season to train a model.")
    st.stop()

X = data[feature_cols]
y = data[selected_target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Ridge": Ridge(),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
}

best_model = None
best_score = -999
results = []

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        results.append((name, score))
        if score > best_score:
            best_model = model
            best_score = score
    except Exception:
        continue

results_df = pd.DataFrame(results, columns=["Model","R²"])
st.write("Model Performance Comparison:")
st.dataframe(results_df, use_container_width=True)

if best_model is None:
    st.error("Model training failed.")
    st.stop()

latest_row = X.iloc[[-1]]
next_pred = best_model.predict(latest_row)[0]
st.success(f"**Predicted {selected_target.replace('_',' ').title()}: {next_pred:.1f}** for next game")
st.caption(f"Best model: {results_df.loc[results_df['R²'].idxmax(), 'Model']} (R²={best_score:.3f})")

# Download player stats
st.download_button("📥 Download Player's 2025 Stats", player_df.to_csv(index=False).encode(), f"{selected_player}_2025.csv")
