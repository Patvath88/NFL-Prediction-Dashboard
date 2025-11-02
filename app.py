import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import nfl_data_py as nfl

st.set_page_config(page_title="NFL Player Predictor", layout="wide")

st.title("🏈 NFL Player Predictor — Free Data Edition")
st.caption("Search any NFL team to view player stats and predict next-game performances using AutoML.")

# ============ LOAD DATA ============
@st.cache_data(ttl=3600)
def load_weekly_data(years):
    df = nfl.import_weekly_data(years=years)
    df = df[df["season_type"] == "REG"]
    return df

years = st.sidebar.multiselect("Seasons to include", [2021, 2022, 2023, 2024], default=[2023, 2024])
df = load_weekly_data(years)

if df.empty:
    st.error("No data found.")
    st.stop()

teams = sorted(df["recent_team"].dropna().unique())
selected_team = st.selectbox("Select a team", teams)

team_df = df[df["recent_team"] == selected_team]

if team_df.empty:
    st.warning("No players found for that team.")
    st.stop()

players = sorted(team_df["player_display_name"].dropna().unique())
selected_player = st.selectbox("Select a player", players)

player_df = team_df[team_df["player_display_name"] == selected_player]
st.subheader(f"📊 Historical Stats for {selected_player}")
st.dataframe(player_df.tail(10), use_container_width=True, height=400)

# ============ PREDICTION ENGINE (Smart Fallback) ============

numeric_cols = player_df.select_dtypes(include=[np.number]).columns.tolist()
target_cols = [
    "passing_yards","passing_tds","rushing_yards",
    "rushing_tds","receiving_yards","receptions","fantasy_points"
]
targets = [t for t in target_cols if t in numeric_cols]
if not targets:
    st.warning("No numeric stats found for prediction.")
    st.stop()

selected_target = st.selectbox("Stat to predict for next game", targets)

# Feature engineering
feature_cols = [c for c in numeric_cols if c not in ["week","season","game_key",selected_target]]
data = player_df[feature_cols + [selected_target]].dropna()

# Fallback if not enough player data
if len(data) < 5:
    st.warning(f"Not enough personal data for {selected_player}. Using {team_df['position'].iloc[0]} group model instead.")
    pos = player_df["position"].iloc[0] if "position" in player_df.columns else None
    if pos:
        group_df = df[df["position"] == pos]
        group_data = group_df[feature_cols + [selected_target]].dropna()
        if len(group_data) > 10:
            data = group_data.copy()
        else:
            st.error("Not enough position-level data available to train a model.")
            st.stop()
    else:
        st.error("Position data not available.")
        st.stop()

X = data[feature_cols]
y = data[selected_target]

if len(X) < 5:
    st.error("Still not enough data to train. Try selecting a different stat.")
    st.stop()

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
            best_score = score
            best_model = model
    except Exception:
        continue

results_df = pd.DataFrame(results, columns=["Model", "R²"])
st.write("Model Performance Comparison:")
st.dataframe(results_df, use_container_width=True)

if best_model is None:
    st.error("Model training failed.")
    st.stop()

# Predict next game
latest_row = X.iloc[[-1]]
next_pred = best_model.predict(latest_row)[0]
st.success(f"**Predicted {selected_target.replace('_',' ').title()}: {next_pred:.1f}** for next game")
st.caption(f"Best model: {results_df.loc[results_df['R²'].idxmax(), 'Model']} (R²={best_score:.3f})")
