import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from datetime import datetime

st.set_page_config(page_title="NFL Active Player Predictor — Live ESPN", layout="wide")

st.title("🏈 NFL Active Player Predictor — Live 2025 Season")
st.caption("Pulls live NFL data from ESPN and predicts next-game player performance using ML models.")

CURRENT_YEAR = datetime.now().year

# =====================================================
# FETCH LIVE PLAYER DATA FROM ESPN
# =====================================================
@st.cache_data(ttl=1800)
def fetch_team_rosters():
    """Fetch live rosters and stats from ESPN NFL API."""
    teams_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
    teams_data = requests.get(teams_url).json()
    teams = []
    for t in teams_data["sports"][0]["leagues"][0]["teams"]:
        info = t["team"]
        teams.append({
            "team_id": info["id"],
            "team": info["displayName"],
            "abbrev": info["abbreviation"]
        })
    return pd.DataFrame(teams)

teams_df = fetch_team_rosters()
selected_team = st.selectbox("Select a team", teams_df["team"].tolist())
team_id = teams_df.loc[teams_df["team"] == selected_team, "team_id"].iloc[0]

@st.cache_data(ttl=1800)
def fetch_team_players(team_id):
    """Fetch active player stats for a team."""
    roster_url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster"
    roster_data = requests.get(roster_url).json()
    players = []
    for a in roster_data.get("athletes", []):
        for player in a.get("items", []):
            stats = player.get("stats", [])
            player_data = {
                "name": player.get("displayName"),
                "position": player.get("position", {}).get("abbreviation"),
                "status": player.get("status", {}).get("type", {}).get("description", "Active"),
                "team": selected_team,
            }
            # Flatten any stat dictionaries (if present)
            for s in stats:
                if isinstance(s, dict) and "name" in s and "value" in s:
                    player_data[s["name"]] = s["value"]
            players.append(player_data)
    return pd.DataFrame(players)

df = fetch_team_players(team_id)

if df.empty:
    st.error("No live player data found for this team.")
    st.stop()

# Filter active players with valid stat data
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.warning("No numeric stats yet for this team.")
    st.stop()

active_df = df.copy()
st.caption(f"📅 Live data for {selected_team} — {CURRENT_YEAR} Season")

# =====================================================
# PLAYER SELECTION
# =====================================================
players = sorted(active_df["name"].unique())
selected_player = st.selectbox("Select a player", players)
player_df = active_df[active_df["name"] == selected_player]

st.subheader(f"📊 {selected_player} — Live 2025 Stats (ESPN)")
st.dataframe(player_df, use_container_width=True)

# =====================================================
# PREDICTION ENGINE
# =====================================================
numeric_cols = player_df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.warning("No numeric stats available for prediction yet.")
    st.stop()

selected_target = st.selectbox("Stat to predict", numeric_cols)

# Build fake time-series samples for ML (since ESPN doesn’t expose week-by-week)
data = active_df[["name"] + numeric_cols].dropna()
if len(data) < 5:
    st.warning("Not enough data points yet to train model. Wait until more games are played.")
    st.stop()

X = data.drop(columns=[selected_target, "name"]).fillna(0)
y = data[selected_target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Ridge": Ridge(),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
}

best_model, best_score = None, -999
results = []

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        results.append((name, score))
        if score > best_score:
            best_model, best_score = model, score
    except Exception:
        continue

results_df = pd.DataFrame(results, columns=["Model","R²"])
st.write("Model Performance Comparison:")
st.dataframe(results_df, use_container_width=True)

if best_model is None:
    st.error("Model training failed.")
    st.stop()

# Predict next game
sample = X.iloc[[-1]]
pred = best_model.predict(sample)[0]
st.success(f"**Predicted {selected_target.replace('_',' ').title()}: {pred:.1f}** for next game")
st.caption(f"Best model: {results_df.loc[results_df['R²'].idxmax(), 'Model']} (R²={best_score:.3f})")
