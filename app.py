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
st.caption("Pulls live NFL player data directly from ESPN and predicts next-game performance using machine learning.")

CURRENT_YEAR = datetime.now().year

# =====================================================
# FETCH TEAM LIST
# =====================================================
@st.cache_data(ttl=1800)
def fetch_team_rosters():
    """Fetch team info from ESPN NFL API."""
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
    resp = requests.get(url)
    data = resp.json()
    teams = []
    for t in data["sports"][0]["leagues"][0]["teams"]:
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

# =====================================================
# FETCH LIVE PLAYER DATA (DEFENSIVE)
# =====================================================
@st.cache_data(ttl=1800)
def fetch_team_players(team_id, selected_team):
    """Fetch active players from ESPN API and normalize safely."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster"
    try:
        r = requests.get(url)
        data = r.json()
    except Exception:
        return pd.DataFrame()

    players = []
    for group in data.get("athletes", []):
        for p in group.get("items", []):
            stats = p.get("stats", [])
            status_field = p.get("status", {})

            # Handle different formats of the "status" field
            if isinstance(status_field, str):
                status_desc = status_field
            elif isinstance(status_field, dict):
                status_desc = status_field.get("type", {}).get("description", "Unknown")
            elif isinstance(status_field, list):
                status_desc = (
                    status_field[0].get("type", {}).get("description", "Unknown")
                    if status_field and isinstance(status_field[0], dict)
                    else "Unknown"
                )
            else:
                status_desc = "Unknown"

            player_info = {
                "name": p.get("displayName"),
                "position": p.get("position", {}).get("abbreviation", ""),
                "status": status_desc,
                "team": selected_team
            }

            # Flatten any numeric stats if available
            if isinstance(stats, list):
                for s in stats:
                    if isinstance(s, dict) and "name" in s and "value" in s:
                        player_info[s["name"]] = s["value"]

            players.append(player_info)

    df = pd.DataFrame(players)
    return df

df = fetch_team_players(team_id, selected_team)

if df.empty:
    st.error("No live player data found for this team (ESPN returned empty). Try another team.")
    st.stop()

# =====================================================
# FILTER ACTIVE PLAYERS
# =====================================================
if "status" in df.columns:
    df = df[df["status"].astype(str).str.lower().str.contains("active", na=False)]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.warning("No numeric stats yet for this team.")
    st.stop()

st.caption(f"📅 Live {CURRENT_YEAR} data for {selected_team} (via ESPN API)")

# =====================================================
# PLAYER SELECTION
# =====================================================
players = sorted(df["name"].dropna().unique())
selected_player = st.selectbox("Select a player", players)
player_df = df[df["name"] == selected_player]

st.subheader(f"📊 {selected_player} — Live {CURRENT_YEAR} Stats")
st.dataframe(player_df, use_container_width=True)

# =====================================================
# PREDICTION ENGINE
# =====================================================
numeric_cols = player_df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.warning("No numeric stats found for this player.")
    st.stop()

selected_target = st.selectbox("Stat to predict", numeric_cols)

# Train using team-level data
data = df[["name"] + numeric_cols].dropna()
if len(data) < 5:
    st.warning("Not enough data to train a model yet.")
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

sample = X.iloc[[-1]]
pred = best_model.predict(sample)[0]
st.success(f"**Predicted {selected_target.replace('_',' ').title()}: {pred:.1f}** for next game")
st.caption(f"Best model: {results_df.loc[results_df['R²'].idxmax(), 'Model']} (R²={best_score:.3f})")
