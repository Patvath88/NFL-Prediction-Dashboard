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
st.caption("Pulls live NFL data from ESPN and predicts next-game player performance using machine learning models.")

CURRENT_YEAR = datetime.now().year

# =====================================================
# TEAM LIST
# =====================================================
@st.cache_data(ttl=1800)
def fetch_team_rosters():
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
    data = requests.get(url).json()
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
# FETCH LIVE PLAYER DATA SAFELY
# =====================================================
@st.cache_data(ttl=1800)
def fetch_team_players(team_id, selected_team):
    """Fetch players from ESPN safely regardless of how 'status' is formatted."""
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
            raw_status = p.get("status", None)

            # ✨ Universal safe conversion — covers str, dict, list, or None
            status_desc = "Unknown"
            if isinstance(raw_status, str):
                status_desc = raw_status
            elif isinstance(raw_status, dict):
                # Try multiple keys since ESPN isn't consistent
                if "type" in raw_status and isinstance(raw_status["type"], dict):
                    status_desc = raw_status["type"].get("description", str(raw_status["type"]))
                elif "type" in raw_status:
                    status_desc = str(raw_status["type"])
                elif "description" in raw_status:
                    status_desc = str(raw_status["description"])
                else:
                    status_desc = "Unknown"
            elif isinstance(raw_status, list):
                first = raw_status[0] if raw_status else None
                if isinstance(first, dict):
                    status_desc = first.get("description") or first.get("type") or "Unknown"

            player_info = {
                "name": p.get("displayName"),
                "position": p.get("position", {}).get("abbreviation", ""),
                "status": status_desc,
                "team": selected_team
            }

            # Flatten numeric stats if available
            if isinstance(stats, list):
                for s in stats:
                    if isinstance(s, dict):
                        name = s.get("name")
                        val = s.get("value")
                        if name and isinstance(val, (int, float)):
                            player_info[name] = val

            players.append(player_info)

    return pd.DataFrame(players)

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

data = df[["name"] + numeric_cols].dropna()
if len(data) < 5:
    st.warning("Not enough team data to train a model yet.")
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

results_df = pd.DataFrame(results, columns=["Model", "R²"])
st.write("Model Performance Comparison:")
st.dataframe(results_df, use_container_width=True)

if best_model is None:
    st.error("Model training failed.")
    st.stop()

sample = X.iloc[[-1]]
pred = best_model.predict(sample)[0]
st.success(f"**Predicted {selected_target.replace('_',' ').title()}: {pred:.1f}** for next game")
st.caption(f"Best model: {results_df.loc[results_df['R²'].idxmax(), 'Model']} (R²={best_score:.3f})")
