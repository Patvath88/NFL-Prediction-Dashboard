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
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="NFL Live Predictor — 2025 Season",
    page_icon="🏈",
    layout="wide",
)

# Custom CSS for a sleek dark layout
st.markdown("""
    <style>
    .main {background-color: #0F172A;}
    h1, h2, h3, h4, h5 {color: #FACC15 !important;}
    .stMetric {background-color: #1E293B !important; border-radius: 10px; padding: 10px;}
    .stDataFrame {background-color: #1E293B !important;}
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🏈 NFL Live Predictor — 2025 Season")
st.caption("Real-time stats, live predictions, and team insights — powered by ESPN API and AI modeling.")

CURRENT_YEAR = datetime.now().year

# ---------------------------------------------------------
# FETCH TEAMS
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_team_rosters():
    """Fetch team info from ESPN API."""
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
    data = requests.get(url).json()
    teams = []
    for t in data["sports"][0]["leagues"][0]["teams"]:
        info = t["team"]
        teams.append({
            "team_id": info["id"],
            "team": info["displayName"],
            "abbrev": info["abbreviation"],
            "logo": info["logos"][0]["href"] if info.get("logos") else None
        })
    return pd.DataFrame(teams)

teams_df = fetch_team_rosters()
selected_team = st.sidebar.selectbox("Select Team", teams_df["team"].tolist())
team_id = teams_df.loc[teams_df["team"] == selected_team, "team_id"].iloc[0]
team_abbrev = teams_df.loc[teams_df["team"] == selected_team, "abbrev"].iloc[0]
team_logo = teams_df.loc[teams_df["team"] == selected_team, "logo"].iloc[0]

# ---------------------------------------------------------
# FETCH LIVE TEAM STATS
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_team_stats(team_id, selected_team):
    """Pull live team stats from ESPN."""
    url = f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/teams/{team_id}/statistics"
    try:
        data = requests.get(url).json()
    except Exception:
        return pd.DataFrame()

    players = []
    for cat in data.get("athletes", []):
        p = cat.get("athlete", {})
        stats = cat.get("stats", [])
        if not p or not stats:
            continue
        player_info = {
            "name": p.get("displayName"),
            "position": p.get("position", {}).get("abbreviation"),
            "team": selected_team,
            "headshot": p.get("headshot", {}).get("href") if p.get("headshot") else None,
        }
        for s in stats:
            if isinstance(s, dict) and "name" in s and "value" in s:
                player_info[s["name"]] = s["value"]
        players.append(player_info)
    return pd.DataFrame(players)

df = fetch_team_stats(team_id, selected_team)

if df.empty:
    st.warning("No live stats found for this team yet.")
    st.stop()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.warning("No numeric data available yet for this team.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR SELECTIONS
# ---------------------------------------------------------
st.sidebar.image(team_logo, width=120)
st.sidebar.markdown(f"### {selected_team}")

players = sorted(df["name"].dropna().unique())
selected_player = st.sidebar.selectbox("Select Player", players)
player_df = df[df["name"] == selected_player]

# ---------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------
col_logo, col_info = st.columns([1, 4])
with col_logo:
    player_img = player_df["headshot"].iloc[0] if "headshot" in player_df.columns and pd.notna(player_df["headshot"].iloc[0]) else None
    if player_img:
        st.image(player_img, width=150)
    else:
        st.image(team_logo, width=150)

with col_info:
    pos = player_df["position"].iloc[0] if "position" in player_df.columns else "—"
    st.subheader(f"{selected_player} ({pos}) — {selected_team}")
    st.caption(f"📅 Live {CURRENT_YEAR} Season | Data Source: ESPN API")

# ---------------------------------------------------------
# DISPLAY PLAYER STATS TABLE
# ---------------------------------------------------------
st.markdown("### 📊 Current Player Stats")
st.dataframe(player_df[numeric_cols].T.rename(columns={player_df.index[0]: "Value"}), use_container_width=True)

# ---------------------------------------------------------
# MODEL TRAINING & PREDICTION
# ---------------------------------------------------------
selected_target = st.selectbox("Select Stat to Predict", numeric_cols, index=0)
data = df[["name"] + numeric_cols].dropna()

if len(data) >= 5:
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

    results_df = pd.DataFrame(results, columns=["Model", "R²"]).sort_values(by="R²", ascending=False)

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("### 🧠 Best Model")
        best_row = results_df.iloc[0]
        st.metric(label="Top Model", value=best_row["Model"])
        st.metric(label="R² Score", value=f"{best_row['R²']:.3f}")

    with col2:
        st.markdown("### ⚙️ Model Comparison")
        st.dataframe(results_df, use_container_width=True)

    # Predict next game
    latest = X.iloc[[-1]]
    pred = best_model.predict(latest)[0]
    avg = y.mean()
    diff = ((pred - avg) / avg * 100) if avg else 0

    st.markdown("---")
    st.markdown("### 🔮 Prediction Summary")
    st.success(f"**{selected_player} is projected for {pred:.1f} {selected_target.replace('_', ' ')} next game**")
    st.caption(f"That's {diff:+.1f}% compared to their season average of {avg:.1f}.")

    # -----------------------------------------------------
    # VISUALIZATIONS
    # -----------------------------------------------------
    st.markdown("### 📈 Stat Distribution (Team Level)")
    fig = px.box(df, y=selected_target, x="position", color="position",
                 title=f"Team Distribution of {selected_target.replace('_',' ').title()}")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Not enough player data to train the model yet.")
