import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import plotly.express as px

st.set_page_config(page_title="NFL Live Predictor — 2025 Season", page_icon="🏈", layout="wide")

st.title("🏈 NFL Live Predictor — Sleeper API Edition")
st.caption("Live player data and next-game performance predictions — powered by Sleeper's public API and AI modeling.")

# ==========================================================
# LOAD DATA FROM SLEEPER
# ==========================================================
@st.cache_data(ttl=1800)
def load_sleeper_data():
    """Load current player and stat data from Sleeper."""
    try:
        players_url = "https://api.sleeper.app/v1/players/nfl"
        players = requests.get(players_url).json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

    player_data = []
    for pid, info in players.items():
        if info.get("status") == "Active" and info.get("team") and info.get("fantasy_positions"):
            player_data.append({
                "player_id": pid,
                "name": info.get("full_name"),
                "team": info.get("team"),
                "position": ",".join(info.get("fantasy_positions")),
                "age": info.get("age"),
                "height": info.get("height"),
                "weight": info.get("weight"),
                "college": info.get("college"),
            })

    return pd.DataFrame(player_data)

df = load_sleeper_data()

if df.empty:
    st.error("No NFL player data available from Sleeper API.")
    st.stop()

# ==========================================================
# FILTER BY TEAM AND PLAYER
# ==========================================================
teams = sorted(df["team"].dropna().unique())
selected_team = st.sidebar.selectbox("Select a Team", teams)
team_df = df[df["team"] == selected_team]

players = sorted(team_df["name"].unique())
selected_player = st.sidebar.selectbox("Select a Player", players)
player_df = team_df[team_df["name"] == selected_player]

# ==========================================================
# DISPLAY PLAYER PROFILE
# ==========================================================
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown(f"### 🧍 {selected_player}")
    st.caption(f"{player_df['position'].iloc[0]} — {selected_team}")
    st.metric("Age", player_df["age"].iloc[0])
    st.metric("Height", player_df["height"].iloc[0])
    st.metric("Weight", player_df["weight"].iloc[0])

with col2:
    st.markdown(f"**College:** {player_df['college'].iloc[0]}")
    st.dataframe(player_df, use_container_width=True)

# ==========================================================
# SIMULATED PERFORMANCE METRICS (since Sleeper API lacks per-game stats)
# ==========================================================
np.random.seed(42)
player_df["recent_yards"] = np.random.randint(30, 150, 5)
player_df["recent_tds"] = np.random.randint(0, 3, 5)
player_df["fantasy_points"] = player_df["recent_yards"] * 0.1 + player_df["recent_tds"] * 6

st.markdown("### 📊 Simulated Recent Game Performance")
sim_data = pd.DataFrame({
    "Game Week": [f"Week {i+1}" for i in range(5)],
    "Yards": player_df["recent_yards"],
    "TDs": player_df["recent_tds"],
    "Fantasy Points": player_df["fantasy_points"]
})
fig = px.bar(sim_data, x="Game Week", y="Fantasy Points", color="Yards", title=f"{selected_player} Recent Game Simulation")
st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# SIMPLE MODEL PREDICTION (simulate next game output)
# ==========================================================
st.markdown("### 🔮 Predict Next Game Output")

X = sim_data[["Yards", "TDs"]].rename(columns={"Yards": "yards", "TDs": "tds"})
y = sim_data["Fantasy Points"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)
pred = model.predict([[np.mean(X["yards"]), np.mean(X["tds"])]])[0]

st.success(f"Predicted Fantasy Points Next Game: **{pred:.2f}**")

# ==========================================================
# TEAM SUMMARY VISUAL
# ==========================================================
st.markdown("### 🧠 Team-Level View")
fig2 = px.histogram(df[df["team"] == selected_team], x="position", title=f"{selected_team} Active Roster Composition")
st.plotly_chart(fig2, use_container_width=True)
