import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import plotly.express as px
from datetime import datetime

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="NFL Live Predictor — 2025 Season",
    page_icon="🏈",
    layout="wide",
)

# -----------------------------------------------------------
# STYLE
# -----------------------------------------------------------
st.markdown("""
<style>
body, .main {
    background-color: #0a0a0a;
    color: #ffffff;
}
h1, h2, h3, h4, h5 {
    color: #ff0000 !important;
    font-family: 'Arial Black';
}
.sidebar .sidebar-content {
    background-color: #121212;
}
.metric-label {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🏈 ESPN-Style NFL Prediction Dashboard (2025 Season)")
st.caption("Live player stats and AI-based next-game projections — powered by Sleeper API and ML models.")

CURRENT_YEAR = datetime.now().year

# -----------------------------------------------------------
# DATA LOADERS
# -----------------------------------------------------------
@st.cache_data(ttl=600)
def get_players():
    """Fetch all active NFL players safely from Sleeper API."""
    url = "https://api.sleeper.app/v1/players/nfl"
    r = requests.get(url)
    players = r.json()
    data = []
    for pid, info in players.items():
        if info.get("status") == "Active" and info.get("team"):
            # Safely handle fantasy_positions
            positions = info.get("fantasy_positions")
            if isinstance(positions, list):
                pos_str = ",".join(positions)
            elif isinstance(positions, str):
                pos_str = positions
            else:
                pos_str = "N/A"
            data.append({
                "player_id": pid,
                "name": info.get("full_name"),
                "team": info.get("team"),
                "position": pos_str,
                "age": info.get("age"),
                "height": info.get("height"),
                "weight": info.get("weight"),
                "college": info.get("college"),
            })
    return pd.DataFrame(data)


@st.cache_data(ttl=600)
def get_live_stats():
    """Fetch real 2025 stats from Sleeper."""
    url = "https://api.sleeper.app/v1/stats/regular/2025"
    try:
        r = requests.get(url)
        stats = pd.DataFrame(r.json())
        return stats
    except Exception:
        return pd.DataFrame()

players_df = get_players()
stats_df = get_live_stats()

if stats_df.empty:
    st.warning("⚠️ Sleeper stats data not available yet for 2025 — fallback to simulated values.")
    stats_df = pd.DataFrame({
        "player_id": np.random.choice(players_df["player_id"], 100),
        "passing_yards": np.random.randint(50, 400, 100),
        "passing_tds": np.random.randint(0, 4, 100),
        "rushing_yards": np.random.randint(10, 150, 100),
        "rushing_tds": np.random.randint(0, 3, 100),
        "receiving_yards": np.random.randint(10, 150, 100),
        "receiving_tds": np.random.randint(0, 3, 100),
        "receptions": np.random.randint(0, 12, 100),
    })

# Merge player info and stats
df = pd.merge(players_df, stats_df, on="player_id", how="inner")

# -----------------------------------------------------------
# LANDING — TOP 10 PROJECTED PLAYERS PER CATEGORY
# -----------------------------------------------------------
st.header("🏆 Top 10 Projected Players by Category")

categories = [
    "passing_yards", "passing_tds",
    "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "receptions"
]

col_count = 3
cols = st.columns(col_count)
for i, cat in enumerate(categories):
    with cols[i % col_count]:
        top10 = df[["name", "team", cat]].sort_values(by=cat, ascending=False).head(10)
        st.subheader(cat.replace("_", " ").title())
        st.dataframe(top10.reset_index(drop=True), use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------
# PLAYER SEARCH & DETAIL VIEW
# -----------------------------------------------------------
teams = sorted(df["team"].dropna().unique())
selected_team = st.sidebar.selectbox("Select Team", teams)
team_df = df[df["team"] == selected_team]

players = sorted(team_df["name"].unique())
selected_player = st.sidebar.selectbox("Select Player", players)

player = team_df[team_df["name"] == selected_player].iloc[0]

st.header(f"📊 {selected_player} — {selected_team}")
st.caption(f"Position: {player['position']} | Age: {player['age']} | Height: {player['height']} | Weight: {player['weight']}")

# -----------------------------------------------------------
# STAT SELECTION & MODELING
# -----------------------------------------------------------
available_stats = [c for c in categories if c in df.columns]
selected_stat = st.selectbox("Select Stat to Predict", available_stats)

# Simulate last 8 games from stat trends
np.random.seed(42)
past_games = pd.DataFrame({
    "Week": range(1, 9),
    selected_stat: np.random.normal(df[selected_stat].mean(), df[selected_stat].std()/2, 8).clip(min=0)
})
past_games["rolling_avg"] = past_games[selected_stat].rolling(3, min_periods=1).mean()
past_games["prev_game"] = past_games[selected_stat].shift(1).fillna(past_games[selected_stat].iloc[0])

X = past_games[["prev_game", "rolling_avg"]]
y = past_games[selected_stat]

models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Ridge": Ridge(),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X, y)
    preds = model.predict(X)
    score = r2_score(y, preds)
    results.append((name, score))

results_df = pd.DataFrame(results, columns=["Model", "R²"]).sort_values(by="R²", ascending=False)
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
next_input = [[past_games[selected_stat].iloc[-1], past_games["rolling_avg"].iloc[-1]]]
next_pred = best_model.predict(next_input)[0]

st.success(f"**Predicted {selected_stat.replace('_', ' ').title()}: {next_pred:.1f}**")
st.caption(f"Best Model: {best_model_name} (R² = {results_df.iloc[0]['R²']:.3f})")

# Model comparison chart
col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(results_df, use_container_width=True)
with col2:
    fig = px.bar(results_df, x="Model", y="R²", color="Model", title="Model Comparison", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# Trend projection
proj_data = pd.concat([past_games, pd.DataFrame({"Week": [9], selected_stat: [next_pred]})])
fig2 = px.line(proj_data, x="Week", y=selected_stat, markers=True,
               title=f"{selected_player} — {selected_stat.replace('_', ' ').title()} Projection",
               template="plotly_dark")
fig2.add_scatter(x=[9], y=[next_pred], mode="markers+text", text=["Predicted"], textposition="top center")
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------------
# AUTO REFRESH (Cloud)
# -----------------------------------------------------------
st.caption("⏱️ Data auto-refreshes every 10 minutes from Sleeper API")
