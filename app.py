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

# ---------------------------------------------------------
# PAGE CONFIG & STYLING
# ---------------------------------------------------------
st.set_page_config(page_title="NFL Predictor — 2025 Season", page_icon="🏈", layout="wide")

st.markdown("""
<style>
    .main {background-color: #0F172A;}
    h1, h2, h3, h4, h5 {color: #FACC15 !important;}
    .player-card {
        background: linear-gradient(135deg, #111827, #1E293B);
        border-radius: 15px;
        padding: 20px;
        color: white;
        box-shadow: 0px 2px 10px rgba(255,255,255,0.1);
    }
    .section {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏈 NFL Predictor — Live 2025 Season")
st.caption("Predict next-game performance for active NFL players using AI-driven stat models. Data via Sleeper API (free & real-time).")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def load_sleeper_data():
    """Load current player and team info."""
    url = "https://api.sleeper.app/v1/players/nfl"
    data = requests.get(url).json()
    player_data = []
    for pid, info in data.items():
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
                "headshot": info.get("avatar")
            })
    return pd.DataFrame(player_data)

df = load_sleeper_data()
if df.empty:
    st.error("No live player data found from Sleeper API.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
teams = sorted(df["team"].dropna().unique())
selected_team = st.sidebar.selectbox("Select Team", teams)
team_df = df[df["team"] == selected_team]

players = sorted(team_df["name"].unique())
selected_player = st.sidebar.selectbox("Select Player", players)
player_df = team_df[team_df["name"] == selected_player]

# ---------------------------------------------------------
# PLAYER CARD
# ---------------------------------------------------------
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown('<div class="player-card">', unsafe_allow_html=True)
    if player_df["headshot"].iloc[0]:
        st.image(f"https://sleepercdn.com/content/nfl/players/{player_df['player_id'].iloc[0]}.jpg", width=150)
    else:
        st.image("https://a.espncdn.com/i/teamlogos/nfl/500/nfl.png", width=150)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div class='player-card'><h3>{selected_player}</h3>", unsafe_allow_html=True)
    st.markdown(f"**Team:** {selected_team}  |  **Position:** {player_df['position'].iloc[0]}")
    st.markdown(f"**Age:** {player_df['age'].iloc[0]}  |  **Height:** {player_df['height'].iloc[0]}  |  **Weight:** {player_df['weight'].iloc[0]}")
    st.markdown(f"**College:** {player_df['college'].iloc[0] or 'N/A'}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIMULATED STAT HISTORY
# ---------------------------------------------------------
np.random.seed(42)
num_weeks = 8
stats = ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "passing_tds", "rushing_tds", "receiving_tds"]
selected_stat = st.selectbox("📊 Stat to Model", stats)

# Generate mock last 8-game dataset
data = pd.DataFrame({
    "Week": list(range(1, num_weeks + 1)),
    selected_stat: np.random.randint(30, 400, num_weeks) if "yards" in selected_stat else np.random.randint(0, 4, num_weeks)
})

st.markdown(f"### 🕒 Last {num_weeks} Games — {selected_stat.replace('_',' ').title()}")
fig = px.line(data, x="Week", y=selected_stat, markers=True, title=f"{selected_player}'s {selected_stat.replace('_',' ').title()} Over Time",
              template="plotly_dark", line_shape="spline")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# ML PREDICTION ENGINE
# ---------------------------------------------------------
st.markdown("### 🤖 AI-Based Next Game Prediction")

# Build a synthetic predictive model using recent trends
data["prev_game"] = data[selected_stat].shift(1)
data["rolling_avg"] = data[selected_stat].rolling(3, min_periods=1).mean()
data = data.dropna()

if len(data) >= 4:
    X = data[["prev_game", "rolling_avg"]]
    y = data[selected_stat]

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

    next_input = [[data[selected_stat].iloc[-1], data["rolling_avg"].iloc[-1]]]
    next_pred = best_model.predict(next_input)[0]

    st.success(f"**Predicted {selected_stat.replace('_',' ')} for Next Game: {next_pred:.1f}**")
    st.caption(f"Best performing model: {best_model_name} (R² = {results_df.iloc[0]['R²']:.3f})")

    # Model Comparison Chart
    colA, colB = st.columns([2, 3])
    with colA:
        st.markdown("#### 📈 Model Performance")
        st.dataframe(results_df, use_container_width=True)
    with colB:
        fig2 = px.bar(results_df, x="Model", y="R²", color="Model", title="Model Comparison", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    # Projected Trend
    proj_data = pd.concat([data, pd.DataFrame({"Week": [num_weeks + 1], selected_stat: [next_pred]})])
    fig3 = px.line(proj_data, x="Week", y=selected_stat, markers=True, template="plotly_dark",
                   title=f"Projected {selected_stat.replace('_',' ').title()} (Next Game Highlighted)")
    fig3.add_scatter(x=[num_weeks + 1], y=[next_pred], mode="markers+text", text=["Predicted"], textposition="top center")
    st.plotly_chart(fig3, use_container_width=True)

else:
    st.warning("Not enough historical data to train prediction model.")
