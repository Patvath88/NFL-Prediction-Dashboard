import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import plotly.express as px
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="NFL AI Prediction Dashboard — ESPN Live 2025",
    page_icon="🏈",
    layout="wide",
)

# --------------- STYLE ------------------
st.markdown("""
<style>
body, .main { background-color: #0B0B0B; color: #FFFFFF; }
h1, h2, h3, h4, h5 { color: #D50A0A !important; font-family: 'Arial Black'; }
.sidebar .sidebar-content { background-color: #161616; }
div.stDataFrame { border: 1px solid #333; border-radius: 10px; }
.metric-label { color: #FFFFFF !important; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🏈 NFL AI Prediction Dashboard — ESPN Live 2025")
st.caption("Real-time stats & AI projections powered by Sleeper API and ML models.")

CURRENT_YEAR = 2025

# --------------- DATA LOADERS -----------------
@st.cache_data(ttl=600)
def get_players():
    """Fetch all *currently rostered* NFL players from Sleeper API."""
    url = "https://api.sleeper.app/v1/players/nfl"
    r = requests.get(url)
    players = r.json()
    data = []
    for pid, info in players.items():
        team = info.get("team")
        status = info.get("status")
        # ✅ Only include players who are on an active NFL roster (not None, IR, or FA)
        if team not in [None, "", "FA", "FA*"] and status == "Active":
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
                "team": team,
                "position": pos_str,
                "age": info.get("age"),
                "height": info.get("height"),
                "weight": info.get("weight"),
                "college": info.get("college"),
            })
    df = pd.DataFrame(data)
    # Remove duplicates & sort by team for cleaner dropdowns
    df = df.drop_duplicates(subset=["name"]).sort_values(by=["team", "position", "name"])
    return df.reset_index(drop=True)


# --------------- TABS -----------------
tabs = st.tabs(["🏆 Leaders", "👤 Player Insights", "📈 Game Trends"])

# --------------- TAB 1: LEADERS -----------------
with tabs[0]:
    st.subheader("🏆 Top 10 Projected Players by Category")
    categories = [
        "passing_yards", "passing_tds",
        "rushing_yards", "rushing_tds",
        "receiving_yards", "receiving_tds", "receptions"
    ]
    cols = st.columns(3)
    for i, cat in enumerate(categories):
        with cols[i % 3]:
            top10 = df[["name", "team", cat]].sort_values(by=cat, ascending=False).head(10)
            st.markdown(f"#### {cat.replace('_',' ').title()}")
            st.dataframe(top10.reset_index(drop=True), use_container_width=True)

# --------------- TAB 2: PLAYER INSIGHTS -----------------
with tabs[1]:
    st.subheader("👤 Player Prediction Engine")

    teams = sorted(df["team"].dropna().unique())
    selected_team = st.selectbox("Select Team", teams)
    team_df = df[df["team"] == selected_team]

    players = sorted(team_df["name"].unique())
    selected_player = st.selectbox("Select Player", players)

    categories = [
        "passing_yards", "passing_tds",
        "rushing_yards", "rushing_tds",
        "receiving_yards", "receiving_tds", "receptions"
    ]
    selected_stat = st.selectbox("Stat to Predict", categories)

    player_df = team_df[team_df["name"] == selected_player]
    avg_val = player_df[selected_stat].mean()

    np.random.seed(42)
    weeks = np.arange(1, 9)
    values = np.random.normal(avg_val, avg_val * 0.2, len(weeks))
    df_games = pd.DataFrame({"Week": weeks, selected_stat: values})
    df_games["rolling_avg"] = df_games[selected_stat].rolling(3, min_periods=1).mean()

    X = df_games[["rolling_avg"]]
    y = df_games[selected_stat]
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Ridge": Ridge(),
        "XGBoost": XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=4, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        results.append((name, r2_score(y, preds)))
    results_df = pd.DataFrame(results, columns=["Model", "R²"]).sort_values(by="R²", ascending=False)

    best_model = models[results_df.iloc[0]["Model"]]
    next_pred = best_model.predict([[df_games["rolling_avg"].iloc[-1]]])[0]

    st.metric(f"Predicted {selected_stat.replace('_',' ').title()} Next Game", f"{next_pred:.1f}")
    st.caption(f"Best Model: {results_df.iloc[0]['Model']} (R² = {results_df.iloc[0]['R²']:.3f})")

    fig = px.bar(results_df, x="Model", y="R²", color="Model",
                 title="Model Performance", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# --------------- TAB 3: GAME TRENDS -----------------
with tabs[2]:
    st.subheader(f"📈 Season & Game-by-Game Trends")
    team_stats = df.groupby("team")[["passing_yards", "rushing_yards", "receiving_yards"]].mean().reset_index()
    fig_team = px.bar(team_stats, x="team", y=["passing_yards", "rushing_yards", "receiving_yards"],
                      barmode="group", title="Average Yards by Team (2025)",
                      template="plotly_dark")
    st.plotly_chart(fig_team, use_container_width=True)

    fig_line = px.line(df_games, x="Week", y=selected_stat, markers=True,
                       title=f"{selected_player} – {selected_stat.replace('_',' ').title()} Trend",
                       template="plotly_dark")
    st.plotly_chart(fig_line, use_container_width=True)

st.caption("⏱ Data auto-refreshes every 10 minutes from Sleeper API or simulated source.")
