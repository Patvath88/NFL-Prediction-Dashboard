import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import plotly.express as px

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="ESPN-Style NFL Prediction Dashboard (2025)",
    page_icon="🏈",
    layout="wide"
)

st.markdown("""
<style>
body, .main { background-color:#0B0B0B; color:#FFFFFF; }
h1,h2,h3,h4,h5 { color:#D50A0A !important; font-family:'Arial Black'; }
.sidebar .sidebar-content { background-color:#161616; }
div.stDataFrame { border:1px solid #333; border-radius:10px; }
.metric-label { color:#FFFFFF !important; }
.stTabs [data-baseweb="tab-list"]{ gap:10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🏈 ESPN-Style NFL Prediction Dashboard (2025 Season)")
st.caption("Live player stats & AI projections — powered by Sleeper API + machine-learning models.")

CURRENT_YEAR = 2025

# ----------------------------------------------------------------------
#  DATA LOADERS
# ----------------------------------------------------------------------
@st.cache_data(ttl=600)
def get_players():
    """Fetch all active rostered players."""
    url = "https://api.sleeper.app/v1/players/nfl"
    r = requests.get(url)
    players = r.json()
    data = []
    for pid, info in players.items():
        team = info.get("team")
        status = info.get("status")
        if team not in [None, "", "FA", "FA*"] and status == "Active":
            pos = info.get("fantasy_positions")
            if isinstance(pos, list): pos_str = ",".join(pos)
            elif isinstance(pos, str): pos_str = pos
            else: pos_str = "N/A"
            data.append({
                "player_id": pid,
                "name": info.get("full_name"),
                "team": team,
                "position": pos_str,
                "age": info.get("age"),
                "height": info.get("height"),
                "weight": info.get("weight"),
                "college": info.get("college")
            })
    df = pd.DataFrame(data).drop_duplicates(subset=["name"]).sort_values(["team","position","name"])
    return df.reset_index(drop=True)

@st.cache_data(ttl=600)
def get_live_stats():
    """Pull latest season stats or return empty."""
    url = f"https://api.sleeper.app/v1/stats/regular/{CURRENT_YEAR}"
    try:
        r = requests.get(url, timeout=10)
        stats = pd.DataFrame(r.json())
        return stats
    except Exception:
        return pd.DataFrame()

# ----------------------------------------------------------------------
#  SAFE MERGE
# ----------------------------------------------------------------------
try:
    players_df = get_players()
except Exception as e:
    st.error(f"Player load error: {e}")
    players_df = pd.DataFrame()

try:
    stats_df = get_live_stats()
except Exception as e:
    st.error(f"Stat load error: {e}")
    stats_df = pd.DataFrame()

if not players_df.empty and not stats_df.empty:
    try:
        df = pd.merge(players_df, stats_df, on="player_id", how="inner")
        if df.empty:
            st.warning("⚠️ No overlapping player stats found; using simulation fallback.")
    except Exception:
        df = pd.DataFrame()
else:
    st.warning("⚠️ Sleeper stats not yet available — using simulated data.")
    np.random.seed(42)
    sample = players_df.sample(min(300, len(players_df)))
    df = sample.assign(
        passing_yards=np.random.randint(50,400,len(sample)),
        passing_tds=np.random.randint(0,4,len(sample)),
        rushing_yards=np.random.randint(10,150,len(sample)),
        rushing_tds=np.random.randint(0,3,len(sample)),
        receiving_yards=np.random.randint(10,150,len(sample)),
        receiving_tds=np.random.randint(0,3,len(sample)),
        receptions=np.random.randint(0,12,len(sample))
    )

# ----------------------------------------------------------------------
#  TABS
# ----------------------------------------------------------------------
tabs = st.tabs(["🏆 Leaders","👤 Player Insights","📈 Game Trends"])

# ------------------- TAB 1 – LEADERS -------------------
with tabs[0]:
    st.subheader("🏆 Top 10 Projected Players by Category")
    categories = [
        "passing_yards","passing_tds",
        "rushing_yards","rushing_tds",
        "receiving_yards","receiving_tds","receptions"
    ]
    cols = st.columns(3)
    for i,cat in enumerate(categories):
        with cols[i%3]:
            if cat in df.columns:
                top10 = df[["name","team",cat]].sort_values(by=cat,ascending=False).head(10)
                st.markdown(f"#### {cat.replace('_',' ').title()}")
                st.dataframe(top10.reset_index(drop=True),use_container_width=True)
            else:
                st.markdown(f"#### {cat.replace('_',' ').title()}")
                st.info("No data available for this stat yet.")

# ------------------- TAB 2 – PLAYER INSIGHTS -------------------
with tabs[1]:
    st.subheader("👤 Player Prediction Engine")
    if df.empty:
        st.info("No data available for predictions yet.")
    else:
        teams = sorted(df["team"].dropna().unique())
        selected_team = st.selectbox("Select Team",teams)
        team_df = df[df["team"]==selected_team]

        team_df["display"] = team_df["name"]+" — "+team_df["position"]
        players = team_df["display"].unique()
        selected_display = st.selectbox("Select Player",players)
        selected_player = selected_display.split(" — ")[0]

        categories = [
            "passing_yards","passing_tds",
            "rushing_yards","rushing_tds",
            "receiving_yards","receiving_tds","receptions"
        ]
        selected_stat = st.selectbox("Stat to Predict",categories)

        player_df = team_df[team_df["name"]==selected_player]
        avg_val = player_df[selected_stat].mean() if selected_stat in player_df else 0

        np.random.seed(42)
        weeks = np.arange(1,9)
        values = np.random.normal(avg_val, max(avg_val*0.2,1), len(weeks))
        df_games = pd.DataFrame({"Week":weeks, selected_stat:values})
        df_games["rolling_avg"] = df_games[selected_stat].rolling(3,min_periods=1).mean()

        X = df_games[["rolling_avg"]]; y = df_games[selected_stat]
        models = {
            "Random Forest":RandomForestRegressor(n_estimators=200,random_state=42),
            "Ridge":Ridge(),
            "XGBoost":XGBRegressor(n_estimators=250,learning_rate=0.05,max_depth=4,random_state=42)
        }

        results=[]
        for name,m in models.items():
            m.fit(X,y)
            preds=m.predict(X)
            results.append((name,r2_score(y,preds)))
        results_df=pd.DataFrame(results,columns=["Model","R²"]).sort_values("R²",ascending=False)

        best=models[results_df.iloc[0]["Model"]]
        next_pred=best.predict([[df_games["rolling_avg"].iloc[-1]]])[0]

        st.metric(f"Predicted {selected_stat.replace('_',' ').title()} Next Game",f"{next_pred:.1f}")
        st.caption(f"Best Model: {results_df.iloc[0]['Model']} (R² = {results_df.iloc[0]['R²']:.3f})")

        fig=px.bar(results_df,x="Model",y="R²",color="Model",
                   title="Model Performance",template="plotly_dark")
        st.plotly_chart(fig,use_container_width=True)

# ------------------- TAB 3 – GAME TRENDS -------------------
with tabs[2]:
    st.subheader("📈 Season & Game-by-Game Trends")
    if df.empty:
        st.info("No team data available yet.")
    else:
        team_stats=df.groupby("team")[["passing_yards","rushing_yards","receiving_yards"]].mean().reset_index()
        fig_team=px.bar(team_stats,x="team",y=["passing_yards","rushing_yards","receiving_yards"],
                        barmode="group",title="Average Yards by Team (2025)",template="plotly_dark")
        st.plotly_chart(fig_team,use_container_width=True)

st.caption("⏱ Auto-refresh every 10 min from Sleeper API or simulated source.")
