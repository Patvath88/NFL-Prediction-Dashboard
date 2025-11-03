# --- ESPN + Sleeper NFL Dashboard (Base Scaffold) ---
import streamlit as st, pandas as pd, numpy as np, requests, plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="NFL ESPN-AI Dashboard", page_icon="🏈", layout="wide")

st.markdown("""
<style>
body, .main {background-color:#0B0B0B;color:#fff;}
h1,h2,h3,h4,h5 {color:#D50A0A!important;font-family:'Arial Black'}
.sidebar .sidebar-content {background-color:#161616;}
</style>
""", unsafe_allow_html=True)
st.markdown("## 🏈 NFL ESPN × Sleeper Prediction Dashboard — 2025 Season")
st.caption("Live player stats + opponent-adjusted AI projections")

@st.cache_data(ttl=600)
def get_players():
    """Active rostered players from Sleeper."""
    url = "https://api.sleeper.app/v1/players/nfl"
    r = requests.get(url); players = r.json(); data=[]
    for pid,info in players.items():
        team, status = info.get("team"), info.get("status")
        if team and status=="Active" and team not in ["FA","FA*"]:
            pos=info.get("fantasy_positions")
            pos_str=",".join(pos) if isinstance(pos,list) else pos or "N/A"
            data.append({
                "player_id":pid,"name":info.get("full_name"),
                "team":team,"position":pos_str})
    return pd.DataFrame(data)

@st.cache_data(ttl=600)
def get_espn_season_stats():
    """Season totals from ESPN."""
    url="https://site.api.espn.com/apis/site/v2/sports/football/nfl/statistics"
    r=requests.get(url,timeout=10)
    js=r.json(); cats=[]
    for cat in js.get("categories",[]): cats+=cat.get("stats",[])
    df=pd.json_normalize(cats)
    df=df.rename(columns={"athlete.displayName":"name"})
    keep=[c for c in df.columns if c.endswith("value") or c=="name"]
    return df[keep].dropna(subset=["name"])

@st.cache_data(ttl=600)
def get_espn_defense_ranks():
    """Pull defensive rankings safely from ESPN stats API."""
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/statistics"
    try:
        js = requests.get(url, timeout=10).json()
        cats = []
        # ESPN sometimes nests stats inside 'children'
        if "categories" in js:
            cats = js["categories"]
        elif "children" in js and isinstance(js["children"], list):
            for child in js["children"]:
                if "categories" in child:
                    cats += child["categories"]
        else:
            return pd.DataFrame()

        # Flatten & find any category that contains team defensive ranks
        df = pd.json_normalize(cats, errors="ignore")
        defense_cols = [c for c in df.columns if "defense" in c.lower() or "rank" in c.lower()]
        keep_cols = ["name", "displayName"] + defense_cols if defense_cols else ["name", "displayName"]
        return df[keep_cols].fillna("-")
    except Exception as e:
        st.warning(f"⚠️ Could not load defense ranks from ESPN: {e}")
        return pd.DataFrame()


# --- Merge player + stats ---
players_df=get_players()
try:
    stats_df=get_espn_season_stats()
except Exception: stats_df=pd.DataFrame()
if stats_df.empty:
    st.warning("ESPN stats currently limited; showing roster only.")
df=pd.merge(players_df,stats_df,on="name",how="inner") if not stats_df.empty else players_df

tabs=st.tabs(["🏆 Leaders","👤 Player Insights","📈 Game Trends","🧱 Defense Ranks"])

# --- Leaders ---
with tabs[0]:
    st.subheader("Top Players by Stat (Live ESPN Season Totals)")
    if not stats_df.empty:
        for cat in [c for c in stats_df.columns if c.endswith("value")][:6]:
            sub=stats_df[["name",cat]].sort_values(cat,ascending=False).head(10)
            st.markdown(f"#### {cat.replace('.value','').title()}")
            st.dataframe(sub,use_container_width=True)
    else: st.info("No ESPN data returned.")

# --- Player Insights / Trends placeholders (ML logic next) ---
with tabs[1]: st.info("ML prediction + opponent rank view loads once weekly data integrated.")
with tabs[2]: st.info("Weekly game-by-game chart will appear here.")
with tabs[3]:
    st.subheader("Defense Rankings (ESPN)")
    st.dataframe(get_espn_defense_ranks(),use_container_width=True)

st.caption("Auto-refresh every 10 min | Data: ESPN & Sleeper")
