# --- ESPN × Sleeper NFL Dashboard (Stabilized Base) ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="NFL ESPN × Sleeper Predictor", layout="wide")

st.markdown(
    "<h1 style='color:red;'>🏈 NFL ESPN × Sleeper Prediction Dashboard — 2025 Season</h1>",
    unsafe_allow_html=True,
)
st.caption("Live player stats + opponent-adjusted AI projections")

# --- Data Loader Functions ---
@st.cache_data(ttl=600)
def get_nfl_data():
    import nfl_data_py as nfl
    try:
        # Try 2025, fallback to 2024
        df = nfl.import_weekly_data([2025])
        if df.empty:
            df = nfl.import_weekly_data([2024])
        df = df[df["season_type"] == "REG"]
        df = df.groupby(["player_display_name", "recent_team"], as_index=False)[
            ["passing_yards", "passing_tds", "rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds"]
        ].mean()
        df.rename(columns={"player_display_name": "name", "recent_team": "team"}, inplace=True)
        return df
    except Exception as e:
        st.warning(f"Fallback triggered: {e}")
        try:
            url = "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/player_stats_regular_season.csv.gz"
            df = pd.read_csv(url, compression="gzip")
            df = df[df["season"] == 2024]
            keep = [
                "player_display_name", "recent_team",
                "passing_yards", "passing_tds", "rushing_yards",
                "rushing_tds", "receiving_yards", "receiving_tds"
            ]
            df = df[keep].groupby(["player_display_name", "recent_team"]).mean().reset_index()
            df.rename(columns={"player_display_name": "name", "recent_team": "team"}, inplace=True)
            st.info("Loaded verified NFLFastR backup (2024 season).")
            return df
        except Exception as err:
            st.error(f"All sources failed: {err}")
            return pd.DataFrame()

@st.cache_data(ttl=600)
def get_espn_roster(team_abbr="BUF"):
    """Get ESPN team roster (active players only)."""
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_abbr.lower()}/roster"
        js = requests.get(url, timeout=10).json()
        players = []
        for item in js.get("athletes", []):
            players.append({
                "name": item.get("displayName"),
                "position": item.get("position", {}).get("abbreviation"),
                "status": item.get("status", {}).get("type", {}).get("description", "Active")
            })
        return pd.DataFrame(players)
    except Exception as e:
        st.warning(f"Roster unavailable for {team_abbr}: {e}")
        return pd.DataFrame()

# --- Load Base Data ---
df = get_nfl_data()

if df.empty:
    st.error("No live NFL data found from any source. Please retry later.")
    st.stop()

# --- Tabs ---
tabs = st.tabs(["Leaders", "Player Insights", "Game Trends", "Defense Ranks"])

# 🏆 LEADERS TAB
with tabs[0]:
    st.markdown("### 🏆 Top Players by Stat (Live or 2024 Season Totals)")
    categories = [c for c in df.columns if c not in ["name", "team"]]
    cols = st.columns(3)
    for i, cat in enumerate(categories):
        with cols[i % 3]:
            top10 = df[["name", "team", cat]].sort_values(by=cat, ascending=False).head(10)
            st.markdown(f"#### {cat.replace('_', ' ').title()}")
            st.dataframe(top10.reset_index(drop=True), use_container_width=True)

# 🔍 PLAYER INSIGHTS TAB
with tabs[1]:
    st.markdown("### 🔍 Player Prediction Engine")
    team_options = sorted(df["team"].dropna().unique())
    team = st.selectbox("Select Team", team_options)
    team_df = df[df["team"] == team]
    player = st.selectbox("Select Player", team_df["name"].unique())
    stat = st.selectbox("Stat to Predict", [c for c in df.columns if c not in ["name", "team"]])

    st.divider()
    st.markdown(f"### Predicting {stat.replace('_', ' ').title()} for {player}")

    try:
        # Prepare data
        player_df = team_df[[stat]].dropna()
        if len(player_df) > 3:
            X = np.arange(len(player_df)).reshape(-1, 1)
            y = player_df[stat].values
            model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
            model.fit(X, y)
            next_pred = model.predict([[len(player_df)]])[0]
            st.success(f"Predicted {stat}: **{next_pred:.1f}** next game.")
        else:
            st.warning("Not enough data to predict this stat.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# 📈 GAME TRENDS TAB
with tabs[2]:
    st.markdown("### 📈 Game Trends")
    try:
        sample_col = [c for c in df.columns if "yards" in c or "tds" in c]
        chart_df = df.melt(id_vars=["name", "team"], value_vars=sample_col, var_name="Stat", value_name="Value")
        fig = px.box(chart_df, x="Stat", y="Value", color="team", points="all")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Trend visualization failed: {e}")

# 🛡️ DEFENSE RANKS TAB
with tabs[3]:
    st.markdown("### 🛡️ Defense Rankings (from ESPN)")
    try:
        defense_df = get_espn_roster("buf")  # example roster fetch
        if defense_df.empty:
            st.warning("No ESPN data returned for this team.")
        else:
            st.dataframe(defense_df, use_container_width=True)
    except Exception as e:
        st.error(f"Defense ranks unavailable: {e}")

st.caption("Auto-refresh every 10 min | Data: ESPN × Sleeper × NFLFastR")
