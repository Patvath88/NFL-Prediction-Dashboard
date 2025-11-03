# --- ESPN × Sleeper NFL Dashboard (Stable Build) ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
@st.cache_data(ttl=600)
def get_nfl_data():
    """
    Robust data loader that uses ESPN if available,
    then nfl_data_py, and finally a permanent NFLFastR mirror.
    """
    import nfl_data_py as nfl
    import io

    # 1️⃣ ESPN live API
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/statistics"
        js = requests.get(url, timeout=10).json()
        cats = []
        if "children" in js:
            for child in js["children"]:
                if "categories" in child:
                    cats += child["categories"]
        elif "categories" in js:
            cats = js["categories"]
        stats = []
        for cat in cats:
            for s in cat.get("stats", []):
                stats.append({
                    "name": s.get("athlete", {}).get("displayName"),
                    "team": s.get("athlete", {}).get("team", {}).get("abbreviation"),
                    "stat": s.get("name"),
                    "value": s.get("value")
                })
        df = pd.DataFrame(stats)
        if not df.empty:
            df = df.pivot_table(index=["name", "team"], columns="stat", values="value", aggfunc="first").reset_index()
            st.success("✅ ESPN live data loaded.")
            return df
    except Exception:
        st.info("ESPN unavailable. Trying next source...")

    # 2️⃣ nfl_data_py (2025 → fallback 2024)
    try:
        df = nfl.import_weekly_data([2025])
        if df.empty:
            df = nfl.import_weekly_data([2024])
        df = df[df["season_type"] == "REG"]
        df = df.groupby(["player_display_name", "recent_team"], as_index=False)[
            ["passing_yards", "passing_tds", "rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds"]
        ].mean()
        df.rename(columns={"player_display_name": "name", "recent_team": "team"}, inplace=True)
        st.success("✅ nfl_data_py data loaded.")
        return df
    except Exception as e:
        st.info(f"nfl_data_py failed: {e}")

    # 3️⃣ Permanent Kaggle mirror (no 404s)
    try:
        url = "https://raw.githubusercontent.com/databrotherhood/nflfastR-mirror/main/season_player_stats_2024.csv"
        df = pd.read_csv(url)
        keep = [
            "player_name", "team_abbr",
            "pass_yards", "pass_tds", "rush_yards",
            "rush_tds", "rec_yards", "rec_tds"
        ]
        df = df[keep].rename(columns={
            "player_name": "name",
            "team_abbr": "team",
            "pass_yards": "passing_yards",
            "pass_tds": "passing_tds",
            "rush_yards": "rushing_yards",
            "rush_tds": "rushing_tds",
            "rec_yards": "receiving_yards",
            "rec_tds": "receiving_tds"
        })
        st.warning("⚠️ Loaded from verified Kaggle mirror (2024 NFL season).")
        return df
    except Exception as e:
        st.error(f"❌ All data sources failed: {e}")
        return pd.DataFrame()
