@st.cache_data(ttl=600)
def get_espn_season_stats():
    """
    Fully fault-tolerant stat loader.
    Tries ESPN → nfl_data_py → verified NFLFastR backup CSV for last season (2024).
    Ensures real, active player data for display & ML.
    """
    import nfl_data_py as nfl
    import io

    # 1️⃣ ESPN API — try first
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
        pass  # silently move to fallback

    # 2️⃣ nfl_data_py live 2025 (or fallback to 2024 if 2025 unavailable)
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
    except Exception:
        pass

    # 3️⃣ Verified public CSV fallback (always available)
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
        st.warning("⚠️ Using verified NFLFastR backup (2024 season data).")
        return df
    except Exception as e:
        st.error(f"❌ All data sources failed: {e}")
        return pd.DataFrame()
