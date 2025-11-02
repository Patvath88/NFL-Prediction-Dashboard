
import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from datetime import datetime
from rapidfuzz import process, fuzz
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import nfl_data_py as nfl
import requests

st.set_page_config(page_title="NFL Prop Predictor — Free Stats", layout="wide")
st.title("🏈 NFL Prop Predictor — Free Stats Edition")
st.caption("Live player prop predictor using free NFL data sources (nfl_data_py) and odds feeds.")

# Sidebar Settings
ODDS_KEY = st.sidebar.text_input("The Odds API Key", type="password")
markets = st.sidebar.multiselect("Prop Markets", ["pass_yds","rush_yds","rec_yds","receptions","any_td"],
                                 default=["pass_yds","rush_yds","rec_yds"])
lookback_weeks = st.sidebar.slider("Rolling Lookback (weeks)", 1, 10, 5)
today = datetime.now().date()

# Load free player data
@st.cache_data(ttl=3600)
def load_recent_stats(years):
    df = nfl.import_weekly_data(years=years)
    df = df[df["season_type"] == "REG"]
    return df

st.sidebar.markdown("---")
st.sidebar.write("Data Source: [nfl_data_py](https://pypi.org/project/nfl-data-py/)")

stats_df = load_recent_stats([2023, 2024])
st.success(f"Loaded {len(stats_df)} rows of free NFL player data.")
st.dataframe(stats_df.head(10))

# Fetch prop odds
@st.cache_data(ttl=600)
def fetch_odds(key, markets):
    if not key:
        return pd.DataFrame()
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {"apiKey": key, "regions": "us", "markets": ",".join(["player_" + m for m in markets])}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    rows = []
    for g in data:
        for bk in g.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for out in mk.get("outcomes", []):
                    rows.append({
                        "player": out.get("description"),
                        "market": mk.get("key"),
                        "line": out.get("point"),
                        "price": out.get("price"),
                        "book": bk.get("key"),
                        "game": f"{g.get('away_team')} @ {g.get('home_team')}"
                    })
    return pd.DataFrame(rows)

odds_df = fetch_odds(ODDS_KEY, markets)
if odds_df.empty:
    st.warning("No live props found (check API key).")
else:
    st.write("Live Prop Sample:")
    st.dataframe(odds_df.head(10))

# Name mapping
# Safer, more robust name mapping with fallbacks and None checks
def get_stats_name_series(df):
    # Try common name columns from nfl_data_py / nflfastR style datasets
    candidates = [
        "player_name",
        "player_display_name",
        "name",
        "full_name",
        "gsis_name",
        "espn_name"
    ]
    for c in candidates:
        if c in df.columns:
            s = df[c].astype(str).fillna("")
            if s.str.len().sum() > 0:
                return s
    # If none found, create a placeholder empty series
    return pd.Series([""] * len(df))

def map_names(odds_df, stats_df):
    if odds_df.empty or stats_df.empty:
        return pd.DataFrame()

    # Clean player strings
    odds_df = odds_df.copy()
    stats_df = stats_df.copy()

    odds_df["player"] = odds_df["player"].astype(str).fillna("")
    odds_df["player_clean"] = odds_df["player"].str.lower().str.replace(".","",regex=False)

    stats_names = get_stats_name_series(stats_df)
    stats_df["player_clean"] = stats_names.str.lower().str.replace(".","",regex=False)

    # Build candidate list (drop empties)
    candidates = stats_df["player_clean"].dropna().unique().tolist()
    if not candidates:
        # No candidates to match against
        odds_df["Name_clean"] = np.nan
        merged = odds_df.merge(stats_df, left_on="player_clean", right_on="player_clean", how="left")
        return merged

    mapping = {}
    for p in odds_df["player_clean"].dropna().unique():
        # Guard RapidFuzz extractOne possibly returning None
        result = process.extractOne(p, candidates, scorer=fuzz.token_sort_ratio)
        if result is None:
            continue
        match, score = result[0], result[1]
        if score is not None and score >= 80:
            mapping[p] = match

    odds_df["Name_clean"] = odds_df["player_clean"].map(mapping)
    # Prefer join on Name_clean if present; else fallback to player_clean
    key_left = "Name_clean" if odds_df["Name_clean"].notna().any() else "player_clean"
    key_right = "player_clean"
    merged = odds_df.merge(stats_df, left_on=key_left, right_on=key_right, how="left", suffixes=("","_stat"))
    return merged

if not odds_df.empty:
    joined_df = map_names(odds_df, stats_df)
    st.write("Merged Data Sample:")
    st.dataframe(joined_df.head(10))
else:
    joined_df = pd.DataFrame()

# Model training
def train_model(df, target_col):
    num = df.select_dtypes(include=[np.number]).fillna(0)
    if target_col not in num.columns:
        return None
    y = num[target_col]
    X = num.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    return rf, xgb, X.columns

# --- Market map and selection ---
# --- Safer market map and selection ---
market_map = {
    "pass_yds": ["passing_yards", "pass_yards", "pass_yds"],
    "rush_yds": ["rushing_yards", "rush_yards", "rush_yds"],
    "rec_yds": ["receiving_yards", "rec_yards", "rec_yds"],
    "receptions": ["receptions", "targets"],
    "any_td": ["rushing_tds", "receiving_tds", "passing_tds", "touchdowns"]
}

selected_market = st.selectbox("Market to model", markets, index=0 if markets else 0)

# Try to find a matching column dynamically
target_candidates = market_map.get(selected_market, [])
target_col = next((c for c in target_candidates if c in stats_df.columns), None)

if target_col is None:
    st.warning(f"No valid stat column found for {selected_market}. Available columns: {list(stats_df.columns)[:20]}")
    model_tuple = None
else:
    st.info(f"Using '{target_col}' as the target variable.")
    model_tuple = train_model(stats_df, target_col)
    if model_tuple:
        st.success(f"Model trained successfully for {target_col}")
    else:
        st.warning("Model training failed — no valid numeric data found.")


# Determine target column safely
target_col = market_map.get(selected_market, None)
if target_col is None:
    st.warning("Selected market not recognized or unavailable.")
    model_tuple = None
else:
    if target_col in stats_df.columns:
        model_tuple = train_model(stats_df, target_col)
        if model_tuple:
            st.success(f"Model trained successfully for {target_col}")
        else:
            st.warning("Model training failed — no valid numeric data found.")
    else:
        st.warning(f"Target column '{target_col}' not found in player stats.")
        model_tuple = None


# Predictions and slip builder
if model_tuple and not odds_df.empty:
    rf, xgb, features = model_tuple
    X = stats_df[features].fillna(0)
    preds = (rf.predict(X) + xgb.predict(X)) / 2
    stats_df["pred_" + target_col] = preds

    # Merge predictions
    merged = map_names(odds_df, stats_df)
    st.subheader("Predictions Joined with Live Props")
    st.dataframe(merged.head(10))

    # Slip builder
    st.header("🎯 Build Your Parlay Slip")
    selected_players = st.multiselect("Select player props:", merged["player"].unique())
    slip = merged[merged["player"].isin(selected_players)]
    if not slip.empty:
        slip["decimal"] = np.where(slip["price"] > 0, 1 + slip["price"]/100, 1 + 100/abs(slip["price"]))
        payout = slip["decimal"].prod()
        st.metric("Projected Parlay Payout (decimal)", round(payout, 2))
        st.download_button("Download Slip CSV", slip.to_csv(index=False).encode(), file_name="nfl_slip.csv")
    else:
        st.info("Select at least one player to create a slip.")
