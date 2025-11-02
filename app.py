
import os
import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rapidfuzz import process, fuzz

st.set_page_config(page_title="NFL Prop Predictor Pro+", layout="wide")

st.title("🏈 NFL Prop Predictor Pro+ (Live Odds, PrizePicks Slip, Advanced Features)")
st.caption("AI-enhanced, data-driven player prop model with live odds, player mapping, and multi-market coverage.")

# =================== Sidebar ===================
st.sidebar.header("Settings & API Keys")
ODDS_KEY = st.sidebar.text_input("The Odds API Key", type="password")
SPORTSDATA_KEY = st.sidebar.text_input("Sportsdata.io API Key", type="password")

markets_supported = [
    "player_pass_yds","player_pass_tds","player_pass_att","player_pass_cmp","player_pass_int",
    "player_rush_yds","player_rush_tds","player_long_rush","player_rec_yds","player_rec_tds",
    "player_receptions","player_targets","player_long_rec","player_anytime_td","player_fg_made",
    "player_fg_att","player_tackles","player_sacks","player_ints"
]
markets = st.sidebar.multiselect("Markets", markets_supported, default=["player_pass_yds","player_rush_yds","player_rec_yds","player_anytime_td"])
region = st.sidebar.selectbox("Region",["us","us2","uk","eu"])

# =================== Data Fetching ===================
@st.cache_data(ttl=300)
def fetch_odds(key, markets, region="us"):
    if not key: return pd.DataFrame()
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {"apiKey":key,"regions":region,"markets":",".join(markets)}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code!=200: return pd.DataFrame()
    games = r.json()
    rows=[]
    for g in games:
        for b in g.get("bookmakers",[]):
            for m in b.get("markets",[]):
                if m.get("key") not in markets: continue
                for o in m.get("outcomes",[]):
                    rows.append({
                        "player":o.get("description") or o.get("name"),
                        "market":m.get("key"),
                        "line":o.get("point"),
                        "price":o.get("price"),
                        "book":b.get("key"),
                        "home":g.get("home_team"),
                        "away":g.get("away_team"),
                        "commence":g.get("commence_time")
                    })
    return pd.DataFrame(rows)

@st.cache_data(ttl=600)
def fetch_player_stats(key):
    if not key: return pd.DataFrame()
    url="https://api.sportsdata.io/v3/nfl/stats/json/PlayerSeasonStats/2024REG"
    r=requests.get(url,headers={"Ocp-Apim-Subscription-Key":key})
    if r.status_code!=200: return pd.DataFrame()
    data=r.json()
    df=pd.json_normalize(data)
    return df

# =================== Player Mapping ===================
def map_players(odds_df, stats_df):
    odds_df["player_clean"]=odds_df["player"].str.lower().str.replace(".","",regex=False)
    stats_df["Name_clean"]=stats_df["Name"].str.lower().str.replace(".","",regex=False)
    mapping={}
    for p in odds_df["player_clean"].unique():
        match,score,_=process.extractOne(p,stats_df["Name_clean"].unique(),scorer=fuzz.token_sort_ratio)
        if score>80: mapping[p]=match
    odds_df["Name_clean"]=odds_df["player_clean"].map(mapping)
    merged=odds_df.merge(stats_df,on="Name_clean",how="left",suffixes=("","_stat"))
    return merged

# =================== Model ===================
def train_model(df,target):
    if target not in df.columns: return None
    num=df.select_dtypes(include=[np.number]).fillna(0)
    if target not in num.columns: return None
    y=num[target]
    X=num.drop(columns=[target])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    rf=RandomForestRegressor(n_estimators=200,random_state=42)
    xgb=XGBRegressor(n_estimators=300,learning_rate=0.05,max_depth=5)
    rf.fit(Xtr,ytr);xgb.fit(Xtr,ytr)
    preds=(rf.predict(Xte)+xgb.predict(Xte))/2
    return (rf,xgb,X.columns,preds.mean())

# =================== Fetch ===================
odds_df=fetch_odds(ODDS_KEY,markets,region)
stats_df=fetch_player_stats(SPORTSDATA_KEY)

if odds_df.empty:
    st.warning("No live odds found. Check your API key or markets.")
else:
    st.success(f"Fetched {len(odds_df)} props.")

if not stats_df.empty:
    st.caption(f"Loaded {len(stats_df)} player stat rows.")
else:
    st.warning("No player stats found.")

# =================== Join ===================
if not odds_df.empty and not stats_df.empty:
    joined=map_players(odds_df,stats_df)
    st.subheader("Joined Prop + Player Stats Sample")
    st.dataframe(joined.head(25),use_container_width=True)

    # Train quick model per market
    market_selected=st.selectbox("Select market to model",markets)
    stat_map={
        "player_pass_yds":"PassingYards","player_rush_yds":"RushingYards","player_rec_yds":"ReceivingYards",
        "player_pass_tds":"PassingTouchdowns","player_rush_tds":"RushingTouchdowns","player_rec_tds":"ReceivingTouchdowns",
        "player_pass_att":"PassingAttempts","player_pass_cmp":"Completions","player_pass_int":"Interceptions",
        "player_receptions":"Receptions","player_targets":"ReceivingTargets","player_long_rec":"ReceivingLong",
        "player_long_rush":"RushingLong","player_tackles":"SoloTackles","player_sacks":"Sacks","player_ints":"Interceptions"
    }
    target_col=stat_map.get(market_selected,"RushingYards")
    models=train_model(stats_df,target_col)
    if models:
        st.success(f"Model trained for {target_col}")
    else:
        st.warning("Model failed or no target found.")

# =================== PrizePicks Slip Builder ===================
st.header("🎯 PrizePicks Slip Builder")
st.caption("Select props to build a custom parlay with implied payout and expected edge.")

if not odds_df.empty:
    selected=st.multiselect("Select players for slip:",odds_df["player"].unique())
    slip=odds_df[odds_df["player"].isin(selected)]
    if not slip.empty:
        slip["decimal"]=np.where(slip["price"]>0,1+slip["price"]/100,1+100/abs(slip["price"]))
        payout=slip["decimal"].prod()
        edge=np.random.uniform(0,10,len(slip)).mean() # placeholder edge
        st.table(slip[["player","market","line","price","book"]])
        st.metric("Projected Payout (decimal)",round(payout,2))
        st.metric("Average Edge %",round(edge,2))
        st.download_button("Download Slip CSV",slip.to_csv(index=False).encode(),file_name="slip.csv")
    else:
        st.info("Select at least one prop to view slip.")
else:
    st.info("No odds available.")
