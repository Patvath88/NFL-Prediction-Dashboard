import streamlit as st
import pandas as pd
from sportsipy.nfl.teams import Teams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import plotly.express as px

st.set_page_config(page_title="NFL Predictor — Live 2025 Season", page_icon="🏈", layout="wide")

st.title("🏈 NFL Live Predictor — Always-On Version")
st.caption("Live player & team stats with machine learning predictions. Powered by SportsReference.com")

@st.cache_data(ttl=3600)
def load_team_data():
    """Fetch team-level and player-level data using sportsipy."""
    teams = Teams()
    data = []
    for team in teams:
        roster = team.roster
        for player_id, player in roster.players.items():
            stats = player.dataframe
            if stats is not None and not stats.empty:
                latest = stats.iloc[-1].to_dict()
                latest.update({
                    "player_name": player.name,
                    "team": team.name,
                    "position": getattr(player, "position", None)
                })
                data.append(latest)
    return pd.DataFrame(data)

df = load_team_data()

if df.empty:
    st.error("No NFL data could be retrieved from SportsReference.")
    st.stop()

teams = sorted(df["team"].unique())
selected_team = st.sidebar.selectbox("Select Team", teams)
team_df = df[df["team"] == selected_team]

players = sorted(team_df["player_name"].unique())
selected_player = st.sidebar.selectbox("Select Player", players)
player_df = team_df[team_df["player_name"] == selected_player]

st.subheader(f"{selected_player} — {selected_team}")
st.dataframe(player_df.T, use_container_width=True)

numeric_cols = player_df.select_dtypes("number").columns.tolist()
if len(team_df) > 10 and numeric_cols:
    target = st.selectbox("Stat to Predict", numeric_cols)
    X = team_df[numeric_cols].fillna(0)
    y = team_df[target].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = RandomForestRegressor(n_estimators=300)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    st.metric("Model R² Score", f"{r2:.3f}")
    pred = model.predict([player_df[numeric_cols].iloc[0]])[0]
    st.success(f"Predicted {target}: **{pred:.1f}**")
    fig = px.histogram(team_df, x=target, title=f"{selected_team} {target} Distribution")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Not enough numeric data to model this player yet.")
