import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import nfl_data_py as nfl

st.set_page_config(page_title="NFL Active Player Predictor", layout="wide")

st.title("🏈 NFL Active Player Predictor — Current Season")
st.caption("View active players' stats and predict next-game performance using adaptive ML models.")

# =====================================================
# LOAD DATA WITH FALLBACK
# =====================================================
@st.cache_data(ttl=1800)
def load_latest_available_data(preferred_year=2025):
    """Load latest available NFL weekly data via nfl_data_py, fallback to previous year."""
    try:
        df = nfl.import_weekly_data([preferred_year])
        if not df.empty:
            return df, preferred_year
    except Exception:
        pass

    for y in range(preferred_year - 1, 2018, -1):
        try:
            df = nfl.import_weekly_data([y])
            if not df.empty:
                return df, y
        except Exception:
            continue
    raise RuntimeError("No valid NFL data found.")

df, active_year = load_latest_available_data()
st.caption(f"📅 Using {active_year} season data (latest available on nfl_data_py)")

# keep only players who recorded fantasy points (played)
df = df[df["season_type"] == "REG"]
df = df[df["fantasy_points"].notna()]
df = df[df["player_display_name"].notna()]

# =====================================================
# TEAM + PLAYER SELECTION
# =====================================================
teams = sorted(df["recent_team"].dropna().unique())
selected_team = st.selectbox("Select a team", teams)
if not selected_team:
    st.stop()

team_df = df[df["recent_team"] == selected_team]
players = sorted(team_df["player_display_name"].unique())
selected_player = st.selectbox("Select a player", players)
if not selected_player:
    st.stop()

player_df = team_df[team_df["player_display_name"] == selected_player]
if player_df.empty:
    st.warning("No data found for this player.")
    st.stop()

# =====================================================
# DISPLAY PLAYER STATS
# =====================================================
st.subheader(f"📊 {selected_player} — {active_year} Season Stats")
cols = [
    "week","opponent_team","passing_yards","passing_tds",
    "rushing_yards","rushing_tds","receiving_yards","receptions","fantasy_points"
]
cols = [c for c in cols if c in player_df.columns]
st.dataframe(player_df[cols].fillna(0), use_container_width=True, height=400)

# =====================================================
# AUTO PREDICTION ENGINE (with fallback)
# =====================================================
numeric_cols = player_df.select_dtypes(include=[np.number]).columns.tolist()
target_candidates = [
    "passing_yards","passing_tds","rushing_yards",
    "rushing_tds","receiving_yards","receptions","fantasy_points"
]
targets = [t for t in target_candidates if t in numeric_cols]
if not targets:
    st.warning("No numeric stats found for this player.")
    st.stop()

selected_target = st.selectbox("Stat to predict for next game", targets)

feature_cols = [c for c in numeric_cols if c not in ["week","season","game_key",selected_target]]
data = player_df[feature_cols + [selected_target]].dropna()

# === Fallback if player has too few games ===
if len(data) < 3:
    pos = player_df["position"].iloc[0] if "position" in player_df.columns else None
    if pos:
        st.info(f"Not enough personal data for {selected_player}. Using {pos} group model.")
        data = df[df["position"] == pos][feature_cols + [selected_target]].dropna()
    if len(data) < 3:
        st.info("Using league-wide data for fallback prediction.")
        data = df[feature_cols + [selected_target]].dropna()

# === Train model ===
X = data[feature_cols]
y = data[selected_target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Ridge": Ridge(),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
}

best_model, best_score = None, -999
results = []

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        results.append((name, score))
        if score > best_score:
            best_model, best_score = model, score
    except Exception:
        continue

results_df = pd.DataFrame(results, columns=["Model","R²"])
st.write("Model Performance Comparison:")
st.dataframe(results_df, use_container_width=True)

if best_model is None:
    st.error("Model training failed.")
    st.stop()

# === Predict next game ===
latest_row = X.iloc[[-1]]
next_pred = best_model.predict(latest_row)[0]
st.success(f"**Predicted {selected_target.replace('_',' ').title()}: {next_pred:.1f}** for next game")
st.caption(f"Best model: {results_df.loc[results_df['R²'].idxmax(), 'Model']} (R²={best_score:.3f})")

# =====================================================
# DOWNLOAD BUTTON
# =====================================================
st.download_button(
    "📥 Download Player's Season Stats",
    player_df.to_csv(index=False).encode(),
    file_name=f"{selected_player}_{active_year}_stats.csv"
)
