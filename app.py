import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor

# --- Helper functions to fetch ESPN API data ---

def get_player_id(player_name):
    """Fetch player ID for given name from ESPN search API"""
    url = f'https://site.api.espn.com/apis/common/v3/search?region=us&lang=en&query={player_name}&limit=5&mode=prefix&type=player'
    r = requests.get(url)
    data = r.json()
    try:
        return data['items'][0]['id']
    except (IndexError, KeyError):
        return None

def get_recent_game_stats(athlete_id):
    """Fetch last game stats for the player from ESPN API"""
    url = f'https://site.api.espn.com/apis/site/v2/sports/football/nfl/athletes/{athlete_id}/gamelog'
    r = requests.get(url)
    data = r.json()
    # Find latest game stats in the gamelog
    try:
        latest_game = data['gamelogs'][0]
        stats = latest_game['stats']
        # Extract key stats safely with defaults
        passing_yards = next((s['value'] for s in stats if s['name'] == 'passYds'), 0)
        receiving_yards = next((s['value'] for s in stats if s['name'] == 'recYds'), 0)
        receptions = next((s['value'] for s in stats if s['name'] == 'receptions'), 0)
        touchdowns = next((s['value'] for s in stats if s['name'] == 'touchdowns'), 0)
        return {
            'passing_yards': passing_yards,
            'receiving_yards': receiving_yards,
            'receptions': receptions,
            'touchdowns': touchdowns,
        }
    except (IndexError, KeyError):
        return None

# --- Train lightweight example models using sample data ---
def train_models():
    data = {
        'opponent_defense_rank': [10, 15, 8, 20, 5, 12, 18, 7, 14, 11],
        'player_snap_count': [50, 55, 48, 60, 52, 58, 49, 53, 57, 51],
        'recent_stat': [270, 230, 290, 250, 310, 260, 240, 280, 265, 275],
        'weather_factor': [1, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)

    # Target stats to predict
    targets = {
        'passing_yards': [280, 240, 300, 230, 320, 250, 260, 290, 270, 285],
        'receiving_yards': [60, 40, 70, 30, 80, 50, 55, 45, 65, 70],
        'receptions': [5, 3, 6, 2, 7, 4, 4, 3, 6, 7],
        'touchdowns': [2, 1, 3, 0, 3, 1, 1, 1, 2, 2],
    }

    models = {}

    for stat in targets:
        X = df[['opponent_defense_rank', 'player_snap_count', 'recent_stat', 'weather_factor']]
        y = targets[stat]
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X, y)
        models[stat] = model
    
    return models

# Train models
models = train_models()

# --- Streamlit UI ---
st.title("NFL Player Prop Predictor with ESPN Live Stats")

player_name = st.text_input("Enter NFL Player Name (e.g., Josh Allen):", "")

if player_name:
    athlete_id = get_player_id(player_name)
    if athlete_id is None:
        st.error(f"Player '{player_name}' not found.")
    else:
        st.write(f"Found player ID: {athlete_id}. Fetching last game stats...")

        stats = get_recent_game_stats(athlete_id)
        if stats is None:
            st.error("Could not retrieve recent game stats for the player.")
        else:
            # You may want inputs for opponent defense rank, snap count, weather factor:
            opponent_defense_rank = st.slider("Opponent Rush Def Rank (1 best, 32 worst):", 1, 32, 15)
            player_snap_count = st.slider("Player Snap Count (last game):", 30, 80, 55)
            weather_select = st.selectbox("Weather Condition:", ["Favorable", "Unfavorable"])
            weather_factor = 1 if weather_select == "Favorable" else 0

            # Prepare input for models
            def predict_stat(stat_name, recent_stat_value):
                input_df = pd.DataFrame([{
                    'opponent_defense_rank': opponent_defense_rank,
                    'player_snap_count': player_snap_count,
                    'recent_stat': recent_stat_value,
                    'weather_factor': weather_factor,
                }])
                return models[stat_name].predict(input_df)[0]

            passing_pred = predict_stat('passing_yards', stats['passing_yards'])
            receiving_pred = predict_stat('receiving_yards', stats['receiving_yards'])
            receptions_pred = predict_stat('receptions', stats['receptions'])
            touchdowns_pred = predict_stat('touchdowns', stats['touchdowns'])

            st.subheader(f"Prop Predictions for {player_name} (based on last game stats):")
            st.write(f"Passing Yards: {passing_pred:.1f}")
            st.write(f"Receiving Yards: {receiving_pred:.1f}")
            st.write(f"Receptions: {receptions_pred:.1f}")
            st.write(f"Touchdowns: {touchdowns_pred:.1f}")

            st.markdown("""
            **Disclaimer:** This is a sample app using ESPN's unofficial API. Real betting models require extensive data, validation, and ethical use.
            """)

