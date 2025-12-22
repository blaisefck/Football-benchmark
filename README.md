# ⚽ FC Versailles Benchmark Dashboard

A Streamlit dashboard to benchmark football teams, players, and positions based on high-intensity running metrics (normalized per 90 minutes).

## 📁 Project Structure
- `data/prepped_players_data.csv` – Cleaned and normalized dataset used by the app
- `app/streamlit_app.py` – Main Streamlit app
- `notebooks/` – Exploratory and data cleaning notebooks

## ▶️ Running the App

```bash
# Create environment
python -m venv venv
source venv/bin/activate  # or .\\venv\\Scripts\\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py

---------------------------------------------------------------------------------


🔎 Metrics Tracked
run_15_20 = 15–20 km/h
hi_20_25 = 20–25 km/h
sprint_25_plus = >25 km/h
All normalized per 90 minutes of play.

🧠 Dashboard Views
Team Benchmark
Match Dynamics
Player Ranking
Position Profile

📦 Requirements
streamlit
pandas
altair
