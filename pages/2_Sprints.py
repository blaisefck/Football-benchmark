import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Sprints", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("prepped_players_data.csv")

df = load_data()

st.sidebar.header("Filters")
teams = sorted(df["team"].dropna().unique().tolist())
positions = sorted(df["Position"].dropna().unique().tolist())

selected_team = st.sidebar.selectbox("Team", ["All"] + teams)
selected_position = st.sidebar.selectbox("Position", ["All"] + positions)

f = df.copy()
if selected_team != "All":
    f = f[f["team"] == selected_team]
if selected_position != "All":
    f = f[f["Position"] == selected_position]

st.title("⚡ Sprint Distance per 90")
metric_col = "SprintDist_per90"  # >25 km/h

if f.empty or metric_col not in f.columns:
    st.info("Sem dados para os filtros selecionados.")
else:
    base = f.dropna(subset=[metric_col, "player"])
    chart_df = base.sort_values(metric_col, ascending=False).head(25)

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            y=alt.Y("player:N", sort="-x", title="Player"),
            x=alt.X(f"{metric_col}:Q", title="Meters per 90"),
            tooltip=[alt.Tooltip("player:N", title="Player"),
                     alt.Tooltip("team:N", title="Team"),
                     alt.Tooltip(f"{metric_col}:Q", title="Sprint/90", format=".1f")]
        )
    )
    st.altair_chart(chart, use_container_width=True)
