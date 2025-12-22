import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="High Intensity", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("prepped_players_data.csv")

df = load_data()

st.sidebar.header("Filters")
leagues = sorted(df["newestLeague"].dropna().unique().tolist())
teams = sorted(df["team"].dropna().unique().tolist())
positions = sorted(df["Position"].dropna().unique().tolist())

selected_league = st.sidebar.selectbox("League", ["All"] + leagues)
selected_team = st.sidebar.selectbox("Team", ["All"] + teams)
selected_position = st.sidebar.selectbox("Position", ["All"] + positions)

f = df.copy()
if selected_league != "All":
    f = f[f["newestLeague"] == selected_league]
if selected_team != "All":
    f = f[f["team"] == selected_team]
if selected_position != "All":
    f = f[f["Position"] == selected_position]

st.title("🏃 High-Intensity (HSR) per 90")
metric_col = "HiSpeedRunDist_per90"  # 20–25 km/h (ajuste se o teu nome for outro)

if f.empty or metric_col not in f.columns:
    st.info("Sem dados para os filtros selecionados.")
else:
    base = f.dropna(subset=[metric_col, "team"])
    chart_df = base.groupby("team", as_index=False)[metric_col].mean().sort_values(metric_col, ascending=False)

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            y=alt.Y("team:N", sort="-x", title="Team"),
            x=alt.X(f"{metric_col}:Q", title="Meters per 90"),
            tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip(f"{metric_col}:Q", title="HSR/90", format=".1f")]
        )
    )
    st.altair_chart(chart, use_container_width=True)
