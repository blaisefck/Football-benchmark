import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Top Speeds", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("prepped_players_data.csv")

df = load_data()

st.sidebar.header("Filters")
teams = sorted(df["team"].dropna().unique().tolist())
selected_team = st.sidebar.selectbox("Team", ["All"] + teams)

f = df.copy()
if selected_team != "All":
    f = f[f["team"] == selected_team]

st.title("🏎️ Top Speed (distribution)")
metric_col = "TopSpeed_kmh"  # ajuste pro teu nome real

if f.empty or metric_col not in f.columns:
    st.info("Sem dados (ou coluna não encontrada). Ajuste o nome de TopSpeed no código.")
else:
    base = f.dropna(subset=[metric_col])

    chart = (
        alt.Chart(base)
        .mark_bar()
        .encode(
            x=alt.X(f"{metric_col}:Q", bin=alt.Bin(maxbins=30), title="km/h"),
            y=alt.Y("count():Q", title="Players"),
            tooltip=[alt.Tooltip("count():Q", title="Players")]
        )
    )
    st.altair_chart(chart, use_container_width=True)
