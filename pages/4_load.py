import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Load", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("prepped_players_data.csv")

df = load_data()

st.sidebar.header("Filters")
positions = sorted(df["Position"].dropna().unique().tolist())
selected_position = st.sidebar.selectbox("Position", ["All"] + positions)

f = df.copy()
if selected_position != "All":
    f = f[f["Position"] == selected_position]

st.title("📈 Load / Volume (Total Distance per 90)")
metric_col = "TotalDist_per90"  # ajuste pro teu nome real (às vezes é TotalDistance_per90)

if f.empty or metric_col not in f.columns:
    st.info("Sem dados (ou coluna não encontrada). Ajuste o nome do TotalDist no código.")
else:
    base = f.dropna(subset=[metric_col, "team"])
    chart_df = base.groupby("team", as_index=False)[metric_col].mean().sort_values(metric_col, ascending=False)

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            y=alt.Y("team:N", sort="-x", title="Team"),
            x=alt.X(f"{metric_col}:Q", title="Meters per 90"),
            tooltip=["team", alt.Tooltip(metric_col, format=".0f")]
        )
    )
    st.altair_chart(chart, use_container_width=True)
