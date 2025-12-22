import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Accelerations", layout="wide")

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

st.title("💥 Accelerations vs Decelerations (per 90)")

acc_col = "AccCount_per90"   # ajuste pro teu nome real
dec_col = "DecCount_per90"   # ajuste pro teu nome real

if f.empty or acc_col not in f.columns or dec_col not in f.columns:
    st.info("Sem dados (ou colunas não encontradas). Ajuste os nomes acc/dec no código.")
else:
    base = f.dropna(subset=[acc_col, dec_col, "player"])
    chart_df = base.groupby(["player", "team"], as_index=False)[[acc_col, dec_col]].mean()

    chart = (
        alt.Chart(chart_df)
        .mark_circle(size=90)
        .encode(
            x=alt.X(f"{acc_col}:Q", title="Accelerations per 90"),
            y=alt.Y(f"{dec_col}:Q", title="Decelerations per 90"),
            tooltip=["player", "team", alt.Tooltip(acc_col, format=".1f"), alt.Tooltip(dec_col, format=".1f")]
        )
    )
    st.altair_chart(chart, use_container_width=True)
