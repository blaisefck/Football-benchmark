import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path


# Page config

st.set_page_config(page_title="Football Benchmark Dashboard", layout="wide")



LEAGUE_COL = "newestLeague"
TEAM_COL = "team"
POSITION_COL = "Position"
PLAYER_COL = "player"

METRIC_MAP = {
    "High-Speed Total (per90)": "TotalHighSpeedDist_per90",
    "20–25 km/h (per90)": "HiSpeedRunDist_per90",
    "Sprint >25 km/h (per90)": "SprintDist_per90",
}

# Optional columns (only used if present)
MINUTES_COL = "MinIncET"          # minutes played incl. ET (if present)
MATCH_COL = "gameId"              # unique match id (if present)

# Data loading

DATA_PATH = Path(__file__).parent.parent / "data" / "prepped_players_data.csv"
# If your file is elsewhere, change DATA_PATH accordingly.

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

if not DATA_PATH.exists():
    st.error(f"Missing file: {DATA_PATH}")
    st.info("Put prepped_players_data.csv next to app.py (or update DATA_PATH in the code).")
    st.stop()

df = load_data(DATA_PATH)


# Validate expected columns exist

required = [LEAGUE_COL, TEAM_COL, POSITION_COL, PLAYER_COL] + list(METRIC_MAP.values())
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Your prepped_players_data.csv is missing expected columns:")
    st.write(missing)
    st.stop()

has_minutes = MINUTES_COL in df.columns
has_match = MATCH_COL in df.columns

# Ensure types are safe
for m in METRIC_MAP.values():
    df[m] = pd.to_numeric(df[m], errors="coerce")

if has_minutes:
    df[MINUTES_COL] = pd.to_numeric(df[MINUTES_COL], errors="coerce")


# Auto-focus FC Versailles

def find_versailles_team(options):
    patterns = ["versailles", "versaille", "fc versailles", "versailles 78"]
    for p in patterns:
        for t in options:
            if p in str(t).lower():
                return t
    return None

teams_all = sorted(df[TEAM_COL].dropna().unique().tolist())
versailles_team = find_versailles_team(teams_all)


# Sidebar (filters + your robustness sliders)

st.sidebar.header("Filters")

page = st.sidebar.radio("Page", ["Overview", "Team Benchmark", "Player Table"])

leagues = sorted(df[LEAGUE_COL].dropna().unique().tolist())
positions = sorted(df[POSITION_COL].dropna().unique().tolist())

selected_league = st.sidebar.selectbox("Select League", ["All"] + leagues)

# Auto-select Versailles if found, otherwise "All"
default_team = versailles_team if versailles_team else "All"
selected_team = st.sidebar.selectbox(
    "Select Team",
    ["All"] + teams_all,
    index=(["All"] + teams_all).index(default_team)
)

selected_position = st.sidebar.selectbox("Select Position", ["All"] + positions)

metric_label = st.sidebar.selectbox("Metric", list(METRIC_MAP.keys()))
metric_col = METRIC_MAP[metric_label]

# Minutes slider (your feature) — only if minutes exist in cleaned file
if has_minutes:
    min_minutes = st.sidebar.slider("Minimum minutes played", 0, 120, 30, 5)
else:
    min_minutes = None
    st.sidebar.info("Minutes not available in cleaned dataset → minutes filter disabled.")

# Games slider (your feature) — uses unique matches if gameId exists, else row count
min_games = st.sidebar.slider("Minimum games (n_obs)", min_value=1, max_value=30, value=5, step=1)
if not has_match:
    st.sidebar.info("gameId not available → n_obs will use row count (less ideal than unique matches).")

# Small auto-focus hint
if versailles_team:
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Auto-focus team: **{versailles_team}**")


# Apply filters

filtered_df = df.copy()

if selected_league != "All":
    filtered_df = filtered_df[filtered_df[LEAGUE_COL] == selected_league]

if selected_team != "All":
    filtered_df = filtered_df[filtered_df[TEAM_COL] == selected_team]

if selected_position != "All":
    filtered_df = filtered_df[filtered_df[POSITION_COL] == selected_position]

if has_minutes and min_minutes is not None:
    filtered_df = filtered_df[filtered_df[MINUTES_COL].notna() & (filtered_df[MINUTES_COL] >= min_minutes)]


# Helpers

def safe_metric_mean(df_in: pd.DataFrame, column: str, unit: str = " m"):
    if df_in.empty or column not in df_in.columns:
        return "—"
    val = df_in[column].mean(skipna=True)
    if pd.isna(val):
        return "—"
    return f"{val:.1f}{unit}"


# UI

st.title("High-Intensity Running Benchmark")

st.caption(
    "Ce tableau de bord permet de comparer les performances de course à haute intensité "
    "(20–25 km/h et >25 km/h) sur une base de 90 minutes. "
    "Utilisez les filtres pour comparer le FC Versailles aux autres équipes du championnat "
    "et ajuster les seuils de minutes jouées ou de matchs afin d’obtenir des comparaisons robustes."
)

st.caption(
    "Les indicateurs sont exprimés en mètres par 90 minutes • "
    "Les données sont filtrées selon un minimum de minutes et de matchs joués."
)

# Focus box (Versailles) — shows regardless of Team filter, but respects league/position/minutes
if versailles_team:
    focus_df = df[df[TEAM_COL] == versailles_team].copy()

    if selected_league != "All":
        focus_df = focus_df[focus_df[LEAGUE_COL] == selected_league]
    if selected_position != "All":
        focus_df = focus_df[focus_df[POSITION_COL] == selected_position]
    if has_minutes and min_minutes is not None:
        focus_df = focus_df[focus_df[MINUTES_COL].notna() & (focus_df[MINUTES_COL] >= min_minutes)]

    with st.container(border=True):
        st.subheader(f"Focus: {versailles_team}")
        c1, c2, c3 = st.columns(3)
        c1.metric("High-Speed Total (per90)", safe_metric_mean(focus_df, "TotalHighSpeedDist_per90"))
        c2.metric("20–25 km/h (per90)", safe_metric_mean(focus_df, "HiSpeedRunDist_per90"))
        c3.metric("Sprint >25 (per90)", safe_metric_mean(focus_df, "SprintDist_per90"))


# Pages

if page == "Overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg High-Speed Total (per90)", safe_metric_mean(filtered_df, "TotalHighSpeedDist_per90"))
    c2.metric("Avg 20–25 km/h (per90)", safe_metric_mean(filtered_df, "HiSpeedRunDist_per90"))
    c3.metric("Avg Sprint >25 (per90)", safe_metric_mean(filtered_df, "SprintDist_per90"))

    st.markdown("---")
    st.subheader("Distribution by Team")

    if filtered_df.empty:
        st.info("No data available for the selected filters.")
    else:
        team_stats = (
            filtered_df
            .groupby(TEAM_COL)[["TotalHighSpeedDist_per90", "HiSpeedRunDist_per90", "SprintDist_per90"]]
            .mean()
            .reset_index()
            .sort_values("TotalHighSpeedDist_per90", ascending=False)
        )

        chart = (
            alt.Chart(team_stats)
            .transform_fold(
                ["TotalHighSpeedDist_per90", "HiSpeedRunDist_per90", "SprintDist_per90"],
                as_=["Metric", "Value"]
            )
            .mark_bar()
            .encode(
                y=alt.Y(f"{TEAM_COL}:N", sort="-x", title="Team"),
                x=alt.X("Value:Q", title="Meters per 90 min"),
                color=alt.Color("Metric:N", title="Metric"),
                tooltip=[
                    alt.Tooltip(f"{TEAM_COL}:N", title="Team"),
                    alt.Tooltip("Metric:N", title="Metric"),
                    alt.Tooltip("Value:Q", title="Value (per90)", format=".1f")
                ]
            )
        )

        st.altair_chart(chart, use_container_width=True)

elif page == "Team Benchmark":
    st.subheader("Team Benchmark (mean / std / games + ranking)")

    if filtered_df.empty:
        st.info("No data available for the selected filters.")
        st.stop()

    # n_obs = unique matches if possible, else row count
    if has_match:
        bench = (
            filtered_df
            .groupby(TEAM_COL, as_index=False)
            .agg(
                mean=(metric_col, "mean"),
                std=(metric_col, "std"),
                n_obs=(MATCH_COL, pd.Series.nunique),
            )
            .sort_values("mean", ascending=False)
            .reset_index(drop=True)
        )
    else:
        bench = (
            filtered_df
            .groupby(TEAM_COL, as_index=False)
            .agg(
                mean=(metric_col, "mean"),
                std=(metric_col, "std"),
                n_obs=(metric_col, "count"),
            )
            .sort_values("mean", ascending=False)
            .reset_index(drop=True)
        )

    # Apply min games filter
    bench = bench[bench["n_obs"] >= min_games].reset_index(drop=True)
    bench["rank"] = bench.index + 1

    # Highlight column for visuals + table pinning
    bench["highlight"] = "Other"
    if versailles_team and versailles_team in set(bench[TEAM_COL]):
        bench.loc[bench[TEAM_COL] == versailles_team, "highlight"] = "FC Versailles"

    # Pin Versailles row to top of the table (if present)
    if versailles_team and versailles_team in set(bench[TEAM_COL]):
        versa_row = bench[bench[TEAM_COL] == versailles_team]
        rest = bench[bench[TEAM_COL] != versailles_team]
        bench_display = pd.concat([versa_row, rest], ignore_index=True)
        st.success(
            f"{versailles_team} rank = {int(versa_row['rank'].iloc[0])} "
            f"(games={int(versa_row['n_obs'].iloc[0])})"
        )
    else:
        bench_display = bench
        st.info("FC Versailles not present under current filters (league/min minutes/min games).")

    st.dataframe(bench_display.drop(columns=["highlight"]), use_container_width=True)

    # Download
    csv = bench.drop(columns=["highlight"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download benchmark table (CSV)",
        data=csv,
        file_name=f"team_benchmark_{metric_col}_{selected_league}.csv".replace(" ", "_"),
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("Ranking chart")

    top_n = st.slider("Show top N teams", 5, 30, 20, 1)
    chart_df = bench.head(top_n).copy()

    bar = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            y=alt.Y(f"{TEAM_COL}:N", sort="-x", title="Team"),
            x=alt.X("mean:Q", title=f"Mean {metric_label}"),
            color=alt.Color(
                "highlight:N",
                scale=alt.Scale(domain=["FC Versailles", "Other"]),
                legend=alt.Legend(title="")
            ),
            tooltip=[
                alt.Tooltip(f"{TEAM_COL}:N", title="Team"),
                alt.Tooltip("rank:Q", title="Rank"),
                alt.Tooltip("mean:Q", title="Mean", format=".2f"),
                alt.Tooltip("std:Q", title="Std", format=".2f"),
                alt.Tooltip("n_obs:Q", title="Games"),
            ],
        )
    )
    st.altair_chart(bar, use_container_width=True)

elif page == "Player Table":
    st.subheader("📋 Player Statistics (per 90 mins)")

    if filtered_df.empty:
        st.info("No players match the selected filters.")
        st.stop()

    cols = [
        PLAYER_COL, TEAM_COL, POSITION_COL,
        "TotalHighSpeedDist_per90",
        "HiSpeedRunDist_per90",
        "SprintDist_per90",
    ]
    cols = [c for c in cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[cols].sort_values(metric_col, ascending=False),
        use_container_width=True
    )
