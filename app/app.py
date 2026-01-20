import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Football Benchmark Dashboard", layout="wide")

DATA_DIR = Path(__file__).parent.parent / "data"

PLAYERS_PATH = DATA_DIR / "prepped_players_v2.csv"

# File paths (with spaces as in your project)
TEAM_L2_PATH = DATA_DIR / "Equipes Ligue 2.csv"
TEAM_N1_PATH = DATA_DIR / "Equipes National1.csv"

# Colonnes (players prepped) - original mixed case
LEAGUE_COL = "newestLeague"
TEAM_COL = "team"
POSITION_COL = "Position"
PLAYER_COL = "player"
MINUTES_COL = "MinIncET"
MATCH_COL = "gameId"

# Colonnes (brut équipes)
TEAM_LEAGUE_COL_RAW = "leagueName"
TEAM_DATE_COL_RAW = "date"
TEAM_OPP_COL_RAW = "opponent"
TEAM_SCORE_COL_RAW = "score"
TEAM_SCORE_OPP_COL_RAW = "finalScoreOpponent"
TEAM_MATCH_COL_RAW = "gameId"
TEAM_MINUTES_COL_RAW = "MinIncET"

TEAM_DIST_TOTAL_RAW = "DistanceRun"
TEAM_RUN_15_20_RAW = "RunDist"
TEAM_HI_20_25_RAW = "HiSpeedRunDist"
TEAM_SPRINT_25_RAW = "SprintDist"


# -----------------------------
# Helpers (nettoyage + robustesse)
# -----------------------------
def normalize_columns(df: pd.DataFrame, target_cols: dict = None) -> pd.DataFrame:
    """
    ✅ FIX: Normalize column names to handle mixed case in team files.
    If target_cols provided, maps lowercase to target case.
    """
    df = df.copy()
    if target_cols:
        # Create mapping from lowercase to target
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower in target_cols:
                col_map[col] = target_cols[lower]
        df = df.rename(columns=col_map)
    return df


def clean_team_col(df: pd.DataFrame, team_col: str) -> pd.DataFrame:
    """Supprime les valeurs 'poubelles' type '.', '-', vide, ponctuation, etc."""
    df = df.copy()
    if team_col not in df.columns:
        return df

    s = df[team_col].astype(str).str.strip()
    s = s.replace({"": None, "-": None, ".": None, "nan": None, "None": None})

    # strings composées uniquement de ponctuation/espaces
    s = s.where(~s.fillna("").str.match(r"^[\W_]+$"), None)

    df[team_col] = s
    df = df[df[team_col].notna()]
    return df


def filter_meta_rows(df: pd.DataFrame) -> pd.DataFrame:
    """✅ FIX 3: Filter out TOTAL, AVERAGE rows and duplicate header rows"""
    df = df.copy()
    # Check for 'rank' or 'Rank' column
    rank_col = None
    for c in df.columns:
        if c.lower() == "rank":
            rank_col = c
            break

    if rank_col:
        rank_upper = df[rank_col].astype(str).str.upper().str.strip()
        df = df[~rank_upper.isin(["TOTAL", "AVERAGE", "RANK"])]
    return df


def to_num(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Convert columns to numeric, handling European format (comma as thousand separator)"""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)  # ✅ FIX: Remove thousand separator first
                .str.replace(" ", "", regex=False)  # Remove spaces
                .str.strip()
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_mean(df_in: pd.DataFrame, col: str):
    if df_in.empty or col not in df_in.columns:
        return None
    v = df_in[col].mean(skipna=True)
    return None if pd.isna(v) else float(v)


def fmt_meters(v, decimals=1):
    if v is None or pd.isna(v):
        return "—"
    if decimals == 0:
        return f"{v:,.0f} m".replace(",", " ")
    return f"{v:,.{decimals}f} m".replace(",", " ")


def fmt_minutes(v):
    if v is None or pd.isna(v):
        return "—"
    return f"{v:,.0f} min".replace(",", " ")


def safe_metric_display(df_in: pd.DataFrame, col: str, unit: str = " m", decimals: int = 1):
    v = safe_mean(df_in, col)
    if v is None:
        return "—"
    if unit.strip() == "m":
        return fmt_meters(v, decimals=decimals)
    if unit.strip() == "min":
        return fmt_minutes(v)
    return f"{v:.{decimals}f}{unit}"


def find_versailles_team(options):
    patterns = ["versailles", "versaille", "fc versailles", "versailles 78", "fc versailles 78"]
    for t in options:
        low = str(t).lower()
        if any(p in low for p in patterns):
            return t
    return None


def zscore(series: pd.Series) -> pd.Series:
    mu = series.mean(skipna=True)
    sd = series.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series([pd.NA] * len(series), index=series.index)
    return (series - mu) / sd


def ensure_meters_team_distance(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Corrige l'unité de DistanceRun (Team Match).
    Heuristique robuste:
      - Si la médiane de DistanceRun est < 500 -> on suppose km -> *1000.
      - Sinon on suppose déjà en mètres.
    """
    df = df.copy()
    if col not in df.columns:
        return df
    vals = pd.to_numeric(df[col], errors="coerce")
    med = vals.dropna().median()
    if pd.notna(med) and med > 0 and med < 500:
        df[col] = vals * 1000
    else:
        df[col] = vals
    return df


# -----------------------------
# Load data (players prepped)
# -----------------------------
@st.cache_data
def load_players(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = filter_meta_rows(df)   # ✅ FIX: Remove TOTAL/AVERAGE rows if present
    df = clean_team_col(df, TEAM_COL)

    num_candidates = [
        MINUTES_COL,
        "RunDist", "HiSpeedRunDist", "SprintDist", "DistanceRun",
        "RunDist_per90", "HiSpeedRunDist_per90", "SprintDist_per90", "DistanceRun_per90",
        "TotalHighSpeedDist_per90",
    ]
    df = to_num(df, [c for c in num_candidates if c in df.columns])
    return df


def build_metrics_players(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    has_minutes = MINUTES_COL in df.columns
    if has_minutes:
        df = df[df[MINUTES_COL].notna() & (df[MINUTES_COL] > 0)]

    # ✅ FIX: Handle DistanceRun with European comma format (e.g., "10,342" → 10342)
    if "DistanceRun" in df.columns:
        df["DistanceRun"] = (
            df["DistanceRun"]
            .astype(str)
            .str.replace(",", "", regex=False)  # Remove thousand separator
            .str.strip()
        )
        df["DistanceRun"] = pd.to_numeric(df["DistanceRun"], errors="coerce")

        # ✅ FIX: Filter out rows where DistanceRun = 0 (missing/bad data)
        df = df[df["DistanceRun"].notna() & (df["DistanceRun"] > 0)]

    # Compute DistanceRun_per90 if missing
    if "DistanceRun_per90" not in df.columns:
        if "DistanceRun" in df.columns and has_minutes:
            df["DistanceRun_per90"] = df["DistanceRun"] / df[MINUTES_COL] * 90

    # 20–25 per90 (already exists in prepped file, but compute if missing)
    if "HiSpeedRunDist_per90" not in df.columns:
        if "HiSpeedRunDist" in df.columns and has_minutes:
            df["HiSpeedRunDist_per90"] = df["HiSpeedRunDist"] / df[MINUTES_COL] * 90

    # Sprint per90 (already exists in prepped file, but compute if missing)
    if "SprintDist_per90" not in df.columns:
        if "SprintDist" in df.columns and has_minutes:
            df["SprintDist_per90"] = df["SprintDist"] / df[MINUTES_COL] * 90

    # ✅ FIX: The prepped file uses "HighIntensity15plus_per90" not "HighIntensity15plus_per90"
    # If it doesn't exist, compute it
    if "HighIntensity15plus_per90" not in df.columns:
        if has_minutes:
            if all(c in df.columns for c in ["RunDist", "HiSpeedRunDist", "SprintDist"]):
                df["HighIntensity15plus"] = df["RunDist"] + df["HiSpeedRunDist"] + df["SprintDist"]
                df["HighIntensity15plus_per90"] = df["HighIntensity15plus"] / df[MINUTES_COL] * 90
            elif all(c in df.columns for c in ["HiSpeedRunDist_per90", "SprintDist_per90"]):
                # Fallback: at minimum add 20-25 + >25
                df["HighIntensity15plus_per90"] = df["HiSpeedRunDist_per90"] + df["SprintDist_per90"]

    return df


if not PLAYERS_PATH.exists():
    st.error(f"Fichier manquant: {PLAYERS_PATH}")
    st.stop()

df_players_all = load_players(PLAYERS_PATH)
df_players_all = build_metrics_players(df_players_all)

# Colonnes minimum
required_min = [LEAGUE_COL, TEAM_COL, POSITION_COL, PLAYER_COL]
missing_min = [c for c in required_min if c not in df_players_all.columns]
if missing_min:
    st.error("Le fichier joueurs (prepped) ne contient pas les colonnes attendues :")
    st.write(missing_min)
    st.write("Colonnes disponibles:", list(df_players_all.columns))
    st.stop()

teams_all = sorted(df_players_all[TEAM_COL].dropna().unique().tolist())
versailles_team = find_versailles_team(teams_all)

# 4 paramètres (noms from screenshot - more readable)
METRIC_MAP = {
    ">15 km/h — per90": "HighIntensity15plus_per90",
    "20-25 km/h — per90": "HiSpeedRunDist_per90",
    ">25 km/h — per90": "SprintDist_per90",
    "Distance totale — per90": "DistanceRun_per90",  # Will be computed if missing
}
available_metrics = {k: v for k, v in METRIC_MAP.items() if v in df_players_all.columns}

if len(available_metrics) < 3:
    st.error("Il manque trop de colonnes pour construire les métriques (HI 15+, 20–25, Sprint, Distance totale).")
    st.write("Métriques disponibles:", available_metrics)
    st.write("Colonnes dans le fichier:", list(df_players_all.columns))
    st.stop()


# -----------------------------
# Load team raw (Team Match)
# -----------------------------
@st.cache_data
def load_team_raw(path: Path, comp_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ✅ FIX: Normalize column names (team files have inconsistent casing)
    # Map lowercase -> target case
    target_cols = {
        "team": "team",
        "minincet": "MinIncET",
        "distancerun": "DistanceRun",
        "rundist": "RunDist",
        "hispeedrundist": "HiSpeedRunDist",
        "sprintdist": "SprintDist",
        "score": "score",
        "finalscoreopponent": "finalScoreOpponent",
        "gameid": "gameId",
        "date": "date",
        "opponent": "opponent",
        "home": "home",
        "away": "away",
        "rank": "Rank",
        "leaguename": "leagueName",
    }
    df = normalize_columns(df, target_cols)

    df = filter_meta_rows(df)   # ✅ FIX: Remove TOTAL/AVERAGE/duplicate header rows

    df = clean_team_col(df, TEAM_COL)

    df = to_num(
        df,
        [TEAM_MINUTES_COL_RAW, TEAM_DIST_TOTAL_RAW, TEAM_RUN_15_20_RAW, TEAM_HI_20_25_RAW, TEAM_SPRINT_25_RAW,
         TEAM_SCORE_COL_RAW, TEAM_SCORE_OPP_COL_RAW]
    )

    df = df[df[TEAM_MINUTES_COL_RAW].notna() & (df[TEAM_MINUTES_COL_RAW] > 0)]

    # ✅ CORRECTION UNITÉS: DistanceRun souvent en km
    df = ensure_meters_team_distance(df, TEAM_DIST_TOTAL_RAW)

    # HI15+ + per90
    df["HI15plus"] = df[TEAM_RUN_15_20_RAW] + df[TEAM_HI_20_25_RAW] + df[TEAM_SPRINT_25_RAW]

    per90_cols = [TEAM_DIST_TOTAL_RAW, TEAM_RUN_15_20_RAW, TEAM_HI_20_25_RAW, TEAM_SPRINT_25_RAW, "HI15plus"]
    for c in per90_cols:
        df[c + "_per90"] = df[c] / df[TEAM_MINUTES_COL_RAW] * 90

    df["competition"] = comp_name
    return df


def build_match_selector_meta(df_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Construit une table unique par match (gameId) :
      - date
      - home_team / away_team
      - score_home / score_away (si dispo)
      - label lisible
    """
    if df_comp.empty:
        return pd.DataFrame(columns=[TEAM_MATCH_COL_RAW, "match_label", "date", "home_team", "away_team"])

    def pick_meta(g: pd.DataFrame):
        g = g.copy()

        # essayer de récupérer home/away
        home_row = None
        away_row = None
        if "home" in g.columns:
            home_rows = g[g["home"] == True]
            if home_rows.empty:
                # ✅ FIX: Also check for string 'true'
                home_rows = g[g["home"].astype(str).str.lower() == "true"]
            if not home_rows.empty:
                home_row = home_rows.iloc[0]
        if "away" in g.columns:
            away_rows = g[g["away"] == True]
            if away_rows.empty:
                away_rows = g[g["away"].astype(str).str.lower() == "true"]
            if not away_rows.empty:
                away_row = away_rows.iloc[0]

        # fallback
        if home_row is None:
            home_row = g.iloc[0]
        if away_row is None:
            # si on a opponent, on peut déduire away_team depuis home_row
            if TEAM_OPP_COL_RAW in g.columns and pd.notna(home_row.get(TEAM_OPP_COL_RAW, None)):
                # on crée un pseudo away_row
                away_team = home_row.get(TEAM_OPP_COL_RAW, "")
            else:
                # sinon prendre une autre ligne si possible
                away_team = g.iloc[1][TEAM_COL] if len(g) > 1 else ""
        else:
            away_team = away_row.get(TEAM_COL, "")

        home_team = home_row.get(TEAM_COL, "")
        if "away" in g.columns and away_row is not None and away_team == "":
            away_team = away_row.get(TEAM_COL, "")

        # date
        date_val = home_row.get(TEAM_DATE_COL_RAW, "")
        date_txt = str(date_val) if pd.notna(date_val) else ""

        # score
        s_home = home_row.get(TEAM_SCORE_COL_RAW, None)
        s_away = home_row.get(TEAM_SCORE_OPP_COL_RAW, None)
        score_txt = ""
        if pd.notna(s_home) and pd.notna(s_away):
            try:
                score_txt = f" ({int(s_home)}-{int(s_away)})"
            except Exception:
                score_txt = f" ({s_home}-{s_away})"

        label = f"{date_txt} — {home_team} vs {away_team}{score_txt}".strip()
        if label == "—":
            label = str(g[TEAM_MATCH_COL_RAW].iloc[0])

        return pd.Series(
            {
                TEAM_MATCH_COL_RAW: g[TEAM_MATCH_COL_RAW].iloc[0],
                "match_label": label,
                "date": date_txt,
                "home_team": str(home_team),
                "away_team": str(away_team),
            }
        )

    meta = df_comp.groupby(TEAM_MATCH_COL_RAW, as_index=False).apply(pick_meta).reset_index(drop=True)

    # enlever labels vides / équipe poubelle
    meta = meta[meta["match_label"].astype(str).str.strip().ne("")]
    meta = meta[meta["home_team"].astype(str).str.strip().ne(".")]
    meta = meta[meta["away_team"].astype(str).str.strip().ne(".")]

    # tri par date si possible
    return meta


# -----------------------------
# UI — Tabs
# -----------------------------
tabs = st.tabs(["📊 Benchmark", "👤 Joueurs", "🆚 Team Match"])


# ==========================================================
# TAB 1 — Benchmark
# ==========================================================
with tabs[0]:
    st.title("Benchmark — intensité & volume (par 90)")

    # =============================================
    # FOCUS VERSAILLES — EN HAUT (placeholder)
    # =============================================
    focus_placeholder = st.empty()

    # =============================================
    # FILTRES
    # =============================================
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1.6])

    leagues = sorted(df_players_all[LEAGUE_COL].dropna().unique().tolist())
    positions = sorted(df_players_all[POSITION_COL].dropna().unique().tolist())
    teams = sorted(df_players_all[TEAM_COL].dropna().unique().tolist())

    with f1:
        selected_league = st.selectbox("Championnat", ["All"] + leagues, index=0, key="bm_league")
    with f2:
        selected_position = st.selectbox("Poste", ["All"] + positions, index=0, key="bm_pos")
    with f3:
        default_team = versailles_team if versailles_team else "All"
        team_choices = ["All"] + teams
        team_idx = team_choices.index(default_team) if default_team in team_choices else 0
        selected_team = st.selectbox("Équipe", team_choices, index=team_idx, key="bm_team")
    with f4:
        metric_label = st.selectbox("Paramètre", list(available_metrics.keys()), index=0, key="bm_metric")

    metric_col = available_metrics[metric_label]

    # slider minutes
    if MINUTES_COL in df_players_all.columns:
        min_minutes = st.slider("Minimum minutes jouées", 0, 120, 30, 5, key="bm_min")
    else:
        min_minutes = None
        st.info("Colonne minutes absente → filtre minutes désactivé.")

    # Filtrage
    df_f = df_players_all.copy()
    if selected_league != "All":
        df_f = df_f[df_f[LEAGUE_COL] == selected_league]
    if selected_position != "All":
        df_f = df_f[df_f[POSITION_COL] == selected_position]
    if selected_team != "All":
        df_f = df_f[df_f[TEAM_COL] == selected_team]
    if min_minutes is not None and MINUTES_COL in df_f.columns:
        df_f = df_f[df_f[MINUTES_COL].notna() & (df_f[MINUTES_COL] >= min_minutes)]

    # =============================================
    # FOCUS VERSAILLES — Affiché dans le placeholder en haut
    # =============================================
    if versailles_team:
        focus_df = df_players_all[df_players_all[TEAM_COL] == versailles_team].copy()
        if selected_league != "All":
            focus_df = focus_df[focus_df[LEAGUE_COL] == selected_league]
        if selected_position != "All":
            focus_df = focus_df[focus_df[POSITION_COL] == selected_position]
        if min_minutes is not None and MINUTES_COL in focus_df.columns:
            focus_df = focus_df[focus_df[MINUTES_COL].notna() & (focus_df[MINUTES_COL] >= min_minutes)]

        with focus_placeholder.container():
            with st.container(border=True):
                st.subheader(f"Focus — {versailles_team}")
                c1, c2, c3, c4 = st.columns(4)
                if "HighIntensity15plus_per90" in df_players_all.columns:
                    c1.metric(">15 km/h — per90", safe_metric_display(focus_df, "HighIntensity15plus_per90", unit="m", decimals=0))
                c2.metric("20-25 km/h — per90", safe_metric_display(focus_df, "HiSpeedRunDist_per90", unit="m", decimals=0))
                c3.metric(">25 km/h — per90", safe_metric_display(focus_df, "SprintDist_per90", unit="m", decimals=0))
                if "DistanceRun_per90" in df_players_all.columns:
                    c4.metric("Distance totale — per90", safe_metric_display(focus_df, "DistanceRun_per90", unit="m", decimals=0))

    st.markdown("---")

    if df_f.empty:
        st.info("Aucune donnée avec ces filtres.")
        st.stop()

    # Agrégation équipe (moyenne + z-score)
    bench = (
        df_f.groupby(TEAM_COL, as_index=False)
        .agg(mean=(metric_col, "mean"))
        .dropna(subset=["mean"])
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )
    bench = clean_team_col(bench, TEAM_COL)
    bench = bench[bench[TEAM_COL].astype(str).str.strip().ne(".")]

    bench = bench.reset_index(drop=True)
    bench["rank"] = bench.index + 1
    bench["z_score"] = zscore(bench["mean"])

    top_n = st.slider("Top N équipes", 5, 30, 20, 1, key="bm_topn")

    # =============================================
    # GRAPHIQUE 1 — Paramètre sélectionné
    # =============================================
    st.subheader(f"Graphique — {metric_label}")

    chart_df = bench.head(top_n).copy()
    chart_df["highlight"] = "Other"
    if versailles_team and versailles_team in set(chart_df[TEAM_COL]):
        chart_df.loc[chart_df[TEAM_COL] == versailles_team, "highlight"] = "FC Versailles"

    bar = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{TEAM_COL}:N", sort="-y", title="Équipe", axis=alt.Axis(labelAngle=-60)),
            y=alt.Y("mean:Q", title=f"Moyenne — {metric_label} (m/90)"),
            color=alt.Color(
                "highlight:N",
                scale=alt.Scale(domain=["FC Versailles", "Other"]),
                legend=alt.Legend(title="")
            ),
            tooltip=[
                alt.Tooltip(f"{TEAM_COL}:N", title="Équipe"),
                alt.Tooltip("rank:Q", title="Rang"),
                alt.Tooltip("mean:Q", title="Moyenne (m/90)", format=".0f"),
                alt.Tooltip("z_score:Q", title="Z-score", format=".2f"),
            ],
        )
    )
    st.altair_chart(bar, use_container_width=True)

    # =============================================
    # GRAPHIQUE 2 — Comparaison des 3 paramètres d'intensité (grouped bar)
    # =============================================
    st.subheader("Comparaison des paramètres d'intensité")

    # Only use intensity metrics (exclude Distance totale which has different scale)
    intensity_metrics = {k: v for k, v in available_metrics.items() if "Distance totale" not in k}
    cols3 = list(intensity_metrics.values())

    team_stats3 = df_f.groupby(TEAM_COL, as_index=False)[cols3].mean()
    team_stats3 = clean_team_col(team_stats3, TEAM_COL)
    team_stats3 = team_stats3[team_stats3[TEAM_COL].astype(str).str.strip().ne(".")]

    # topN déterminé par la métrique choisie
    team_stats3 = team_stats3.sort_values(cols3[0], ascending=False).head(top_n)

    folded = team_stats3.melt(
        id_vars=[TEAM_COL],
        value_vars=cols3,
        var_name="Metric",
        value_name="Value"
    )
    label_map = {v: k for k, v in intensity_metrics.items()}
    folded["MetricLabel"] = folded["Metric"].map(label_map).fillna(folded["Metric"])

    bar3 = (
        alt.Chart(folded)
        .mark_bar()
        .encode(
            x=alt.X(f"{TEAM_COL}:N", sort="-y", title="Équipe", axis=alt.Axis(labelAngle=-60)),
            y=alt.Y("Value:Q", title="Mètres / 90"),
            color=alt.Color("MetricLabel:N", title="Paramètre"),
            xOffset="MetricLabel:N",  # Grouped bar chart
            tooltip=[
                alt.Tooltip(f"{TEAM_COL}:N", title="Équipe"),
                alt.Tooltip("MetricLabel:N", title="Paramètre"),
                alt.Tooltip("Value:Q", title="Valeur (m/90)", format=".0f"),
            ]
        )
    )
    st.altair_chart(bar3, use_container_width=True)

    # =============================================
    # TABLEAU
    # =============================================
    st.subheader("Tableau (ranking + z-score)")
    show_tbl = bench[[TEAM_COL, "rank", "mean", "z_score"]].head(top_n).copy()
    st.dataframe(show_tbl, use_container_width=True, height=738)

    csv = bench[[TEAM_COL, "rank", "mean", "z_score"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger le benchmark (CSV)",
        data=csv,
        file_name=f"benchmark_{metric_col}_{selected_league}.csv".replace(" ", "_"),
        mime="text/csv"
    )


# ==========================================================
# TAB 2 — Joueurs (table + fiche joueur)
# ==========================================================
with tabs[1]:
    st.title("Joueurs — table + focus individuel")

    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1.6])

    leagues = sorted(df_players_all[LEAGUE_COL].dropna().unique().tolist())
    positions = sorted(df_players_all[POSITION_COL].dropna().unique().tolist())
    teams = sorted(df_players_all[TEAM_COL].dropna().unique().tolist())

    with f1:
        selected_league = st.selectbox("Championnat", ["All"] + leagues, index=0, key="pl_league")
    with f2:
        selected_team = st.selectbox("Équipe", ["All"] + teams, index=0, key="pl_team")
    with f3:
        selected_position = st.selectbox("Poste", ["All"] + positions, index=0, key="pl_pos")
    with f4:
        metric_label = st.selectbox("Tri (paramètre)", list(available_metrics.keys()), index=0, key="pl_metric")

    metric_col = available_metrics[metric_label]

    if MINUTES_COL in df_players_all.columns:
        min_minutes = st.slider("Minimum minutes jouées", 0, 120, 30, 5, key="pl_min")
    else:
        min_minutes = None

    df_f = df_players_all.copy()
    if selected_league != "All":
        df_f = df_f[df_f[LEAGUE_COL] == selected_league]
    if selected_team != "All":
        df_f = df_f[df_f[TEAM_COL] == selected_team]
    if selected_position != "All":
        df_f = df_f[df_f[POSITION_COL] == selected_position]
    if min_minutes is not None and MINUTES_COL in df_f.columns:
        df_f = df_f[df_f[MINUTES_COL].notna() & (df_f[MINUTES_COL] >= min_minutes)]

    df_f = clean_team_col(df_f, TEAM_COL)

    if df_f.empty:
        st.info("Aucun joueur avec ces filtres.")
        st.stop()

    cols_show = [
        PLAYER_COL, TEAM_COL, POSITION_COL, LEAGUE_COL,
        MINUTES_COL if MINUTES_COL in df_f.columns else None,
        "HighIntensity15plus_per90" if "HighIntensity15plus_per90" in df_f.columns else None,
        "HiSpeedRunDist_per90" if "HiSpeedRunDist_per90" in df_f.columns else None,
        "SprintDist_per90" if "SprintDist_per90" in df_f.columns else None,
        "DistanceRun_per90" if "DistanceRun_per90" in df_f.columns else None,
    ]
    cols_show = [c for c in cols_show if c is not None and c in df_f.columns]

    st.subheader("Table joueurs")
    st.dataframe(
        df_f[cols_show].sort_values(metric_col, ascending=False),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Fiche joueur (détails)")
    player_list = sorted(df_f[PLAYER_COL].dropna().unique().tolist())

    default_player = None
    if versailles_team:
        versa_players = df_f[df_f[TEAM_COL] == versailles_team][PLAYER_COL].dropna().unique().tolist()
        if versa_players:
            default_player = sorted(versa_players)[0]

    chosen_player = st.selectbox(
        "Choisir un joueur",
        player_list,
        index=(player_list.index(default_player) if default_player in player_list else 0),
        key="pl_player_select"
    )

    # ✅ FIX: For player detail, get ALL games for this player (ignore position filter)
    # Apply only league, team, and minutes filters - not position
    p_df = df_players_all[df_players_all[PLAYER_COL] == chosen_player].copy()
    if selected_league != "All":
        p_df = p_df[p_df[LEAGUE_COL] == selected_league]
    if selected_team != "All":
        p_df = p_df[p_df[TEAM_COL] == selected_team]
    if min_minutes is not None and MINUTES_COL in p_df.columns:
        p_df = p_df[p_df[MINUTES_COL].notna() & (p_df[MINUTES_COL] >= min_minutes)]

    p_team = p_df[TEAM_COL].iloc[0] if not p_df.empty else "—"
    p_league = p_df[LEAGUE_COL].iloc[0] if not p_df.empty else "—"

    # ✅ Show all positions played by this player
    p_positions = p_df[POSITION_COL].unique().tolist() if not p_df.empty else ["—"]
    p_positions_str = ", ".join(p_positions)

    # Game count
    n_games = len(p_df)

    with st.container(border=True):
        st.subheader(chosen_player)
        st.caption(f"Équipe: {p_team} • Poste(s): {p_positions_str} • Championnat: {p_league} • Matchs: {n_games}")

        c1, c2, c3, c4 = st.columns(4)
        if "HighIntensity15plus_per90" in p_df.columns:
            c1.metric(">15 km/h — per90", safe_metric_display(p_df, "HighIntensity15plus_per90", unit="m", decimals=0))
        if "HiSpeedRunDist_per90" in p_df.columns:
            c2.metric("20-25 km/h — per90", safe_metric_display(p_df, "HiSpeedRunDist_per90", unit="m", decimals=0))
        if "SprintDist_per90" in p_df.columns:
            c3.metric(">25 km/h — per90", safe_metric_display(p_df, "SprintDist_per90", unit="m", decimals=0))
        if "DistanceRun_per90" in p_df.columns:
            c4.metric("Distance totale — per90", safe_metric_display(p_df, "DistanceRun_per90", unit="m", decimals=0))

        if MINUTES_COL in p_df.columns:
            st.caption(f"Minutes (moyenne sur les lignes filtrées) : {safe_metric_display(p_df, MINUTES_COL, unit='min', decimals=0)}")

    if MATCH_COL in p_df.columns:
        st.subheader("Détail (par match)")
        detail_cols = [MATCH_COL]
        # ✅ Add date, game info, and Position
        for c in ["date", "game", POSITION_COL, "opponent"]:
            if c in p_df.columns:
                detail_cols.append(c)

        for c in ["HighIntensity15plus_per90", "HiSpeedRunDist_per90", "SprintDist_per90", "DistanceRun_per90"]:
            if c in p_df.columns:
                detail_cols.append(c)

        detail_cols = [c for c in detail_cols if c in p_df.columns]
        st.dataframe(p_df[detail_cols], use_container_width=True)
    else:
        st.info("Colonne gameId absente côté joueurs → pas de détail par match disponible.")


# ==========================================================
# TAB 3 — Team Match (Option B = fichiers équipes bruts)
# ==========================================================
with tabs[2]:
    st.title("Team Match — comparaison match")

    if not TEAM_L2_PATH.exists() or not TEAM_N1_PATH.exists():
        st.error("Fichiers équipes bruts manquants dans /data")
        st.write({"L2": str(TEAM_L2_PATH), "N1": str(TEAM_N1_PATH)})
        st.stop()

    df_l2 = load_team_raw(TEAM_L2_PATH, "Ligue 2")
    df_n1 = load_team_raw(TEAM_N1_PATH, "National 1")
    df_team_all = pd.concat([df_l2, df_n1], ignore_index=True)

    df_team_all = clean_team_col(df_team_all, TEAM_COL)

    c1, c2 = st.columns([1.2, 3.8])

    with c1:
        comp = st.selectbox("Championnat", ["Ligue 2", "National 1"], index=0, key="tm_comp")

    df_comp = df_team_all[df_team_all["competition"] == comp].copy()

    meta = build_match_selector_meta(df_comp)
    if meta.empty:
        st.info("Aucun match exploitable dans ces fichiers (vérifie les colonnes date/home/away/opponent/score).")
        st.stop()

    # label -> gameId
    labels = meta["match_label"].tolist()
    label_to_game = dict(zip(meta["match_label"], meta[TEAM_MATCH_COL_RAW]))

    with c2:
        chosen_label = st.selectbox("Match", labels, key="tm_match_label")

    game_id = label_to_game[chosen_label]
    match_df = df_comp[df_comp[TEAM_MATCH_COL_RAW] == game_id].copy()

    # Agrégation par équipe - ✅ FIX: Use RAW TOTALS, not per90
    agg = (
        match_df.groupby(TEAM_COL, as_index=False)
        .agg(
            hi15=("HI15plus", "sum"),  # Raw total HI distance
            dist_total=(TEAM_DIST_TOTAL_RAW, "sum"),  # Raw total distance
            hs_20_25=(TEAM_HI_20_25_RAW, "sum"),  # Raw 20-25 km/h
            sprint_25=(TEAM_SPRINT_25_RAW, "sum"),  # Raw >25 km/h
        )
    )
    agg = clean_team_col(agg, TEAM_COL)
    agg = agg[agg[TEAM_COL].astype(str).str.strip().ne(".")]

    if agg.empty:
        st.info("Aucune donnée match exploitable (valeurs manquantes ou équipes invalides).")
        st.stop()

    # 🔥 Forcer Home à gauche / Away à droite si possible
    home_team = None
    away_team = None
    meta_row = meta[meta[TEAM_MATCH_COL_RAW] == game_id]
    if not meta_row.empty:
        home_team = meta_row["home_team"].iloc[0]
        away_team = meta_row["away_team"].iloc[0]

    def get_row(team_name: str):
        if team_name is None:
            return None
        rows = agg[agg[TEAM_COL] == team_name]
        return rows.iloc[0] if not rows.empty else None

    left_row = get_row(home_team)
    right_row = get_row(away_team)

    # fallback si la déduction home/away échoue
    if left_row is None or right_row is None:
        agg_sorted = agg.sort_values("hi15", ascending=False).reset_index(drop=True)
        left_row = agg_sorted.iloc[0]
        right_row = agg_sorted.iloc[1] if len(agg_sorted) > 1 else None

    left, right = st.columns(2)

    def render_team_card(container, row):
        with container:
            st.subheader(str(row[TEAM_COL]))

            a, b = st.columns(2)
            a.metric(">15 km/h", fmt_meters(row["hi15"], decimals=0))
            b.metric("Distance totale", fmt_meters(row["dist_total"], decimals=0))

            c, d = st.columns(2)
            c.metric("20-25 km/h", fmt_meters(row["hs_20_25"], decimals=0))
            d.metric(">25 km/h", fmt_meters(row["sprint_25"], decimals=0))

    render_team_card(left, left_row)

    if right_row is not None:
        render_team_card(right, right_row)
    else:
        st.info("Une seule équipe disponible pour ce match (donnée incomplète côté adversaire).")

    st.markdown("---")
    st.caption("Distances exprimées en mètres (totaux équipe sur le match).")
