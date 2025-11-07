
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

st.set_page_config(page_title="New Model Dashboard", layout="wide")

# =====================================================
# Season & Sources
# =====================================================
SEASON = 2025  # user confirmed 2025-2026 season

# Your existing sheet that carries: home_team, away_team, favored_team, spread, over_under (+ week & scores if you have them)
SCORE_URL = "https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv"

# Pro Football Reference tables for the season (players)
PFR_PASSING = f"https://www.pro-football-reference.com/years/{SEASON}/passing.htm"
PFR_RUSHING = f"https://www.pro-football-reference.com/years/{SEASON}/rushing.htm"
PFR_RECEIVING = f"https://www.pro-football-reference.com/years/{SEASON}/receiving.htm"

# Pro Football Reference team pages
PFR_TEAM_OFF_DEF = f"https://www.pro-football-reference.com/years/{SEASON}/"

# =====================================================
# Team alias map (handles full names, nicknames, abbreviations)
# =====================================================
TEAM_ALIAS_TO_CODE = {
    # Cardinals
    "arizona cardinals": "ARI", "cardinals": "ARI", "arizona": "ARI", "ari": "ARI",
    # Falcons
    "atlanta falcons": "ATL", "falcons": "ATL", "atlanta": "ATL", "atl": "ATL",
    # Ravens
    "baltimore ravens": "BAL", "ravens": "BAL", "baltimore": "BAL", "bal": "BAL",
    # Bills
    "buffalo bills": "BUF", "bills": "BUF", "buffalo": "BUF", "buf": "BUF",
    # Panthers
    "carolina panthers": "CAR", "panthers": "CAR", "carolina": "CAR", "car": "CAR",
    # Bears
    "chicago bears": "CHI", "bears": "CHI", "chicago": "CHI", "chi": "CHI",
    # Bengals
    "cincinnati bengals": "CIN", "bengals": "CIN", "cincinnati": "CIN", "cin": "CIN",
    # Browns
    "cleveland browns": "CLE", "browns": "CLE", "cleveland": "CLE", "cle": "CLE",
    # Cowboys
    "dallas cowboys": "DAL", "cowboys": "DAL", "dallas": "DAL", "dal": "DAL",
    # Broncos
    "denver broncos": "DEN", "broncos": "DEN", "denver": "DEN", "den": "DEN",
    # Lions
    "detroit lions": "DET", "lions": "DET", "detroit": "DET", "det": "DET",
    # Packers
    "green bay packers": "GB", "packers": "GB", "green bay": "GB", "gb": "GB",
    # Texans
    "houston texans": "HOU", "texans": "HOU", "houston": "HOU", "hou": "HOU",
    # Colts
    "indianapolis colts": "IND", "colts": "IND", "indianapolis": "IND", "ind": "IND",
    # Jaguars
    "jacksonville jaguars": "JAX", "jaguars": "JAX", "jacksonville": "JAX", "jax": "JAX", "jacs": "JAX",
    # Chiefs
    "kansas city chiefs": "KC", "chiefs": "KC", "kansas city": "KC", "kc": "KC",
    # Raiders (incl. legacy)
    "las vegas raiders": "LV", "raiders": "LV", "las vegas": "LV", "lv": "LV",
    "oakland raiders": "LV", "oakland": "LV",
    # Chargers (incl. legacy)
    "los angeles chargers": "LAC", "la chargers": "LAC", "chargers": "LAC", "lac": "LAC",
    "san diego chargers": "LAC", "san diego": "LAC",
    # Rams (incl. legacy)
    "los angeles rams": "LAR", "la rams": "LAR", "rams": "LAR", "lar": "LAR",
    "st. louis rams": "LAR", "st louis rams": "LAR", "st louis": "LAR",
    # Dolphins
    "miami dolphins": "MIA", "dolphins": "MIA", "miami": "MIA", "mia": "MIA",
    # Vikings
    "minnesota vikings": "MIN", "vikings": "MIN", "minnesota": "MIN", "min": "MIN",
    # Patriots
    "new england patriots": "NE", "patriots": "NE", "new england": "NE", "ne": "NE",
    # Saints
    "new orleans saints": "NO", "saints": "NO", "new orleans": "NO", "no": "NO", "nos": "NO",
    # Giants
    "new york giants": "NYG", "ny giants": "NYG", "giants": "NYG", "nyg": "NYG",
    # Jets
    "new york jets": "NYJ", "ny jets": "NYJ", "jets": "NYJ", "nyj": "NYJ",
    # Eagles
    "philadelphia eagles": "PHI", "eagles": "PHI", "philadelphia": "PHI", "phi": "PHI",
    # Steelers
    "pittsburgh steelers": "PIT", "steelers": "PIT", "pittsburgh": "PIT", "pit": "PIT",
    # 49ers
    "san francisco 49ers": "SF", "49ers": "SF", "niners": "SF", "san francisco": "SF", "sf": "SF",
    # Seahawks
    "seattle seahawks": "SEA", "seahawks": "SEA", "seattle": "SEA", "sea": "SEA",
    # Buccaneers
    "tampa bay buccaneers": "TB", "buccaneers": "TB", "bucs": "TB", "tampa bay": "TB", "tb": "TB",
    # Titans
    "tennessee titans": "TEN", "titans": "TEN", "tennessee": "TEN", "ten": "TEN",
    # Commanders (incl. legacy)
    "washington commanders": "WAS", "commanders": "WAS", "washington": "WAS", "was": "WAS", "wsh": "WAS",
    "washington football team": "WAS", "redskins": "WAS"
}

CODE_TO_FULLNAME = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders"
}
FULLNAME_TO_CODE = {v: k for k, v in CODE_TO_FULLNAME.items()}

def team_key(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name).strip().lower()
    return TEAM_ALIAS_TO_CODE.get(s, s)

def normalize_header(name: str) -> str:
    name = str(name) if not isinstance(name, str) else name
    name = name.strip().replace(" ", "_").lower()
    name = re.sub(r"[^0-9a-z_]", "", name)
    return name

# =====================================================
# Scrapers (PFR)  — via pandas.read_html
# =====================================================
@st.cache_data(show_spinner=False)
def read_pfr_table(url: str, match: str = None) -> pd.DataFrame:
    tables = pd.read_html(url, match=match) if match else pd.read_html(url)
    if not tables:
        return pd.DataFrame()
    df = tables[0].copy()
    df = df[df.columns[df.columns.notnull()]]
    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"]
    return df

@st.cache_data(show_spinner=False)
def scrape_players():
    pass_df = read_pfr_table(PFR_PASSING, match="Passing")
    if not pass_df.empty:
        cols = {c: normalize_header(c) for c in pass_df.columns}
        pass_df.rename(columns=cols, inplace=True)
        keep = ["Player", "Tm", "G", "Yds"]
        # Column names may be already normalized; handle both
        mapping = {
            "player": "player", "tm": "team", "g": "games_played", "yds": "passing_yards_total",
            "Player": "player", "Tm": "team", "G": "games_played", "Yds": "passing_yards_total"
        }
        for k,v in mapping.items():
            if k in pass_df.columns:
                pass_df.rename(columns={k:v}, inplace=True)
        pass_df = pass_df[[c for c in ["player","team","games_played","passing_yards_total"] if c in pass_df.columns]].copy()
        pass_df["team_key"] = pass_df["team"].apply(team_key)
        pass_df["position"] = "qb"
    else:
        pass_df = pd.DataFrame(columns=["player","team","games_played","passing_yards_total","team_key","position"])

    rush_df = read_pfr_table(PFR_RUSHING, match="Rushing")
    if not rush_df.empty:
        cols = {c: normalize_header(c) for c in rush_df.columns}
        rush_df.rename(columns=cols, inplace=True)
        mapping = {
            "player": "player", "tm": "team", "g": "games_played", "yds": "rushing_yards_total",
            "att": "rushing_attempts_total"
        }
        for k,v in mapping.items():
            if k in rush_df.columns:
                rush_df.rename(columns={k:v}, inplace=True)
        keep = [c for c in ["player","team","games_played","rushing_yards_total","rushing_attempts_total"] if c in rush_df.columns]
        rush_df = rush_df[keep].copy()
        rush_df["team_key"] = rush_df["team"].apply(team_key)
        rush_df["position"] = "rb"
    else:
        rush_df = pd.DataFrame(columns=["player","team","games_played","rushing_yards_total","rushing_attempts_total","team_key","position"])

    rec_df = read_pfr_table(PFR_RECEIVING, match="Receiving")
    if not rec_df.empty:
        cols = {c: normalize_header(c) for c in rec_df.columns}
        rec_df.rename(columns=cols, inplace=True)
        mapping = {
            "player": "player", "tm": "team", "g": "games_played",
            "yds": "receiving_yards_total", "rec": "receiving_receptions_total", "tgt": "receiving_targets_total"
        }
        for k,v in mapping.items():
            if k in rec_df.columns:
                rec_df.rename(columns={k:v}, inplace=True)
        keep = [c for c in ["player","team","games_played","receiving_yards_total","receiving_receptions_total","receiving_targets_total"] if c in rec_df.columns]
        rec_df = rec_df[keep].copy()
        rec_df["team_key"] = rec_df["team"].apply(team_key)
        rec_df["position"] = "wr"
    else:
        rec_df = pd.DataFrame(columns=["player","team","games_played","receiving_yards_total","receiving_receptions_total","receiving_targets_total","team_key","position"])

    # ensure numeric
    for d in [pass_df, rush_df, rec_df]:
        for c in d.columns:
            if any(tok in c for tok in ["yards","games","attempts","receptions","targets"]):
                d[c] = pd.to_numeric(d[c], errors="coerce")
    return pass_df, rush_df, rec_df

@st.cache_data(show_spinner=False)
def scrape_team_off_def():
    try:
        off = pd.read_html(f"https://www.pro-football-reference.com/years/{SEASON}/team.htm", match="Team Offense")[0]
        defn = pd.read_html(f"https://www.pro-football-reference.com/years/{SEASON}/opp.htm", match="Opponent")[0]
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    off = off[off.columns[off.columns.notnull()]]
    if "Rk" in off.columns:
        off = off[off["Rk"] != "Rk"]
    off.columns = [normalize_header(c) for c in off.columns]

    defn = defn[defn.columns[defn.columns.notnull()]]
    if "Rk" in defn.columns:
        defn = defn[defn["Rk"] != "Rk"]
    defn.columns = [normalize_header(c) for c in defn.columns]

    off_slim = off.rename(columns={"tm": "team", "pts": "points_for", "g": "games_played_off"})
    off_slim = off_slim[["team","points_for","games_played_off"]]

    cand = {"tot_yds":"tot_yds","pass_yds":"pass_yds","rush_yds":"rush_yds","pts":"pts","g":"g"}
    for k in list(cand.keys()):
        if cand[k] not in defn.columns:
            for c in defn.columns:
                if k in c:
                    cand[k] = c
                    break
    def_slim = defn.rename(columns={"tm": "team"})
    keep = [c for c in ["team", cand["pass_yds"], cand["rush_yds"], cand["pts"], cand["g"]] if c in def_slim.columns]
    def_slim = def_slim[keep].copy()
    def_slim.rename(columns={
        cand["pass_yds"]: "passing_yards_allowed_total",
        cand["rush_yds"]: "rushing_yards_allowed_total",
        cand["pts"]: "points_allowed",
        cand["g"]: "games_played_def"
    }, inplace=True)

    off_slim["team_key"] = off_slim["team"].apply(team_key)
    def_slim["team_key"] = def_slim["team"].apply(team_key)

    team_sum = pd.merge(def_slim, off_slim[["team_key","points_for","games_played_off"]], on="team_key", how="left")
    return team_sum, off_slim, def_slim

@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    df = pd.read_csv(SCORE_URL)
    df.columns = [normalize_header(c) for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_all_player_dfs():
    p_pass, p_rush, p_rec = scrape_players()
    return {"player_passing": p_pass, "player_rushing": p_rush, "player_receiving": p_rec}

@st.cache_data(show_spinner=False)
def load_defense_tables(assume_wr_share=0.67, assume_te_share=0.17):
    team_sum, off_slim, def_slim = scrape_team_off_def()
    if team_sum is None or team_sum.empty:
        blank = pd.DataFrame(columns=["team_key","games_played_def"])
        return {"def_qb": blank, "def_rb": blank, "def_wr": blank, "def_te": blank}
    out = team_sum.copy()
    # QB/Passing
    d_qb = out[["team_key","passing_yards_allowed_total","games_played_def"]].copy()
    # RB/Rushing
    d_rb = out[["team_key","rushing_yards_allowed_total","games_played_def"]].copy()
    # WR/TE receiving approximations
    if "passing_yards_allowed_total" in out.columns:
        out["wr_rec_yards_allowed_total"] = out["passing_yards_allowed_total"] * assume_wr_share
        out["te_rec_yards_allowed_total"] = out["passing_yards_allowed_total"] * assume_te_share
    else:
        out["wr_rec_yards_allowed_total"] = np.nan
        out["te_rec_yards_allowed_total"] = np.nan
    d_wr = out[["team_key","wr_rec_yards_allowed_total","games_played_def"]].copy().rename(
        columns={"wr_rec_yards_allowed_total":"receiving_yards_allowed"}
    )
    d_te = out[["team_key","te_rec_yards_allowed_total","games_played_def"]].copy().rename(
        columns={"te_rec_yards_allowed_total":"receiving_yards_allowed"}
    )
    return {"def_qb": d_qb, "def_rb": d_rb, "def_wr": d_wr, "def_te": d_te}

# ----------------- modeling -----------------
def avg_scoring(df: pd.DataFrame, team_label: str):
    scored_home = df.loc[df["home_team"] == team_label, "home_score"].mean() if "home_score" in df.columns else np.nan
    scored_away = df.loc[df["away_team"] == team_label, "away_score"].mean() if "away_score" in df.columns else np.nan
    allowed_home = df.loc[df["home_team"] == team_label, "away_score"].mean() if "away_score" in df.columns else np.nan
    allowed_away = df.loc[df["away_team"] == team_label, "home_score"].mean() if "home_score" in df.columns else np.nan
    return np.nanmean([scored_home, scored_away]), np.nanmean([allowed_home, allowed_away])

def predict_scores(df: pd.DataFrame, team_label: str, opponent_label: str):
    team_avg_scored, team_avg_allowed = avg_scoring(df, team_label)
    opp_avg_scored, opp_avg_allowed = avg_scoring(df, opponent_label)
    raw_team_pts = (team_avg_scored + opp_avg_allowed) / 2
    raw_opp_pts = (opp_avg_scored + team_avg_allowed) / 2
    if "home_score" in df.columns and "away_score" in df.columns:
        league_avg_pts = df[["home_score", "away_score"]].stack().mean()
    else:
        league_avg_pts = 22.3
    cal_factor = 22.3 / league_avg_pts if not np.isnan(league_avg_pts) and league_avg_pts > 0 else 1.0
    team_pts = float(raw_team_pts * cal_factor) if pd.notna(raw_team_pts) else 22.3
    opp_pts = float(raw_opp_pts * cal_factor) if pd.notna(raw_opp_pts) else 22.3
    return team_pts, opp_pts

def american_to_decimal(odds: float) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 1 + (o / 100.0)
    else:
        return 1 + (100.0 / abs(o))

def decimal_to_american(dec: float) -> float:
    if dec <= 1:
        return np.nan
    if dec >= 2:
        return round((dec - 1) * 100)
    else:
        return round(-100 / (dec - 1))

def prob_to_decimal(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1-1e-6))
    return 1.0 / (1.0 - p + 1e-12)

def prob_to_american(p: float) -> float:
    dec = prob_to_decimal(p)
    return decimal_to_american(dec)

def prob_total_over_under(scores_df: pd.DataFrame, home: str, away: str, line_total: float):
    team_pts, opp_pts = predict_scores(scores_df, home, away)
    pred_total = team_pts + opp_pts
    stdev_total = max(6.0, pred_total * 0.18)
    z = (line_total - pred_total) / stdev_total
    p_over = float(np.clip(1 - norm.cdf(z), 0.001, 0.999))
    p_under = float(np.clip(norm.cdf(z), 0.001, 0.999))
    return pred_total, p_over, p_under

def prob_spread_cover(scores_df: pd.DataFrame, home: str, away: str, spread_signed: float, side: str):
    home_pts, away_pts = predict_scores(scores_df, home, away)
    pred_margin = home_pts - away_pts
    stdev_margin = max(5.0, abs(pred_margin) * 0.9 + 6.0)
    target = -spread_signed
    z = (target - pred_margin) / stdev_margin
    p_home_cover = float(np.clip(1 - norm.cdf(z), 0.001, 0.999))
    p_away_cover = 1.0 - p_home_cover
    return (pred_margin, p_home_cover if side == "home" else p_away_cover)

def detect_stat_col(df: pd.DataFrame, prop: str):
    cols = list(df.columns)
    normc = [normalize_header(c) for c in cols]
    mapping = {
        "rushing_yards": ["rushing_yards_total", "rushing_yards_per_game"],
        "receiving_yards": ["receiving_yards_total", "receiving_yards_per_game"],
        "passing_yards": ["passing_yards_total", "passing_yards_per_game"],
        "receptions": ["receiving_receptions_total"],
        "targets": ["receiving_targets_total"],
        "carries": ["rushing_attempts_total", "rushing_carries_per_game"],
    }
    pri = mapping.get(prop, [])
    for cand in pri:
        if cand in normc:
            return cols[normc.index(cand)]
    for i, c in enumerate(normc):
        if prop.split("_")[0] in c and ("per_game" in c or "total" in c):
            return cols[i]
    return None

def pick_def_df(prop: str, pos: str, d_qb, d_rb, d_wr, d_te):
    if prop == "passing_yards":
        return d_qb
    if prop in ["rushing_yards", "carries"]:
        return d_rb if pos != "qb" else d_qb
    if prop in ["receiving_yards", "receptions", "targets"]:
        if pos == "te":
            return d_te
        if pos == "rb":
            return d_rb
        return d_wr
    return None

def prop_prediction_and_probs(
    player_name: str,
    selected_prop: str,
    line_val: float,
    selected_team_key: str,
    opponent_key: str,
    p_rec: pd.DataFrame,
    p_rush: pd.DataFrame,
    p_pass: pd.DataFrame,
    d_qb: pd.DataFrame,
    d_rb: pd.DataFrame,
    d_wr: pd.DataFrame,
    d_te: pd.DataFrame,
):
    if selected_prop in ["receiving_yards", "receptions", "targets"]:
        player_df = p_rec; fallback_pos = "wr"
    elif selected_prop in ["rushing_yards", "carries"]:
        player_df = p_rush; fallback_pos = "rb"
    elif selected_prop == "passing_yards":
        player_df = p_pass; fallback_pos = "qb"
    else:
        player_df = p_rec; fallback_pos = "wr"

    if player_df is None or player_df.empty or "player" not in player_df.columns:
        return {"error": "No player data available."}
    row = player_df[player_df["player"].astype(str).str.lower() == str(player_name).lower()]
    if row.empty:
        return {"error": "Player not found."}
    row = row.iloc[0]
    pos = row.get("position", fallback_pos)
    stat_col = detect_stat_col(player_df, selected_prop)
    if not stat_col or stat_col not in player_df.columns:
        return {"error": "No matching stat column for this prop."}

    season_total = float(row[stat_col]) if pd.notna(row[stat_col]) else 0.0
    games = float(row.get("games_played", 1)) or 1.0
    player_pg = season_total / games if games > 0 else 0.0

    def_df = pick_def_df(selected_prop, pos, d_qb, d_rb, d_wr, d_te)
    def_col = None
    if def_df is not None and not def_df.empty:
        for c in def_df.columns:
            if "allowed" in c:
                def_col = c
                break
    if def_df is not None and def_col is not None:
        if "games_played_def" in def_df.columns:
            league_pg = (def_df[def_col] / def_df["games_played_def"].replace(0, np.nan)).mean()
        else:
            league_pg = def_df[def_col].mean()
        opp_row = def_df[def_df["team_key"] == opponent_key]
        if not opp_row.empty:
            if "games_played_def" in opp_row.columns and float(opp_row.iloc[0]["games_played_def"]) > 0:
                opp_pg = float(opp_row.iloc[0][def_col]) / float(opp_row.iloc[0]["games_played_def"])
            else:
                opp_pg = float(opp_row.iloc[0][def_col])
        else:
            opp_pg = league_pg
        adj = (opp_pg / league_pg) if (league_pg and league_pg > 0) else 1.0
    else:
        adj = 1.0

    predicted_pg = player_pg * adj
    stdev = max(3.0, predicted_pg * 0.35)
    z = (line_val - predicted_pg) / stdev
    prob_over = float(np.clip(1 - norm.cdf(z), 0.001, 0.999))
    prob_under = float(np.clip(norm.cdf(z), 0.001, 0.999))
    return {
        "predicted_pg": predicted_pg,
        "prob_over": prob_over,
        "prob_under": prob_under,
        "player_pg": player_pg,
        "season_total": season_total,
        "games_played": games
    }

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Data")
    if st.button("🔄 Scrape now (clear cache)", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("**Receiving splits (for defense vs WR/TE approximation)**")
    wr_share = st.slider("WR share of passing yards", 0.0, 1.0, 0.67, 0.01)
    te_share = st.slider("TE share of passing yards", 0.0, 1.0, 0.17, 0.01)
    rb_share = max(0.0, 1.0 - (wr_share + te_share))
    st.markdown(f"RB receiving share (computed): **{rb_share:.2f}**")

# ------------- Load -------------
scores_df = load_scores()
if scores_df.empty:
    st.error("Could not load NFL game sheet (spreads & totals).")
    st.stop()
players = load_all_player_dfs()
p_pass, p_rush, p_rec = players["player_passing"], players["player_rushing"], players["player_receiving"]
defs = load_defense_tables(assume_wr_share=wr_share, assume_te_share=te_share)
d_qb, d_rb, d_wr, d_te = defs["def_qb"], defs["def_rb"], defs["def_wr"], defs["def_te"]

# ------------ UI -------------
st.title("🏈 New Model Dashboard (PFR Scrape + Your Lines)")

section_names = ["1) Game Selection + Prediction","2) Top Edges This Week","3) Player Props","4) Parlay Builder"]
selected_section = st.selectbox("Jump to section", section_names, index=0)

with st.expander(section_names[0], expanded=(selected_section == section_names[0])):
    st.subheader("Select Game")
    cols = st.columns([1,1,2])
    with cols[0]:
        if "week" in scores_df.columns:
            week_list = sorted(scores_df["week"].dropna().unique())
            selected_week = st.selectbox("Week", week_list, key="sec1_week")
            week_mask = (scores_df["week"] == selected_week)
        else:
            selected_week = None
            week_mask = np.ones(len(scores_df), dtype=bool)
    with cols[1]:
        teams_in_week = sorted(
            set(scores_df.loc[week_mask, "home_team"].dropna().unique()) |
            set(scores_df.loc[week_mask, "away_team"].dropna().unique())
        )
        selected_team = st.selectbox("Team", teams_in_week, key="sec1_team")

    row_mask = ((scores_df["home_team"] == selected_team) | (scores_df["away_team"] == selected_team))
    if selected_week is not None and "week" in scores_df.columns:
        row_mask &= (scores_df["week"] == selected_week)
    game_row = scores_df[row_mask]
    if game_row.empty:
        st.warning("No game found for that team/week.")
    else:
        g = game_row.iloc[0]
        opponent = g["away_team"] if g["home_team"] == selected_team else g["home_team"]
        with cols[2]:
            st.markdown(f"**Matchup:** {selected_team} vs {opponent}")
        default_ou = float(g.get("over_under", 45.0)) if pd.notna(g.get("over_under", np.nan)) else 45.0
        default_spread = float(g.get("spread", 0.0)) if pd.notna(g.get("spread", np.nan)) else 0.0
        cL, cR = st.columns(2)
        with cL:
            over_under = st.number_input("Over/Under (Vegas or yours)", value=default_ou, step=0.5, key="sec1_ou")
        with cR:
            spread = st.number_input("Home-based Spread (signed: favorite = negative)", value=default_spread, step=0.5, key="sec1_spread")

        st.subheader("Game Prediction (Calibrated)")
        team_pts, opp_pts = predict_scores(scores_df, selected_team, opponent)
        total_pred = team_pts + opp_pts
        margin = team_pts - opp_pts
        total_diff = total_pred - over_under
        spread_diff = margin - (-spread)

        mrow1 = st.columns(2)
        mrow1[0].metric(f"{selected_team} Predicted", f"{team_pts:.1f} pts")
        mrow1[1].metric(f"{opponent} Predicted", f"{opp_pts:.1f} pts")
        mrow2 = st.columns(2)
        mrow2[0].metric("Predicted Total", f"{total_pred:.1f}", f"{total_diff:+.1f} vs O/U")
        mrow2[1].metric("Predicted Margin", f"{margin:+.1f}", f"{spread_diff:+.1f} vs Spread")

        st.plotly_chart(px.bar(x=["Predicted Total","O/U"], y=[total_pred, over_under], title="Predicted Total vs O/U"), use_container_width=True)
        st.plotly_chart(px.bar(x=["Predicted Margin","Home Spread"], y=[margin, -spread], title="Predicted Margin vs Spread"), use_container_width=True)

with st.expander(section_names[1], expanded=(selected_section == section_names[1])):
    if 'sec1_week' in st.session_state and "week" in scores_df.columns:
        selected_week_for_edges = st.session_state['sec1_week']
        wk = scores_df[scores_df["week"] == selected_week_for_edges].copy()
        st.caption(f"Week shown: {selected_week_for_edges}")
    else:
        wk = scores_df.copy()
        st.caption("Showing all games (no week selected).")

    def strength_badge(edge_val):
        if pd.isna(edge_val): return "⬜"
        a = abs(edge_val)
        if a >= 4: return "🟩"
        if a >= 2: return "🟨"
        return "🟥"

    rows = []
    for _, r in wk.iterrows():
        h, a = r.get("home_team"), r.get("away_team")
        if pd.isna(h) or pd.isna(a): continue
        h_pts, a_pts = predict_scores(scores_df, h, a)
        tot_pred = h_pts + a_pts
        mar_pred = h_pts - a_pts
        ou = float(r.get("over_under")) if pd.notna(r.get("over_under", np.nan)) else np.nan
        sp = float(r.get("spread")) if pd.notna(r.get("spread", np.nan)) else np.nan
        total_edge = np.nan if pd.isna(ou) else (tot_pred - ou)
        spread_edge = np.nan if pd.isna(sp) else (mar_pred - (-sp))

        if pd.isna(total_edge):
            total_pick, total_badge = "", "⬜"
        else:
            direction = "OVER" if total_edge > 0 else "UNDER"
            total_badge = strength_badge(total_edge)
            total_pick = f"{total_badge} {direction}"

        if pd.isna(spread_edge) or pd.isna(sp):
            spread_pick = ""
        else:
            if mar_pred > -sp:
                spread_pick = f"{strength_badge(spread_edge)} {h} {sp:+.1f}"
            else:
                spread_pick = f"{strength_badge(spread_edge)} {a} {(-sp):+.1f}"

        rows.append({
            "Matchup": f"{a} @ {h}",
            "Pred Total": round(tot_pred, 1),
            "O/U": ou if not pd.isna(ou) else "",
            "Total Edge (pts)": None if pd.isna(total_edge) else round(total_edge, 1),
            "Total Pick": total_pick,
            "Pred Margin": round(mar_pred, 1),
            "Spread (home)": sp if not pd.isna(sp) else "",
            "Spread Edge (pts)": None if pd.isna(spread_edge) else round(spread_edge, 1),
            "Spread Pick": spread_pick,
        })

    edges_df = pd.DataFrame(rows)
    if not edges_df.empty:
        def best_edge(row):
            vals = [abs(v) for v in [row.get("Total Edge (pts)"), row.get("Spread Edge (pts)")] if pd.notna(v)]
            return max(vals) if vals else 0.0
        edges_df["Rank Score"] = edges_df.apply(best_edge, axis=1)
        edges_df = edges_df.sort_values("Rank Score", ascending=False).drop(columns=["Rank Score"])
        show_cols = ["Matchup","Pred Total","O/U","Total Edge (pts)","Total Pick","Pred Margin","Spread (home)","Spread Edge (pts)","Spread Pick"]
        st.dataframe(edges_df[show_cols], use_container_width=True)
    else:
        st.info("No games found.")

with st.expander(section_names[2], expanded=(selected_section == section_names[2])):
    if 'sec1_team' not in st.session_state:
        st.info("Pick a game in Section 1 first.")
    else:
        selected_team = st.session_state['sec1_team']
        if 'sec1_week' in st.session_state and "week" in scores_df.columns:
            mask = ((scores_df["home_team"] == selected_team) | (scores_df["away_team"] == selected_team)) & (scores_df["week"] == st.session_state['sec1_week'])
        else:
            mask = ((scores_df["home_team"] == selected_team) | (scores_df["away_team"] == selected_team))
        game_row = scores_df[mask]
        if game_row.empty:
            st.warning("No game found for that team/week.")
        else:
            g = game_row.iloc[0]
            opponent = g["away_team"] if g["home_team"] == selected_team else g["home_team"]
            sel_key = team_key(selected_team)
            opp_key = team_key(opponent)

            def players_for_team(df, team_name_or_label):
                key = team_key(team_name_or_label)
                if df is None or df.empty or "team_key" not in df.columns or "player" not in df.columns:
                    return []
                return list(df.loc[df["team_key"] == key, "player"].dropna().unique())

            team_players = set(players_for_team(p_rec, selected_team) + players_for_team(p_rush, selected_team) + players_for_team(p_pass, selected_team))
            opp_players = set(players_for_team(p_rec, opponent) + players_for_team(p_rush, opponent) + players_for_team(p_pass, opponent))
            both_players = sorted(team_players.union(opp_players))

            c1, c2, c3 = st.columns([2,1.2,1.2])
            with c1:
                player_name = st.selectbox("Select Player", [""] + both_players, key="player_pick_props")
            with c2:
                prop_choices = ["passing_yards","rushing_yards","receiving_yards","receptions","targets","carries"]
                selected_prop = st.selectbox("Prop Type", prop_choices, index=2, key="prop_type_props")
            with c3:
                line_val = st.number_input("Sportsbook Line", value=50.0, step=0.5, key="prop_line")

            if player_name:
                res = prop_prediction_and_probs(
                    player_name=player_name,
                    selected_prop=selected_prop,
                    line_val=line_val,
                    selected_team_key=sel_key,
                    opponent_key=opp_key,
                    p_rec=p_rec, p_rush=p_rush, p_pass=p_pass,
                    d_qb=d_qb, d_rb=d_rb, d_wr=d_wr, d_te=d_te
                )
                if "error" in res:
                    st.warning(res["error"])
                else:
                    st.subheader(selected_prop.replace("_"," ").title())
                    st.write(f"**Season Total:** {res['season_total']:.1f}")
                    st.write(f"**Games Played:** {res['games_played']:.0f}")
                    st.write(f"**Per Game (season):** {res['player_pg']:.2f}")
                    st.write(f"**Adjusted prediction (this game):** {res['predicted_pg']:.2f}")
                    st.write(f"**Line:** {line_val:.1f}")
                    st.write(f"**Probability of OVER:** {res['prob_over']*100:.1f}%")
                    st.write(f"**Probability of UNDER:** {res['prob_under']*100:.1f}%")
                    st.plotly_chart(px.bar(x=["Predicted (this game)","Line"], y=[res["predicted_pg"], line_val], title=f"{player_name} – {selected_prop.replace('_',' ').title()}"), use_container_width=True)
            else:
                st.info("Select a player to evaluate props.")

with st.expander(section_names[3], expanded=(selected_section == section_names[3])):
    if "parlay_legs" not in st.session_state:
        st.session_state.parlay_legs = []
    g1, g2, g3, g4, g5, g6 = st.columns([1.0,2.2,1.6,1.2,1.2,1.2])
    with g1:
        week_for_market = st.selectbox("Week", sorted(scores_df["week"].dropna().unique()) if "week" in scores_df.columns else [], key="gm_week")
    with g2:
        wk_df = scores_df[scores_df["week"] == week_for_market] if "week" in scores_df.columns else scores_df.copy()
        matchups, meta = [], []
        for _, row in wk_df.iterrows():
            h = row.get("home_team"); a = row.get("away_team")
            if pd.isna(h) or pd.isna(a): continue
            matchups.append(f"{a} @ {h}")
            meta.append((h, a))
        gm_match = st.selectbox("Matchup", matchups, key="gm_matchup")
        idx = matchups.index(gm_match) if gm_match in matchups else -1
        home_team = meta[idx][0] if idx >= 0 else None
        away_team = meta[idx][1] if idx >= 0 else None
    with g3:
        gm_market = st.selectbox("Market", ["Total","Spread"], key="gm_market")
    with g4:
        if gm_market == "Total":
            default_tot_line = float(wk_df.iloc[0].get("over_under", 45.0)) if not wk_df.empty else 45.0
            gm_total = st.number_input("O/U Line", value=default_tot_line, step=0.5, key="gm_total_line")
            gm_side = st.selectbox("Side", ["over","under"], key="gm_total_side")
        else:
            default_sp_line = float(wk_df.iloc[0].get("spread", 0.0)) if not wk_df.empty else 0.0
            gm_spread = st.number_input("Home Spread (signed)", value=default_sp_line, step=0.5, key="gm_spread_line")
            gm_side_spread = st.selectbox("Side", ["home","away"], key="gm_spread_side")
    with g5:
        if gm_market == "Total" and home_team and away_team:
            _, p_over, p_under = prob_total_over_under(scores_df, home_team, away_team, gm_total)
            st.metric("Model Pr.", f"{(p_over if gm_side=='over' else p_under)*100:.1f}%")
        elif gm_market == "Spread" and home_team and away_team:
            _, p_cov = prob_spread_cover(scores_df, home_team, away_team, gm_spread, gm_side_spread)
            st.metric("Model Pr.", f"{p_cov*100:.1f}%")
        else:
            st.write(" ")
    with g6:
        if st.button("➕ Add Game Leg", use_container_width=True, key="gm_add"):
            if not (home_team and away_team):
                st.warning("Pick a valid matchup.")
            else:
                if gm_market == "Total":
                    _, p_over, p_under = prob_total_over_under(scores_df, home_team, away_team, gm_total)
                    prob = float(p_over if gm_side == "over" else p_under)
                    label = f"{away_team} @ {home_team} Total {gm_side.title()} {gm_total}"
                else:
                    _, p_cov = prob_spread_cover(scores_df, home_team, away_team, gm_spread, gm_side_spread)
                    prob = float(p_cov)
                    side_text = f"{gm_side_spread.title()} Cover {gm_spread:+.1f}"
                    label = f"{away_team} @ {home_team} Spread {side_text}"
                st.session_state.parlay_legs.append({"kind":"game","label":label,"prob":prob})
                st.rerun()

    if st.session_state.parlay_legs:
        st.subheader("Your Legs")
        for i, leg in enumerate(st.session_state.parlay_legs):
            c1, c2 = st.columns([8,1])
            c1.markdown(f"• **{leg.get('label','Leg')}** — Model Pr: **{leg.get('prob',0.0)*100:.1f}%**")
            if c2.button("🗑 Remove", key=f"rm_leg_{i}"):
                st.session_state.parlay_legs.pop(i); st.rerun()
        probs = [float(leg.get("prob", 0.0)) for leg in st.session_state.parlay_legs]
        parlay_hit_prob = float(np.prod(probs)) if probs else 0.0
        model_am_odds = prob_to_american(parlay_hit_prob)
        b1, b2, b3 = st.columns([1.2,1,1])
        with b1:
            book_total_american = st.text_input("Book Total Parlay Odds (American, e.g. +650)", value="", key="book_any_odds")
        with b2:
            stake = st.number_input("Stake ($)", value=100.0, step=10.0, min_value=0.0, key="book_any_stake")
        with b3:
            st.metric("Model Parlay Prob.", f"{parlay_hit_prob*100:.1f}%")
        if book_total_american.strip():
            try:
                book_am = float(book_total_american.strip())
                book_dec = american_to_decimal(book_am)
                payout = stake * (book_dec - 1.0)
                ev = parlay_hit_prob * payout - (1 - parlay_hit_prob) * stake
                st.metric("Model Fair Odds", f"{int(model_am_odds):+d}")
                st.metric("Expected Value ($)", f"{ev:,.2f}")
            except Exception:
                st.warning("Could not parse the book odds you entered. Use a number like +650 or -120.")
        else:
            st.metric("Model Fair Odds", f"{int(model_am_odds):+d}")
    else:
        st.info("Add legs above to build your parlay.")
