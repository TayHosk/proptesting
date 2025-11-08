import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re
from datetime import datetime

st.set_page_config(page_title="NFL Betting Model", layout="wide")

# =========================
# Data Sources (Google Sheets)
# =========================
# Games & lines sheet you already use
SCORE_URL = "https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv"

# NEW: Team game log sheet (abbreviations; one row per team per game)
# (Converted to CSV export form using your doc id)
TEAM_GAME_LOG_URL = "https://docs.google.com/spreadsheets/d/1PmVC_rWKdHNISbZHe0uQmv65d3fJx6vBtA2ydHl1qQA/export?format=csv"

SHEETS = {
    "total_offense": "https://docs.google.com/spreadsheets/d/1DFZRqOiMXbIoEeLaNaWh-4srxeWaXscqJxIAHt9yq48/export?format=csv",
    "total_passing": "https://docs.google.com/spreadsheets/d/1QclB5ajymBsCC09j8s4Gie_bxj4ebJwEw4kihG6uCng/export?format=csv",
    "total_rushing": "https://docs.google.com/spreadsheets/d/14NgUntobNrL1AZg3U85yZInArFkHyf9mi1csVFodu90/export?format=csv",
    "total_scoring": "https://docs.google.com/spreadsheets/d/1SJ_Y1ljU44lOjbNHuXGyKGiF3mgQxjAjX8H3j-CCqSw/export?format=csv",
    "player_receiving": "https://docs.google.com/spreadsheets/d/1Gwb2A-a4ge7UKHnC7wUpJltgioTuCQNuwOiC5ecZReM/export?format=csv",
    "player_rushing": "https://docs.google.com/spreadsheets/d/1c0xpi_wZSf8VhkSPzzchxvhzAQHK0tFetakdRqb3e6k/export?format=csv",
    "player_passing": "https://docs.google.com/spreadsheets/d/1I9YNSQMylW_waJs910q4S6SM8CZE--hsyNElrJeRfvk/export?format=csv",
    "def_rb": "https://docs.google.com/spreadsheets/d/1xTP8tMnEVybu9vYuN4i6IIrI71q1j60BuqVC40fjNeY/export?format=csv",
    "def_qb": "https://docs.google.com/spreadsheets/d/1SEwUdExz7Px61FpRNQX3bUsxVFtK97JzuQhTddVa660/export?format=csv",
    "def_wr": "https://docs.google.com/spreadsheets/d/14klXrrHHCLlXhW6-F-9eJIz3dkp_ROXVSeehlM8TYAo/export?format=csv",
    "def_te": "https://docs.google.com/spreadsheets/d/1yMpgtx1ObYLDVufTMR5Se3KrMi1rG6UzMzLcoptwhi4/export?format=csv",
}

# =========================
# Team alias map (full names + nicknames + abbreviations)
# =========================
TEAM_ALIAS_TO_CODE = {
    "arizona cardinals": "ARI", "cardinals": "ARI", "arizona": "ARI", "ari": "ARI",
    "atlanta falcons": "ATL", "falcons": "ATL", "atlanta": "ATL", "atl": "ATL",
    "baltimore ravens": "BAL", "ravens": "BAL", "baltimore": "BAL", "bal": "BAL",
    "buffalo bills": "BUF", "bills": "BUF", "buffalo": "BUF", "buf": "BUF",
    "carolina panthers": "CAR", "panthers": "CAR", "carolina": "CAR", "car": "CAR",
    "chicago bears": "CHI", "bears": "CHI", "chicago": "CHI", "chi": "CHI",
    "cincinnati bengals": "CIN", "bengals": "CIN", "cincinnati": "CIN", "cin": "CIN",
    "cleveland browns": "CLE", "browns": "CLE", "cleveland": "CLE", "cle": "CLE",
    "dallas cowboys": "DAL", "cowboys": "DAL", "dallas": "DAL", "dal": "DAL",
    "denver broncos": "DEN", "broncos": "DEN", "denver": "DEN", "den": "DEN",
    "detroit lions": "DET", "lions": "DET", "detroit": "DET", "det": "DET",
    "green bay packers": "GB", "packers": "GB", "green bay": "GB", "gb": "GB",
    "houston texans": "HOU", "texans": "HOU", "houston": "HOU", "hou": "HOU",
    "indianapolis colts": "IND", "colts": "IND", "indianapolis": "IND", "ind": "IND",
    "jacksonville jaguars": "JAX", "jaguars": "JAX", "jacksonville": "JAX", "jax": "JAX", "jacs": "JAX",
    "kansas city chiefs": "KC", "chiefs": "KC", "kansas city": "KC", "kc": "KC",
    "las vegas raiders": "LV", "raiders": "LV", "las vegas": "LV", "lv": "LV",
    "oakland raiders": "LV", "oakland": "LV",
    "los angeles chargers": "LAC", "la chargers": "LAC", "chargers": "LAC", "lac": "LAC",
    "san diego chargers": "LAC", "san diego": "LAC",
    "los angeles rams": "LAR", "la rams": "LAR", "rams": "LAR", "lar": "LAR",
    "st. louis rams": "LAR", "st louis rams": "LAR", "st louis": "LAR",
    "miami dolphins": "MIA", "dolphins": "MIA", "miami": "MIA", "mia": "MIA",
    "minnesota vikings": "MIN", "vikings": "MIN", "minnesota": "MIN", "min": "MIN",
    "new england patriots": "NE", "patriots": "NE", "new england": "NE", "ne": "NE",
    "new orleans saints": "NO", "saints": "NO", "new orleans": "NO", "no": "NO", "nos": "NO",
    "new york giants": "NYG", "ny giants": "NYG", "giants": "NYG", "nyg": "NYG",
    "new york jets": "NYJ", "ny jets": "NYJ", "jets": "NYJ", "nyj": "NYJ",
    "philadelphia eagles": "PHI", "eagles": "PHI", "philadelphia": "PHI", "phi": "PHI",
    "pittsburgh steelers": "PIT", "steelers": "PIT", "pittsburgh": "PIT", "pit": "PIT",
    "san francisco 49ers": "SF", "49ers": "SF", "niners": "SF", "san francisco": "SF", "sf": "SF",
    "seattle seahawks": "SEA", "seahawks": "SEA", "seattle": "SEA", "sea": "SEA",
    "tampa bay buccaneers": "TB", "buccaneers": "TB", "bucs": "TB", "tampa bay": "TB", "tb": "TB",
    "tennessee titans": "TEN", "titans": "TEN", "tennessee": "TEN", "ten": "TEN",
    "washington commanders": "WAS", "commanders": "WAS", "washington": "WAS", "was": "WAS", "wsh": "WAS",
    "washington football team": "WAS", "redskins": "WAS"
}
CODE_TO_FULLNAME = {
    "ARI": "Arizona Cardinals","ATL": "Atlanta Falcons","BAL": "Baltimore Ravens","BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers","CHI": "Chicago Bears","CIN": "Cincinnati Bengals","CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys","DEN": "Denver Broncos","DET": "Detroit Lions","GB": "Green Bay Packers",
    "HOU": "Houston Texans","IND": "Indianapolis Colts","JAX": "Jacksonville Jaguars","KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers","LAR": "Los Angeles Rams","LV": "Las Vegas Raiders","MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings","NE": "New England Patriots","NO": "New Orleans Saints","NYG": "New York Giants",
    "NYJ": "New York Jets","PHI": "Philadelphia Eagles","PIT": "Pittsburgh Steelers","SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers","TB": "Tampa Bay Buccaneers","TEN": "Tennessee Titans","WAS": "Washington Commanders"
}
FULLNAME_TO_CODE = {v: k for k, v in CODE_TO_FULLNAME.items()}

def team_key(name: str) -> str:
    if pd.isna(name): return ""
    s = str(name).strip().lower()
    return TEAM_ALIAS_TO_CODE.get(s, s)

# =========================
# Sidebar: Cache refresh
# =========================
with st.sidebar:
    st.header("Data")
    if st.button("🔄 Clear cache & reload data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# =========================
# Helpers
# =========================
def normalize_header(name: str) -> str:
    name = str(name) if not isinstance(name, str) else name
    name = name.strip().replace(" ", "_").lower()
    name = re.sub(r"[^0-9a-z_]", "", name)
    return name

def find_col(cols_norm, prefer_any_of, contains_ok=False):
    """
    cols_norm: list of normalized column names
    prefer_any_of: list of normalized candidate names or substrings
    contains_ok: if True, match by substring if exact not found
    returns index or None
    """
    for cand in prefer_any_of:
        if cand in cols_norm:
            return cols_norm.index(cand)
    if contains_ok:
        for i, c in enumerate(cols_norm):
            if any(cand in c for cand in prefer_any_of):
                return i
    return None

def consolidate_duplicate_columns(df, base_normalized):
    """
    If df has duplicate columns like over_under_result, over_under_result_1, etc.,
    create a single column with the first non-null value and drop the extras.
    """
    cols = [c for c in df.columns if c == base_normalized or c.startswith(base_normalized + "_")]
    if len(cols) <= 1:
        return df
    out = df.copy()
    out[base_normalized + "_final"] = pd.NA
    for c in cols:
        out[base_normalized + "_final"] = out[base_normalized + "_final"].fillna(out[c])
    # keep a single clean name
    out.drop(columns=cols, inplace=True)
    out.rename(columns={base_normalized + "_final": base_normalized}, inplace=True)
    return out

def parse_yes_no_push(val: str):
    if pd.isna(val): return None
    s = str(val).strip().lower()
    if s in ["over","o","u","under","push","p","tie"]:
        return s  # for OU we want over/under/push
    # for spread cover
    if any(x in s for x in ["cover","covered","yes","y"]):
        return "covered"
    if any(x in s for x in ["no","not","didnt","didn't","failed"]):
        return "not_covered"
    if "push" in s or "tie" in s:
        return "push"
    # sometimes books use W/L/Push relative to spread
    if s in ["w","win","won"]: return "covered"
    if s in ["l","lose","lost"]: return "not_covered"
    return s

# =========================
# Loaders
# =========================
def compute_home_spread(row: pd.Series) -> float:
    """
    Convert (favored_team, spread) into HOME-based spread.
      - If home is the favorite  → negative abs(spread)
      - If away is the favorite  → positive abs(spread)
    """
    try:
        home = row.get("home_team", None)
        fav = row.get("favored_team", None)
        sp = row.get("spread", np.nan)
        if pd.isna(sp) or not fav or not home:
            return np.nan
        hk = team_key(home); fk = team_key(fav)
        return -abs(sp) if (fk and hk and fk == hk) else abs(sp)
    except Exception:
        return np.nan

@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    df = pd.read_csv(SCORE_URL)
    df.columns = [normalize_header(c) for c in df.columns]
    if "over_under" in df.columns:
        df["over_under"] = pd.to_numeric(df["over_under"], errors="coerce")
    if "spread" in df.columns:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
    if "home_team" in df.columns:
        df["home_key"] = df["home_team"].apply(team_key)
    if "away_team" in df.columns:
        df["away_key"] = df["away_team"].apply(team_key)
    if "favored_team" in df.columns:
        df["favored_key"] = df["favored_team"].apply(team_key)
    if {"home_team","favored_team","spread"}.issubset(df.columns):
        df["home_spread"] = df.apply(compute_home_spread, axis=1)
    else:
        df["home_spread"] = np.nan
    return df

def load_and_clean(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [normalize_header(c) for c in df.columns]
    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.strip()
    elif "teams" in df.columns:
        df["team"] = df["teams"].astype(str).str.strip()
    else:
        df["team"] = ""
    df["team_key"] = df["team"].apply(team_key)
    return df

@st.cache_data(show_spinner=False)
def load_all_player_dfs():
    return {name: load_and_clean(url) for name, url in SHEETS.items()}

@st.cache_data(show_spinner=False)
def load_team_game_log(url: str) -> pd.DataFrame:
    raw = pd.read_csv(url)
    raw_cols = list(raw.columns)
    norm_cols = [normalize_header(c) for c in raw_cols]
    df = raw.copy()
    df.columns = norm_cols

    # Consolidate the duplicate "Over Under Result"
    df = consolidate_duplicate_columns(df, "over_under_result")

    # Build canonical columns if present
    # team, date, day, week, opponent, is_away (column with '@' when away), ou_line, ou_result, spread, cover_result
    mapper = {}

    idx = find_col(norm_cols, ["team"])
    if idx is not None: mapper["team"] = df.columns[idx]

    idx = find_col(norm_cols, ["date"])
    if idx is not None: mapper["date"] = df.columns[idx]

    idx = find_col(norm_cols, ["day"])
    if idx is not None: mapper["day"] = df.columns[idx]

    idx = find_col(norm_cols, ["week"])
    if idx is not None: mapper["week"] = df.columns[idx]

    # "If Away Team it states: @. If home team it says nothing"
    idx = find_col(norm_cols, ["if_away_team_it_states", "if_away_team_it_states_if_home_team_it_says_nothing"], contains_ok=True)
    if idx is not None: mapper["home_away_flag"] = df.columns[idx]

    idx = find_col(norm_cols, ["opponent"])
    if idx is not None: mapper["opponent"] = df.columns[idx]

    idx = find_col(norm_cols, ["over_under_line"])
    if idx is not None: mapper["ou_line"] = df.columns[idx]

    # we renamed duplicates to a single "over_under_result" above
    if "over_under_result" in df.columns:
        mapper["ou_result"] = "over_under_result"

    idx = find_col(norm_cols, ["spread"])
    if idx is not None: mapper["spread"] = df.columns[idx]

    idx = find_col(norm_cols, ["did_they_cover_the_spread"], contains_ok=True)
    if idx is not None: mapper["cover_result"] = df.columns[idx]

    out = pd.DataFrame()
    for k, col in mapper.items():
        out[k] = df[col]

    # Normalize types
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "week" in out.columns:
        out["week"] = pd.to_numeric(out["week"], errors="coerce")
    if "ou_line" in out.columns:
        out["ou_line"] = pd.to_numeric(out["ou_line"], errors="coerce")
    if "spread" in out.columns:
        out["spread"] = pd.to_numeric(out["spread"], errors="coerce")

    # Flags
    if "home_away_flag" in out.columns:
        out["is_away"] = out["home_away_flag"].fillna("").astype(str).str.contains("@")
    else:
        out["is_away"] = pd.NA

    # Team/opponent keys
    out["team_key"] = out["team"].apply(team_key) if "team" in out.columns else ""
    if "opponent" in out.columns:
        out["opponent_key"] = out["opponent"].apply(team_key)

    # Parse OU + cover result to consistent labels
    if "ou_result" in out.columns:
        out["ou_result_norm"] = out["ou_result"].apply(parse_yes_no_push)
    if "cover_result" in out.columns:
        out["cover_result_norm"] = out["cover_result"].apply(parse_yes_no_push)

    return out

# =========================
# Model logic (existing)
# =========================
def avg_scoring(df: pd.DataFrame, team_label: str):
    scored_home = df.loc[df["home_team"] == team_label, "home_score"].mean()
    scored_away = df.loc[df["away_team"] == team_label, "away_score"].mean()
    allowed_home = df.loc[df["home_team"] == team_label, "away_score"].mean()
    allowed_away = df.loc[df["away_team"] == team_label, "home_score"].mean()
    return np.nanmean([scored_home, scored_away]), np.nanmean([allowed_home, allowed_away])

def predict_scores(df: pd.DataFrame, team_label: str, opponent_label: str):
    team_avg_scored, team_avg_allowed = avg_scoring(df, team_label)
    opp_avg_scored, opp_avg_allowed = avg_scoring(df, opponent_label)
    raw_team_pts = (team_avg_scored + opp_avg_allowed) / 2
    raw_opp_pts = (opp_avg_scored + team_avg_allowed) / 2
    league_avg_pts = df[["home_score", "away_score"]].stack().mean()
    cal_factor = 22.3 / league_avg_pts if not np.isnan(league_avg_pts) and league_avg_pts > 0 else 1.0
    team_pts = float(raw_team_pts * cal_factor) if pd.notna(raw_team_pts) else 22.3
    opp_pts = float(raw_opp_pts * cal_factor) if pd.notna(raw_opp_pts) else 22.3
    return team_pts, opp_pts

# ===== Prop helpers (existing) =====
def find_player_in(df: pd.DataFrame, player_name: str):
    if "player" not in df.columns: return None
    mask = df["player"].astype(str).str.lower() == str(player_name).lower()
    return df[mask].copy() if mask.any() else None

def detect_stat_col(df: pd.DataFrame, prop: str):
    cols = list(df.columns); norm = [normalize_header(c) for c in cols]
    mapping = {
        "rushing_yards": ["rushing_yards_total","rushing_yards_per_game"],
        "receiving_yards": ["receiving_yards_total","receiving_yards_per_game"],
        "passing_yards": ["passing_yards_total","passing_yards_per_game"],
        "receptions": ["receiving_receptions_total"],
        "targets": ["receiving_targets_total"],
        "carries": ["rushing_attempts_total","rushing_carries_per_game"],
    }
    pri = mapping.get(prop, [])
    for cand in pri:
        if cand in norm: return cols[norm.index(cand)]
    for i, c in enumerate(norm):
        if prop.split("_")[0] in c and ("per_game" in c or "total" in c):
            return cols[i]
    return None

def pick_def_df(prop: str, pos: str, d_qb, d_rb, d_wr, d_te):
    if prop == "passing_yards": return d_qb
    if prop in ["rushing_yards","carries"]: return d_rb if pos != "qb" else d_qb
    if prop in ["receiving_yards","receptions","targets"]:
        if pos == "te": return d_te
        if pos == "rb": return d_rb
        return d_wr
    return None

def detect_def_col(def_df: pd.DataFrame, prop: str):
    cols = list(def_df.columns); norm = [normalize_header(c) for c in cols]
    prefs = []
    if prop in ["rushing_yards","carries"]:
        prefs = ["rushing_yards_allowed_total","rushing_yards_allowed"]
    elif prop in ["receiving_yards","receptions","targets"]:
        prefs = ["receiving_yards_allowed_total","receiving_yards_allowed"]
    elif prop == "passing_yards":
        prefs = ["passing_yards_allowed_total","passing_yards_allowed"]
    for cand in prefs:
        if cand in norm: return cols[norm.index(cand)]
    for i, nc in enumerate(norm):
        if "allowed" in nc: return cols[i]
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
    opponent_key_override: str = None
):
    def pick_player_df(prop):
        if prop in ["receiving_yards","receptions","targets"]: return p_rec, "wr"
        if prop in ["rushing_yards","carries"]: return p_rush, "rb"
        if prop == "passing_yards": return p_pass, "qb"
        return p_rec, "wr"

    if selected_prop == "anytime_td":
        rec_row = find_player_in(p_rec, player_name)
        rush_row = find_player_in(p_rush, player_name)
        total_tds = 0.0; total_games = 0.0
        for df_ in [rec_row, rush_row]:
            if df_ is not None and not df_.empty:
                td_cols = [c for c in df_.columns if "td" in c and "allowed" not in c]
                games_col = "games_played" if "games_played" in df_.columns else None
                if td_cols and games_col:
                    tds = sum(float(df_.iloc[0][col]) for col in td_cols if pd.notna(df_.iloc[0][col]))
                    total_tds += tds
                    total_games = max(total_games, float(df_.iloc[0][games_col]))
        if total_games == 0:
            return {"error": "No touchdown data found for this player."}
        def_dfs = [d_rb.copy(), d_wr.copy(), d_te.copy()]
        for d in def_dfs:
            if "games_played" not in d.columns: d["games_played"] = 1
            td_cols = [c for c in d.columns if "td" in c and "allowed" in c]
            if len(td_cols) == 0:
                d["tds_pg"] = np.nan
            else:
                d["tds_pg"] = d[td_cols].sum(axis=1) / d["games_played"].replace(0, np.nan)
            if "team_key" not in d.columns: d["team_key"] = d["team"].apply(team_key)
        league_td_pg = np.nanmean([d["tds_pg"].mean() for d in def_dfs if "tds_pg" in d.columns])
        player_team_key = None
        for df_ in [p_rec, p_rush, p_pass]:
            row_ = find_player_in(df_, player_name)
            if row_ is not None and not row_.empty:
                tk = row_.iloc[0].get("team_key", "")
                if tk: player_team_key = tk; break
        if not player_team_key: player_team_key = selected_team_key
        opp_key_for_player = opponent_key_override if opponent_key_override else (opponent_key if player_team_key == selected_team_key else selected_team_key)
        opp_td_list = []
        for d in def_dfs:
            mask = d["team_key"] == opp_key_for_player
            val = d.loc[mask, "tds_pg"].mean()
            opp_td_list.append(val)
        opp_td_pg = np.nanmean(opp_td_list)
        if np.isnan(opp_td_pg) or league_td_pg is None or np.isnan(league_td_pg) or league_td_pg <= 0:
            adj_factor = 1.0
        else:
            adj_factor = opp_td_pg / league_td_pg
        adj_td_rate = (total_tds / total_games) * adj_factor
        prob_anytime = 1 - np.exp(-adj_td_rate)
        prob_anytime = float(np.clip(prob_anytime, 0.0, 1.0))
        return {"prob_anytime": prob_anytime, "adj_rate": adj_td_rate, "player_rate": (total_tds/total_games)}

    player_df_source, fallback_pos = pick_player_df(selected_prop)
    this_player_df = find_player_in(player_df_source, player_name)
    if this_player_df is None or this_player_df.empty:
        return {"error": "Player not found in the selected stat table."}

    player_pos = this_player_df.iloc[0].get("position", fallback_pos)
    stat_col = detect_stat_col(this_player_df, selected_prop)
    if not stat_col:
        return {"error": "No matching stat column found for this prop."}

    season_val = float(this_player_df.iloc[0][stat_col]) if pd.notna(this_player_df.iloc[0][stat_col]) else 0.0
    games_played = float(this_player_df.iloc[0].get("games_played", 1)) or 1.0
    player_pg = season_val / games_played if games_played > 0 else 0.0

    def_df = pick_def_df(selected_prop, player_pos, d_qb, d_rb, d_wr, d_te)
    def_col = detect_def_col(def_df, selected_prop) if def_df is not None else None

    player_team_key = str(this_player_df.iloc[0].get("team_key", "")).strip()
    opp_key_for_player = opponent_key_override if opponent_key_override else (opponent_key if player_team_key == selected_team_key else selected_team_key)

    opp_allowed_pg = None; league_allowed_pg = None
    if def_df is not None and def_col is not None:
        if "team_key" not in def_df.columns:
            def_df["team_key"] = def_df["team"].apply(team_key)
        if "games_played" in def_df.columns:
            league_allowed_pg = (def_df[def_col] / def_df["games_played"].replace(0, np.nan)).mean()
        else:
            league_allowed_pg = def_df[def_col].mean()
        opp_row = def_df[def_df["team_key"] == opp_key_for_player]
        if not opp_row.empty:
            if "games_played" in opp_row.columns and float(opp_row.iloc[0]["games_played"]) > 0:
                opp_allowed_pg = float(opp_row.iloc[0][def_col]) / float(opp_row.iloc[0]["games_played"])
            else:
                opp_allowed_pg = float(opp_row.iloc[0][def_col])
        else:
            opp_allowed_pg = league_allowed_pg

    adj_factor = (opp_allowed_pg / league_allowed_pg) if (league_allowed_pg and league_allowed_pg > 0) else 1.0
    predicted_pg = player_pg * adj_factor

    stdev = max(3.0, predicted_pg * 0.35)
    z = (line_val - predicted_pg) / stdev
    prob_over = float(np.clip(1 - norm.cdf(z), 0.001, 0.999))
    prob_under = float(np.clip(norm.cdf(z), 0.001, 0.999))

    return {
        "predicted_pg": predicted_pg,
        "prob_over": prob_over,
        "prob_under": prob_under,
        "player_pg": player_pg,
        "season_total": season_val,
        "games_played": games_played
    }

# Odds + market helpers (existing)
def american_to_decimal(odds: float) -> float:
    try: o = float(odds)
    except Exception: return np.nan
    return 1 + (o / 100.0) if o > 0 else 1 + (100.0 / abs(o))

def decimal_to_american(dec: float) -> float:
    if dec <= 1: return np.nan
    if dec >= 2: return round((dec - 1) * 100)
    return round(-100 / (dec - 1))

def prob_to_decimal(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1-1e-6))
    return 1.0 / (1.0 - p + 1e-12)

def prob_to_american(p: float) -> float:
    dec = prob_to_decimal(p)
    return decimal_to_american(dec)

def prob_total_over_under(scores_df: pd.DataFrame, home: str, away: str, line_total: float):
    home_pts, away_pts = predict_scores(scores_df, home, away)
    pred_total = home_pts + away_pts
    stdev_total = max(6.0, pred_total * 0.18)
    z = (line_total - pred_total) / stdev_total
    p_over = float(np.clip(1 - norm.cdf(z), 0.001, 0.999))
    p_under = float(np.clip(norm.cdf(z), 0.001, 0.999))
    return pred_total, p_over, p_under

def prob_spread_cover(scores_df: pd.DataFrame, home: str, away: str, home_spread: float, side: str):
    home_pts, away_pts = predict_scores(scores_df, home, away)
    pred_margin = home_pts - away_pts  # home - away
    stdev_margin = max(5.0, abs(pred_margin) * 0.9 + 6.0)
    line_margin = -home_spread
    z = (line_margin - pred_margin) / stdev_margin
    p_home_cover = float(np.clip(1 - norm.cdf(z), 0.001, 0.999))
    p_away_cover = 1.0 - p_home_cover
    return (pred_margin, p_home_cover) if side == "home" else (pred_margin, p_away_cover)

# =========================
# UI – Main
# =========================
st.title("🏈 The Official un-official NFL Betting Model")

with st.expander("📘 How This Model Works", expanded=False):
    st.markdown("""
**Welcome to the New Model Dashboard — Where Data Picks the Winners.**

We project team scoring using historical efficiency and calibrate to league scoring,
then compare to Vegas lines to find edges on totals and spreads. Player props layer
in opponent-allowed adjustments.

Use the **Team Game Log & Trends** section to see historical O/U & cover rates for any team—
overall and by day (Sun/Mon/Thu).
""")

with st.expander("📱 Add This App to Your Home Screen (Recommended)", expanded=False):
    st.markdown("""
**iPhone/iPad:** Share → Add to Home Screen  
**Android (Chrome):** ⋮ → Add to Home Screen
""")

scores_df = load_scores()
if scores_df.empty:
    st.error("Could not load NFL game data.")
    st.stop()

player_data = load_all_player_dfs()
p_rec, p_rush, p_pass = player_data["player_receiving"], player_data["player_rushing"], player_data["player_passing"]
d_rb, d_qb, d_wr, d_te = player_data["def_rb"], player_data["def_qb"], player_data["def_wr"], player_data["def_te"]

team_log_df = load_team_game_log(TEAM_GAME_LOG_URL)

section_names = [
    "1) Game Selection + Prediction",
    "2) Top Edges This Week",
    "3) Player Props",
    "4) Parlay Builder (Players + Game Markets)",
    "5) Team Game Log & Trends (NEW)"
]
selected_section = st.selectbox("Jump to section", section_names, index=0, help="Pick a section to open")

# -------------------------
# Section 1: Game Selection + Prediction
# -------------------------
with st.expander("1) Game Selection + Prediction", expanded=(selected_section == section_names[0])):
    st.subheader("Select Game")
    cols = st.columns([1, 1, 2])
    with cols[0]:
        week_list = sorted(scores_df["week"].dropna().unique())
        selected_week = st.selectbox("Week", week_list, key="sec1_week")
    with cols[1]:
        teams_in_week = sorted(
            set(scores_df.loc[scores_df["week"] == selected_week, "home_team"].dropna().unique())
            | set(scores_df.loc[scores_df["week"] == selected_week, "away_team"].dropna().unique())
        )
        selected_team = st.selectbox("Team", teams_in_week, key="sec1_team")

    game_row = scores_df[
        ((scores_df["home_team"] == selected_team) | (scores_df["away_team"] == selected_team))
        & (scores_df["week"] == selected_week)
    ]
    if game_row.empty:
        st.warning("No game found for that team/week.")
    else:
        g = game_row.iloc[0]
        opponent = g["away_team"] if g["home_team"] == selected_team else g["home_team"]

        with cols[2]:
            st.markdown(f"**Matchup:** {selected_team} vs {opponent}")

        default_ou = float(g["over_under"]) if pd.notna(g.get("over_under", np.nan)) else 45.0
        default_home_spread = float(g["home_spread"]) if pd.notna(g.get("home_spread", np.nan)) else 0.0

        cL, cR = st.columns(2)
        with cL:
            over_under = st.number_input("Over/Under (Vegas or yours)", value=default_ou, step=0.5, key="sec1_ou")
        with cR:
            home_spread_val = st.number_input("Home-based Spread (home team perspective)", value=default_home_spread, step=0.5, key="sec1_spread")

        st.subheader("Game Prediction (Vegas-Calibrated)")
        team_pts, opp_pts = predict_scores(scores_df, selected_team, opponent)
        total_pred = team_pts + opp_pts
        margin = team_pts - opp_pts
        total_diff = total_pred - over_under
        spread_diff = margin - (-home_spread_val)

        mrow1 = st.columns(2)
        mrow1[0].metric(f"{selected_team} Predicted", f"{team_pts:.1f} pts")
        mrow1[1].metric(f"{opponent} Predicted", f"{opp_pts:.1f} pts")
        mrow2 = st.columns(2)
        mrow2[0].metric("Predicted Total", f"{total_pred:.1f}", f"{total_diff:+.1f} vs O/U")
        mrow2[1].metric("Predicted Margin", f"{margin:+.1f}", f"{spread_diff:+.1f} vs Home Spread")

        fig_total = px.bar(x=["Predicted Total", "Vegas O/U"], y=[total_pred, over_under], title="Predicted Total vs O/U")
        st.plotly_chart(fig_total, use_container_width=True)
        fig_margin = px.bar(x=["Predicted Margin", "Home Spread Target"], y=[margin, -home_spread_val], title="Predicted Margin vs Home Spread")
        st.plotly_chart(fig_margin, use_container_width=True)

# -------------------------
# Section 2: Top Edges This Week
# -------------------------
with st.expander("2) Top Edges This Week", expanded=(selected_section == section_names[1])):
    selected_week_for_edges = st.session_state.get('sec1_week', sorted(scores_df["week"].dropna().unique())[0])
    st.caption(f"Week shown: {selected_week_for_edges} — Legend: 🟩 strong | 🟨 lean | 🟥 pass")

    wk = scores_df[scores_df["week"] == selected_week_for_edges].copy()
    if "home_spread" not in wk.columns:
        wk["home_spread"] = wk.apply(compute_home_spread, axis=1)

    def strength_badge(edge_val):
        if pd.isna(edge_val): return "⬜"
        a = abs(edge_val)
        if a >= 4: return "🟩"
        elif a >= 2: return "🟨"
        else: return "🟥"

    rows = []
    for _, r in wk.iterrows():
        h, a = r.get("home_team"), r.get("away_team")
        if pd.isna(h) or pd.isna(a): continue
        h_pts, a_pts = predict_scores(scores_df, h, a)
        tot_pred = h_pts + a_pts
        mar_pred = h_pts - a_pts

        ou = float(r.get("over_under")) if pd.notna(r.get("over_under", np.nan)) else np.nan
        home_spread = float(r.get("home_spread")) if pd.notna(r.get("home_spread", np.nan)) else np.nan

        total_edge = np.nan if pd.isna(ou) else (tot_pred - ou)
        spread_edge = np.nan if pd.isna(home_spread) else (mar_pred - (-home_spread))

        if pd.isna(total_edge):
            total_pick = ""
        else:
            direction = "OVER" if total_edge > 0 else "UNDER"
            total_pick = f"{strength_badge(total_edge)} {direction}"

        if pd.isna(spread_edge) or pd.isna(home_spread):
            spread_pick = ""
        else:
            home_covers = mar_pred > -home_spread
            pick_text = f"{h} {home_spread:+.1f}" if home_covers else f"{a} {(-home_spread):+.1f}"
            spread_pick = f"{strength_badge(spread_edge)} {pick_text}"

        rows.append({
            "Matchup": f"{a} @ {h}",
            "Pred Total": round(tot_pred, 1),
            "O/U": ou if not pd.isna(ou) else "",
            "Total Edge (pts)": None if pd.isna(total_edge) else round(total_edge, 1),
            "Total Pick": total_pick,
            "Pred Margin": round(mar_pred, 1),
            "Home Spread": home_spread if not pd.isna(home_spread) else "",
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
        display_cols = ["Matchup","Pred Total","O/U","Total Edge (pts)","Total Pick","Pred Margin","Home Spread","Spread Edge (pts)","Spread Pick"]
        st.dataframe(edges_df[display_cols], use_container_width=True)
    else:
        st.info("No games found for this week.")

# -------------------------
# Section 3: Player Props
# -------------------------
with st.expander("3) Player Props", expanded=(selected_section == section_names[2])):
    if 'sec1_team' not in st.session_state or 'sec1_week' not in st.session_state:
        st.info("Pick a game in Section 1 first.")
    else:
        selected_team = st.session_state['sec1_team']
        selected_week = st.session_state['sec1_week']
        game_row = scores_df[
            ((scores_df["home_team"] == selected_team) | (scores_df["away_team"] == selected_team))
            & (scores_df["week"] == selected_week)
        ]
        if game_row.empty:
            st.warning("No game found for that team/week.")
        else:
            g = game_row.iloc[0]
            opponent = g["away_team"] if g["home_team"] == selected_team else g["home_team"]

            def players_for_team(df, team_name_or_label):
                key = team_key(team_name_or_label)
                if "team_key" not in df.columns or "player" not in df.columns:
                    return []
                mask = df["team_key"] == key
                return list(df.loc[mask, "player"].dropna().unique())

            team_players = set(
                players_for_team(p_rec, selected_team) +
                players_for_team(p_rush, selected_team) +
                players_for_team(p_pass, selected_team)
            )
            opp_players = set(
                players_for_team(p_rec, opponent) +
                players_for_team(p_rush, opponent) +
                players_for_team(p_pass, opponent)
            )
            both_players = sorted(team_players.union(opp_players))

            if not both_players:
                st.info(f"No players found for this matchup. Keys — You: **{team_key(selected_team)}**, Opp: **{team_key(opponent)}**.")

            c1, c2, c3 = st.columns([2, 1.2, 1.2])
            with c1:
                player_name = st.selectbox("Select Player", [""] + both_players, key="player_pick_props")
            with c2:
                prop_choices = ["passing_yards","rushing_yards","receiving_yards","receptions","targets","carries","anytime_td"]
                selected_prop = st.selectbox("Prop Type", prop_choices, index=2, key="prop_type_props")
            with c3:
                default_line = 50.0 if selected_prop != "anytime_td" else 0.0
                line_val = st.number_input("Sportsbook Line", value=float(default_line), key="prop_line") if selected_prop != "anytime_td" else 0.0

            if player_name:
                res = prop_prediction_and_probs(
                    player_name=player_name,
                    selected_prop=selected_prop,
                    line_val=line_val,
                    selected_team_key=team_key(selected_team),
                    opponent_key=team_key(opponent),
                    p_rec=p_rec, p_rush=p_rush, p_pass=p_pass,
                    d_qb=d_qb, d_rb=d_rb, d_wr=d_wr, d_te=d_te
                )

                if "error" in res:
                    st.warning(res["error"])
                elif selected_prop == "anytime_td":
                    st.subheader("Anytime TD Probability")
                    st.write(f"Estimated Anytime TD Probability: **{res['prob_anytime']*100:.1f}%**")
                    bar_df = pd.DataFrame({"Category": ["Player TDs/Game", "Adj. vs Opponent"], "TDs/Game": [res["player_rate"], res["adj_rate"]]})
                    st.plotly_chart(px.bar(bar_df, x="Category", y="TDs/Game", title=f"{player_name} – Anytime TD"), use_container_width=True)
                else:
                    st.subheader(selected_prop.replace("_", " ").title())
                    st.write(f"**Season Total:** {res['season_total']:.1f}")
                    st.write(f"**Games Played:** {res['games_played']:.0f}")
                    st.write(f"**Per Game (season):** {res['player_pg']:.2f}")
                    st.write(f"**Adjusted prediction (this game):** {res['predicted_pg']:.2f}")
                    st.write(f"**Line:** {line_val:.1f}")
                    st.write(f"**Probability of OVER:** {res['prob_over']*100:.1f}%")
                    st.write(f"**Probability of UNDER:** {res['prob_under']*100:.1f}%")
                    st.plotly_chart(
                        px.bar(x=["Predicted (this game)", "Line"], y=[res['predicted_pg'], line_val], title=f"{player_name} – {selected_prop.replace('_', ' ').title()}"),
                        use_container_width=True
                    )
            else:
                st.info("Select a player to evaluate props.")

# -------------------------
# Section 4: Parlay Builder
# -------------------------
with st.expander("4) Parlay Builder (Players + Game Markets)", expanded=(selected_section == section_names[3])):
    if "parlay_legs" not in st.session_state:
        st.session_state.parlay_legs = []

    def unique_players(*dfs):
        names = []
        for df in dfs:
            if "player" in df.columns:
                names.extend(list(df["player"].dropna().astype(str).unique()))
        return sorted(pd.unique(names))

    all_players = unique_players(p_rec, p_rush, p_pass)
    full_team_names = sorted(list(CODE_TO_FULLNAME.values()))

    st.markdown("**Add Player Prop Leg**")
    a1, a2, a3, a4, a5 = st.columns([2.2, 1.6, 1.2, 1.6, 1.2])
    with a1:
        pb_player = st.selectbox("Player", [""] + all_players, key="pb_any_player")
    with a2:
        pb_prop = st.selectbox("Prop", ["passing_yards","rushing_yards","receiving_yards","receptions","targets","carries","anytime_td"], key="pb_any_prop")
    with a3:
        if pb_prop == "anytime_td":
            pb_line = 0.0; pb_side = "yes"
            st.text_input("Line", "—", disabled=True, key="pb_any_line_disabled")
        else:
            pb_line = st.number_input("Line", value=50.0, step=0.5, key="pb_any_line")
            pb_side = st.selectbox("Side", ["over","under"], key="pb_any_side")
    with a4:
        pb_opp_full = st.selectbox("Opponent (Full Name)", full_team_names, key="pb_any_opp_full")
        pb_opp_key = FULLNAME_TO_CODE.get(pb_opp_full, "")
    with a5:
        if st.button("➕ Add Player Leg", use_container_width=True, key="pb_any_add"):
            if not pb_player:
                st.warning("Pick a player first.")
            elif not pb_opp_key:
                st.warning("Pick an opponent team.")
            else:
                res = prop_prediction_and_probs(
                    player_name=pb_player,
                    selected_prop=pb_prop,
                    line_val=pb_line,
                    selected_team_key="SEL",
                    opponent_key="OPP",
                    p_rec=p_rec, p_rush=p_rush, p_pass=p_pass,
                    d_qb=d_qb, d_rb=d_rb, d_wr=d_wr, d_te=d_te,
                    opponent_key_override=pb_opp_key
                )
                if "error" in res:
                    st.warning(res["error"])
                else:
                    if pb_prop == "anytime_td":
                        prob = float(res["prob_anytime"])
                        label = f"{pb_player} Anytime TD vs {pb_opp_full}"
                    else:
                        prob = float(res["prob_over"] if pb_side == "over" else res["prob_under"])
                        label = f"{pb_player} {pb_prop.replace('_',' ').title()} {pb_side.title()} {pb_line} vs {pb_opp_full}"
                    st.session_state.parlay_legs.append({"kind": "player", "label": label, "prob": prob})
                    st.rerun()

    st.markdown("---")

    st.markdown("**Add Game Market Leg**")
    g1, g2, g3, g4, g5, g6 = st.columns([1.0, 2.2, 1.6, 1.2, 1.2, 1.2])
    with g1:
        week_for_market = st.selectbox("Week", sorted(scores_df["week"].dropna().unique()), key="gm_week")
    with g2:
        wk_df = scores_df[scores_df["week"] == week_for_market].copy()
        if "home_spread" not in wk_df.columns:
            wk_df["home_spread"] = wk_df.apply(compute_home_spread, axis=1)

        matchups, meta, home_spreads_for_match, ous_for_match = [], [], [], []
        for _, row in wk_df.iterrows():
            h = row.get("home_team"); a = row.get("away_team")
            if pd.isna(h) or pd.isna(a): continue
            matchups.append(f"{a} @ {h}")
            meta.append((h, a))
            home_spreads_for_match.append(float(row.get("home_spread")) if pd.notna(row.get("home_spread", np.nan)) else 0.0)
            ous_for_match.append(float(row.get("over_under")) if pd.notna(row.get("over_under", np.nan)) else 45.0)

        gm_match = st.selectbox("Matchup", matchups, key="gm_matchup")
        idx = matchups.index(gm_match) if gm_match in matchups else -1
        home_team = meta[idx][0] if idx >= 0 else None
        away_team = meta[idx][1] if idx >= 0 else None
        default_home_sp = home_spreads_for_match[idx] if idx >= 0 else 0.0
        default_ou = ous_for_match[idx] if idx >= 0 else 45.0
    with g3:
        gm_market = st.selectbox("Market", ["Total","Spread"], key="gm_market")
    with g4:
        if gm_market == "Total":
            gm_total = st.number_input("O/U Line", value=float(default_ou), step=0.5, key="gm_total_line")
            gm_side = st.selectbox("Side", ["over","under"], key="gm_total_side")
        else:
            gm_spread = st.number_input("Home-based Spread", value=float(default_home_sp), step=0.5, key="gm_spread_line")
            gm_side_spread = st.selectbox("Side", ["home","away"], key="gm_spread_side")
    with g5:
        if gm_market == "Total" and home_team and away_team:
            _, p_over, p_under = prob_total_over_under(scores_df, home_team, away_team, gm_total)
            prev_prob = p_over if gm_side == "over" else p_under
            st.metric("Model Pr.", f"{prev_prob*100:.1f}%")
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
                    if gm_side_spread == "home":
                        side_text = f"{home_team} {gm_spread:+.1f}"
                    else:
                        side_text = f"{away_team} {(-gm_spread):+.1f}"
                    label = f"{away_team} @ {home_team} Spread {side_text}"
                st.session_state.parlay_legs.append({"kind": "game", "label": label, "prob": prob})
                st.rerun()

    if st.session_state.parlay_legs:
        st.subheader("Your Legs")
        for i, leg in enumerate(st.session_state.parlay_legs):
            c1, c2 = st.columns([8, 1])
            c1.markdown(f"• **{leg.get('label','Leg')}** — Model Pr: **{leg.get('prob',0.0)*100:.1f}%**")
            if c2.button("🗑 Remove", key=f"rm_leg_{i}"):
                st.session_state.parlay_legs.pop(i); st.rerun()

        probs = [float(leg.get("prob", 0.0)) for leg in st.session_state.parlay_legs]
        parlay_hit_prob = float(np.prod(probs)) if probs else 0.0
        model_dec_odds = prob_to_decimal(parlay_hit_prob)
        model_am_odds = prob_to_american(parlay_hit_prob)

        st.markdown("---")
        b1, b2, b3 = st.columns([1.2, 1, 1])
        with b1:
            book_total_american = st.text_input("Book Total Parlay Odds (American, e.g. +650)", value="", key="book_any_odds")
        with b2:
            stake = st.number_input("Stake ($)", value=100.0, step=10.0, min_value=0.0, key="book_any_stake")
        with b3:
            st.metric("Model Parlay Prob.", f"{parlay_hit_prob*100:.1f}%")

        if book_total_american.strip():
            try:
                text = book_total_american.strip()
                book_am = float(text)
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
        st.info("Add legs above to build your parlay. Mix player props and game markets. We'll multiply probabilities for the parlay hit rate.")

# -------------------------
# Section 5: Team Game Log & Trends (NEW)
# -------------------------
with st.expander("5) Team Game Log & Trends (NEW)", expanded=(selected_section == section_names[4])):
    if team_log_df is None or team_log_df.empty:
        st.info("No team game log data found.")
    else:
        # Team selector (abbreviations)
        teams = sorted([t for t in team_log_df["team"].dropna().unique()] if "team" in team_log_df.columns else [])
        if not teams:
            st.info("Team column not present in sheet. Check your sharing/export.")
        else:
            tcol1, tcol2, tcol3 = st.columns([1.2, 1, 1])
            with tcol1:
                pick_team = st.selectbox("Team (abbr.)", teams, key="tgl_team")
            with tcol2:
                day_filter = st.selectbox("Day filter", ["All","Sunday","Monday","Thursday"], index=0, key="tgl_day")
            with tcol3:
                last_n = st.number_input("Last N games (0 = all)", value=0, min_value=0, step=1, key="tgl_lastn")

            sub = team_log_df.copy()
            sub = sub[sub["team"] == pick_team]

            # Day filter (your sheet uses full day names like Sunday/Monday/Thursday)
            if day_filter != "All" and "day" in sub.columns:
                sub = sub[sub["day"].astype(str).str.lower() == day_filter.lower()]

            # Sort by date then limit to last N
            if "date" in sub.columns:
                sub = sub.sort_values("date")
            if last_n and last_n > 0:
                sub = sub.tail(last_n)

            # Metrics
            ou_over = ou_under = ou_total = 0
            if "ou_result_norm" in sub.columns:
                vals = sub["ou_result_norm"].dropna().astype(str).str.lower()
                ou_over = (vals == "over").sum()
                ou_under = (vals == "under").sum()
                ou_total = (vals.isin(["over","under"])).sum()  # exclude pushes
            cover_yes = cover_no = cover_total = 0
            if "cover_result_norm" in sub.columns:
                cvals = sub["cover_result_norm"].dropna().astype(str).str.lower()
                cover_yes = (cvals == "covered").sum()
                cover_no = (cvals == "not_covered").sum()
                cover_total = (cvals.isin(["covered","not_covered"])).sum()

            m1, m2, m3 = st.columns(3)
            with m1:
                pct_over = (ou_over / ou_total * 100.0) if ou_total else 0.0
                st.metric("O/U Over Hit %", f"{pct_over:.1f}%", f"{ou_over}/{ou_total} (excl. pushes)")
            with m2:
                pct_under = (ou_under / ou_total * 100.0) if ou_total else 0.0
                st.metric("O/U Under Hit %", f"{pct_under:.1f}%", f"{ou_under}/{ou_total} (excl. pushes)")
            with m3:
                pct_cover = (cover_yes / cover_total * 100.0) if cover_total else 0.0
                st.metric("Spread Cover %", f"{pct_cover:.1f}%", f"{cover_yes}/{cover_total} (excl. pushes)")

            # By-day breakdown chart
            if "day" in team_log_df.columns and "ou_result_norm" in team_log_df.columns:
                day_data = team_log_df[team_log_df["team"] == pick_team].copy()
                day_data = day_data[day_data["ou_result_norm"].isin(["over","under"])]
                if not day_data.empty:
                    grp = day_data.groupby(["day","ou_result_norm"]).size().reset_index(name="count")
                    fig = px.bar(grp, x="day", y="count", color="ou_result_norm",
                                 title=f"{pick_team} — O/U Results by Day",
                                 barmode="group")
                    st.plotly_chart(fig, use_container_width=True)

            # Recent games table
            show_cols = []
            label_map = []
            # build nice column selections if present
            if "date" in sub.columns: show_cols.append("date"); label_map.append("Date")
            if "day" in sub.columns: show_cols.append("day"); label_map.append("Day")
            if "opponent" in sub.columns: show_cols.append("opponent"); label_map.append("Opponent")
            if "is_away" in sub.columns: show_cols.append("is_away"); label_map.append("Away?")
            if "ou_line" in sub.columns: show_cols.append("ou_line"); label_map.append("O/U Line")
            if "ou_result" in sub.columns: show_cols.append("ou_result"); label_map.append("O/U Result")
            if "spread" in sub.columns: show_cols.append("spread"); label_map.append("Spread")
            if "cover_result" in sub.columns: show_cols.append("cover_result"); label_map.append("Cover Result")
            if not show_cols:
                st.info("Table columns not found in sheet. Check headers.")
            else:
                st.subheader("Recent Games")
                pretty = sub[show_cols].copy()
                # format columns
                if "is_away" in pretty.columns:
                    pretty["is_away"] = pretty["is_away"].map({True:"@", False:"home"}).fillna("")
                if "date" in pretty.columns:
                    pretty["date"] = pd.to_datetime(pretty["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                # rename headers
                pretty.columns = label_map
                st.dataframe(pretty, use_container_width=True)
