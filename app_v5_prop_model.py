# New Model Dashboard (single-file, hybrid: PFR scraper + Google Sheets fallback)
# -------------------------------------------------------------------------------
# How to run locally:
#   1) pip install streamlit pandas numpy scipy plotly requests beautifulsoup4 lxml
#   2) streamlit run New_Model_Dashboard.py
#
# What's inside:
#   - Robust Pro-Football-Reference (PFR) scraper for 2025 season tables you listed
#     * Works with PFR's comment-wrapped tables (no JS needed)
#     * Retries with randomized user-agent to reduce 403s
#   - Local CSV cache at data/pfr_2025/
#   - App uses PFR CSVs if present; otherwise falls back to your Google Sheets
#   - Sections: Game Prediction, Top Edges, Player Props, Parlay Builder
#
# Notes for spreads:
#   - Your sheet columns are exactly: home_team, away_team, favored_team, spread, over_under
#   - We normalize the line to **HOME-BASED** format:
#       home_spread = -abs(spread) if home is favored
#       home_spread = +abs(spread) if away is favored
#   - "Spread Pick" prints the team *with the same signed line you’d see at the book*
#     (e.g., "Broncos -9.5" if home is favored and model likes home).
#
import os
import io
import re
import math
import json
import time
import random
import string
import shutil
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import streamlit as st
from bs4 import BeautifulSoup, Comment

# =========================
# Scraper (inline)
# =========================
SEASON = 2025
BASE = f"https://www.pro-football-reference.com/years/{SEASON}"
URLS = {
    "fpa_rb": f"{BASE}/fantasy-points-against-RB.htm",
    "fpa_wr": f"{BASE}/fantasy-points-against-WR.htm",
    "fpa_te": f"{BASE}/fantasy-points-against-TE.htm",
    "fpa_qb": f"{BASE}/fantasy-points-against-QB.htm",
    "passing": f"{BASE}/passing.htm",
    "rushing": f"{BASE}/rushing.htm",
    "receiving": f"{BASE}/receiving.htm",
    "opp": f"{BASE}/opp.htm",
}
DEFAULT_OUT_DIR = os.path.join("data", f"pfr_{SEASON}")
os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)

# Rotating user-agents to reduce chance of 403s
UA_LIST = [
    # Common desktop UAs
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

def _headers():
    return {
        "User-Agent": random.choice(UA_LIST),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }

TEAM_NAME_TO_CODE = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR",
    "Miami Dolphins":"MIA","Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO",
    "New York Giants":"NYG","New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT",
    "San Francisco 49ers":"SF","Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN",
    "Washington Commanders":"WAS",
}

def _normalize_header(c: str) -> str:
    c = (c or "").strip()
    c = c.replace("\xa0", " ")
    c = c.lower()
    c = c.replace("%", "pct").replace("/", "_").replace(" ", "_")
    c = re.sub(r"[^0-9a-z_]", "", c)
    c = re.sub(r"_+", "_", c)
    return c

def _http_get(url: str, max_retries: int = 4, backoff: float = 1.0) -> str:
    """
    Robust GET with rotating UAs + small randomized backoff.
    Returns decoded HTML (str). Raises on failure after retries.
    """
    last_err = None
    for i in range(max_retries):
        try:
            r = requests.get(url, headers=_headers(), timeout=30)
            # Some PFR pages return 200 but set anti-scrape; parsing still works via comments.
            if r.status_code == 200:
                # Use text to preserve encoding
                return r.text
            last_err = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last_err = e
        # backoff with jitter
        time.sleep(backoff * (1.0 + random.random()))
    raise RuntimeError(f"Failed GET {url}: {last_err}")

def _read_main_table_from_html(html: str) -> pd.DataFrame:
    """
    PFR frequently wraps tables in HTML comments. This parses both commented and visible tables.
    Prefers the largest table by area.
    """
    soup = BeautifulSoup(html, "lxml")

    tables = []

    # 1) Comment-wrapped tables
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        try:
            frag = BeautifulSoup(c, "lxml")
            t = frag.find("table")
            if t is not None:
                tables.append(pd.read_html(str(t))[0])
        except Exception:
            pass

    # 2) Fallback to visible tables
    if not tables:
        try:
            tables = pd.read_html(html)
        except Exception:
            tables = []

    if not tables:
        raise RuntimeError("No parseable table found in HTML.")

    # pick largest table
    df = max(tables, key=lambda t: t.shape[0] * t.shape[1])

    # Normalize multiindex
    if df.columns.tolist() and hasattr(df.columns, "droplevel"):
        try:
            df.columns = df.columns.droplevel(0)
        except Exception:
            pass

    # Drop any repeated header rows
    if len(df) and isinstance(df.columns[0], str):
        mask_header = df.iloc[:, 0].astype(str) == str(df.columns[0])
        df = df[~mask_header]

    df.columns = [_normalize_header(c) for c in df.columns]
    # Drop unnamed columns
    df = df.loc[:, ~pd.Series(df.columns).astype(str).str.contains("unnamed")]
    return df.reset_index(drop=True)

def _clean_player_name(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s)
    s = re.sub(r"[\*\+]+", "", s)         # star/plus markers
    s = re.sub(r"\s+\(.+\)$", "", s)      # footnote suffix
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def _team_to_code(s: str) -> str:
    return TEAM_NAME_TO_CODE.get(str(s), str(s))

def _coerce_float(s):
    try:
        if s in (None, "", "-", "—"):
            return math.nan
        return float(str(s).replace(",", ""))
    except Exception:
        return math.nan

def scrape_defense_allowed_by_position():
    out = {}
    for pos_key, url in [
        ("RB", URLS["fpa_rb"]), ("WR", URLS["fpa_wr"]),
        ("TE", URLS["fpa_te"]), ("QB", URLS["fpa_qb"])
    ]:
        html = _http_get(url)
        df = _read_main_table_from_html(html)

        if "tm" in df.columns:
            df.rename(columns={"tm": "team"}, inplace=True)
        if "team" not in df.columns:
            df.rename(columns={df.columns[0]: "team"}, inplace=True)

        # remove totals / conference rows
        df = df[~df["team"].astype(str).str.contains("AFC|NFC|League", na=False)].copy()
        df["team_key"] = df["team"].map(_team_to_code)

        if pos_key == "RB":
            candidates = {
                "games_played": ["g"],
                "rushing_yards_allowed": ["rushing_yds", "rush_yds", "yds_rush"],
                "rushing_tds_allowed": ["rushing_td", "rush_td", "td_rush"],
                "receptions_allowed": ["rec", "receptions"],
                "receiving_yards_allowed": ["rec_yds", "receiving_yds", "yds_rec"],
                "receiving_tds_allowed": ["rec_td", "receiving_td", "td_rec"],
            }
        elif pos_key == "WR":
            candidates = {
                "games_played": ["g"],
                "receptions_allowed": ["rec", "receptions"],
                "receiving_yards_allowed": ["rec_yds", "receiving_yds", "yds_rec"],
                "receiving_tds_allowed": ["rec_td", "receiving_td", "td_rec"],
            }
        elif pos_key == "TE":
            candidates = {
                "games_played": ["g"],
                "receptions_allowed": ["rec", "receptions"],
                "receiving_yards_allowed": ["rec_yds", "receiving_yds", "yds_rec"],
                "receiving_tds_allowed": ["rec_td", "receiving_td", "td_rec"],
            }
        else:  # QB
            candidates = {
                "games_played": ["g"],
                "passing_yards_allowed": ["pass_yds", "passing_yds", "yds_pass"],
                "passing_tds_allowed": ["pass_td", "passing_td", "td_pass"],
                "ints_made": ["int", "ints"],
            }

        colmap = {}
        cols = set(df.columns)
        for out_name, choices in candidates.items():
            for c in choices:
                if c in cols:
                    colmap[out_name] = c
                    break

        keep = ["team", "team_key"] + list(colmap.values())
        keep_existing = [c for c in keep if c in df.columns]
        dfo = df[keep_existing].copy()
        inv = {v: k for k, v in colmap.items()}
        dfo.rename(columns=inv, inplace=True)

        for c in dfo.columns:
            if c in ("team", "team_key"): 
                continue
            dfo[c] = dfo[c].apply(_coerce_float)

        if "games_played" in dfo.columns:
            gp = dfo["games_played"].replace(0, pd.NA)
            for metric in [c for c in dfo.columns if c not in ("team", "team_key", "games_played")]:
                dfo[f"{metric}_pg"] = dfo[metric] / gp

        out[pos_key] = dfo.reset_index(drop=True)
    return out

def scrape_players_totals():
    out = {}
    for key in ("passing", "rushing", "receiving"):
        html = _http_get(URLS[key])
        df = _read_main_table_from_html(html)

        if "player" not in df.columns:
            df.rename(columns={df.columns[0]: "player"}, inplace=True)
        if "tm" in df.columns and "team" not in df.columns:
            df.rename(columns={"tm": "team"}, inplace=True)
        if "g" in df.columns and "games_played" not in df.columns:
            df.rename(columns={"g": "games_played"}, inplace=True)

        df["player"] = df["player"].astype(str).map(_clean_player_name)
        df["team"] = df["team"].astype(str)
        df = df[(df["player"] != "") & (~df["team"].isin(["TM", "2TM", "3TM", "4TM"]))].copy()
        df["team_key"] = df["team"].map(lambda t: TEAM_NAME_TO_CODE.get(t, t))

        for c in df.columns:
            if c in ("player", "team", "team_key", "position"):
                continue
            df[c] = df[c].apply(_coerce_float)

        out[key] = df.reset_index(drop=True)
    return out

def scrape_team_opp_summary():
    html = _http_get(URLS["opp"])
    df = _read_main_table_from_html(html)
    if "team" not in df.columns:
        df.rename(columns={df.columns[0]: "team"}, inplace=True)
    df = df[~df["team"].astype(str).str.contains("AFC|NFC|League", na=False)].copy()
    df["team_key"] = df["team"].map(_team_to_code)

    rename = {}
    for src, dst in [
        ("pts", "points_for"), ("points", "points_for"),
        ("pts_opp", "points_against"), ("points_opp", "points_against"),
        ("g", "games_played"),
    ]:
        if src in df.columns:
            rename[src] = dst
    df.rename(columns=rename, inplace=True)

    for c in df.columns:
        if c in ("team", "team_key"):
            continue
        df[c] = df[c].apply(_coerce_float)

    if "games_played" in df.columns:
        gp = df["games_played"].replace(0, pd.NA)
        if "points_for" in df.columns:
            df["pf_pg"] = df["points_for"] / gp
        if "points_against" in df.columns:
            df["pa_pg"] = df["points_against"] / gp

    return df.reset_index(drop=True)

def scrape_all(out_dir: str = DEFAULT_OUT_DIR) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    def_pos = scrape_defense_allowed_by_position()
    def_pos["RB"].to_csv(os.path.join(out_dir, "def_rb.csv"), index=False)
    def_pos["WR"].to_csv(os.path.join(out_dir, "def_wr.csv"), index=False)
    def_pos["TE"].to_csv(os.path.join(out_dir, "def_te.csv"), index=False)
    def_pos["QB"].to_csv(os.path.join(out_dir, "def_qb.csv"), index=False)

    players = scrape_players_totals()
    players["passing"].to_csv(os.path.join(out_dir, "player_passing.csv"), index=False)
    players["rushing"].to_csv(os.path.join(out_dir, "player_rushing.csv"), index=False)
    players["receiving"].to_csv(os.path.join(out_dir, "player_receiving.csv"), index=False)

    opp = scrape_team_opp_summary()
    opp.to_csv(os.path.join(out_dir, "team_opp.csv"), index=False)

    return {
        "def_rb": os.path.join(out_dir, "def_rb.csv"),
        "def_wr": os.path.join(out_dir, "def_wr.csv"),
        "def_te": os.path.join(out_dir, "def_te.csv"),
        "def_qb": os.path.join(out_dir, "def_qb.csv"),
        "player_passing": os.path.join(out_dir, "player_passing.csv"),
        "player_rushing": os.path.join(out_dir, "player_rushing.csv"),
        "player_receiving": os.path.join(out_dir, "player_receiving.csv"),
        "team_opp": os.path.join(out_dir, "team_opp.csv"),
    }

# =========================
# App Config + Fallback Google Sheets (Hybrid)
# =========================
st.set_page_config(page_title="New Model Dashboard", layout="wide")

# Your Google Sheets fallbacks
SCORE_URL = "https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv"
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

TEAM_ALIAS_TO_CODE = {
    "arizona cardinals": "ARI","cardinals": "ARI","arizona": "ARI","ari": "ARI",
    "atlanta falcons": "ATL","falcons": "ATL","atlanta": "ATL","atl": "ATL",
    "baltimore ravens": "BAL","ravens": "BAL","baltimore": "BAL","bal": "BAL",
    "buffalo bills": "BUF","bills": "BUF","buffalo": "BUF","buf": "BUF",
    "carolina panthers": "CAR","panthers": "CAR","carolina": "CAR","car": "CAR",
    "chicago bears": "CHI","bears": "CHI","chicago": "CHI","chi": "CHI",
    "cincinnati bengals": "CIN","bengals": "CIN","cincinnati": "CIN","cin": "CIN",
    "cleveland browns": "CLE","browns": "CLE","cleveland": "CLE","cle": "CLE",
    "dallas cowboys": "DAL","cowboys": "DAL","dallas": "DAL","dal": "DAL",
    "denver broncos": "DEN","broncos": "DEN","denver": "DEN","den": "DEN",
    "detroit lions": "DET","lions": "DET","detroit": "DET","det": "DET",
    "green bay packers": "GB","packers": "GB","green bay": "GB","gb": "GB",
    "houston texans": "HOU","texans": "HOU","houston": "HOU","hou": "HOU",
    "indianapolis colts": "IND","colts": "IND","indianapolis": "IND","ind": "IND",
    "jacksonville jaguars": "JAX","jaguars": "JAX","jacksonville": "JAX","jax": "JAX","jacs":"JAX",
    "kansas city chiefs": "KC","chiefs": "KC","kansas city": "KC","kc": "KC",
    "las vegas raiders": "LV","raiders": "LV","las vegas": "LV","lv": "LV",
    "los angeles chargers": "LAC","la chargers": "LAC","chargers": "LAC","lac": "LAC","san diego chargers":"LAC","san diego":"LAC",
    "los angeles rams": "LAR","la rams": "LAR","rams": "LAR","lar": "LAR","st. louis rams":"LAR","st louis":"LAR",
    "miami dolphins": "MIA","dolphins": "MIA","miami": "MIA","mia": "MIA",
    "minnesota vikings": "MIN","vikings": "MIN","minnesota": "MIN","min": "MIN",
    "new england patriots": "NE","patriots": "NE","new england": "NE","ne": "NE",
    "new orleans saints": "NO","saints": "NO","new orleans": "NO","no": "NO","nos":"NO",
    "new york giants": "NYG","ny giants": "NYG","giants": "NYG","nyg": "NYG",
    "new york jets": "NYJ","ny jets": "NYJ","jets": "NYJ","nyj": "NYJ",
    "philadelphia eagles": "PHI","eagles": "PHI","philadelphia": "PHI","phi": "PHI",
    "pittsburgh steelers": "PIT","steelers": "PIT","pittsburgh": "PIT","pit": "PIT",
    "san francisco 49ers": "SF","49ers": "SF","niners": "SF","san francisco": "SF","sf": "SF",
    "seattle seahawks": "SEA","seahawks": "SEA","seattle": "SEA","sea": "SEA",
    "tampa bay buccaneers": "TB","buccaneers": "TB","bucs": "TB","tampa bay": "TB","tb": "TB",
    "tennessee titans": "TEN","titans": "TEN","tennessee": "TEN","ten": "TEN",
    "washington commanders": "WAS","commanders": "WAS","washington": "WAS","was": "WAS","wsh": "WAS",
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
    if pd.isna(name): return ""
    s = str(name).strip().lower()
    return TEAM_ALIAS_TO_CODE.get(s, s)

# ===== Helpers =====
def normalize_header(name: str) -> str:
    name = str(name) if not isinstance(name, str) else name
    name = name.strip().replace(" ", "_").lower()
    name = re.sub(r"[^0-9a-z_]", "", name)
    return name

@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    df = pd.read_csv(SCORE_URL)
    df.columns = [normalize_header(c) for c in df.columns]
    # Expecting columns exactly: home_team, away_team, favored_team, spread, over_under (week optional)
    return df

def _read_csv_auto(local_path: str, backup_url: str) -> pd.DataFrame:
    try:
        if local_path and os.path.exists(local_path):
            return pd.read_csv(local_path)
        return pd.read_csv(backup_url)
    except Exception:
        return pd.DataFrame()

LOCAL = {
    "player_receiving": os.path.join(DEFAULT_OUT_DIR, "player_receiving.csv"),
    "player_rushing":   os.path.join(DEFAULT_OUT_DIR, "player_rushing.csv"),
    "player_passing":   os.path.join(DEFAULT_OUT_DIR, "player_passing.csv"),
    "def_rb":           os.path.join(DEFAULT_OUT_DIR, "def_rb.csv"),
    "def_qb":           os.path.join(DEFAULT_OUT_DIR, "def_qb.csv"),
    "def_wr":           os.path.join(DEFAULT_OUT_DIR, "def_wr.csv"),
    "def_te":           os.path.join(DEFAULT_OUT_DIR, "def_te.csv"),
}

@st.cache_data(show_spinner=False)
def load_all_player_dfs():
    dfs = {}
    for key, backup_url in SHEETS.items():
        df = _read_csv_auto(LOCAL.get(key, ""), backup_url)
        if df.empty:
            dfs[key] = df
            continue
        df.columns = [normalize_header(c) for c in df.columns]
        if "team" in df.columns:
            df["team"] = df["team"].astype(str).str.strip()
            df["team_key"] = df["team"].apply(team_key)
        elif "teams" in df.columns:
            df["team"] = df["teams"].astype(str).str.strip()
            df["team_key"] = df["team"].apply(team_key)
        else:
            if "team_key" not in df.columns:
                df["team_key"] = ""
        dfs[key] = df
    return dfs

def avg_scoring(df: pd.DataFrame, team_label: str):
    scored_home = df.loc[df["home_team"] == team_label, "home_score"].mean() if "home_score" in df.columns else np.nan
    scored_away = df.loc[df["away_team"] == team_label, "away_score"].mean() if "away_score" in df.columns else np.nan
    allowed_home = df.loc[df["home_team"] == team_label, "away_score"].mean() if "away_score" in df.columns else np.nan
    allowed_away = df.loc[df["away_team"] == team_label, "home_score"].mean() if "home_score" in df.columns else np.nan
    return np.nanmean([scored_home, scored_away]), np.nanmean([allowed_home, allowed_away])

def predict_scores(df: pd.DataFrame, home_label: str, away_label: str):
    # Simple calibration method (can be upgraded later)
    home_scored, home_allowed = avg_scoring(df, home_label)
    away_scored, away_allowed = avg_scoring(df, away_label)
    raw_home = (home_scored + away_allowed) / 2
    raw_away = (away_scored + home_allowed) / 2
    league_avg_pts = df[["home_score", "away_score"]].stack().mean() if set(["home_score","away_score"]).issubset(df.columns) else 44.0
    cal = 22.3 / league_avg_pts if league_avg_pts and league_avg_pts > 0 else 1.0
    return float(raw_home * cal if not np.isnan(raw_home) else 22.3), float(raw_away * cal if not np.isnan(raw_away) else 22.3)

# ===== Prop helpers =====
def find_player_in(df: pd.DataFrame, player_name: str):
    if "player" not in df.columns: return None
    mask = df["player"].astype(str).str.lower() == str(player_name).lower()
    return df[mask].copy() if mask.any() else None

def detect_stat_col(df: pd.DataFrame, prop: str):
    cols = list(df.columns)
    norm = [normalize_header(c) for c in cols]
    mapping = {
        "rushing_yards": ["rushing_yards_total","rushing_yards","rush_yds","rushing_yds","rushing_yards_per_game","rush_yds_pg"],
        "receiving_yards": ["receiving_yards_total","receiving_yds","rec_yds","receiving_yards_per_game","rec_yds_pg"],
        "passing_yards": ["passing_yards_total","passing_yds","pass_yds","passing_yards_per_game","pass_yds_pg"],
        "receptions": ["receiving_receptions_total","rec","receptions"],
        "targets": ["receiving_targets_total","tgt","targets"],
        "carries": ["rushing_attempts_total","rush_att","rushing_att","rushing_carries_per_game"]
    }
    pri = mapping.get(prop, [])
    for cand in pri:
        if cand in norm:
            return cols[norm.index(cand)]
    for i, c in enumerate(norm):
        if prop.split("_")[0] in c and ("per_game" in c or "total" in c or c.endswith("_yds") or c in ("rec","tgt","rush_att")):
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
    cols = list(def_df.columns)
    norm = [normalize_header(c) for c in cols]
    prefs = []
    if prop in ["rushing_yards","carries"]:
        prefs = ["rushing_yards_allowed_total","rushing_yards_allowed","rush_yds_allowed"]
    elif prop in ["receiving_yards","receptions","targets"]:
        prefs = ["receiving_yards_allowed_total","receiving_yards_allowed","rec_yds_allowed"]
    elif prop == "passing_yards":
        prefs = ["passing_yards_allowed_total","passing_yards_allowed","pass_yds_allowed"]
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
        total_tds, total_games = 0.0, 0.0
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

        if opponent_key_override:
            opp_key_for_player = opponent_key_override
        else:
            opp_key_for_player = opponent_key if player_team_key == selected_team_key else selected_team_key

        opp_td_list = []
        for d in def_dfs:
            mask = d["team_key"] == opp_key_for_player
            val = d.loc[mask, "tds_pg"].mean()
            opp_td_list.append(val)
        opp_td_pg = np.nanmean(opp_td_list)
        adj_factor = 1.0 if (np.isnan(opp_td_pg) or not league_td_pg or np.isnan(league_td_pg) or league_td_pg <= 0) else (opp_td_pg / league_td_pg)
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
    if opponent_key_override:
        opp_key_for_player = opponent_key_override
    else:
        opp_key_for_player = opponent_key if player_team_key == selected_team_key else selected_team_key

    opp_allowed_pg, league_allowed_pg = None, None
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

# ===== Odds helpers =====
def american_to_decimal(odds: float) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0: return 1 + (o / 100.0)
    else:     return 1 + (100.0 / abs(o))

def decimal_to_american(dec: float) -> float:
    if dec <= 1: return np.nan
    if dec >= 2: return round((dec - 1) * 100)
    else:        return round(-100 / (dec - 1))

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
    # home_spread is the book line from HOME perspective (negative => home is favored by |line|)
    home_pts, away_pts = predict_scores(scores_df, home, away)
    pred_margin = home_pts - away_pts                          # positive means home wins by that many
    line_margin = -home_spread                                 # target margin implied by book
    stdev_margin = max(5.0, abs(pred_margin) * 0.9 + 6.0)
    z = (line_margin - pred_margin) / stdev_margin
    p_home_cover = float(np.clip(1 - norm.cdf(z), 0.001, 0.999))
    p_away_cover = 1.0 - p_home_cover
    return (pred_margin, p_home_cover if side == "home" else p_away_cover)

# =========================
# UI – Single Page
# =========================
st.title("🏈 New Model Dashboard")

# Sidebar controls
with st.sidebar:
    st.header("Data")
    if st.button("🔄 Clear cache & reload data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.button("🧲 Refresh from Pro-Football-Reference (2025)", use_container_width=True):
        try:
            with st.spinner("Scraping PFR (2025)…"):
                paths = scrape_all(DEFAULT_OUT_DIR)
            st.success("PFR data refreshed!")
            st.caption(str(paths))
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Scrape failed: {e}")

scores_df = load_scores()
if scores_df.empty:
    st.error("Could not load NFL game data (score sheet).")
    st.stop()

player_data = load_all_player_dfs()
p_rec  = player_data.get("player_receiving", pd.DataFrame())
p_rush = player_data.get("player_rushing", pd.DataFrame())
p_pass = player_data.get("player_passing", pd.DataFrame())
d_rb   = player_data.get("def_rb", pd.DataFrame())
d_qb   = player_data.get("def_qb", pd.DataFrame())
d_wr   = player_data.get("def_wr", pd.DataFrame())
d_te   = player_data.get("def_te", pd.DataFrame())

section_names = [
    "1) Game Selection + Prediction",
    "2) Top Edges This Week",
    "3) Player Props",
    "4) Parlay Builder (Players + Game Markets)"
]
selected_section = st.selectbox("Jump to section", section_names, index=0, help="Pick a section to open")

# -------------------------
# Section 1: Game Selection + Prediction
# -------------------------
with st.expander("1) Game Selection + Prediction", expanded=(selected_section == section_names[0])):
    st.subheader("Select Game")
    cols = st.columns([1, 1, 2])
    with cols[0]:
        week_list = sorted(scores_df["week"].dropna().unique()) if "week" in scores_df.columns else ["All"]
        selected_week = st.selectbox("Week", week_list, key="sec1_week")
    with cols[1]:
        if "week" in scores_df.columns:
            teams_in_week = sorted(
                set(scores_df.loc[scores_df["week"] == selected_week, "home_team"].dropna().unique())
                | set(scores_df.loc[scores_df["week"] == selected_week, "away_team"].dropna().unique())
            )
        else:
            teams_in_week = sorted(
                set(scores_df["home_team"].dropna().unique()) | set(scores_df["away_team"].dropna().unique())
            )
        selected_team = st.selectbox("Team", teams_in_week, key="sec1_team")

    if "week" in scores_df.columns:
        game_row = scores_df[
            ((scores_df["home_team"] == selected_team) | (scores_df["away_team"] == selected_team))
            & (scores_df["week"] == selected_week)
        ]
    else:
        game_row = scores_df[
            (scores_df["home_team"] == selected_team) | (scores_df["away_team"] == selected_team)
        ]

    if game_row.empty:
        st.warning("No game found for that team/week.")
    else:
        g = game_row.iloc[0]
        home = g["home_team"]; away = g["away_team"]
        with cols[2]:
            st.markdown(f"**Matchup:** {away} @ {home}")

        default_ou = float(g.get("over_under", 45.0)) if pd.notna(g.get("over_under", np.nan)) else 45.0

        # Convert favored_team + spread into HOME-BASED spread (negative if home is favored)
        fav = str(g.get("favored_team", "")).strip()
        raw_spread = float(g.get("spread", 0.0)) if pd.notna(g.get("spread", np.nan)) else 0.0
        if fav == home:
            default_home_spread = -abs(raw_spread) if raw_spread != 0 else -0.0
        elif fav == away:
            default_home_spread = +abs(raw_spread)
        else:
            default_home_spread = float(raw_spread)  # fallback

        cL, cR = st.columns(2)
        with cL:
            over_under = st.number_input("Over/Under (Vegas or yours)", value=default_ou, step=0.5, key="sec1_ou")
        with cR:
            home_spread = st.number_input("Spread (home format: negative=favorite)", value=float(default_home_spread), step=0.5, key="sec1_spread")

        st.subheader("Game Prediction (Vegas-Calibrated)")
        home_pts, away_pts = predict_scores(scores_df, home, away)
        total_pred = home_pts + away_pts
        margin_pred = home_pts - away_pts
        total_diff = total_pred - over_under
        spread_diff = margin_pred - (-home_spread)

        mrow1 = st.columns(2)
        mrow1[0].metric(f"{home} Predicted", f"{home_pts:.1f} pts")
        mrow1[1].metric(f"{away} Predicted", f"{away_pts:.1f} pts")
        mrow2 = st.columns(2)
        mrow2[0].metric("Predicted Total", f"{total_pred:.1f}", f"{total_diff:+.1f} vs O/U")
        mrow2[1].metric("Predicted Margin (home)", f"{margin_pred:+.1f}", f"{spread_diff:+.1f} vs Spread")

        fig_total = px.bar(x=["Predicted Total", "Vegas O/U"], y=[total_pred, over_under], title="Predicted Total vs O/U")
        st.plotly_chart(fig_total, use_container_width=True)
        fig_margin = px.bar(x=["Predicted Margin (home)", "Vegas Spread (home)"], y=[margin_pred, -home_spread], title="Predicted Margin vs Home Spread")
        st.plotly_chart(fig_margin, use_container_width=True)

# -------------------------
# Section 2: Top Edges This Week
# -------------------------
with st.expander("2) Top Edges This Week", expanded=(selected_section == section_names[1])):
    if "week" in scores_df.columns:
        selected_week_for_edges = st.session_state.get("sec1_week", sorted(scores_df["week"].dropna().unique())[0])
        wk = scores_df[scores_df["week"] == selected_week_for_edges].copy()
        st.caption(f"Week shown: {selected_week_for_edges}")
    else:
        wk = scores_df.copy()
        st.caption("Week column not found; showing all rows.")

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

        fav = str(r.get("favored_team", "")).strip()
        raw_spread = float(r.get("spread", np.nan)) if pd.notna(r.get("spread", np.nan)) else np.nan
        ou = float(r.get("over_under", np.nan)) if pd.notna(r.get("over_under", np.nan)) else np.nan

        # Convert to HOME-BASED spread
        if pd.notna(raw_spread):
            if fav == h:
                home_spread = -abs(raw_spread)
            elif fav == a:
                home_spread = +abs(raw_spread)
            else:
                home_spread = raw_spread
        else:
            home_spread = np.nan

        # Model predictions
        h_pts, a_pts = predict_scores(scores_df, h, a)
        tot_pred = h_pts + a_pts
        mar_pred = h_pts - a_pts

        # Edges
        total_edge = np.nan if pd.isna(ou) else (tot_pred - ou)
        spread_edge = np.nan if pd.isna(home_spread) else (mar_pred - (-home_spread))

        # Total pick
        if pd.isna(total_edge):
            total_pick = ""
        else:
            direction = "OVER" if total_edge > 0 else "UNDER"
            total_pick = f"{strength_badge(total_edge)} {direction}"

        # Spread pick — show number as book shows it (home format)
        if pd.isna(spread_edge) or pd.isna(home_spread):
            spread_pick = ""
        else:
            if mar_pred > -home_spread:
                spread_pick_text = f"{h} {home_spread:+.1f}"
            else:
                spread_pick_text = f"{a} {(-home_spread):+.1f}"
            spread_pick = f"{strength_badge(spread_edge)} {spread_pick_text}"

        rows.append({
            "Matchup": f"{a} @ {h}",
            "Pred Total": round(tot_pred, 1),
            "O/U": ou if not pd.isna(ou) else "",
            "Total Edge (pts)": None if pd.isna(total_edge) else round(total_edge, 1),
            "Total Pick": total_pick,
            "Pred Margin (home)": round(mar_pred, 1),
            "Spread (home)": home_spread if not pd.isna(home_spread) else "",
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
        display_cols = ["Matchup","Pred Total","O/U","Total Edge (pts)","Total Pick","Pred Margin (home)","Spread (home)","Spread Edge (pts)","Spread Pick"]
        st.dataframe(edges_df[display_cols], use_container_width=True)
    else:
        st.info("No games found.")

# -------------------------
# Section 3: Player Props
# -------------------------
with st.expander("3) Player Props", expanded=(selected_section == section_names[2])):
    if 'sec1_team' not in st.session_state or ('week' in scores_df.columns and 'sec1_week' not in st.session_state):
        st.info("Pick a game in Section 1 first.")
    else:
        if "week" in scores_df.columns:
            selected_team = st.session_state['sec1_team']
            selected_week = st.session_state['sec1_week']
            game_row = scores_df[
                ((scores_df["home_team"] == selected_team) | (scores_df["away_team"] == selected_team))
                & (scores_df["week"] == selected_week)
            ]
        else:
            selected_team = st.session_state['sec1_team']
            game_row = scores_df[
                (scores_df["home_team"] == selected_team) | (scores_df["away_team"] == selected_team)
            ]

        if game_row.empty:
            st.warning("No game found for that team/week.")
        else:
            g = game_row.iloc[0]
            home, away = g["home_team"], g["away_team"]

            def players_for_team(df, team_name_or_label):
                key = team_key(team_name_or_label)
                if "team_key" not in df.columns or "player" not in df.columns:
                    return []
                mask = df["team_key"] == key
                return list(df.loc[mask, "player"].dropna().unique())

            team_players = set(
                players_for_team(p_rec, home) + players_for_team(p_rush, home) + players_for_team(p_pass, home) +
                players_for_team(p_rec, away) + players_for_team(p_rush, away) + players_for_team(p_pass, away)
            )
            both_players = sorted(team_players)

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
                selected_team_key = team_key(home)  # anchor on home for opponent logic
                opponent_key = team_key(away)
                res = prop_prediction_and_probs(
                    player_name=player_name,
                    selected_prop=selected_prop,
                    line_val=line_val,
                    selected_team_key=selected_team_key,
                    opponent_key=opponent_key,
                    p_rec=p_rec, p_rush=p_rush, p_pass=p_pass,
                    d_qb=d_qb, d_rb=d_rb, d_wr=d_wr, d_te=d_te
                )
                if "error" in res:
                    st.warning(res["error"])
                elif selected_prop == "anytime_td":
                    st.subheader("Anytime TD Probability")
                    st.write(f"Estimated Anytime TD Probability: **{res['prob_anytime']*100:.1f}%**")
                    bar_df = pd.DataFrame({"Category":["Player TDs/Game","Adj. vs Opponent"], "TDs/Game":[res["player_rate"], res["adj_rate"]]})
                    st.plotly_chart(px.bar(bar_df, x="Category", y="TDs/Game", title=f"{player_name} – Anytime TD"), use_container_width=True)
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
            pb_line = 0.0
            pb_side = "yes"
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
                    st.session_state.parlay_legs.append({"kind":"player","label":label,"prob":prob})
                    st.rerun()

    st.markdown("---")

    st.markdown("**Add Game Market Leg**")
    g1, g2, g3, g4, g5, g6 = st.columns([1.0, 2.2, 1.6, 1.2, 1.2, 1.2])
    with g1:
        week_for_market = st.selectbox("Week", sorted(scores_df["week"].dropna().unique()) if "week" in scores_df.columns else ["All"], key="gm_week")
    with g2:
        wk_df = scores_df[scores_df["week"] == week_for_market] if "week" in scores_df.columns and week_for_market != "All" else scores_df.copy()
        matchups, meta, home_spreads = [], [], []
        for _, row in wk_df.iterrows():
            h, a = row.get("home_team"), row.get("away_team")
            if pd.isna(h) or pd.isna(a): continue
            fav = str(row.get("favored_team","")).strip()
            raw_sp = float(row.get("spread", np.nan)) if pd.notna(row.get("spread", np.nan)) else np.nan
            if pd.notna(raw_sp):
                if fav == h: hs = -abs(raw_sp)
                elif fav == a: hs = +abs(raw_sp)
                else: hs = raw_sp
            else:
                hs = np.nan
            matchups.append(f"{a} @ {h}")
            meta.append((h, a))
            home_spreads.append(hs)
        gm_match = st.selectbox("Matchup", matchups, key="gm_matchup")
        idx = matchups.index(gm_match) if gm_match in matchups else -1
        home_team = meta[idx][0] if idx >= 0 else None
        away_team = meta[idx][1] if idx >= 0 else None
        default_home_sp = home_spreads[idx] if idx >= 0 and not pd.isna(home_spreads[idx]) else 0.0
    with g3:
        gm_market = st.selectbox("Market", ["Total","Spread"], key="gm_market")
    with g4:
        if gm_market == "Total":
            default_tot_line = float(wk_df.iloc[0].get("over_under", 45.0)) if not wk_df.empty else 45.0
            gm_total = st.number_input("O/U Line", value=default_tot_line, step=0.5, key="gm_total_line")
            gm_side = st.selectbox("Side", ["over","under"], key="gm_total_side")
        else:
            gm_spread = st.number_input("Home Spread (negative=favorite)", value=float(default_home_sp), step=0.5, key="gm_spread_line")
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
                    side_text = f"{gm_side_spread.title()} Cover {gm_spread:+.1f}"
                    label = f"{away_team} @ {home_team} Spread {side_text}"
                st.session_state.parlay_legs.append({"kind":"game","label":label,"prob":prob})
                st.rerun()

    if st.session_state.parlay_legs:
        st.subheader("Your Legs")
        for i, leg in enumerate(st.session_state.parlay_legs):
            c1, c2 = st.columns([8, 1])
            c1.markdown(f"• **{leg.get('label','Leg')}** — Model Pr: **{leg.get('prob',0.0)*100:.1f}%**")
            if c2.button("🗑 Remove", key=f"rm_leg_{i}"):
                st.session_state.parlay_legs.pop(i)
                st.rerun()

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
        st.info("Add legs above to build your parlay. Mix player props and game markets.")
