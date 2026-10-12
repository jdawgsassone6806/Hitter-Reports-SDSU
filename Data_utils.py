from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants / reference data
# --------------------------------------------------------------------------

# Rulebook plate width is 17in -> +/-8.5in from center = 0.708 ft.
# Batter-specific top/bottom of zone is not present in this export, so we
# use a standard average zone (1.50 ft - 3.50 ft) as is common practice
# when a per-batter zone isn't available. This is documented in the report.
ZONE_HALF_WIDTH_FT = 0.708
ZONE_BOTTOM_FT = 1.50
ZONE_TOP_FT = 3.50

BALL_CALLS = {"BallCalled", "BallinDirt", "AutomaticBall"}
HBP_CALLS = {"HitByPitch"}
CALLED_STRIKE = {"StrikeCalled", "AutomaticStrike"}
SWING_CALLS = {
    "StrikeSwinging",
    "FoulBall",
    "FoulBallNotFieldable",
    "FoulBallFieldable",
    "InPlay",
}
WHIFF_CALLS = {"StrikeSwinging"}
INPLAY_CALLS = {"InPlay"}

# TaggedPitchType -> broad pitch group, matches standard scouting convention
PITCH_GROUP_MAP = {
    "Fastball": "Fastball",
    "FourSeamFastBall": "Fastball",
    "Four-Seam": "Fastball",
    "TwoSeamFastBall": "Fastball",
    "Sinker": "Fastball",
    "Cutter": "Fastball",
    "Slider": "Breaking",
    "Curveball": "Breaking",
    "Sweeper": "Breaking",
    "Knuckleball": "Breaking",
    "ChangeUp": "OffSpeed",
    "Changeup": "OffSpeed",
    "Splitter": "OffSpeed",
}

PITCH_LABEL_MAP = {
    "Fastball": "Fastball",
    "FourSeamFastBall": "Four-Seam",
    "TwoSeamFastBall": "Two-Seam",
    "Sinker": "Sinker",
    "Cutter": "Cutter",
    "Slider": "Slider",
    "Curveball": "Curveball",
    "Sweeper": "Sweeper",
    "ChangeUp": "Changeup",
    "Changeup": "Changeup",
    "Splitter": "Splitter",
    "Knuckleball": "Knuckleball",
}

PLAY_RESULT_COLORS = {
    "Single": "#1a7a3c",
    "Double": "#7b5fd6",
    "Triple": "#d63b9b",
    "HomeRun": "#e0672a",
    "Out": "#8a8f98",
    "Error": "#8a8f98",
    "FieldersChoice": "#8a8f98",
    "Sacrifice": "#a3762b",
}

PITCH_TYPE_COLORS = {
    "Fastball": "#c8102e",
    "Four-Seam": "#c8102e",
    "Two-Seam": "#e0672a",
    "Sinker": "#e0672a",
    "Cutter": "#7a1f2b",
    "Slider": "#f2c500",
    "Curveball": "#1f9ad6",
    "Sweeper": "#e8b400",
    "Changeup": "#1a7a3c",
    "Splitter": "#2fb5a0",
    "Knuckleball": "#8a8f98",
}

RESULT_MARKERS = {
    "Ball": "o",
    "CalledStrike": "^",
    "SwingingStrike": "D",
    "Foul": "s",
    "InPlay": "P",
    "HitByPitch": "X",
}


def _classify_result(pitch_call: str) -> str:
    if pitch_call in BALL_CALLS:
        return "Ball"
    if pitch_call in HBP_CALLS:
        return "HitByPitch"
    if pitch_call in CALLED_STRIKE:
        return "CalledStrike"
    if pitch_call in WHIFF_CALLS:
        return "SwingingStrike"
    if pitch_call in {"FoulBall", "FoulBallNotFieldable", "FoulBallFieldable"}:
        return "Foul"
    if pitch_call in INPLAY_CALLS:
        return "InPlay"
    return "Ball"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_pitch_data(path: str) -> pd.DataFrame:
    """Load a Trackman-style export (.xlsx or .csv) into a DataFrame."""
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
    else:
        df = pd.read_excel(path)
    return df


def list_batters(df: pd.DataFrame) -> list[str]:
    return sorted(df["Batter"].dropna().unique().tolist())


# --------------------------------------------------------------------------
# Per-pitch enrichment
# --------------------------------------------------------------------------

def enrich_batter_pitches(df: pd.DataFrame, batter_name: str) -> pd.DataFrame:
    """Filter to one batter's pitches for the game and add derived columns."""
    sub = df[df["Batter"] == batter_name].copy()
    if sub.empty:
        raise ValueError(f"No pitches found for batter '{batter_name}' in this file.")
    sub = sub.sort_values("PitchNo").reset_index(drop=True).copy()

    # Prefer the human-tagged pitch type; fall back to the auto classifier.
    tagged = sub["TaggedPitchType"].astype(str)
    auto = sub["AutoPitchType"].astype(str)
    raw_type = tagged.where(~tagged.isin(["Undefined", "nan", "Other", ""]), auto)
    sub["PitchGroup"] = raw_type.map(PITCH_GROUP_MAP).fillna("Other")
    sub["PitchLabel"] = raw_type.map(PITCH_LABEL_MAP).fillna(raw_type)

    sub["ResultClass"] = sub["PitchCall"].apply(_classify_result)
    sub["IsStrike"] = sub["ResultClass"] != "Ball"
    sub["IsSwing"] = sub["PitchCall"].isin(SWING_CALLS)
    sub["IsWhiff"] = sub["PitchCall"].isin(WHIFF_CALLS)
    sub["IsInPlay"] = sub["PitchCall"].isin(INPLAY_CALLS)

    side = pd.to_numeric(sub["PlateLocSide"], errors="coerce")
    height = pd.to_numeric(sub["PlateLocHeight"], errors="coerce")
    sub["InZone"] = (
        side.abs().le(ZONE_HALF_WIDTH_FT)
        & height.between(ZONE_BOTTOM_FT, ZONE_TOP_FT)
    )
    sub["IsChase"] = (~sub["InZone"]) & sub["IsSwing"]

    balls = pd.to_numeric(sub["Balls"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(sub["Strikes"], errors="coerce").fillna(0).astype(int)
    sub["CountStr"] = balls.astype(str) + "-" + strikes.astype(str)
    sub["IsFirstPitch"] = (balls == 0) & (strikes == 0)
    sub["IsTwoStrike"] = strikes >= 2
    sub["IsAhead"] = balls > strikes
    sub["IsBehind"] = strikes > balls
    sub["IsEven"] = balls == strikes

    for col in ("ExitSpeed", "Angle", "Distance", "Bearing", "RelSpeed"):
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")

    return sub


# --------------------------------------------------------------------------
# Aggregate tables
# --------------------------------------------------------------------------

def _agg_block(g: pd.DataFrame) -> dict:
    pitches = len(g)
    swings = int(g["IsSwing"].sum())
    out_of_zone = int((~g["InZone"]).sum())
    bbe = g[g["IsInPlay"]]
    return {
        "P": pitches,
        "Strike%": g["IsStrike"].mean() * 100 if pitches else 0.0,
        "Zone%": g["InZone"].mean() * 100 if pitches else 0.0,
        "Swing%": swings / pitches * 100 if pitches else 0.0,
        "Whiff%": (g["IsWhiff"].sum() / swings * 100) if swings else 0.0,
        "Chase%": (g["IsChase"].sum() / out_of_zone * 100) if out_of_zone else 0.0,
        "BBE": int(len(bbe)),
        "EV": bbe["ExitSpeed"].mean() if len(bbe) else float("nan"),
        "LA": bbe["Angle"].mean() if len(bbe) else float("nan"),
    }


def pitch_type_summary(pitches: pd.DataFrame) -> pd.DataFrame:
    order = ["Fastball", "Breaking", "OffSpeed", "Other"]
    total = len(pitches)
    rows = []
    for grp in order:
        g = pitches[pitches["PitchGroup"] == grp]
        if g.empty:
            continue
        rec = _agg_block(g)
        rec["Usage%"] = rec["P"] / total * 100 if total else 0.0
        rec["Group"] = grp
        rows.append(rec)
    cols = ["Group", "P", "Usage%", "Strike%", "Zone%", "Swing%", "Whiff%", "Chase%", "BBE", "EV", "LA"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


def count_situation_summary(pitches: pd.DataFrame) -> pd.DataFrame:
    situations = [
        ("1st Pitch", pitches["IsFirstPitch"]),
        ("2 Strikes", pitches["IsTwoStrike"]),
        ("Ahead", pitches["IsAhead"]),
        ("Behind", pitches["IsBehind"]),
        ("Even", pitches["IsEven"]),
    ]
    rows = []
    for label, mask in situations:
        g = pitches[mask]
        if g.empty:
            continue
        rec = _agg_block(g)
        rec["Situation"] = label
        rows.append(rec)
    cols = ["Situation", "P", "Strike%", "Zone%", "Swing%", "Whiff%", "Chase%", "BBE", "EV", "LA"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------
# Plate-appearance grouping
# --------------------------------------------------------------------------

@dataclass
class PlateAppearance:
    pa_number: int
    inning: int
    top_bottom: str
    pitcher: str
    pitcher_throws: str
    outs_start: int
    pitches: pd.DataFrame
    result: str
    rbi: int
    final_count: str


def _pa_result_label(g: pd.DataFrame) -> str:
    last = g.iloc[-1]
    play_result = str(last.get("PlayResult", "Undefined"))
    korbb = str(last.get("KorBB", "Undefined"))
    if play_result not in ("Undefined", "nan", ""):
        return play_result
    if korbb not in ("Undefined", "nan", ""):
        return korbb
    return _classify_result(last.get("PitchCall", ""))


def build_plate_appearances(pitches: pd.DataFrame) -> list[PlateAppearance]:
    pas = []
    key_cols = ["Inning", "PAofInning"]
    groups = pitches.groupby(key_cols, sort=False)
    # preserve chronological order (by first PitchNo in each group)
    order = pitches.groupby(key_cols)["PitchNo"].min().sort_values()
    for i, (key, _) in enumerate(order.items(), start=1):
        g = groups.get_group(key).sort_values("PitchNo").reset_index(drop=True)
        first = g.iloc[0]
        runs = pd.to_numeric(g.get("RunsScored", pd.Series([0])), errors="coerce").fillna(0).sum()
        pas.append(
            PlateAppearance(
                pa_number=i,
                inning=int(first["Inning"]),
                top_bottom=str(first.get("Top/Bottom", "")),
                pitcher=str(first["Pitcher"]),
                pitcher_throws=str(first.get("PitcherThrows", "")),
                outs_start=int(first.get("Outs", 0)),
                pitches=g,
                result=_pa_result_label(g),
                rbi=int(runs),
                final_count=f'{int(g.iloc[-1]["Balls"])}-{int(g.iloc[-1]["Strikes"])}',
            )
        )
    return pas


# --------------------------------------------------------------------------
# Game-level slash line / KPI summary
# --------------------------------------------------------------------------

@dataclass
class GameStats:
    pa: int = 0
    ab: int = 0
    h: int = 0
    doubles: int = 0
    triples: int = 0
    hr: int = 0
    bb: int = 0
    so: int = 0
    hbp: int = 0
    sac: int = 0
    rbi: int = 0
    avg: float = float("nan")
    obp: float = float("nan")
    slg: float = float("nan")
    ops: float = float("nan")
    avg_ev: float = float("nan")
    max_ev: float = float("nan")
    hard_hit_pct: float = float("nan")
    chase_pct: float = float("nan")
    whiff_pct: float = float("nan")
    zone_pct: float = float("nan")
    k_pct: float = float("nan")
    bb_pct: float = float("nan")


def compute_game_stats(pitches: pd.DataFrame, pas: list[PlateAppearance]) -> GameStats:
    stats = GameStats()
    stats.pa = len(pas)
    singles = 0
    for p in pas:
        r = p.result
        if r == "Walk":
            stats.bb += 1
        elif r == "HitByPitch":
            stats.hbp += 1
        elif r == "Strikeout":
            stats.so += 1
            stats.ab += 1
        elif r == "Sacrifice":
            stats.sac += 1
        elif r == "Single":
            singles += 1
            stats.h += 1
            stats.ab += 1
        elif r == "Double":
            stats.doubles += 1
            stats.h += 1
            stats.ab += 1
        elif r == "Triple":
            stats.triples += 1
            stats.h += 1
            stats.ab += 1
        elif r == "HomeRun":
            stats.hr += 1
            stats.h += 1
            stats.ab += 1
        elif r in ("Out", "Error", "FieldersChoice"):
            stats.ab += 1
        stats.rbi += p.rbi

    tb = singles + 2 * stats.doubles + 3 * stats.triples + 4 * stats.hr
    stats.avg = stats.h / stats.ab if stats.ab else float("nan")
    obp_den = stats.ab + stats.bb + stats.hbp + stats.sac
    stats.obp = (stats.h + stats.bb + stats.hbp) / obp_den if obp_den else float("nan")
    stats.slg = tb / stats.ab if stats.ab else float("nan")
    if not math.isnan(stats.obp) and not math.isnan(stats.slg):
        stats.ops = stats.obp + stats.slg

    bbe = pitches[pitches["IsInPlay"]]
    if len(bbe):
        stats.avg_ev = bbe["ExitSpeed"].mean()
        stats.max_ev = bbe["ExitSpeed"].max()
        stats.hard_hit_pct = (bbe["ExitSpeed"] >= 95).mean() * 100

    out_of_zone = pitches[~pitches["InZone"]]
    swings = pitches[pitches["IsSwing"]]
    stats.chase_pct = out_of_zone["IsSwing"].mean() * 100 if len(out_of_zone) else float("nan")
    stats.whiff_pct = (pitches["IsWhiff"].sum() / len(swings) * 100) if len(swings) else float("nan")
    stats.zone_pct = pitches["InZone"].mean() * 100 if len(pitches) else float("nan")
    stats.k_pct = stats.so / stats.pa * 100 if stats.pa else float("nan")
    stats.bb_pct = stats.bb / stats.pa * 100 if stats.pa else float("nan")
    return stats


# --------------------------------------------------------------------------
# Top level container
# --------------------------------------------------------------------------

@dataclass
class GameReport:
    batter: str
    batter_team: str
    opponent_team: str
    batter_side: str
    date: str
    stadium: str
    pitches: pd.DataFrame
    pas: list[PlateAppearance]
    stats: GameStats
    pitch_summary: pd.DataFrame
    situation_summary: pd.DataFrame


def build_game_report(path: str, batter_name: str) -> GameReport:
    raw = load_pitch_data(path)
    pitches = enrich_batter_pitches(raw, batter_name)
    pas = build_plate_appearances(pitches)
    stats = compute_game_stats(pitches, pas)
    pitch_summary = pitch_type_summary(pitches)
    situation_summary = count_situation_summary(pitches)

    first = pitches.iloc[0]
    batter_team = str(first.get("BatterTeam", ""))
    pitcher_team = str(first.get("PitcherTeam", ""))
    date_val = first.get("Date", "")
    try:
        date_str = pd.to_datetime(date_val).strftime("%B %d, %Y")
    except Exception:
        date_str = str(date_val)

    return GameReport(
        batter=batter_name,
        batter_team=batter_team,
        opponent_team=pitcher_team,
        batter_side=str(first.get("BatterSide", "")),
        date=date_str,
        stadium=str(first.get("Stadium", "")),
        pitches=pitches,
        pas=pas,
        stats=stats,
        pitch_summary=pitch_summary,
        situation_summary=situation_summary,
    )
