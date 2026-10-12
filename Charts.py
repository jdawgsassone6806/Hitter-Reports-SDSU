from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from data_utils import (
    ZONE_HALF_WIDTH_FT,
    ZONE_BOTTOM_FT,
    ZONE_TOP_FT,
    PITCH_TYPE_COLORS,
    PLAY_RESULT_COLORS,
    RESULT_MARKERS,
)

# SDSU Aztecs official brand palette 
Black = "#000000"        
Dark_Red= "#A6192E"  
Bright_Red = "#D41736"       
GRID_GRAY = "#DADADA"
GRASS = "#E7EEE4"
DIRT = "#D9C6A3"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = Dark_Red
plt.rcParams["text.color"] = Black
plt.rcParams["axes.labelcolor"] = Black
plt.rcParams["xtick.color"] = Dark_Red
plt.rcParams["ytick.color"] = Dark_Red


def _new_ax(figsize, dpi=300):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    return fig, ax


def _style_spines(ax, keep=("left", "bottom")):
    for spine in ax.spines:
        if spine in keep:
            ax.spines[spine].set_color(GRID_GRAY)
            ax.spines[spine].set_linewidth(0.8)
        else:
            ax.spines[spine].set_visible(False)


# --------------------------------------------------------------------------
# Pitch location ("strike zone") chart
# --------------------------------------------------------------------------

def _draw_zone(ax, lw=1.6):
    zone = mpatches.Rectangle(
        (-ZONE_HALF_WIDTH_FT, ZONE_BOTTOM_FT),
        2 * ZONE_HALF_WIDTH_FT,
        ZONE_TOP_FT - ZONE_BOTTOM_FT,
        fill=False,
        edgecolor=Black,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(zone)
    # 3x3 interior grid
    xs = np.linspace(-ZONE_HALF_WIDTH_FT, ZONE_HALF_WIDTH_FT, 4)
    ys = np.linspace(ZONE_BOTTOM_FT, ZONE_TOP_FT, 4)
    for x in xs[1:-1]:
        ax.plot([x, x], [ZONE_BOTTOM_FT, ZONE_TOP_FT], color=Black, lw=0.6, alpha=0.5, zorder=2)
    for y in ys[1:-1]:
        ax.plot([-ZONE_HALF_WIDTH_FT, ZONE_HALF_WIDTH_FT], [y, y], color=Black, lw=0.6, alpha=0.5, zorder=2)

    # home plate reference at the bottom, drawn as seen from catcher
    hp_w = 0.83
    plate_y = -0.55
    plate = [
        (-hp_w / 2, plate_y + 0.25),
        (hp_w / 2, plate_y + 0.25),
        (hp_w / 2, plate_y),
        (0, plate_y - 0.22),
        (-hp_w / 2, plate_y),
    ]
    ax.add_patch(mpatches.Polygon(plate, closed=True, facecolor="white",
                                   edgecolor=Dark_Red, linewidth=1.0, zorder=2))


def pitch_location_chart(pitches: pd.DataFrame, save_path: str, figsize=(3.6, 3.9)):
    fig, ax = _new_ax(figsize)
    _draw_zone(ax)

    for seq, (_, row) in enumerate(pitches.iterrows(), start=1):
        x, y = row.get("PlateLocSide"), row.get("PlateLocHeight")
        if pd.isna(x) or pd.isna(y):
            continue
        color = PITCH_TYPE_COLORS.get(row["PitchLabel"], "#8a8f98")
        marker = RESULT_MARKERS.get(row["ResultClass"], "o")
        ax.scatter([x], [y], s=280, marker=marker, color=color,
                   edgecolor="white", linewidth=1.3, zorder=5)
        ax.text(x, y, str(seq), ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="white", zorder=6)

    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-1.1, 4.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("PITCH LOCATION", fontsize=10.5, fontweight="bold", color=Black, pad=8, loc="left")
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, transparent=True)
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------
# Field / spray chart
# --------------------------------------------------------------------------

def _draw_field(ax, max_r=420):
    theta = np.linspace(-45, 45, 100)
    # outfield grass wedge
    r_wall = max_r * 0.92
    xs = np.concatenate([[0], r_wall * np.sin(np.radians(theta)), [0]])
    ys = np.concatenate([[0], r_wall * np.cos(np.radians(theta)), [0]])
    ax.fill(xs, ys, color=GRASS, zorder=0)

    # infield dirt (rotated square)
    base_r = 95
    diamond = [(0, 0), (base_r, base_r), (0, 2 * base_r), (-base_r, base_r)]
    ax.add_patch(mpatches.Polygon(diamond, closed=True, facecolor=DIRT, edgecolor="none", zorder=1))
    infield_grass = [(0, 20), (base_r - 20, base_r), (0, 2 * base_r - 20), (-(base_r - 20), base_r)]
    ax.add_patch(mpatches.Polygon(infield_grass, closed=True, facecolor=GRASS, edgecolor="none", zorder=1))

    # foul lines
    ax.plot([0, r_wall * math.sin(math.radians(45))], [0, r_wall * math.cos(math.radians(45))],
            color="white", lw=1.6, zorder=2)
    ax.plot([0, -r_wall * math.sin(math.radians(45))], [0, r_wall * math.cos(math.radians(45))],
            color="white", lw=1.6, zorder=2)
    # outfield wall arc
    ax.plot(r_wall * np.sin(np.radians(theta)), r_wall * np.cos(np.radians(theta)),
            color=Dark_Red, lw=1.2, zorder=2)

    # bases
    for bx, by in [(base_r, base_r), (0, 2 * base_r), (-base_r, base_r), (0, 0)]:
        ax.add_patch(mpatches.RegularPolygon((bx, by), 4, radius=6.5, orientation=math.radians(45),
                                              facecolor="white", edgecolor=Dark_Red, linewidth=0.7, zorder=3))
    # mound
    ax.add_patch(mpatches.Circle((0, 60.5), 8, facecolor=DIRT, edgecolor=Dark_Red, linewidth=0.6, zorder=2))

    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-15, max_r)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def spray_chart(pitches: pd.DataFrame, save_path: str, figsize=(4.6, 3.9), annotate_ev=True):
    fig, ax = _new_ax(figsize)
    _draw_field(ax)

    bbe = pitches[pitches["IsInPlay"]].copy()
    for _, row in bbe.iterrows():
        dist, bearing = row.get("Distance"), row.get("Bearing")
        if pd.isna(dist) or pd.isna(bearing):
            continue
        x = dist * math.sin(math.radians(bearing))
        y = dist * math.cos(math.radians(bearing))
        play_result = str(row.get("PlayResult", "Out"))
        color = PLAY_RESULT_COLORS.get(play_result, "#8a8f98")
        ev = row.get("ExitSpeed")
        size = 180 if pd.isna(ev) else float(np.clip(90 + (ev - 60) * 3.2, 90, 420))
        ax.scatter([x], [y], s=size, color=color, edgecolor="white", linewidth=1.3, zorder=6)
        if annotate_ev and not pd.isna(ev):
            ax.annotate(f"{ev:.0f} mph", (x, y), textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=6.6, color=Black, fontweight="bold", zorder=7)

    ax.set_title("BATTED BALL SPRAY", fontsize=10.5, fontweight="bold", color=Black, pad=8, loc="left")
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, transparent=True)
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------
# Contact point chart (bird's-eye extension relative to the plate)
# --------------------------------------------------------------------------

def contact_point_chart(pitches: pd.DataFrame, save_path: str, figsize=(3.2, 3.9)):
    fig, ax = _new_ax(figsize)

    # home plate, bird's-eye, point toward the catcher (bottom)
    hp_w = 0.83
    plate = [(-hp_w / 2, 0.6), (hp_w / 2, 0.6), (hp_w / 2, 0.15), (0, -0.15), (-hp_w / 2, 0.15)]
    ax.add_patch(mpatches.Polygon(plate, closed=True, facecolor="white",
                                   edgecolor=Black, linewidth=1.4, zorder=2))

    bbe = pitches[pitches["IsInPlay"]].copy()
    for seq, (_, row) in enumerate(pitches.reset_index(drop=True).iterrows(), start=1):
        if not row.get("IsInPlay", False):
            continue
        x, y = row.get("ContactPositionX"), row.get("ContactPositionY")
        if pd.isna(x) or pd.isna(y):
            continue
        play_result = str(row.get("PlayResult", "Out"))
        color = PLAY_RESULT_COLORS.get(play_result, "#8a8f98")
        ax.scatter([x], [y], s=220, color=color, edgecolor="white", linewidth=1.3, zorder=5)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 4.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("CONTACT POINT", fontsize=10.5, fontweight="bold", color=Black, pad=8, loc="left")
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, transparent=True)
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------
# Cover-page aggregate charts
# --------------------------------------------------------------------------

def aggregate_zone_chart(pitches: pd.DataFrame, save_path: str, figsize=(4.6, 4.9)):
    fig, ax = _new_ax(figsize)
    _draw_zone(ax, lw=2.0)
    for _, row in pitches.iterrows():
        x, y = row.get("PlateLocSide"), row.get("PlateLocHeight")
        if pd.isna(x) or pd.isna(y):
            continue
        color = PITCH_TYPE_COLORS.get(row["PitchLabel"], "#8a8f98")
        marker = RESULT_MARKERS.get(row["ResultClass"], "o")
        ax.scatter([x], [y], s=150, marker=marker, color=color, alpha=0.9,
                   edgecolor="white", linewidth=1.0, zorder=5)
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-1.1, 4.8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("ALL PITCHES SEEN", fontsize=12, fontweight="bold", color=Black, pad=10, loc="left")
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, transparent=True)
    plt.close(fig)
    return save_path


def game_spray_chart(pitches: pd.DataFrame, save_path: str, figsize=(5.6, 4.9)):
    fig, ax = _new_ax(figsize)
    _draw_field(ax)
    bbe = pitches[pitches["IsInPlay"]].copy()
    for _, row in bbe.iterrows():
        dist, bearing = row.get("Distance"), row.get("Bearing")
        if pd.isna(dist) or pd.isna(bearing):
            continue
        x = dist * math.sin(math.radians(bearing))
        y = dist * math.cos(math.radians(bearing))
        play_result = str(row.get("PlayResult", "Out"))
        color = PLAY_RESULT_COLORS.get(play_result, "#8a8f98")
        ev = row.get("ExitSpeed")
        size = 200 if pd.isna(ev) else float(np.clip(110 + (ev - 60) * 3.4, 110, 480))
        ax.scatter([x], [y], s=size, color=color, edgecolor="white", linewidth=1.4, zorder=6)
    ax.set_title("BALLS IN PLAY", fontsize=12, fontweight="bold", color=Black, pad=10, loc="left")
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, transparent=True)
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------
# Shared legend swatch image (pitch type + result markers)
# --------------------------------------------------------------------------

def legend_strip(pitch_labels, save_path, figsize=(13.0, 0.45)):
    fig, ax = _new_ax(figsize)
    ax.axis("off")

    handles = []
    for label in pitch_labels:
        color = PITCH_TYPE_COLORS.get(label, "#8a8f98")
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                               markeredgecolor="white", markersize=10, label=label))
    result_defs = [("Ball", "o"), ("Called Strike", "^"), ("Swinging Strike", "D"),
                   ("Foul", "s"), ("In Play", "P")]
    for label, marker in result_defs:
        handles.append(Line2D([0], [0], marker=marker, color="none", markerfacecolor=Dark_Red,
                               markeredgecolor="white", markersize=9, label=label))

    leg = ax.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
                     fontsize=8.5, handletextpad=0.4, columnspacing=1.3,
                     labelcolor=Black)
    fig.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return save_path


def play_result_legend(save_path, figsize=(13.0, 0.4)):
    fig, ax = _new_ax(figsize)
    ax.axis("off")
    handles = []
    for label, color in PLAY_RESULT_COLORS.items():
        if label in ("Error", "FieldersChoice"):
            continue
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                               markeredgecolor="white", markersize=10, label=label))
    leg = ax.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
                     fontsize=8.5, handletextpad=0.4, columnspacing=1.3,
                     labelcolor=Black)
    fig.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return save_path
