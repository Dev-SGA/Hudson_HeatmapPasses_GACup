import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Pass Heatmap Analysis")

st.title("Pass Heatmap Analysis - Origin & Destination")
st.caption("Heatmaps based on pass origin and destination coordinates across multiple matches.")

# ==========================
# Pass Data (correct passes only)
# ==========================
passes_data = {
    "Vs Connecticut": [
        ((26.75, 68.34), (8.97, 51.05)),
        ((31.24, 51.22), (34.57, 72.50)),
        ((36.06, 46.90), (44.37, 57.04)),
        ((48.36, 64.02), (58.17, 51.72)),
        ((58.17, 64.02), (62.49, 55.21)),
        ((54.51, 49.72), (64.82, 61.69)),
        ((42.21, 70.84), (34.90, 76.49)),
        ((43.54, 75.32), (36.73, 67.84)),
        ((32.24, 53.96), (6.81, 38.50)),
        ((33.57, 65.77), (36.56, 75.57)),
        ((37.39, 61.11), (43.04, 75.41)),
        ((65.49, 53.63), (56.18, 70.42)),
        ((55.68, 48.15), (46.87, 30.86)),
        ((52.02, 22.05), (46.70, 41.99)),
        ((62.16, 35.51), (71.80, 35.18)),
        ((54.02, 33.35), (63.99, 22.55)),
        ((60.00, 22.21), (76.62, 32.85)),
        ((87.10, 9.41), (77.45, 16.23)),
        ((62.66, 20.05), (117.18, 8.25)),
        ((98.90, 43.49), (103.22, 47.15)),
        ((70.31, 45.98), (82.28, 60.11)),
        ((85.10, 75.24), (101.39, 74.08)),
        ((53.18, 67.59), (39.05, 59.62)),
        ((55.18, 49.64), (54.85, 13.07)),
        ((68.64, 19.22), (49.03, 24.37)),
        ((53.35, 22.71), (59.34, 30.19)),
        ((44.37, 24.71), (40.05, 46.82)),
        ((43.88, 39.34), (41.38, 73.08)),
        ((56.84, 53.46), (70.81, 76.24)),
        ((82.77, 12.24), (91.42, 4.59)),
        ((108.04, 11.74), (115.69, 58.29)),
        ((93.08, 3.93), (111.03, 13.74)),
        ((84.60, 17.89), (96.74, 22.05)),
        ((58.34, 16.06), (65.65, 2.43)),
        ((52.02, 8.58), (44.37, 15.73)),
        ((61.00, 23.21), (49.36, 15.23)),
        ((32.74, 30.69), (50.03, 33.02)),
        ((51.85, 33.68), (60.66, 40.00)),
        ((79.95, 60.45), (98.23, 60.28)),
        ((31.24, 52.14), (39.05, 72.08)),
        ((39.72, 48.98), (33.40, 57.62)),
        ((70.64, 51.47), (61.00, 51.64)),
    ],
    "Vs Nashville": [
        ((21.27, 14.23), (29.25, 31.02)),
        ((29.41, 23.38), (34.40, 64.60)),
        ((41.55, 39.67), (41.88, 6.92)),
        ((44.54, 32.52), (43.54, 14.23)),
        ((23.59, 56.46), (34.57, 47.48)),
        ((30.58, 64.44), (21.10, 49.48)),
        ((33.07, 56.79), (49.53, 69.59)),
        ((33.24, 59.78), (44.04, 71.75)),
        ((61.50, 71.58), (54.68, 75.57)),
        ((63.16, 50.81), (78.45, 67.26)),
        ((63.49, 76.90), (84.44, 62.77)),
        ((76.96, 56.96), (86.93, 57.79)),
        ((82.61, 59.12), (96.41, 68.43)),
        ((79.78, 35.35), (106.21, 11.74)),
        ((45.37, 49.64), (40.72, 32.02)),
    ],
    "Vs Seongnam": [
        ((28.08, 28.53), (29.75, 8.25)),
        ((33.74, 26.54), (29.41, 43.82)),
        ((28.08, 47.15), (31.57, 64.60)),
        ((39.39, 43.82), (51.69, 53.46)),
        ((43.88, 46.15), (55.84, 40.66)),
        ((47.03, 49.97), (44.04, 28.03)),
        ((47.53, 50.81), (71.97, 33.18)),
        ((67.65, 52.63), (64.32, 33.85)),
        ((73.63, 65.10), (69.31, 73.25)),
        ((77.29, 63.27), (79.12, 72.91)),
        ((81.61, 56.62), (93.91, 73.75)),
        ((86.43, 66.43), (81.78, 54.96)),
        ((111.03, 71.42), (99.56, 67.59)),
        ((89.76, 59.62), (97.74, 48.98)),
        ((88.43, 52.47), (96.41, 74.24)),
        ((87.93, 50.97), (77.12, 27.70)),
        ((81.61, 53.63), (74.30, 27.03)),
        ((79.28, 51.14), (94.91, 70.42)),
        ((52.85, 32.85), (65.49, 25.37)),
        ((82.77, 33.18), (69.31, 47.65)),
    ],
    "Vs Red Bull": [
        ((39.39, 19.39), (52.35, 4.76)),
        ((63.82, 7.92), (72.63, 1.43)),
        ((70.47, 11.91), (80.95, 13.74)),
        ((64.49, 22.55), (97.24, 10.24)),
        ((32.07, 35.51), (43.04, 28.20)),
        ((53.52, 46.32), (54.02, 33.68)),
        ((77.12, 48.64), (84.94, 50.14)),
        ((78.12, 52.47), (117.52, 69.42)),
        ((88.76, 65.93), (97.40, 76.74)),
        ((82.61, 69.26), (86.60, 77.40)),
        ((78.62, 66.26), (79.62, 78.40)),
        ((83.61, 75.91), (62.49, 57.12)),
        ((34.40, 50.14), (88.76, 75.41)),
        ((56.68, 64.27), (78.29, 64.27)),
        ((51.85, 73.25), (54.18, 78.07)),
        ((41.05, 57.45), (46.04, 74.91)),
        ((37.39, 60.61), (41.71, 73.91)),
        ((30.41, 63.44), (36.89, 77.40)),
        ((26.09, 63.94), (28.42, 76.74)),
        ((22.43, 56.62), (22.10, 76.41)),
        ((33.90, 64.77), (25.42, 73.58)),
    ],
}

# ==========================
# Build DataFrames
# ==========================
def build_pass_dataframes(passes_dict):
    origins_by_match = {}
    destinations_by_match = {}

    for match_name, passes in passes_dict.items():
        origins = []
        destinations = []
        for (ox, oy), (dx, dy) in passes:
            origins.append((ox, oy))
            destinations.append((dx, dy))

        origins_by_match[match_name] = pd.DataFrame(origins, columns=["x", "y"])
        destinations_by_match[match_name] = pd.DataFrame(destinations, columns=["x", "y"])

    df_all_origin = pd.concat(origins_by_match.values(), ignore_index=True)
    df_all_dest = pd.concat(destinations_by_match.values(), ignore_index=True)

    origin_full = {"All Games": df_all_origin}
    origin_full.update(origins_by_match)

    dest_full = {"All Games": df_all_dest}
    dest_full.update(destinations_by_match)

    return origin_full, dest_full


origin_data, dest_data = build_pass_dataframes(passes_data)

# ==========================
# Sidebar
# ==========================
st.sidebar.header("Filter Configuration")
selected_match = st.sidebar.radio(
    "Select a match", list(origin_data.keys()), index=0
)
st.sidebar.divider()
st.sidebar.caption("Heatmaps are filtered by the selected match above.")

# ==========================
# Get Data
# ==========================
df_origins = origin_data[selected_match].copy()
df_dests = dest_data[selected_match].copy()

# ==========================
# Statistics
# ==========================
total_passes = len(df_origins)

st.divider()
st.subheader(f"📊 Pass Statistics — {selected_match}")

c1, c2 = st.columns(2)
c1.metric("Total Correct Passes", total_passes)

# Per-match breakdown
if selected_match == "All Games":
    match_counts = []
    for match_name in passes_data:
        match_counts.append({"Match": match_name, "Passes": len(passes_data[match_name])})
    df_breakdown = pd.DataFrame(match_counts)
    c2.dataframe(df_breakdown, hide_index=True, use_container_width=True)
else:
    avg_per_match = sum(len(p) for p in passes_data.values()) / len(passes_data)
    c2.metric("Avg Passes per Match", f"{avg_per_match:.1f}")

st.divider()

# ==========================
# Heatmap Drawing Function
# ==========================
def draw_heatmap(df, title, cmap="Reds"):
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#6BB36B",
        line_color="white",
    )
    fig, ax = pitch.draw(figsize=(8, 5.6))

    if not df.empty:
        pitch.kdeplot(
            df["x"],
            df["y"],
            ax=ax,
            cmap=cmap,
            fill=True,
            levels=100,
            alpha=0.7,
        )

        pitch.scatter(
            df["x"],
            df["y"],
            ax=ax,
            c="#1a1a1a",
            s=22,
            alpha=0.75,
            zorder=3,
            edgecolors="white",
            linewidths=0.3,
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # Attack direction arrow
    arrow = FancyArrowPatch(
        (0.45, 0.05),
        (0.55, 0.05),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=2,
        color="#333333",
    )
    fig.patches.append(arrow)
    fig.text(
        0.5, 0.03,
        "Attack Direction",
        ha="center",
        va="center",
        fontsize=9,
        color="#333333",
    )

    return fig

# ==========================
# Heatmap Row: Origin + Destination side by side
# ==========================
col_origin, col_dest = st.columns(2)

with col_origin:
    st.subheader("🟢 Pass Origin Heatmap")
    fig_origin = draw_heatmap(
        df_origins,
        f"Pass Origins — {selected_match}",
        cmap="Reds",
    )
    st.pyplot(fig_origin, use_container_width=True)
    plt.close(fig_origin)

with col_dest:
    st.subheader("🔵 Pass Destination Heatmap")
    fig_dest = draw_heatmap(
        df_dests,
        f"Pass Destinations — {selected_match}",
        cmap="Blues",
    )
    st.pyplot(fig_dest, use_container_width=True)
    plt.close(fig_dest)
