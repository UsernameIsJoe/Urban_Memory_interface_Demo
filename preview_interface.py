#!/usr/bin/env python3
# Hover preview over the district timeline (0416 Study).
# Run from the 0416 Study folder:
#   python preview_interface.py
#
# What this does:
# - Loads ``outputs/district_selected_vibes.csv`` + ``outputs/vibe_scores_per_image.csv`` (same logic
#   as ``visualization.run_district_vibe_timeseries_plot`` for weights / smoothing).
# - Plots the confidence-weighted line chart (thin raw + bold smoothed).
# - Right panel shows full-resolution ``outputs/per_image_reports/<stem>.png`` while you hover along x.

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# Paths (0416 Study defaults)
# =========================
_STUDY_DIR = Path(__file__).resolve().parent
# All pipeline CSVs + ``per_image_reports/`` live here:
OUTPUT_FOLDER = _STUDY_DIR / "outputs"
# Optional: absolute path if your outputs are elsewhere
# OUTPUT_FOLDER = Path("/path/to/0416 Study/outputs")

# If set, this file is loaded and sibling ``vibe_scores_per_image.csv`` + ``per_image_reports/``
# are expected in the same directory.
SELECTED_CSV_PATH: str | Path | None = None  # e.g. OUTPUT_FOLDER / "district_selected_vibes.csv"

REPORTS_SUBDIR = "per_image_reports"

# =========================
# Match ``config.py`` / ``run_district_vibe_timeseries_plot``
# =========================
SMOOTH_WINDOW = 7
USE_NORMALIZED = True
USE_WEIGHTED_SMOOTH = True

COLOR_SCHEME = [
    "#66c2a5",
    "#8da0cb",
    "#fc8d62",
    "#a6d854",
    "#80b1d3",
    "#b3b3b3",
    "#ffd92f",
    "#bebada",
    "#e78ac3",
]

PREVIEW_TITLE = True


def _resolve_selected_csv_path() -> Path:
    if SELECTED_CSV_PATH:
        return Path(SELECTED_CSV_PATH).resolve()
    return (OUTPUT_FOLDER / "district_selected_vibes.csv").resolve()


def _study_root_from_selected_csv(csv_path: Path) -> Path:
    return csv_path.parent


def _reports_dir(study_root: Path) -> Path:
    return study_root / REPORTS_SUBDIR


def _build_report_png_paths(df_merged: pd.DataFrame, reports_dir: Path) -> list[str]:
    if "image" not in df_merged.columns:
        raise ValueError("Merged frame must contain an 'image' column.")
    names = df_merged["image"].astype(str).tolist()
    out: list[str] = []
    for n in names:
        stem = Path(n).stem
        out.append(str((reports_dir / f"{stem}.png").resolve()))
    return out


def load_preview_image(path: str) -> Image.Image:
    """Native resolution of the report PNG (no downscaling)."""
    return Image.open(path).convert("RGB")


def main() -> None:
    csv_selected = _resolve_selected_csv_path()
    if not csv_selected.is_file():
        raise FileNotFoundError(f"district CSV not found: {csv_selected}")

    study_root = _study_root_from_selected_csv(csv_selected)
    vibe_csv_path = study_root / "vibe_scores_per_image.csv"
    reports_dir = _reports_dir(study_root)

    if not vibe_csv_path.is_file():
        raise FileNotFoundError(
            f"vibe_scores_per_image.csv not found next to district CSV ({vibe_csv_path}). "
            "Run the pipeline or place exports beside district_selected_vibes.csv.",
        )
    if not reports_dir.is_dir():
        print(f"[warn] Report folder missing ({reports_dir}); previews will fail until reports exist.")

    df_sel = pd.read_csv(csv_selected)
    df_all = pd.read_csv(vibe_csv_path)

    select_vibes = [c for c in df_sel.columns if c not in ["image", "image_path", "confidence"]]
    if not select_vibes:
        raise ValueError("No vibe columns in district_selected_vibes.csv.")

    if "confidence" not in df_all.columns:
        raise ValueError("vibe_scores_per_image.csv must contain a 'confidence' column.")

    if "image" in df_sel.columns and "image" in df_all.columns:
        df_merged = df_sel.merge(df_all[["image", "confidence"]], on="image", how="left", suffixes=("", "_all"))
    elif "image_path" in df_sel.columns and "image_path" in df_all.columns:
        df_merged = df_sel.merge(
            df_all[["image_path", "confidence"]], on="image_path", how="left", suffixes=("", "_all"),
        )
    else:
        raise ValueError("Need common key 'image' or 'image_path' in both CSVs for confidence merge.")

    if df_merged["confidence"].isna().any():
        miss = df_merged[df_merged["confidence"].isna()]
        head = miss[["image", "image_path"]].head(5) if "image_path" in miss.columns else miss.head(5)
        raise ValueError(
            "Some rows have no confidence after merge. Check vibe_scores_per_image.csv.\n" f"{head}",
        )

    x = np.arange(len(df_merged))
    pivot = df_merged.set_index("image")[select_vibes]
    conf = df_merged.set_index("image")["confidence"].reindex(pivot.index).astype(float)

    pivot_weighted = pivot.mul(conf, axis=0)

    if USE_NORMALIZED:
        denom = pivot_weighted.sum(axis=1).replace(0, np.nan)
        pivot_base = (pivot_weighted.T / denom).T.fillna(0.0)
        y_label = "Confidence-weighted vibe proportion"
    else:
        pivot_base = pivot_weighted
        y_label = "Confidence-weighted vibe score"

    if USE_WEIGHTED_SMOOTH:
        conf_roll = conf.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).sum().replace(0, np.nan)
        pivot_smooth = pd.DataFrame(index=pivot_base.index, columns=pivot_base.columns, dtype=float)
        for vibe in pivot_base.columns:
            num = (pivot_base[vibe] * conf).rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).sum()
            pivot_smooth[vibe] = (num / conf_roll).fillna(0.0)
    else:
        pivot_smooth = pivot_base.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()

    image_paths = _build_report_png_paths(df_merged, reports_dir)

    n = min(len(image_paths), len(x))
    if len(image_paths) != len(x):
        print(f"[warn] row count mismatch: x={len(x)} vs report paths={len(image_paths)}; trim to n={n}.")
        x = x[:n]
        image_paths = image_paths[:n]
        pivot_base = pivot_base.iloc[:n]
        pivot_smooth = pivot_smooth.iloc[:n]

    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        print(f"[warn] Missing {len(missing)} per-image report PNGs. Example: {missing[0]}")

    # -------- plot + hover --------
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 3.5])

    ax = fig.add_subplot(gs[0, 0])
    ax_preview = fig.add_subplot(gs[0, 1])
    ax_preview.set_axis_off()

    for i, vibe in enumerate(select_vibes):
        color = COLOR_SCHEME[i % len(COLOR_SCHEME)]
        ax.plot(x, pivot_base[vibe].values, color=color, alpha=0.18, linewidth=1)
        ax.plot(x, pivot_smooth[vibe].values, color=color, linewidth=3, label=vibe)

    ax.set_title("District Selected Vibes (Confidence-weighted, Base + Smoothed)")
    ax.set_xlabel("Image index (sequence order)")
    ax.set_ylabel(y_label)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=9,
    )

    preview_cache: dict[str, Image.Image] = {}
    current_idx: dict[str, int | None] = {"i": None}

    def show_preview(idx: int) -> None:
        if idx < 0 or idx >= len(image_paths):
            return
        if current_idx["i"] == idx:
            return

        path = image_paths[idx]
        if not os.path.exists(path):
            return

        if path not in preview_cache:
            try:
                preview_cache[path] = load_preview_image(path)
            except Exception:
                return

        ax_preview.clear()
        ax_preview.set_axis_off()
        ax_preview.imshow(preview_cache[path])
        if PREVIEW_TITLE:
            ax_preview.set_title(Path(path).name, fontsize=10)
        current_idx["i"] = idx
        fig.canvas.draw_idle()

    def on_move(event):
        if event.inaxes != ax:
            return
        if event.xdata is None:
            return
        ix = int(round(float(event.xdata)))
        ix = max(0, min(ix, len(image_paths) - 1))
        show_preview(ix)

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    if len(image_paths) > 0:
        show_preview(0)

    plt.tight_layout()
    plt.show()

    print("Selected district vibes:", select_vibes)


if __name__ == "__main__":
    main()
