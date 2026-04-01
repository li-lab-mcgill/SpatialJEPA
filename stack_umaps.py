from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env")

import io
import os

import matplotlib.pyplot as plt
import mlflow
from PIL import Image, ImageDraw, ImageFont

RUNS = [
    {
        "name": "20260331_153543",
        "id": "30fbbad29a63496187a5deb14440ed76",
        "subtitle": "Non-spatial = identity graph, low clip weight",
    },
    {
        "name": "20260331_205205",
        "id": "4f9fbf78b70548d88dff5667492a7dc3",
        "subtitle": "Non-spatial = knn graph, high clip weight",
    },
]
BASE_DIR = "/home/mcb/users/dmannk/BAKLAVA_base/mlflow_tracking/MultiGATE/mlflow_artifacts/multigate_mouse_brain_live_zeroshot"
TRACKING_URI = "sqlite:////home/mcb/users/dmannk/BAKLAVA_base/mlflow_tracking/MultiGATE/mlflow.db"
OUTPUT_PATH = os.getenv("OUTPUT_PATH") + "/stacked_umaps_comparison.png"

MODELS = [
    ("Teacher Source", "stage1_teacher/source_concat_adata_umap.png"),
    ("Teacher Target", "stage1_teacher/target_concat_adata_umap.png"),
    ("Student Source", "stage1/source_concat_adata_umap.png"),
    ("Student Target", "stage1/target_concat_adata_umap.png"),
    ("Non-spatial Source", "stage1_nonspatial/source_concat_adata_umap.png"),
    ("Non-spatial Target", "stage1_nonspatial/target_concat_adata_umap.png"),
]

SCIB_KEYS = [
    ("source_scib_silhouette_label", "Source Sil"),
    ("source_scib_ilisi", "Source iLISI"),
    ("target_scib_silhouette_label", "Target Sil"),
    ("target_scib_ilisi", "Target iLISI"),
]

LOSS_KEYS = [
    ("source_train_loss_clip", "Teacher Clip"),
    ("stage1_nonspatial_train_loss_clip", "Non-spatial Clip"),
    ("stage1_source_target_balanced_mmd", "Student MMD"),
    ("stage1_nonspatial_source_target_balanced_mmd", "Non-spatial MMD"),
]

METRIC_PANELS = [
    ("source_scib_silhouette_label", "Source Sil"),
    ("source_scib_ilisi", "Source iLISI"),
    ("target_scib_silhouette_label", "Target Sil"),
    ("target_scib_ilisi", "Target iLISI"),
    ("source_train_loss_clip", "Teacher Clip"),
    ("stage1_nonspatial_train_loss_clip", "Non-spatial Clip"),
    ("stage1_source_target_balanced_mmd", "Student MMD"),
    ("stage1_nonspatial_source_target_balanced_mmd", "Non-spatial MMD"),
]


def load_font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def center_text(draw, text, cx, y, font, fill="black"):
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    w = bbox[2] - bbox[0]
    draw.multiline_text((cx - (w // 2), y), text, fill=fill, font=font, align="center")


def fetch_last_metrics(run_id, key_defs):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    out = {}
    for key, _ in key_defs:
        out[key] = float(run.data.metrics.get(key, float("nan")))
    return out


def build_grouped_panel(metrics_by_run, run_labels, key_defs, title, ylabel, width, height=360, ylim=None):
    labels = [label for _, label in key_defs]
    x = list(range(len(labels)))
    bar_width = 0.36
    run_colors = ["#4C78A8", "#F58518"]

    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)

    for run_idx, metrics in enumerate(metrics_by_run):
        vals = [metrics[key] for key, _ in key_defs]
        x_pos = [xi + (run_idx - 0.5) * bar_width for xi in x]
        bars = ax.bar(
            x_pos,
            vals,
            width=bar_width,
            color=run_colors[run_idx % len(run_colors)],
            label=run_labels[run_idx],
            alpha=0.95,
        )
        for bar, val in zip(bars, vals):
            if val == val:  # not NaN
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    val + 0.02 if ylim is None else min(ylim[1] * 0.98, val + (ylim[1] * 0.02)),
                    "{:.3f}".format(val),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=18, pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    panel = Image.open(buf).convert("RGB")
    if panel.width != width:
        panel = panel.resize((width, panel.height))
    return panel


def build_single_metric_panel(metric_key, metric_label, values, run_labels, width=250, height=260):
    colors = ["#4C78A8", "#F58518"]
    x = list(range(len(run_labels)))

    finite_vals = [v for v in values if v == v]
    if finite_vals:
        vmax = max(finite_vals)
        ymin = 0.0 if min(finite_vals) >= 0 else min(finite_vals) * 1.1
        ymax = vmax * 1.2 if vmax > 0 else 1.0
    else:
        ymin, ymax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
    bars = ax.bar(x, values, color=colors[: len(run_labels)], width=0.65, alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=20, ha="right", fontsize=9)
    ax.set_title(metric_label, fontsize=12, pad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(ymin, ymax)

    for bar, val in zip(bars, values):
        if val == val:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + (ymax - ymin) * 0.03,
                "{:.3f}".format(val),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    panel = Image.open(buf).convert("RGB")
    if panel.width != width:
        panel = panel.resize((width, panel.height))
    return panel


def load_umap_images():
    images = []
    max_w = 0
    max_h = 0
    for run in RUNS:
        run_images = []
        for label, rel_path in MODELS:
            full_path = os.path.join(BASE_DIR, run["id"], "artifacts/umap", rel_path)
            if os.path.exists(full_path):
                img = Image.open(full_path).convert("RGB")
                run_images.append((label, img))
                max_w = max(max_w, img.width)
                max_h = max(max_h, img.height)
            else:
                print("Warning: {} not found".format(full_path))
                run_images.append((label, None))
        images.append(run_images)
    return images, max_w, max_h


def main():
    images, max_w, max_h = load_umap_images()

    header_h = 190
    margin = 28
    row_gap = 18
    col_gap = 140
    metric_panel_h = 260
    metric_col_gap = 16
    metric_panel_w = 250

    title_font = load_font(64)
    row_label_font = load_font(44)

    grid_w = (max_w * 2) + (margin * 2) + col_gap
    grid_h = header_h + (max_h * 6) + (row_gap * 5) + metric_panel_h + margin * 3
    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)

    left_x = margin
    right_x = margin + max_w + col_gap
    col_centers = [left_x + (max_w // 2), right_x + (max_w // 2)]

    for idx, run in enumerate(RUNS):
        title = "Run {}\n({})".format(run["name"], run["subtitle"])
        center_text(draw, title, col_centers[idx], margin, title_font)

    y_offset = header_h
    for row_idx in range(6):
        for col_idx, run_images in enumerate(images):
            label, img = run_images[row_idx]
            x_offset = left_x if col_idx == 0 else right_x
            if img is not None:
                canvas.paste(img, (x_offset, y_offset))
                if col_idx == 0:
                    draw.text((x_offset + 32, y_offset + 24), label, fill="black", font=row_label_font)
        y_offset += max_h + row_gap

    metric_y = y_offset + margin
    run_labels = [run["name"] for run in RUNS]
    all_metric_keys = [(k, lbl) for k, lbl in METRIC_PANELS]
    metrics_by_run = [fetch_last_metrics(run["id"], all_metric_keys) for run in RUNS]

    n_panels = len(METRIC_PANELS)
    total_metrics_w = n_panels * metric_panel_w + (n_panels - 1) * metric_col_gap
    start_x = max(left_x, (grid_w - total_metrics_w) // 2)

    for idx, (metric_key, metric_label) in enumerate(METRIC_PANELS):
        values = [metrics_by_run[ridx][metric_key] for ridx in range(len(RUNS))]
        panel = build_single_metric_panel(
            metric_key=metric_key,
            metric_label=metric_label,
            values=values,
            run_labels=run_labels,
            width=metric_panel_w,
            height=metric_panel_h,
        )
        x = start_x + idx * (metric_panel_w + metric_col_gap)
        canvas.paste(panel, (x, metric_y))

    canvas.save(OUTPUT_PATH)
    print("Saved stacked image with metrics to {}".format(OUTPUT_PATH))


if __name__ == "__main__":
    main()
