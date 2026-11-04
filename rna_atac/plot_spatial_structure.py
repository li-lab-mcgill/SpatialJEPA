"""Joint-latent replacements for the elements of the 28 October Fig. 3 layout.

The Fig. 3 uploaded to Overleaf on 28 October (f22d6cd) keeps the schematic (a)
and was assembled from RNA-side elements:
  - panel b from fig3_panel_b_wide.py (wide ECDF axes, 21 October);
  - the four c maps, the c colour bar, the student d map and the d colour bar from
    the August export of plot_spatial_structure_figure (analyze_rna_atac.py;
    figures/3_spatial_ecdf_analysis.svg in the Overleaf clone);
  - a teacher d map from fig3_student_minus_teacher.py, which ran on the joint latents.
This script redraws those elements on the joint latents (fig3_supp_ecdf_by_latent
build("joint"), the latents of fig3_joint_latents.py) with the geometry and styles of
the elements they replace:
  panel_b_wide_joint.{svg,pdf,png}  page and axes height of panel_b_ecdf_wide.svg; the axes are widened so
                                    their right edge lines up with the panel a explainer boxes above, and the
                                    legend (with the short arm names of panel c) sits in the bottom-right corner
  maps_cd_joint.{svg,pdf,png}       the c and d rows of plot_spatial_structure_figure;
                                    d shows teacher - MultiVI and student - MultiVI on
                                    one colour scale (98th percentile of |student - MultiVI|)
It then pastes them over the RNA-side elements of the Overleaf PNG, element by element,
giving 3_spatial_ecdf_analysis_joint.png. The schematic, the panel letters, the d titles
and the colour-bar labels are left as they are; the c colour bar is unchanged (0-1 with
the random floor, which is the same on every latent). The same pasting is first done
with RNA-side renders, which must reproduce the Overleaf PNG; the residual is reported.

    python plot_spatial_structure.py [--output DIR] [--overleaf-figure PNG]
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
from pathlib import Path
import argparse
import json
import pickle
import re
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import plot_position_recovery_by_latent as S2

BASE = Path(os.environ['SPATIALJEPA_HOME'])
CLONE = Path(os.environ['SPATIALJEPA_MANUSCRIPT'])
DEFAULT_OUT = BASE / "outputs/rna_atac_figures/fig3_joint_new_layout_20261029"
AUGUST_SVG = CLONE / "figures/3_spatial_ecdf_analysis.svg"
ARMS = ["teacher", "student", "non-spatial", "MultiVI"]
SVG = "{http://www.w3.org/2000/svg}"
DPI = 600
# panel b, as in fig3_panel_b_wide.py
B_STYLES = [("teacher", "SpatialJEPA teacher", "#00441b", "--", 1.4),
            ("student", "SpatialJEPA student", "#1b7837", "-", 2.4),
            ("non-spatial", "MultiGATE non-spatial", "#5aae61", "-", 1.4),
            ("MultiVI", "MultiVI", "#762a83", "-", 2.4),
            ("expression PCs", "expression PCs", "#6e6e6e", "-.", 1.2),
            ("random floor", "random floor", "#999999", ":", 1.4)]


def load_stats(modality, out):
    cache = out / f"stats_{modality}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    stats = S2.original_functions()(*S2.build(modality))
    cache.write_bytes(pickle.dumps(stats))
    return stats


def wide_geometry():
    """Axes size (pt) and x range of panel_b_ecdf_wide, recovered from the August SVG as that script does."""
    axes = ET.fromstring(AUGUST_SVG.read_bytes()).find('.//*[@id="axes_1"]')

    def vertices(group_id):
        data = axes.find(f'.//*[@id="{group_id}"]/{SVG}path').get("d")
        return np.array([float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", data)]).reshape(-1, 2)

    box = vertices("patch_2")
    (left, top), (right, bottom) = box.min(axis=0), box.max(axis=0)
    threshold_x = vertices("line2d_32")[0, 0]
    return (right - left) * 1.4, bottom - top, 10 ** ((right - left) / (threshold_x - left) * np.log10(2))


def ecdf_curves(stats):
    """Pooled ECDFs as plot_spatial_structure_figure draws them (same floor draw and log grid)."""
    xy, pitch, errors = stats["xy"], stats["pitch"], stats["errors"]
    rng = np.random.default_rng(0)
    n_e = errors["student"].size
    floor = np.linalg.norm(xy[rng.integers(0, len(xy), n_e)] - xy[rng.integers(0, len(xy), n_e)], axis=1) / pitch
    curves = {a: errors[a][~np.isnan(errors[a])].ravel() for a in ARMS + ["expression PCs"]}
    curves["random floor"] = floor
    grid = np.logspace(0.0, np.log10(stats["xmax"]), 400)
    cdf = {k: np.searchsorted(np.sort(np.clip(e, 1.0, None)), grid, side="right") / len(e) for k, e in curves.items()}
    at2 = {k: float(np.searchsorted(np.sort(np.clip(e, 1.0, None)), 2.0, side="right") / len(e)) for k, e in curves.items()}
    return grid, cdf, at2


def draw_panel_b(stats, out, tag, legend, width=None, short_labels=False):
    wide_width, height, xmax = wide_geometry()
    width = width or wide_width
    grid, cdf, at2 = ecdf_curves(stats)
    pw, ph = width + 53, height + 48
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans", "path.simplify": False})
    fig = plt.figure(figsize=(pw / 72, ph / 72))
    ax = fig.add_axes([43 / pw, 34 / ph, width / pw, height / ph])
    for key, label, color, ls, lw in B_STYLES:
        ax.plot(grid, cdf[key], ls=ls, lw=lw, color=color, label=key if short_labels else label)
    ax.axvline(2, color="0.6", lw=.8, ls=":")
    ax.plot(2, at2["student"], marker="*", markersize=9, markerfacecolor="#a6dba0", markeredgecolor="black", zorder=5)
    ax.annotate(f"{at2['student']:.0%}", (2, at2["student"]), xytext=(5, -7), textcoords="offset points",
                fontsize=9, color="0.45")
    ax.text(2.15, .11, "≤ 2 sp", fontsize=9, color="0.45")
    ax.set(xscale="log", xlim=(1, xmax), ylim=(0, 1.02))
    ax.set_xlabel("position error (spot widths)", fontsize=9)
    ax.set_ylabel("cumulative fraction", fontsize=9)
    ax.tick_params(labelsize=8)
    leg = ax.legend(fontsize=6.5, **{"labelspacing": .32, "handlelength": 1.9, "borderpad": .4, **legend})
    fig.canvas.draw()
    # in the bottom-right corner, no curve may pass through the legend box (the RNA-side rebuild keeps the
    # Overleaf legend, which touches the teacher curve, as it is)
    if short_labels:
        bb = leg.get_window_extent().transformed(ax.transData.inverted())
        inside = (grid >= bb.x0) & (grid <= bb.x1)
        crossing = [k for k, v in cdf.items() if ((v[inside] >= bb.y0 - .01) & (v[inside] <= bb.y1 + .01)).any()]
        assert not crossing, f"legend box covers {crossing}"
    fig.text(3 / pw, (34 + height - 11) / ph, "b", fontsize=15, fontweight="bold")
    for ext in ("svg", "pdf", "png"):
        fig.savefig(out / f"panel_b_wide_{tag}.{ext}", dpi=DPI)
    plt.close(fig)
    return at2


def draw_maps(stats, out, tag):
    """Rows c and d of plot_spatial_structure_figure (figsize, GridSpec, styles), d as teacher/student - MultiVI."""
    xy = stats["xy"]
    nb = NearestNeighbors(n_neighbors=7).fit(xy).kneighbors(xy, return_distance=False)
    sm = lambda v: np.asarray(v)[nb].mean(axis=1)
    seq = mpl.colors.LinearSegmentedColormap.from_list("seq", plt.get_cmap("Blues")(np.linspace(0.05, 1.0, 256)))
    div = mpl.colors.LinearSegmentedColormap.from_list("div", ["#762a83", "#9970ab", "#e7e7e7", "#5aae61", "#1b7837"])
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(11.0, 9.8))
    gs = GridSpec(4, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.09],
                  height_ratios=[1.5, 0.18, 1, 1], hspace=0.25, wspace=0.10)

    def bare(ax):
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6); s.set_color("0.7")
        return ax

    for c, a in enumerate(ARMS):
        ax = bare(fig.add_subplot(gs[2, c]))
        sca = ax.scatter(xy[:, 0], xy[:, 1], c=sm(stats["auc"][a]), s=3.0, linewidths=0, cmap=seq, vmin=0, vmax=1,
                         rasterized=True)
        ax.set_title(f"{a}\narea {np.nanmean(stats['auc'][a]):.2f}", fontsize=9, pad=5)
    cb = fig.colorbar(sca, cax=fig.add_subplot(gs[2, 4]))
    cb.ax.tick_params(labelsize=7)
    cb.ax.axhline(stats["floor_auc"], color="0.25", lw=1.0)

    auc = {a: np.asarray(stats["auc"][a]) for a in ARMS}
    vlim = float(np.percentile(np.abs(auc["student"] - auc["MultiVI"]), 98))
    for c, a in enumerate(("teacher", "student")):
        ax = bare(fig.add_subplot(gs[3, c]))
        sca = ax.scatter(xy[:, 0], xy[:, 1], c=sm(auc[a] - auc["MultiVI"]), s=3.0, linewidths=0, cmap=div,
                         vmin=-vlim, vmax=vlim, rasterized=True)
        ax.set_title(f"Δ ({a} - MultiVI)", fontsize=9, pad=5)
    cb = fig.colorbar(sca, cax=fig.add_subplot(gs[3, 4]), extend="both")
    cb.set_ticks(np.round(np.arange(-0.6, 0.61, 0.2), 1))   # the Overleaf bar's ticks, for any limit above 0.6
    cb.ax.tick_params(labelsize=7)
    cb.ax.axhline(0.0, color="0.35", lw=0.8)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(out / f"maps_cd_{tag}.{ext}", dpi=DPI)
    plt.close(fig)
    ahead = {a: float(np.mean(auc[a] - auc["MultiVI"] > 0)) for a in ("teacher", "student")}
    return vlim, ahead


def runs(v, minlen):
    d = np.diff(np.r_[0, v.astype(int), 0])
    return [(s, e) for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]) if e - s >= minlen]


def boxes(img, mask_fn, minfrac):
    """Outer boxes of rectangles drawn with long lines of one colour: [(x0, y0, x1, y1)], sorted top-left first."""
    m = mask_fn(img)
    found = []
    for y in range(m.shape[0]):
        for s, e in runs(m[y], int(minfrac * m.shape[1])):
            found.append((y, s, e))
    rects = []
    for y, s, e in found:   # a top edge is the first row of a group; pair it with the last row of the same span
        for r in rects:
            if abs(r[0] - s) <= 40 and abs(r[2] - e) <= 12 and y - r[3] <= 3:
                r[3] = y; break
        else:
            rects.append([s, y, e, y])
    # top and bottom edges of one rectangle share x; join them
    out, used = [], set()
    for i, a in enumerate(rects):
        if i in used:
            continue
        for j in range(i + 1, len(rects)):
            b = rects[j]
            # tick marks on a spine can lengthen one edge to the left, so the box takes the inner left edge
            if j not in used and abs(a[0] - b[0]) <= 40 and abs(a[2] - b[2]) <= 12 and b[1] - a[3] > 50:
                out.append((max(a[0], b[0]), a[1], min(a[2], b[2]), b[3])); used |= {i, j}; break
    return sorted(out, key=lambda r: (round(r[1] / 200), r[0]))


gray_frame = lambda im: (np.abs(im - 179).max(axis=2) < 35) & (np.ptp(im, axis=2) < 12)
black_spine = lambda im: im.max(axis=2) < 60
saturated = lambda im: np.ptp(im, axis=2) > 60


def rgb(path):
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[-1]); im = bg
    return np.asarray(im.convert("RGB")).astype(np.int16)


def sub(img, x0, y0, x1, y1):
    return img[y0:y1, x0:x1]


def shift(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy)


def bar_box(img):
    """Bounding box of a colour bar's saturated strip, extend triangles included."""
    xs = np.where(saturated(img).sum(axis=0) > 0.3 * img.shape[0])[0]
    # the bar's grey midpoint isn't saturated, so its rows are found as non-white; the longest run excludes the labels
    filled = (img[:, xs.min():xs.max() + 1].min(axis=2) < 245).any(axis=1)
    y0, y1 = max(runs(filled, 1), key=lambda r: r[1] - r[0])
    return xs.min(), y0, xs.max() + 1, y1


def locate(target, b_png, maps_png):
    """Matching element boxes in the Overleaf PNG (T) and in the renders (S)."""
    T, S = {}, {}
    tf = [shift(r, 2900, 0) for r in boxes(sub(target, 2900, 0, target.shape[1], target.shape[0]), gray_frame, 0.35)]
    T["c"], T["d"] = tf[:4], tf[4:6]
    h = maps_png.shape[0]
    sf = boxes(maps_png[int(0.35 * h):], gray_frame, 0.12)
    sf = [shift(r, 0, int(0.35 * h)) for r in sf]
    S["c"], S["d"] = sf[:4], sf[4:6]
    T["b"] = shift(boxes(sub(target, 0, 3100, 2950, target.shape[0]), black_spine, 0.3)[0], 0, 3100)
    S["b"] = boxes(b_png, black_spine, 0.3)[0]
    # d colour bar: right of the d maps in the render, between panel b and the d maps in the Overleaf PNG
    x0, y0 = S["d"][1][2] + 20, S["c"][0][3] + 40
    S["dbar"] = shift(bar_box(maps_png[y0:, x0:]), x0, y0)
    T["dbar"] = shift(bar_box(sub(target, 2550, 3300, 2950, 4750)), 2550, 3300)
    return T, S


def panel_a_right(target, tb):
    """Right edge of the panel a explainer box directly above panel b (its vertical border)."""
    band = target[tb[1] - 700:tb[1] - 80, tb[0]:tb[0] + 3000]
    cols = np.where((band.min(axis=2) < 235).sum(axis=0) > 0.8 * band.shape[0])[0]
    return int(tb[0] + cols.max() + 1)


def paste(canvas, src, sbox, tbox, pad):
    """Map src region sbox (+pad in source px: left, top, right, bottom) onto tbox, scaling x and y independently."""
    sx = (tbox[2] - tbox[0]) / (sbox[2] - sbox[0])
    sy = (tbox[3] - tbox[1]) / (sbox[3] - sbox[1])
    l, t, r, b = pad
    crop = Image.fromarray(src[sbox[1] - t:sbox[3] + b, sbox[0] - l:sbox[2] + r].astype(np.uint8))
    w, h = round((sbox[2] - sbox[0] + l + r) * sx), round((sbox[3] - sbox[1] + t + b) * sy)
    x, y = round(tbox[0] - l * sx), round(tbox[1] - t * sy)
    canvas.paste(crop.resize((w, h), Image.LANCZOS), (x, y))
    return (x, y, x + w, y + h)


def compose(target, b_png, maps_png, T, S, b_right=None):
    canvas = Image.fromarray(target.astype(np.uint8))
    region = {}
    pt = DPI / 72
    # c maps, then their titles (two 9-pt lines above the frame); the title band skips the frame's outer
    # 15% on each side, where the Overleaf PNG has the panel letter
    for i in range(4):
        sb, tb = S["c"][i], T["c"][i]
        region[f"c{i}"] = paste(canvas, maps_png, sb, tb, [round(v * pt) for v in (2, 2, 2, 2)])
        inset = round(0.15 * (sb[2] - sb[0]))
        sx = (tb[2] - tb[0]) / (sb[2] - sb[0])
        region[f"c{i}_title"] = paste(canvas, maps_png, (sb[0] + inset, sb[1] - round(28 * pt), sb[2] - inset, sb[1] - 4),
                                      (round(tb[0] + inset * sx), round(tb[1] - (28 * pt - 0) * sx),
                                       round(tb[2] - inset * sx), round(tb[1] - 4 * sx)), (0, 0, 0, 0))
    # d maps without titles (the Overleaf titles are kept)
    for i in range(2):
        tb = T["d"][i]
        # the teacher map in the Overleaf PNG came from another script and has another aspect: white it out
        # and paste the new frame at the student map's aspect, centred on the old one
        sb = S["d"][i]
        aspect = (sb[3] - sb[1]) / (sb[2] - sb[0])
        hgt = round((tb[2] - tb[0]) * aspect)
        cy = (tb[1] + tb[3]) / 2
        canvas.paste((255, 255, 255), (tb[0] - 8, tb[1] - 8, tb[2] + 8, tb[3] + 8))
        region[f"d{i}"] = paste(canvas, maps_png, sb, (tb[0], round(cy - hgt / 2), tb[2], round(cy - hgt / 2) + hgt),
                                [round(v * pt) for v in (2, 2, 2, 2)])
    # d colour bar, then its tick labels (the side labels are kept). The Overleaf bar is widened, but its
    # labels aren't, so the labels are scaled by the bar's height only and placed against its right edge
    tb, sb = T["dbar"], S["dbar"]
    canvas.paste((255, 255, 255), (tb[0] - 4, tb[1] - 4, T["d"][0][0] - 30, tb[3] + 4))
    region["dbar"] = paste(canvas, maps_png, sb, tb, (2, 2, 2, 2))
    sy = (tb[3] - tb[1]) / (sb[3] - sb[1])
    w = round(28 * pt)
    region["dbar_ticks"] = paste(canvas, maps_png, (sb[2] + 2, sb[1], sb[2] + 2 + w, sb[3]),
                                 (tb[2] + 2, tb[1], tb[2] + 2 + round(w * sy), tb[3]), (0, 0, 0, 0))
    # panel b, from just above the axes to the bottom of the page; the Overleaf panel letter sits elsewhere,
    # so the top-left corner, which holds only the letter, is restored afterwards. A widened panel b ends at
    # b_right, and the panel d letter beside its top-right corner moves right to clear it
    tb, sb = T["b"], S["b"]
    if b_right:
        tb = (tb[0], tb[1], b_right, tb[3])
        win = target[tb[1] - 60:tb[1] + 120, b_right - 40:b_right + 150]
        ys, xs = np.where(win.max(axis=2) < 80)
        d_box = (b_right - 40 + xs.min() - 4, tb[1] - 60 + ys.min() - 4, b_right - 40 + xs.max() + 5, tb[1] - 60 + ys.max() + 5)
        d_letter = Image.fromarray(target.astype(np.uint8)).crop(d_box)
    pb = (round(42 * DPI / 72), round(6 * DPI / 72), round((1 if b_right else 8) * DPI / 72), b_png.shape[0] - sb[3] - 2)
    corner = (0, round(tb[1] - 6 * pt * (tb[3] - tb[1]) / (sb[3] - sb[1])),
              round(tb[0] - 24 * pt * (tb[2] - tb[0]) / (sb[2] - sb[0])),
              round(tb[1] + 18 * pt * (tb[3] - tb[1]) / (sb[3] - sb[1])))
    letter = Image.fromarray(target.astype(np.uint8)).crop(corner)
    region["b"] = paste(canvas, b_png, sb, tb, pb)
    canvas.paste(letter, corner[:2])
    if b_right and d_box[0] + 4 < b_right + 25:
        canvas.paste((255, 255, 255), (max(d_box[0], region["b"][2]), d_box[1], d_box[2], d_box[3]))
        dx = b_right + 25 - (d_box[0] + 4)
        canvas.paste(d_letter, (d_box[0] + dx, d_box[1]))
        region["d_letter"] = (d_box[0] + dx, d_box[1], d_box[2] + dx, d_box[3])
    return canvas, region


def residual(a, b, regions):
    """Mean absolute difference (0-255) inside each pasted region."""
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    clip = lambda r: (max(r[0], 0), max(r[1], 0), r[2], r[3])
    return {k: round(float(np.abs(a[y0:y1, x0:x1] - b[y0:y1, x0:x1]).mean()), 2)
            for k, (x0, y0, x1, y1) in ((k, clip(r)) for k, r in regions.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--overleaf-figure", type=Path, default=CLONE / "figures/3_spatial_ecdf_analysis.png")
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    target = rgb(args.overleaf_figure)
    legend = {"rna": dict(frameon=True, loc="upper left", bbox_to_anchor=(.14, 1)),
              # the joint teacher curve runs through the RNA legend's place; the band between the student curve
              # and the baselines is clear
              "joint": dict(frameon=True, loc="lower right", bbox_to_anchor=(.995, .012), labelspacing=.2,
                            borderpad=.3, handlelength=1.6, handletextpad=.5)}
    summary = {}
    # the RNA-side pass rebuilds the Overleaf panel b as it is (the check); the joint pass widens it to panel a
    tb = shift(boxes(sub(target, 0, 3100, 2950, target.shape[0]), black_spine, 0.3)[0], 0, 3100)
    b_right = panel_a_right(target, tb)
    joint_width = wide_geometry()[1] * (b_right - tb[0]) / (tb[3] - tb[1])
    for tag in ("rna", "joint"):
        stats = load_stats(tag, out)
        at2 = (draw_panel_b(stats, out, tag, legend[tag]) if tag == "rna" else
               draw_panel_b(stats, out, tag, legend[tag], width=joint_width, short_labels=True))
        vlim, ahead = draw_maps(stats, out, tag)
        b_png, maps_png = rgb(out / f"panel_b_wide_{tag}.png"), rgb(out / f"maps_cd_{tag}.png")
        T, S = locate(target, b_png, maps_png)
        canvas, region = compose(target, b_png, maps_png, T, S, b_right=b_right if tag == "joint" else None)
        canvas.save(out / f"3_spatial_ecdf_analysis_{tag}.png", dpi=(300, 300))
        summary[tag] = dict(ecdf_area={a: round(float(np.nanmean(stats["auc"][a])), 4) for a in ARMS},
                            ecdf_area_expression_pcs=round(float(np.nanmean(stats["auc_pcs"])), 4),
                            floor_area=round(float(stats["floor_auc"]), 4),
                            within_2_pooled_curve={k: round(v, 4) for k, v in at2.items()},
                            d_colour_limit=round(vlim, 4), fraction_of_spots_ahead_of_MultiVI=ahead,
                            residual_vs_overleaf=residual(canvas, target, region))
        summary["panel_b_right_edge_px"] = b_right
        summary["panel_b_axes_width_pt"] = round(joint_width, 2)
        summary["boxes"] = dict(target={k: [list(map(int, r)) for r in (v if isinstance(v, list) else [v])] for k, v in T.items()})
    summary["overleaf_figure"] = str(args.overleaf_figure)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
