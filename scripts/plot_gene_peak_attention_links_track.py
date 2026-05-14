#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_GTF_RELATIVE = "gene_annotations/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz"


def _default_gtf_path() -> Path | None:
    datapath = os.environ.get("DATAPATH")
    if not datapath:
        return None
    return Path(datapath) / DEFAULT_GTF_RELATIVE


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot exported MultiGATE gene-peak BEDPE links with CoolBox.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--links",
        required=True,
        help="Headerless BEDPE file produced by save_gene_peak_links_bedpe.",
    )
    p.add_argument(
        "--gtf",
        default=None,
        help="Gene annotation GTF. Defaults to $DATAPATH/gene_annotations/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz.",
    )
    p.add_argument(
        "--region",
        default=None,
        help="Optional region string, e.g. chr1:1000000-1200000. If omitted, derive from the BEDPE endpoints.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional output image path. If omitted, show the matplotlib window.",
    )
    p.add_argument(
        "--padding-bp",
        type=int,
        default=5000,
        help="Padding applied to the auto-derived region.",
    )
    p.add_argument(
        "--color",
        default="#cc4c02",
        help="Arc color passed to CoolBox Arcs.",
    )
    p.add_argument(
        "--arc-height",
        type=float,
        default=5.0,
        help="CoolBox track height for the arcs track.",
    )
    p.add_argument(
        "--gtf-height",
        type=float,
        default=3.0,
        help="CoolBox track height for the GTF track.",
    )
    p.add_argument(
        "--fig-width-inches",
        type=float,
        default=14.0,
        help=(
            "Final saved figure width in inches; height is rescaled to keep "
            "aspect ratio. Pass 0 to keep CoolBox's native width."
        ),
    )
    p.add_argument(
        "--pad-inches",
        type=float,
        default=0.25,
        help="Padding around the saved figure (in inches).",
    )
    args = p.parse_args()

    if args.padding_bp < 0:
        p.error("--padding-bp must be non-negative.")

    args.links = Path(args.links)
    if not args.links.is_file():
        p.error(f"--links does not exist or is not a file: {args.links}")

    args.gtf = Path(args.gtf) if args.gtf else _default_gtf_path()
    if args.gtf is None:
        p.error("--gtf is required when DATAPATH is not set.")
    if not args.gtf.is_file():
        p.error(f"--gtf does not exist or is not a file: {args.gtf}")

    args.out = Path(args.out) if args.out else None
    return args


def _read_bedpe_bounds(path: Path) -> tuple[str, int, int]:
    chroms: set[str] = set()
    starts: list[int] = []
    ends: list[int] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if len(fields) < 6:
                raise ValueError(
                    f"{path} line {line_no} has {len(fields)} columns; BEDPE needs at least 6."
                )

            chrom1, start1, end1, chrom2, start2, end2 = fields[:6]
            try:
                start1_i = int(start1)
                end1_i = int(end1)
                start2_i = int(start2)
                end2_i = int(end2)
            except ValueError as exc:
                raise ValueError(f"{path} line {line_no} has non-integer BEDPE coordinates.") from exc

            chroms.update([chrom1, chrom2])
            starts.extend([start1_i, start2_i])
            ends.extend([end1_i, end2_i])

    if not starts:
        raise ValueError(f"{path} does not contain any BEDPE rows.")

    if len(chroms) != 1:
        chrom_list = ", ".join(sorted(chroms))
        raise ValueError(
            f"{path} spans multiple chromosomes ({chrom_list}); pass --region explicitly."
        )

    return next(iter(chroms)), min(starts), max(ends)


def _derive_region(path: Path, padding_bp: int) -> str:
    chrom, min_start, max_end = _read_bedpe_bounds(path)
    
    # Adjust padding dynamically if the region is small
    region_len = max_end - min_start
    if region_len < 50000:
        padding_bp = min(padding_bp, 10000) # Cap padding at 10kb for small regions

    start = max(0, min_start - padding_bp)
    end = max_end + padding_bp
    if end <= start:
        end = start + 1
    return f"{chrom}:{start}-{end}"


def _resolve_figure(plot_result):
    import matplotlib.pyplot as plt

    if hasattr(plot_result, "savefig"):
        return plot_result
    if isinstance(plot_result, tuple) and plot_result and hasattr(plot_result[0], "savefig"):
        return plot_result[0]
    return plt.gcf()


def _unclip_text_artists(fig) -> None:
    """Let GTF gene labels at axis edges render beyond the axes clip box."""
    for ax in fig.get_axes():
        for txt in ax.texts:
            txt.set_clip_on(False)


def _clamp_offscreen_text_to_xlim(fig, *, edge_margin_frac: float = 0.02) -> None:
    """Clamp text artists whose x position is outside their axes' xlim.

    pyGenomeTracks/CoolBox's GTF track places gene labels at the gene midpoint,
    which can fall far outside the plotted region for long genes (e.g. Pde10a).
    With ``clip_on=False`` such labels are technically rendered, but they sit
    so far off-canvas that ``Figure.get_tightbbox`` (and ``bbox_inches='tight'``)
    blow the saved page width up to the label's coordinate. Clamping the
    x-position back inside the axes keeps the label visible at the edge of the
    track while preventing runaway horizontal expansion.
    """
    for ax in fig.get_axes():
        x0, x1 = ax.get_xlim()
        if not (x0 < x1):
            continue
        span = x1 - x0
        if span <= 0:
            continue
        margin = span * edge_margin_frac
        lo = x0 + margin
        hi = x1 - margin
        for txt in ax.texts:
            tx, ty = txt.get_position()
            try:
                tx_f = float(tx)
            except (TypeError, ValueError):
                continue
            new_tx = tx_f
            if tx_f < x0:
                new_tx = lo
            elif tx_f > x1:
                new_tx = hi
            if new_tx != tx_f:
                txt.set_position((new_tx, ty))


def _save_or_show(
    plot_result,
    out_path: Path | None,
    *,
    fig_width_inches: float | None = None,
    pad_inches: float = 0.25,
) -> None:
    import matplotlib.pyplot as plt

    fig = _resolve_figure(plot_result)
    _unclip_text_artists(fig)
    _clamp_offscreen_text_to_xlim(fig)

    if fig_width_inches is not None:
        _, current_h = fig.get_size_inches()
        fig.set_size_inches(fig_width_inches, current_h)

    if out_path is None:
        plt.show()
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=pad_inches)


def _check_coolbox_cli_dependencies() -> None:
    required = ["bgzip", "pairix", "tabix"]
    missing = [tool for tool in required if shutil.which(tool) is None]
    if missing:
        missing_str = ", ".join(missing)
        raise SystemExit(
            "CoolBox needs these command-line tools on PATH for BEDPE/GTF tracks: "
            f"{missing_str}. Install htslib (bgzip/tabix) and pairix in the CoolBox "
            "environment, or run this script from an environment that provides them."
        )


def _ensure_sorted_bedpe(path: Path) -> Path:
    """Ensure that pos1 < pos2 for all BEDPE rows so CoolBox doesn't compute negative diameters."""
    import tempfile
    out_path = Path(tempfile.gettempdir()) / (path.name + ".sorted_anchors.bedpe")
    with path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line_s = line.strip()
            if not line_s or line_s.startswith("#"):
                fout.write(line)
                continue
            fields = line_s.split("\t")
            if len(fields) >= 6:
                chrom1, start1, end1, chrom2, start2, end2 = fields[:6]
                pos1 = (int(start1) + int(end1)) / 2.0
                pos2 = (int(start2) + int(end2)) / 2.0
                if chrom1 == chrom2 and pos1 > pos2:
                    # Swap the first two anchors
                    fields[0:3], fields[3:6] = fields[3:6], fields[0:3]
            fout.write("\t".join(fields) + "\n")
    return out_path

def _patch_coolbox_plot_arcs() -> None:
    """Replace CoolBox's plot_arcs with a version that supports fill and respects score_to_width.

    CoolBox's stock implementation uses matplotlib.patches.Arc (stroke-only, so fill is silently
    ignored), and its line_width property always overrides score_to_width. This patched
    implementation:

    * skips line_width when it is None or non-numeric, falling back to score_to_width;
    * fills the area under each arc with a closed Polygon when fill=True is set on the track.
    """
    import numpy as np
    from matplotlib.patches import Arc, Polygon
    from coolbox.core.track.arcs.plot import PlotContacts

    def plot_arcs(self, ax, gr, gr2, intervals):
        properties = self.properties

        def get_height(diameter):
            key = 'diameter_to_height'
            if key in properties:
                try:
                    return eval(properties[key])
                except Exception:
                    pass
            return 0.97 * max_height * diameter / max_diameter

        def get_linewidth(score):
            lw = properties.get('line_width')
            if lw is not None:
                try:
                    return float(lw)
                except (TypeError, ValueError):
                    pass
            if 'score_to_width' in properties:
                try:
                    return eval(properties['score_to_width'])
                except Exception:
                    pass
            return 0.5 * np.sqrt(score)

        color = properties['color']
        alpha = properties['alpha']
        max_height = properties['height']

        fill = properties.get('fill', False)
        if isinstance(fill, str):
            fill = fill.lower() in ('yes', 'true', '1')
        fill_alpha = float(properties.get('fill_alpha', 0.2))
        fill_color = properties.get('fill_color', None)
        if fill_color in (None, '', 'bed_rgb', 'white'):
            fill_color = color

        if properties.get('orientation') == 'inverted':
            ax.set_ylim(max_height, 0.001)
        else:
            ax.set_ylim(-0.001, max_height)
        ax.set_xlim(gr.start, gr.end)
        if len(intervals) == 0:
            return

        max_diameter = (intervals['pos2'] - intervals['pos1']).max()
        for row in intervals.itertuples():
            start, end = row.pos1, row.pos2
            score = row.score if 'score' in intervals.columns else 1
            line_width = get_linewidth(score)
            diameter = (end - start)
            height = 2 * get_height(diameter)
            center = (start + end) / 2

            if fill:
                theta = np.linspace(0.0, np.pi, 200)
                xs = center + (diameter / 2.0) * np.cos(theta)
                ys = (height / 2.0) * np.sin(theta)
                verts = list(zip(xs, ys))
                poly = Polygon(
                    verts, closed=True,
                    facecolor=fill_color, alpha=fill_alpha,
                    edgecolor='none', zorder=1,
                )
                ax.add_patch(poly)

            arc = Arc(
                (center, 0), diameter, height, 0, 0, 180,
                color=color, alpha=alpha, lw=line_width, zorder=2,
            )
            ax.add_patch(arc)

    PlotContacts.plot_arcs = plot_arcs


def main() -> None:
    args = _parse_args()
    region = args.region or _derive_region(args.links, args.padding_bp)
    _check_coolbox_cli_dependencies()

    # Fix negative diameter issue by ensuring start < end
    fixed_links = _ensure_sorted_bedpe(args.links)

    from coolbox.api import Arcs, GTF, TrackHeight, XAxis, ChromName

    _patch_coolbox_plot_arcs()

    arcs_track = Arcs(
        str(fixed_links),
        open_region=False,
        color=args.color,
        line_width=None,
        score_to_width="score * 10",
        diameter_to_height="max_height * (diameter / max_diameter)**0.3",
        fill=True,
        fill_alpha=0.2,
        fill_color=args.color,
    )

    frame = (
        XAxis()
        + ChromName()
        + arcs_track
        + TrackHeight(args.arc_height)
        + GTF(str(args.gtf))
        + TrackHeight(args.gtf_height)
    )
    plot_result = frame.plot(region)
    fig_width = args.fig_width_inches if args.fig_width_inches and args.fig_width_inches > 0 else None
    _save_or_show(
        plot_result,
        args.out,
        fig_width_inches=fig_width,
        pad_inches=args.pad_inches,
    )

    print(f"Plotted region: {region}")
    if args.out is not None:
        print(f"Saved plot: {args.out}")


if __name__ == "__main__":
    main()
