"""Assemble screenshot panels a--g without writing to the manuscript repository.

Uses the revised a/c/d/e panels and the original b/f/g vector exports.
The only crop removes the standalone headline/footer and empty side margins
of f; every scientific plot, title, and colorbar is retained.
Run with system python3 (librsvg and libcairo are used through ctypes).
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
import copy
import ctypes as C
import ctypes.util
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET

REPO = Path(os.environ["SPATIALJEPA_REPO"])
BASE = Path(os.environ["SPATIALJEPA_HOME"])
REVISION = BASE / "outputs/rna_atac_figures/fig4_revision_20261020"
SHORTLIST = BASE / "outputs/rna_atac_figures/fig4_linked_peak_candidates_20261018/bivariate_gene_peak/shortlist"
MANUSCRIPT = Path(os.environ['SPATIALJEPA_MANUSCRIPT'])
SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
INKSCAPE = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", XLINK)
ET.register_namespace("inkscape", INKSCAPE)
PAGE = (878.4, 690.0)  # points


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manuscript_hashes():
    paths = [MANUSCRIPT / "main.tex", MANUSCRIPT / "supplementary.tex"]
    paths += [MANUSCRIPT / f"figures/2_source_target_aligned.{ext}"
              for ext in ("svg", "pdf", "png")]
    return {str(p): digest(p) for p in paths if p.exists()}


def tag(name):
    return f"{{{SVG}}}{name}"


def namespace_ids(root, prefix):
    """Keep clip paths, fonts, and image references independent between panels."""
    mapping = {e.attrib["id"]: prefix + e.attrib["id"]
               for e in root.iter() if "id" in e.attrib}
    for e in root.iter():
        for key, value in list(e.attrib.items()):
            if key == "id":
                e.set(key, mapping[value])
            elif key.endswith("href") and value.startswith("#"):
                e.set(key, "#" + mapping.get(value[1:], value[1:]))
            else:
                e.set(key, re.sub(r"url\(#([^)]+)\)",
                                 lambda m: f"url(#{mapping.get(m[1], m[1])})", value))


def add_panel(root, label, path, x, y, scale, crop=None, label_y=None):
    panel = ET.parse(path).getroot()
    original_view = [float(v) for v in panel.attrib["viewBox"].split()]
    view = original_view if crop is None else crop
    width, height = view[2] * scale, view[3] * scale
    assert x >= 0 and y >= 0 and x+width <= PAGE[0] and y+height <= PAGE[1]
    namespace_ids(panel, label + "_")
    # Ordinary groups are directly selectable/movable in Inkscape. Keep the
    # panel letter in the same group, outside the scaled artwork and crop.
    outer = ET.SubElement(root, tag("g"), {
        "id": f"panel_{label}",
        f"{{{INKSCAPE}}}label": f"Panel {label}",
        "transform": f"translate({x} {y})",
    })
    scaled = ET.SubElement(outer, tag("g"), {"transform": f"scale({scale})"})
    if crop is not None:
        definitions = ET.SubElement(scaled, tag("defs"))
        clip_id = f"{label}_panel_crop"
        clip = ET.SubElement(definitions, tag("clipPath"), {
            "id": clip_id, "clipPathUnits": "userSpaceOnUse"})
        ET.SubElement(clip, tag("rect"), {
            "x": "0", "y": "0", "width": str(view[2]), "height": str(view[3])})
        scaled = ET.SubElement(scaled, tag("g"), {"clip-path": f"url(#{clip_id})"})
    container = ET.SubElement(scaled, tag("g"), {
        "transform": f"translate({-view[0]} {-view[1]})"})
    for element in panel:
        if element.tag != tag("metadata"):
            container.append(copy.deepcopy(element))
    title = ET.SubElement(outer, tag("text"), {
        "x": "0", "y": str(label_y-y if label_y is not None else -8),
        "font-family": "DejaVu Sans", "font-size": "16", "font-weight": "bold",
    })
    title.text = label
    return {
        "input": str(path), "sha256": digest(path),
        "native_viewbox_pt": original_view, "crop_viewbox_pt": view,
        "placement_pt": [x, y, width, height], "uniform_scale": scale,
    }


class Rectangle(C.Structure):
    _fields_ = [(k, C.c_double) for k in ("x", "y", "width", "height")]


def render(svg_path, output, dpi):
    """Render directly from the assembled SVG; PDF retains vector elements."""
    cairo = C.CDLL(ctypes.util.find_library("cairo"))
    rsvg = C.CDLL(ctypes.util.find_library("rsvg-2"))
    gobject = C.CDLL(ctypes.util.find_library("gobject-2.0"))
    cairo.cairo_pdf_surface_create.argtypes = [C.c_char_p, C.c_double, C.c_double]
    cairo.cairo_pdf_surface_create.restype = C.c_void_p
    cairo.cairo_image_surface_create.argtypes = [C.c_int, C.c_int, C.c_int]
    cairo.cairo_image_surface_create.restype = C.c_void_p
    for name in ("cairo_create",):
        getattr(cairo, name).argtypes = [C.c_void_p]
        getattr(cairo, name).restype = C.c_void_p
    for name in ("cairo_destroy", "cairo_surface_destroy", "cairo_surface_finish",
                 "cairo_show_page"):
        getattr(cairo, name).argtypes = [C.c_void_p]
    cairo.cairo_surface_status.argtypes = [C.c_void_p]
    cairo.cairo_surface_status.restype = C.c_int
    cairo.cairo_surface_write_to_png.argtypes = [C.c_void_p, C.c_char_p]
    cairo.cairo_surface_write_to_png.restype = C.c_int
    rsvg.rsvg_handle_new_from_file.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
    rsvg.rsvg_handle_new_from_file.restype = C.c_void_p
    rsvg.rsvg_handle_render_document.argtypes = [
        C.c_void_p, C.c_void_p, C.POINTER(Rectangle), C.POINTER(C.c_void_p)]
    rsvg.rsvg_handle_render_document.restype = C.c_int
    gobject.g_object_unref.argtypes = [C.c_void_p]
    error = C.c_void_p()
    handle = rsvg.rsvg_handle_new_from_file(str(svg_path).encode(), C.byref(error))
    if not handle:
        raise RuntimeError(f"Cannot read SVG: {svg_path}")
    for suffix, resolution in ((".pdf", 72), (".png", dpi), ("_preview.png", 120)):
        width, height = [v*resolution/72 for v in PAGE]
        target = output.parent / (output.name + suffix)
        if suffix == ".pdf":
            surface = cairo.cairo_pdf_surface_create(str(target).encode(), width, height)
        else:
            width, height = round(width), round(height)
            surface = cairo.cairo_image_surface_create(0, width, height)
        context = cairo.cairo_create(surface)
        viewport = Rectangle(0, 0, width, height)
        ok = rsvg.rsvg_handle_render_document(handle, context, C.byref(viewport), C.byref(error))
        if not ok:
            raise RuntimeError(f"Cannot render {target}")
        if suffix == ".pdf":
            cairo.cairo_show_page(context)
        elif cairo.cairo_surface_write_to_png(surface, str(target).encode()):
            raise RuntimeError(f"Cannot write {target}")
        cairo.cairo_surface_finish(surface)
        assert cairo.cairo_surface_status(surface) == 0
        cairo.cairo_destroy(context)
        cairo.cairo_surface_destroy(surface)
        print(target)
    gobject.g_object_unref(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REVISION / "assembly")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    before = manuscript_hashes()
    root = ET.Element(tag("svg"), {
        "width": f"{PAGE[0]}pt", "height": f"{PAGE[1]}pt",
        "viewBox": f"0 0 {PAGE[0]} {PAGE[1]}", "version": "1.1",
    })
    ET.SubElement(root, tag("rect"), {
        "width": str(PAGE[0]), "height": str(PAGE[1]), "fill": "white"})
    manifest = {}
    def panel(label, path, x, y, scale, crop=None, label_y=None):
        manifest[label] = add_panel(root, label, path, x, y, scale, crop, label_y)

    panel("a", REVISION / "panel_a_cluster_spatial_umap.svg", 8, 24, .78, label_y=16)
    panel("b", BASE / "outputs/rna_atac_transfer_clusters/shared_params/"
          "fig4c_shared_k150_r0p9_manuscript.svg", 314, 24, 1.5, label_y=16)
    right_scale = 320/(7.9*72)
    panel("c", REVISION / "panel_c_rna_pls.svg", 550, 24, right_scale, label_y=16)
    panel("d", REVISION / "panel_d_loading_enrichment_wide.svg",
          550, 258, right_scale, label_y=250)
    panel("e", REVISION / "panel_e_marker_scores.svg", 8, 382, .78, label_y=374)
    panel("f", SHORTLIST / "shortlist_peaks_two_row_smooth1_no_mal.svg",
          314, 396.82, .78, crop=[180, 24, 365, 340], label_y=374)
    panel("g", SHORTLIST / "shortlist_paired_bivariate_mean_no_mal.svg",
          630, 399, 1.17, label_y=374)
    assert manifest["e"]["uniform_scale"] == manifest["f"]["uniform_scale"]
    assert manifest["c"]["placement_pt"][0] == manifest["d"]["placement_pt"][0]
    assert manifest["c"]["placement_pt"][2] == manifest["d"]["placement_pt"][2]
    for first, a in manifest.items():
        x, y, w, h = a["placement_pt"]
        for second, b in manifest.items():
            if first >= second:
                continue
            xx, yy, ww, hh = b["placement_pt"]
            assert x+w <= xx or xx+ww <= x or y+h <= yy or yy+hh <= y, (first, second)
    stem = args.output / "figure4_assembled"
    svg_path = stem.with_suffix(".svg")
    ET.ElementTree(root).write(svg_path, encoding="utf-8", xml_declaration=True)
    shutil.copyfile(svg_path, args.output / "figure4_assembled_inkscape.svg")
    render(svg_path, stem, args.dpi)
    assert before == manuscript_hashes(), "A protected manuscript file changed."
    (args.output / "assembly_manifest.json").write_text(json.dumps({
        "canvas_points": PAGE, "png_dpi": args.dpi, "panels": manifest,
        "manuscript_files_unchanged_sha256": before,
        "excluded": ["supplementary ATAC rows", "exploratory Phactr1 RNA maps"],
        "note": "Panel f cropped to scientific content; no anisotropic scaling.",
        "editing": "Seven named ordinary SVG groups; panel letters move with panels.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
