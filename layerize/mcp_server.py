"""Local MCP server for Claude Code / Codex / any MCP client.

Run:  python -m layerize.mcp_server          (needs only `pip install mcp`)
Env:  LAYERIZE_URL, LAYERIZE_TOKEN           (printed by the Colab notebook; or call layerize_connect)
"""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP, Image

from . import adobe
from .client import LayerizeClient, LayerizeError, save_layers

mcp = FastMCP("layerize")
client = LayerizeClient()
OUT_ROOT = os.environ.get("LAYERIZE_OUT", os.path.join(os.path.expanduser("~"), "layerize_out"))


def _fmt_regions(regions: list[dict]) -> str:
    lines = ["id | label | x,y,w,h | area%"]
    for r in regions:
        x, y, w, h = r["bbox"]
        lines.append(f"{r['id']} | {r.get('label') or '-'} | {x},{y},{w},{h} | {r['area_pct']}")
    return "\n".join(lines)


@mcp.tool()
def layerize_connect(url: str, token: str) -> str:
    """Point this MCP at the Colab server (URL + token are printed by the Colab notebook)."""
    client.configure(url, token)
    return "connected" if client.health() else "server reachable but not healthy"


@mcp.tool()
def layerize_process_image(image_path: str) -> str:
    """Upload a flat image to the Colab GPU and split it into every usable region (auto, no clicking).
    Returns a job_id and the region table. Use the ids with the other layerize_* tools."""
    info = client.process(image_path)
    regs = info["regions"]
    return (f"job_id={info['job_id']} canvas={info['width']}x{info['height']} regions={len(regs)}\n"
            + _fmt_regions(regs[:60]) + ("\n(... more: call layerize_list_layers)" if len(regs) > 60 else ""))


@mcp.tool()
def layerize_list_layers(job_id: str, min_area_pct: float = 0.0, label_contains: str = "", limit: int = 60) -> str:
    """List regions of an analysed image. Ids ascend from big/back regions to small/front regions."""
    regs = [r for r in client.job(job_id)["regions"]
            if r["area_pct"] >= min_area_pct and label_contains.lower() in (r.get("label") or "").lower()]
    return _fmt_regions(regs[:limit])


@mcp.tool()
def layerize_contact_sheet(job_id: str, ids: list[int] | None = None, limit: int = 36) -> Image:
    """Return a numbered picture of the regions so you can SEE them and pick ids."""
    data = client.sheet(job_id, ids, limit=limit)
    path = os.path.join(OUT_ROOT, job_id, "sheet.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return Image(path=path)


@mcp.tool()
def layerize_find_layers(job_id: str, query: str, top_k: int = 8) -> str:
    """Find regions by text, e.g. 'the red car', 'logo', 'person's face'."""
    return "\n".join(f"id={m['id']} score={m['score']:.3f} label={m['label']}"
                     for m in client.search(job_id, query, top_k))


@mcp.tool()
def layerize_segment_at(job_id: str, x: float | None = None, y: float | None = None,
                        box: list[float] | None = None) -> str:
    """Create a new region from a pixel position (x,y) or a box [x0,y0,x1,y1] when auto-split missed it."""
    pts = [[x, y, 1]] if x is not None and y is not None else None
    r = client.segment(job_id, pts, box)
    return _fmt_regions([r])


@mcp.tool()
def layerize_get_layers(job_id: str, ids: list[int], full_canvas: bool = True, matte: bool = True,
                        out_dir: str = "") -> list[dict]:
    """Download regions as transparent PNG layers. full_canvas=True makes every PNG the size of the
    source image, so in AE/PS they line up at (0,0) with no repositioning. Returns local file paths
    (ordered bottom -> top) - import these with your Adobe tooling."""
    return save_layers(client, job_id, sorted(ids), out_dir or os.path.join(OUT_ROOT, job_id), full_canvas, matte)


@mcp.tool()
def layerize_export_psd(job_id: str, ids: list[int], fill_background: bool = True, as_zip: bool = False,
                        out_dir: str = "") -> str:
    """Download ONE layered PSD (bottom: background with the chosen regions removed and filled in; then each
    region as its own layer). In After Effects import it as a Composition. as_zip adds PNGs + manifest."""
    data, name = client.export(job_id, ids, fill_background, "zip" if as_zip else "psd")
    d = out_dir or os.path.join(OUT_ROOT, job_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, "wb") as fh:
        fh.write(data)
    return os.path.abspath(path)


@mcp.tool()
def layerize_background(job_id: str, remove_ids: list[int], out_dir: str = "") -> str:
    """Download the background plate with the given regions removed and the holes inpainted."""
    d = out_dir or os.path.join(OUT_ROOT, job_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "background.png")
    with open(path, "wb") as fh:
        fh.write(client.background(job_id, remove_ids))
    return os.path.abspath(path)


@mcp.tool()
def layerize_adobe_script(app: str, layers: list[dict]) -> str:
    """Build ExtendScript (JSX) that imports layers returned by layerize_get_layers into the active
    After Effects comp / Photoshop doc / Illustrator doc. app: aftereffects | photoshop | illustrator.
    Run the returned script with whatever Adobe MCP/tool you have."""
    if app not in adobe.APPS:
        raise LayerizeError(f"app must be one of {sorted(adobe.APPS)}")
    return adobe.APPS[app](layers)


@mcp.tool()
def layerize_after_effects_psd_script(psd_path: str) -> str:
    """JSX that imports a PSD from layerize_export_psd into After Effects as a composition (one AE layer per PSD layer)."""
    return adobe.aftereffects_psd(psd_path)


if __name__ == "__main__":
    mcp.run()
