# Layerize

Turn any flat image (e.g. AI-generated) into a library of **ready-to-use layers** for After Effects,
Photoshop and Illustrator — automatically, with no clicking, and **without downloading models to your PC**.

```
 your PC (tiny)                         Colab GPU (heavy, temporary)
 ┌──────────────┐   HTTPS tunnel   ┌──────────────────────────────────────────┐
 │ Claude Code /│ ───────────────► │ SAM 2.1  find every region automatically │
 │ Codex agent  │                  │ CLIP     label + text search            │
 │   │ MCP      │ ◄─────────────── │ ViTMatte clean transparent edges        │
 │ layerize MCP │  PNG / PSD only  │ LaMa     fill the hole behind a layer   │
 └──────────────┘                  └──────────────────────────────────────────┘
```

## Just want to do it yourself, no agent? Use the Studio
Open `colab/Layerize_Studio.ipynb` in Colab (GPU runtime), run all cells, open the public link. Two tabs:
* **Full scan** — upload, **Process**, **Download**: every element found automatically (PSD + zip of PNG layers).
* **Select an element** — draw a rectangle or scribble over the part you want, **Make layer**: just that element is cut
  out, snapped to its real edges. Repeat, then **Download**.

## 1. (Agent setup) Start the GPU side (Colab)
Open `colab/Layerize_Colab.ipynb` in Colab, select a GPU runtime, run all cells. It prints a
`LAYERIZE_URL` and `LAYERIZE_TOKEN`. Keep the last cell running. (URL changes every session.)

## 2. Add the MCP to your agent (PC side: needs only `pip install mcp` and this folder)
```bash
# Claude Code
claude mcp add layerize -e LAYERIZE_URL=<url> -e LAYERIZE_TOKEN=<token> -- python -m layerize.mcp_server
```
```toml
# Codex  (~/.codex/config.toml)
[mcp_servers.layerize]
command = "python"
args = ["-m", "layerize.mcp_server"]
cwd = "/path/to/Nvidia-claude"
env = { LAYERIZE_URL = "<url>", LAYERIZE_TOKEN = "<token>" }
```
New Colab session = new URL: tell the agent to call `layerize_connect(url, token)`.

## What the agent can do
| Tool | Purpose |
|---|---|
| `layerize_process_image(path)` | Upload + auto-split into every usable region (parts, objects, background areas). Returns `job_id` + region table (id, label, bbox, area%). |
| `layerize_contact_sheet(job_id)` | A numbered picture of the regions so a multimodal agent can *see* and choose. |
| `layerize_find_layers(job_id, "the red car")` | Text search over regions (CLIP). |
| `layerize_segment_at(job_id, x, y \| box)` | Make a region where auto-split missed one. |
| `layerize_get_layers(job_id, ids)` | Transparent PNGs (full-canvas by default → they line up at 0,0). |
| `layerize_export_psd(job_id, ids)` | One layered PSD: background plate with those regions removed & inpainted + one layer per region. |
| `layerize_background(job_id, remove_ids)` | The clean plate alone. |
| `layerize_adobe_script(app, layers)` / `layerize_after_effects_psd_script(psd)` | ExtendScript for AE / PS / Illustrator to import the files, to run via your Adobe MCP. |

Region ids ascend from big/back regions to small/front regions, so sorted ids are already a sensible stack order.

## Notes / limits
* Matting and cut-outs are produced lazily on request, so analysis stays fast.
* Overlap: a region cut from the original contains whatever was visible; hidden parts of a lower
  object are not reconstructed (the background plate *is* inpainted).
* Layers are raster (PNG/PSD). Converting to vector shapes for Illustrator is not included.
* The Cloudflare quick tunnel caps uploads at ~100 MB; images are downscaled to 4096 px on the server.
* Tested here: layer math, PSD writer (strict independent parser), server↔client↔export flow, using a toy engine.
  **Not tested here** (no GPU): the real SAM 2 / ViTMatte / LaMa / CLIP wrappers in `engines.py`, the Cloudflare tunnel,
  the MCP wrapper, and opening the PSD / JSX scripts inside real Adobe apps.
