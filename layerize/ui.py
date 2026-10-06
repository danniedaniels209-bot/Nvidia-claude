"""Colab web UI, two sections:

  1. Full scan        upload -> Process -> every element found automatically -> download PSD / PNG zip
  2. Select an element draw a rectangle (or scribble) over the part you want -> Make layer -> download

    from layerize.ui import launch
    launch(engines)          # prints a public gradio.live link
"""
from __future__ import annotations

import os
import threading

import numpy as np
from PIL import Image

from .pipeline import Job


def mask_to_prompt(mask: np.ndarray) -> tuple[list[float], list[list[float]]]:
    """Painted selection -> SAM prompt: its bounding box plus a positive point at its centre of mass."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("Draw over the element first (a rectangle or a scribble on it).")
    box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    return box, [[float(xs.mean()), float(ys.mean()), 1]]


class Studio:
    """All UI logic as plain methods (no Gradio), so it can be tested without a browser."""

    def __init__(self, engines, out_dir: str = "/content/layerize_out"):
        self.engines, self.out_dir = engines, out_dir
        self.job: Job | None = None
        self.lock = threading.Lock()
        os.makedirs(out_dir, exist_ok=True)

    # ---- shared ------------------------------------------------------------------------
    def load(self, image: np.ndarray) -> Job:
        """Set the working image (no scan yet)."""
        if image is None:
            raise ValueError("Upload an image first.")
        im = Image.fromarray(np.asarray(image)).convert("RGB")
        if max(im.size) > 4096:
            im.thumbnail((4096, 4096), Image.LANCZOS)
        self.job = Job(np.asarray(im), self.engines, "studio")
        self.job.status = "ready"
        return self.job

    def _job(self, image: np.ndarray | None = None) -> Job:
        if image is not None and (self.job is None or not self._same(image)):
            self.load(image)
        if self.job is None:
            raise ValueError("Upload an image first.")
        return self.job

    def _same(self, image: np.ndarray) -> bool:
        im = np.asarray(image)[..., :3]
        return im.shape == self.job.image.shape and np.array_equal(im, self.job.image)

    def _save(self, data: bytes, name: str) -> str:
        path = os.path.join(self.out_dir, name)
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    def _export(self, ids: list[int], fill: bool, stem: str, exact: bool = False) -> list[str]:
        job = self._job()
        if not ids:
            raise ValueError("Nothing to export yet.")
        with self.lock:
            psd, _ = job.export(ids, fill, "psd", exact)
            zipped, _ = job.export(ids, fill, "zip", exact)
        return [self._save(psd, f"{stem}.psd"), self._save(zipped, f"{stem}_png_layers.zip")]

    # ---- 1. full scan ------------------------------------------------------------------
    def process(self, image: np.ndarray) -> list[int]:
        job = self._job(image)
        with self.lock:
            job.analyse()
        return list(job.regions)

    def auto_ids(self) -> list[int]:
        return [r.id for r in self._job().regions.values() if r.source == "auto"]

    def gallery(self, ids: list[int]) -> list[tuple[np.ndarray, str]]:
        job = self._job()
        return [(job.thumb(i), f"#{i} {job.regions[i].label}") for i in ids if i in job.regions]

    def export_full(self, fill: bool = True, exact: bool = True) -> list[str]:
        ids = self.auto_ids()
        if not ids:
            raise ValueError("Press Process first.")
        return self._export(ids, fill, "full_scan", exact)

    # ---- 2. manual selection -----------------------------------------------------------
    def select(self, image: np.ndarray, painted: np.ndarray) -> int:
        """Turn the element under the painted area into a layer. Returns its id."""
        job = self._job(image)
        box, points = mask_to_prompt(np.asarray(painted).astype(bool))
        with self.lock:
            return job.add_prompt_region(points, box).id

    def manual_ids(self) -> list[int]:
        return [r.id for r in self._job().regions.values() if r.source == "prompt"]

    def cutout(self, rid: int) -> np.ndarray:
        job = self._job()
        with self.lock:
            return job.preview([rid])

    def remove(self, rid: int) -> None:
        self._job().regions.pop(rid, None)

    def export_manual(self, fill: bool = True, exact: bool = False) -> list[str]:
        ids = self.manual_ids()
        if not ids:
            raise ValueError("Select at least one element first.")
        return self._export(ids, fill, "selected", exact)


def painted_mask(editor_value) -> np.ndarray:
    """Alpha of what the user drew in a gr.ImageEditor (any layer), as a bool mask."""
    if not editor_value:
        raise ValueError("Upload an image first.")
    layers = [l for l in (editor_value.get("layers") or []) if l is not None]
    if not layers:
        raise ValueError("Draw over the element first (a rectangle or a scribble on it).")
    mask = None
    for layer in layers:
        a = np.asarray(layer)
        a = a[..., 3] if a.ndim == 3 and a.shape[2] == 4 else a.max(axis=2) if a.ndim == 3 else a
        mask = (a > 0) if mask is None else mask | (a > 0)
    return mask


def build_ui(studio: Studio):
    import gradio as gr

    def run(fn):
        def wrapped(*a):
            try:
                return fn(*a)
            except Exception as exc:
                raise gr.Error(str(exc))
        return wrapped

    with gr.Blocks(title="Layerize") as demo:
        gr.Markdown("## Layerize — turn any part of an image into a layer")
        with gr.Tabs():
            # ------------------------------------------------------------- 1. full scan
            with gr.Tab("1. Full scan (automatic)"):
                gr.Markdown("Upload an image, press **Process**. Every element is found and cut out on its own. "
                            "Then **Download**: a layered PSD (background filled in behind the elements) + a zip of transparent PNGs.")
                with gr.Row():
                    with gr.Column(scale=2):
                        img_full = gr.Image(label="Image", type="numpy")
                        btn_process = gr.Button("Process", variant="primary")
                        exact_full = gr.Checkbox(label="Keep the picture exactly as it is (all layers together = original)", value=True)
                        fill_full = gr.Checkbox(label="Fill in the background behind the elements", value=True)
                        btn_dl_full = gr.Button("Download (PSD + PNG layers)")
                        status_full = gr.Markdown("")
                    with gr.Column(scale=3):
                        gal_full = gr.Gallery(label="Elements found", columns=5, height=460,
                                              allow_preview=False, object_fit="contain")
                files_full = gr.File(label="Your files", file_count="multiple")

                def process(img):
                    ids = studio.process(img)
                    auto = studio.auto_ids()
                    return studio.gallery(auto), f"Found **{len(auto)}** elements. Press Download."

                btn_process.click(run(process), [img_full], [gal_full, status_full])
                btn_dl_full.click(run(lambda f, e: studio.export_full(f, e)), [fill_full, exact_full], [files_full])

            # ------------------------------------------------------------ 2. manual select
            with gr.Tab("2. Select an element (you choose)"):
                gr.Markdown("Upload the image, then use the **brush** to draw a rectangle or scribble over the element you want. "
                            "Press **Make layer**: only that element is cut out (AI snaps to its exact edges). "
                            "Repeat for more elements, then **Download**.")
                with gr.Row():
                    with gr.Column(scale=3):
                        editor = gr.ImageEditor(label="Draw over the element", type="numpy",
                                                brush=gr.Brush(default_size=25, colors=["#ff0066"], color_mode="fixed"),
                                                layers=False, eraser=gr.Eraser(default_size=25))
                        with gr.Row():
                            btn_make = gr.Button("Make layer from my selection", variant="primary")
                            btn_undo = gr.Button("Remove last layer")
                        status_sel = gr.Markdown("")
                    with gr.Column(scale=2):
                        last_cut = gr.Image(label="Last element cut out", interactive=False)
                        gal_sel = gr.Gallery(label="Your selected elements", columns=3, height=300,
                                             allow_preview=False, object_fit="contain")
                with gr.Row():
                    fill_sel = gr.Checkbox(label="Fill in the background behind the elements", value=True)
                    exact_sel = gr.Checkbox(label="Keep original pixels (no edge clean-up)", value=False)
                btn_dl_sel = gr.Button("Download (PSD + PNG layers)")
                files_sel = gr.File(label="Your files", file_count="multiple")

                def make(value):
                    if not value or value.get("background") is None:
                        raise ValueError("Upload an image first.")
                    rid = studio.select(value["background"], painted_mask(value))
                    ids = studio.manual_ids()
                    cleared = {"background": value["background"], "layers": [], "composite": value["background"]}
                    return (studio.cutout(rid), studio.gallery(ids), f"Layer #{rid} made ({len(ids)} selected).", cleared)

                def undo():
                    ids = studio.manual_ids()
                    if ids:
                        studio.remove(ids[-1])
                    ids = studio.manual_ids()
                    return studio.gallery(ids), f"{len(ids)} selected."

                btn_make.click(run(make), [editor], [last_cut, gal_sel, status_sel, editor])
                btn_undo.click(run(undo), [], [gal_sel, status_sel])
                btn_dl_sel.click(run(lambda f, e: studio.export_manual(f, e)), [fill_sel, exact_sel], [files_sel])
    return demo


def launch(engines, share: bool = True, out_dir: str = "/content/layerize_out"):
    demo = build_ui(Studio(engines, out_dir))
    demo.queue().launch(share=share, debug=False)
    return demo
