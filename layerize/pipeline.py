"""A Job = one analysed image: every usable region found, labelled, and cut out on demand."""
from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .core import Layer, composite, dilate, guided_alpha, make_layer
from .psd import build_psd

# Zero-shot vocabulary used to give every region a human/agent-readable label.
VOCAB = [
    "person", "face", "hair", "hand", "arm", "leg", "head", "eye", "mouth", "shirt", "jacket", "dress",
    "pants", "shoe", "hat", "glasses", "bag", "watch", "jewelry", "animal", "dog", "cat", "bird", "horse",
    "fish", "flower", "plant", "tree", "grass", "leaf", "mountain", "rock", "water", "sea", "river", "sky",
    "cloud", "sun", "moon", "star", "fire", "smoke", "snow", "road", "building", "house", "window", "door",
    "wall", "floor", "roof", "bridge", "city", "car", "truck", "bike", "motorcycle", "boat", "plane", "train",
    "chair", "table", "sofa", "bed", "lamp", "bottle", "cup", "plate", "food", "fruit", "book", "phone",
    "laptop", "screen", "camera", "guitar", "ball", "toy", "robot", "sword", "weapon", "logo", "text",
    "title text", "icon", "button", "frame", "pattern", "shadow", "light", "background", "texture", "sign",
    "banner", "poster", "painting", "statue", "balloon", "crown", "wing", "tail", "cloth", "ribbon",
]


@dataclass
class Region:
    id: int
    x0: int
    y0: int
    mask: np.ndarray  # cropped bool mask; covers [y0:y0+h, x0:x0+w]
    score: float
    source: str = "auto"
    label: str = ""
    label_score: float = 0.0

    @property
    def h(self) -> int:
        return self.mask.shape[0]

    @property
    def w(self) -> int:
        return self.mask.shape[1]

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def _region_from_mask(mask: np.ndarray, rid: int, score: float, source: str) -> Region | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return Region(rid, int(x0), int(y0), mask[y0:y1, x0:x1].copy(), float(score), source)


def _iou(a: Region, b: Region) -> float:
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x0 + a.w, b.x0 + b.w), min(a.y0 + a.h, b.y0 + b.h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    ca = a.mask[y0 - a.y0:y1 - a.y0, x0 - a.x0:x1 - a.x0]
    cb = b.mask[y0 - b.y0:y1 - b.y0, x0 - b.x0:x1 - b.x0]
    inter = int(np.logical_and(ca, cb).sum())
    return inter / float(a.area + b.area - inter)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "layer"


class Job:
    def __init__(self, image: np.ndarray, engines, job_id: str):
        self.id = job_id
        self.image = np.ascontiguousarray(image[..., :3])
        self.engines = engines
        self.regions: dict[int, Region] = {}
        self.status = "processing"
        self.error: str | None = None
        self._next_id = 1
        self._layers: dict[tuple[int, bool, bool], Layer] = {}
        self._embeds: dict[int, np.ndarray] = {}
        self.analysed = False

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    # ---- analysis -------------------------------------------------------------------------
    def analyse(self, max_regions: int = 120, min_area_frac: float = 0.0005, dedupe_iou: float = 0.9) -> None:
        raw = self.engines.auto_masks(self.image)
        min_area = max(16, int(self.height * self.width * min_area_frac))
        cands = []
        for mask, score in raw:
            if int(mask.sum()) >= min_area:
                reg = _region_from_mask(mask, 0, score, "auto")
                if reg is not None:
                    cands.append(reg)
        cands.sort(key=lambda r: -r.score)
        kept: list[Region] = []
        for reg in cands:
            if all(_iou(reg, k) < dedupe_iou for k in kept):
                kept.append(reg)
            if len(kept) >= max_regions:
                break
        kept.sort(key=lambda r: -r.area)  # big/back first, small/front last
        for reg in kept:
            reg.id = self._next_id
            self._next_id += 1
            self.regions[reg.id] = reg
        self._label(list(self.regions.values()))
        self.analysed = True

    def _crop_preview(self, reg: Region) -> Image.Image:
        sl = (slice(reg.y0, reg.y0 + reg.h), slice(reg.x0, reg.x0 + reg.w))
        crop = np.where(reg.mask[..., None], self.image[sl], 127).astype(np.uint8)
        side = max(reg.w, reg.h)
        canvas = np.full((side, side, 3), 127, np.uint8)
        oy, ox = (side - reg.h) // 2, (side - reg.w) // 2
        canvas[oy:oy + reg.h, ox:ox + reg.w] = crop
        return Image.fromarray(canvas)

    def _label(self, regs: list[Region]) -> None:
        if not regs:
            return
        try:
            img_emb = self.engines.embed_images([self._crop_preview(r) for r in regs])
            vocab_emb = self.engines.embed_text([f"a photo of {w}" for w in VOCAB])
        except Exception as exc:  # labelling is a nicety; never fail the job over it
            print(f"[layerize] labelling skipped: {exc}")
            return
        sims = img_emb @ vocab_emb.T
        for reg, emb, row in zip(regs, img_emb, sims):
            self._embeds[reg.id] = emb
            j = int(np.argmax(row))
            reg.label, reg.label_score = VOCAB[j], float(row[j])

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        if not self._embeds:
            raise RuntimeError("no region embeddings available (labelling failed)")
        q = self.engines.embed_text([query])[0]
        ids = list(self._embeds)
        sims = np.array([float(self._embeds[i] @ q) for i in ids])
        order = np.argsort(-sims)[:top_k]
        return [{"id": ids[i], "score": float(sims[i]), "label": self.regions[ids[i]].label} for i in order]

    def add_prompt_region(self, points: list[list[float]] | None, box: list[float] | None) -> Region:
        pts = [(float(p[0]), float(p[1])) for p in (points or [])]
        lbl = [int(p[2]) if len(p) > 2 else 1 for p in (points or [])]
        mask = self.engines.segment(self.image, self.id, pts, lbl, box)
        reg = _region_from_mask(mask, self._next_id, 1.0, "prompt")
        if reg is None:
            raise ValueError("segmentation returned an empty mask")
        self._next_id += 1
        self.regions[reg.id] = reg
        self._label([reg])
        return reg

    # ---- cutting --------------------------------------------------------------------------
    def full_mask(self, rid: int) -> np.ndarray:
        reg = self.regions[rid]
        out = np.zeros((self.height, self.width), bool)
        out[reg.y0:reg.y0 + reg.h, reg.x0:reg.x0 + reg.w] = reg.mask
        return out

    def layer_name(self, rid: int) -> str:
        reg = self.regions[rid]
        return f"{rid:03d}_{_safe(reg.label or 'region')}"

    def layer(self, rid: int, matte: bool = True, exact: bool = False) -> Layer:
        """exact=True keeps original pixels (stack reproduces the picture); False cleans edge colours."""
        key = (rid, matte, exact)
        if key not in self._layers:
            mask = self.full_mask(rid)
            alpha = None
            if matte:
                try:
                    alpha = self.engines.matte(self.image, mask)
                except Exception as exc:
                    print(f"[layerize] matting failed for {rid}, using guided filter: {exc}")
            if alpha is None:
                alpha = guided_alpha(self.image, mask)
            lyr = make_layer(self.image, alpha, self.layer_name(rid), decontaminate=not exact)
            if lyr is None:
                raise ValueError(f"region {rid} is empty after matting")
            self._layers[key] = lyr
        return self._layers[key]

    def background(self, remove_ids: list[int], grow: int | None = None) -> np.ndarray:
        """The image with the given regions removed and the holes filled in."""
        if not remove_ids:
            return self.image
        hole = np.zeros((self.height, self.width), bool)
        for rid in remove_ids:
            hole |= self.layer(rid).footprint(self.width, self.height)
        grow = grow if grow is not None else max(4, int(0.004 * max(self.height, self.width)))
        return self.engines.inpaint(self.image, dilate(hole, grow))

    # ---- output ---------------------------------------------------------------------------
    def manifest(self) -> dict:
        regs = []
        for reg in self.regions.values():
            regs.append({
                "id": reg.id, "label": reg.label, "label_score": round(reg.label_score, 3),
                "bbox": [reg.x0, reg.y0, reg.w, reg.h], "area": reg.area,
                "area_pct": round(100.0 * reg.area / (self.width * self.height), 2),
                "source": reg.source,
            })
        return {"job_id": self.id, "status": self.status, "width": self.width, "height": self.height,
                "regions": regs}

    def thumb(self, rid: int, tile: int = 160) -> np.ndarray:
        """Region cut-out over a checkerboard, as a tile x tile RGB array."""
        reg = self.regions[rid]
        checker = np.indices((tile, tile)).sum(axis=0) // 12 % 2
        checker = np.where(checker[..., None] == 0, 200, 150).astype(np.uint8).repeat(3, axis=2)
        sl = (slice(reg.y0, reg.y0 + reg.h), slice(reg.x0, reg.x0 + reg.w))
        im = Image.fromarray(np.dstack([self.image[sl], reg.mask.astype(np.uint8) * 255]), "RGBA")
        im.thumbnail((tile - 8, tile - 8))
        bg = Image.fromarray(checker).convert("RGBA")
        bg.alpha_composite(im, ((tile - im.width) // 2, (tile - im.height) // 2))
        return np.asarray(bg.convert("RGB"))

    def preview(self, ids: list[int]) -> np.ndarray:
        """Selected layers stacked over a checkerboard (RGB), to eyeball what you picked."""
        checker = np.indices((self.height, self.width)).sum(axis=0) // 16 % 2
        base = np.where(checker[..., None] == 0, 200, 150).astype(np.uint8).repeat(3, axis=2)
        return composite(base, [self.layer(i) for i in self._ordered(ids)])

    def sheet(self, ids: list[int], cols: int = 6, tile: int = 180) -> bytes:
        """Numbered contact sheet (PNG) so a multimodal agent can *see* and pick regions."""
        ids = [i for i in ids if i in self.regions]
        cols = max(1, min(cols, max(1, len(ids))))
        rows = (len(ids) + cols - 1) // cols
        label_h = 16
        sheet = Image.new("RGB", (cols * tile, max(1, rows) * (tile + label_h)), (40, 40, 40))
        draw = ImageDraw.Draw(sheet)
        for n, rid in enumerate(ids):
            reg = self.regions[rid]
            cx, cy = (n % cols) * tile, (n // cols) * (tile + label_h)
            sheet.paste(Image.fromarray(self.thumb(rid, tile)), (cx, cy))
            draw.text((cx + 3, cy + tile + 2), f"#{rid} {reg.label}"[:28], fill=(255, 255, 255))
        buf = io.BytesIO()
        sheet.save(buf, "PNG")
        return buf.getvalue()

    def _ordered(self, ids: list[int]) -> list[int]:
        return sorted({i for i in ids if i in self.regions})  # id order == big/back -> small/front

    def png(self, rid: int, matte: bool = True, canvas: bool = False, exact: bool = False) -> tuple[bytes, Layer]:
        lyr = self.layer(rid, matte, exact)
        arr = lyr.to_canvas(self.width, self.height) if canvas else lyr.rgba
        buf = io.BytesIO()
        Image.fromarray(arr, "RGBA").save(buf, "PNG")
        return buf.getvalue(), lyr

    def export(self, ids: list[int], fill_background: bool = True, fmt: str = "psd",
               exact: bool = False) -> tuple[bytes, str]:
        """exact=True: all layers visible == the original picture, pixel for pixel (when not filling)."""
        ids = self._ordered(ids)
        if not ids:
            raise ValueError("no valid layer ids")
        layers = [self.layer(i, exact=exact) for i in ids]
        bg_rgb = self.background(ids) if fill_background else self.image
        stack = [Layer("Background" if fill_background else "Original (flat)", np.dstack(
            [bg_rgb, np.full(bg_rgb.shape[:2], 255, np.uint8)]), 0, 0)] + layers
        flat = composite(bg_rgb, layers)
        psd = build_psd(self.width, self.height, stack, flat)
        if fmt == "psd":
            return psd, "layers.psd"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("layers.psd", psd)
            manifest = []
            for n, lyr in enumerate(stack):
                b = io.BytesIO()
                Image.fromarray(lyr.rgba, "RGBA").save(b, "PNG")
                fname = f"layers/{n:03d}_{_safe(lyr.name)}.png"
                z.writestr(fname, b.getvalue())
                b = io.BytesIO()  # full-canvas copy: lines up at (0,0) with no repositioning
                Image.fromarray(lyr.to_canvas(self.width, self.height), "RGBA").save(b, "PNG")
                z.writestr(f"full_canvas/{n:03d}_{_safe(lyr.name)}.png", b.getvalue())
                manifest.append({"file": fname, "name": lyr.name, "x": lyr.x, "y": lyr.y, "w": lyr.w, "h": lyr.h})
            b = io.BytesIO()
            Image.fromarray(flat).save(b, "PNG")
            z.writestr("preview.png", b.getvalue())
            z.writestr("manifest.json", json.dumps(
                {"width": self.width, "height": self.height, "layers": manifest}, indent=2))
        return buf.getvalue(), "layers.zip"
