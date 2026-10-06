"""Layer model and the pixel math. numpy + Pillow only, so it runs anywhere."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHA_CUTOFF = 2  # alpha (0-255) at or below which a pixel counts as empty


@dataclass
class Layer:
    name: str
    rgba: np.ndarray  # h x w x 4 uint8 (straight alpha), cropped to its bounding box
    x: int
    y: int
    visible: bool = True
    opacity: int = 255

    @property
    def h(self) -> int:
        return self.rgba.shape[0]

    @property
    def w(self) -> int:
        return self.rgba.shape[1]

    def to_canvas(self, width: int, height: int) -> np.ndarray:
        out = np.zeros((height, width, 4), np.uint8)
        out[self.y:self.y + self.h, self.x:self.x + self.w] = self.rgba
        return out

    def footprint(self, width: int, height: int) -> np.ndarray:
        out = np.zeros((height, width), bool)
        out[self.y:self.y + self.h, self.x:self.x + self.w] = self.rgba[..., 3] > ALPHA_CUTOFF
        return out


def box_mean(x: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window with edge replication (summed-area table)."""
    x = x.astype(np.float32)
    k = 2 * r + 1
    extra = [(0, 0)] * (x.ndim - 2)
    p = np.pad(x, [(r, r), (r, r)] + extra, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0, dtype=np.float64), axis=1, dtype=np.float64)
    c = np.pad(c, [(1, 0), (1, 0)] + extra)
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s / (k * k)).astype(np.float32)


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    return box_mean(mask.astype(np.float32), r) > 0.0


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    return box_mean(mask.astype(np.float32), r) > 0.9999


def guided_alpha(rgb: np.ndarray, mask: np.ndarray, r: int = 4, eps: float = 1e-3, band: int = 6) -> np.ndarray:
    """Model-free soft alpha: snap a hard mask's edge to image edges (guided filter)."""
    guide = rgb.astype(np.float32).mean(axis=2) / 255.0
    p = mask.astype(np.float32)
    mi, mp = box_mean(guide, r), box_mean(p, r)
    cov = box_mean(guide * p, r) - mi * mp
    var = box_mean(guide * guide, r) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    q = np.clip(box_mean(a, r) * guide + box_mean(b, r), 0.0, 1.0)
    edge = dilate(mask, band) & ~erode(mask, band)
    return np.where(edge, q, p).astype(np.float32)


def _blur_fusion(img, fg, bg, a, r):
    ba = box_mean(a, r)
    bf = box_mean(fg * a, r) / (ba + 1e-5)
    bb = box_mean(bg * (1 - a), r) / ((1 - ba) + 1e-5)
    out = bf + a * (img - a * bf - (1 - a) * bb)
    return np.clip(out, 0.0, 1.0), bb


def estimate_foreground(rgb: np.ndarray, alpha: np.ndarray, r1: int = 45, r2: int = 3) -> np.ndarray:
    """Remove background colour bleeding into soft edges (blur-fusion foreground estimation)."""
    img = rgb.astype(np.float32) / 255.0
    a = alpha.astype(np.float32)[..., None]
    fg, bb = _blur_fusion(img, img, img, a, r1)
    fg, _ = _blur_fusion(img, fg, bb, a, r2)
    return (fg * 255.0 + 0.5).astype(np.uint8)


def make_layer(rgb: np.ndarray, alpha: np.ndarray, name: str) -> Layer | None:
    """Crop to the alpha's bounding box and decontaminate edge colours."""
    a8 = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(a8 > ALPHA_CUTOFF)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    a_crop = alpha[y0:y1, x0:x1]
    a8_crop = a8[y0:y1, x0:x1].copy()
    a8_crop[a8_crop <= ALPHA_CUTOFF] = 0
    fg = estimate_foreground(rgb[y0:y1, x0:x1], a_crop)
    return Layer(name, np.dstack([fg, a8_crop]), int(x0), int(y0))


def composite(base_rgb: np.ndarray, layers: list[Layer]) -> np.ndarray:
    """Alpha-composite layers (bottom to top) over an RGB base."""
    out = base_rgb.astype(np.float32).copy()
    for lyr in layers:
        if not lyr.visible:
            continue
        sl = (slice(lyr.y, lyr.y + lyr.h), slice(lyr.x, lyr.x + lyr.w))
        a = lyr.rgba[..., 3:4].astype(np.float32) / 255.0 * (lyr.opacity / 255.0)
        out[sl] = lyr.rgba[..., :3] * a + out[sl] * (1 - a)
    return (out + 0.5).astype(np.uint8)
