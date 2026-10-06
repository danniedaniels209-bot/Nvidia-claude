"""GPU model wrappers (Colab). Everything is imported lazily so the module loads without torch.

  SAM 2.1 (large)  - find every region automatically + segment on demand from points/box
  ViTMatte         - soft alpha (hair, glass, glow) from a trimap built around the SAM mask
  LaMa             - fill the hole left behind when a layer is lifted out
  CLIP ViT-L/14    - label regions and answer text queries ("the red car")
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .core import dilate, erode

LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"


class Engines:
    def __init__(
        self,
        device: str | None = None,
        sam_id: str = "facebook/sam2.1-hiera-large",
        matte_id: str = "hustvl/vitmatte-base-composition-1k",
        clip_id: str = "openai/clip-vit-large-patch14",
        cache_dir: str = "/content/layerize_models",
    ):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam_id, self.matte_id, self.clip_id, self.cache_dir = sam_id, matte_id, clip_id, cache_dir
        self._predictor = self._amg = self._matte = self._lama = self._clip = None
        self._predictor_key = None

    def warm(self) -> None:
        """Download + load everything now so the first request is fast."""
        self._load_sam()
        self._load_matte()
        self._load_clip()
        self._load_lama()

    # ---- SAM 2 ----------------------------------------------------------------------------
    def _load_sam(self):
        if self._predictor is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2_hf
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2_hf(self.sam_id, device=self.device)
            self._predictor = SAM2ImagePredictor(model)
            self._amg = SAM2AutomaticMaskGenerator(
                model, points_per_side=32, points_per_batch=64, pred_iou_thresh=0.8,
                stability_score_thresh=0.92, box_nms_thresh=0.7, min_mask_region_area=100,
                multimask_output=True,
            )

    def auto_masks(self, image: np.ndarray) -> list[tuple[np.ndarray, float]]:
        self._load_sam()
        torch = self.torch
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            out = self._amg.generate(image)
        return [(m["segmentation"].astype(bool), float(m["predicted_iou"])) for m in out]

    def segment(self, image, key, points, labels, box) -> np.ndarray:
        self._load_sam()
        torch = self.torch
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            if self._predictor_key != key:
                self._predictor.set_image(image)
                self._predictor_key = key
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array(points, np.float32) if points else None,
                point_labels=np.array(labels, np.int32) if points else None,
                box=np.array(box, np.float32) if box else None,
                multimask_output=bool(len(points) == 1 and not box),
            )
        return masks[int(np.argmax(scores))].astype(bool)

    # ---- ViTMatte -------------------------------------------------------------------------
    def _load_matte(self):
        if self._matte is None:
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor

            self._matte_proc = VitMatteImageProcessor.from_pretrained(self.matte_id)
            self._matte = VitMatteForImageMatting.from_pretrained(self.matte_id).to(self.device).eval()

    def matte(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._load_matte()
        torch = self.torch
        ys, xs = np.nonzero(mask)
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        m = max(16, int(0.1 * max(bw, bh)))
        y0, y1 = max(0, ys.min() - m), min(mask.shape[0], ys.max() + 1 + m)
        x0, x1 = max(0, xs.min() - m), min(mask.shape[1], xs.max() + 1 + m)
        crop, mcrop = image[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        k = int(np.clip(0.03 * max(bw, bh), 4, 40))
        sure_fg, sure_bg = erode(mcrop, k), ~dilate(mcrop, k)
        tri = np.full(mcrop.shape, 128, np.uint8)
        tri[sure_fg], tri[sure_bg] = 255, 0

        ch, cw = crop.shape[:2]
        scale = min(1.0, 1280.0 / max(ch, cw))
        size = (max(32, int(cw * scale)), max(32, int(ch * scale)))
        im_s = Image.fromarray(crop).resize(size, Image.BICUBIC)
        tri_s = Image.fromarray(tri).resize(size, Image.NEAREST)
        inputs = self._matte_proc(images=im_s, trimaps=tri_s, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            alphas = self._matte(**inputs).alphas
        a = alphas[0, 0, :size[1], :size[0]].float().cpu().numpy()
        a = np.asarray(Image.fromarray(a, "F").resize((cw, ch), Image.BICUBIC), np.float32)
        a = np.clip(a, 0.0, 1.0)
        a[sure_fg], a[sure_bg] = 1.0, 0.0
        out = np.zeros(mask.shape, np.float32)
        out[y0:y1, x0:x1] = a
        return out

    # ---- LaMa -----------------------------------------------------------------------------
    def _load_lama(self):
        if self._lama is None:
            torch = self.torch
            os.makedirs(self.cache_dir, exist_ok=True)
            path = os.path.join(self.cache_dir, "big-lama.pt")
            if not os.path.exists(path):
                torch.hub.download_url_to_file(LAMA_URL, path)
            self._lama = torch.jit.load(path, map_location=self.device).eval()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        H, W = mask.shape
        try:
            self._load_lama()
            torch = self.torch
            scale = min(1.0, 2048.0 / max(H, W))
            img, msk = image, mask
            if scale < 1.0:
                size = (int(W * scale), int(H * scale))
                img = np.asarray(Image.fromarray(image).resize(size, Image.LANCZOS))
                msk = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0
            h, w = msk.shape
            it = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
            mt = torch.from_numpy(msk.astype(np.float32))[None, None]
            ph, pw = (-h) % 8, (-w) % 8
            it = torch.nn.functional.pad(it, (0, pw, 0, ph), mode="reflect")
            mt = torch.nn.functional.pad(mt, (0, pw, 0, ph), mode="reflect")
            with torch.inference_mode():
                res = self._lama(it.to(self.device), (mt > 0).float().to(self.device))
            res = res[0].permute(1, 2, 0).float().cpu().numpy()[:h, :w]
            res = np.clip(res * 255.0, 0, 255).astype(np.uint8)
            if scale < 1.0:
                res = np.asarray(Image.fromarray(res).resize((W, H), Image.LANCZOS))
        except Exception as exc:
            print(f"[layerize] LaMa unavailable ({exc}); falling back to OpenCV inpaint")
            import cv2

            res = cv2.inpaint(image, mask.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        return np.where(mask[..., None], res, image)

    # ---- CLIP -----------------------------------------------------------------------------
    def _load_clip(self):
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor

            dtype = self.torch.float16 if self.device == "cuda" else self.torch.float32
            self._clip_proc = CLIPProcessor.from_pretrained(self.clip_id)
            self._clip = CLIPModel.from_pretrained(self.clip_id, torch_dtype=dtype).to(self.device).eval()
            self._clip_dtype = dtype

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        self._load_clip()
        torch, outs = self.torch, []
        for i in range(0, len(images), 32):
            inp = self._clip_proc(images=images[i:i + 32], return_tensors="pt")
            with torch.inference_mode():
                f = self._clip.get_image_features(pixel_values=inp["pixel_values"].to(self.device, self._clip_dtype))
            outs.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(outs)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        self._load_clip()
        torch = self.torch
        inp = self._clip_proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            f = self._clip.get_text_features(**inp)
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()
