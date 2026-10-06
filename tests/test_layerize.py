"""Runs without GPU/torch: a toy engine stands in for the models.  `python tests/test_layerize.py`"""
import io
import json
import os
import sys
import tempfile
import zipfile

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from layerize import adobe, psd, server  # noqa: E402
from layerize.client import LayerizeClient, save_layers  # noqa: E402
from layerize.core import box_mean, dilate, erode, estimate_foreground, make_layer  # noqa: E402

COLORS = {"red": (230, 20, 20), "green": (20, 200, 40), "blue": (20, 40, 230)}


class ToyEngines:
    """Colour-based stand-in for SAM/ViTMatte/LaMa/CLIP."""

    def auto_masks(self, image):
        out = []
        for c in COLORS.values():
            m = (np.abs(image.astype(int) - np.array(c)).sum(axis=2) < 40)
            if m.any():
                out.append((m, 0.9))
        out.append((np.ones(image.shape[:2], bool), 0.5))  # whole-image region
        out.append((out[0][0].copy(), 0.8))  # exact duplicate -> must be deduped
        return out

    def segment(self, image, key, points, labels, box):
        x, y = map(int, points[0])
        return np.abs(image.astype(int) - image[y, x].astype(int)).sum(axis=2) < 40

    def matte(self, image, mask):
        return np.clip(box_mean(mask.astype(np.float32), 1) * 1.2, 0, 1)

    def inpaint(self, image, mask):
        out = image.copy()
        out[mask] = 200
        return out

    def embed_images(self, images):
        vs = []
        for im in images:
            a = np.asarray(im, np.float32).reshape(-1, 3)
            a = a[np.abs(a - 127).sum(axis=1) > 3]
            v = a.mean(axis=0) if len(a) else np.ones(3)
            vs.append(v / (np.linalg.norm(v) + 1e-9))
        return np.array(vs)

    def embed_text(self, texts):
        out = []
        for t in texts:
            v = np.ones(3, np.float32) * 0.1
            for i, c in enumerate(COLORS):
                if c in t:
                    v = np.eye(3, dtype=np.float32)[i]
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def make_image():
    im = Image.new("RGB", (200, 160), (200, 200, 200))
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, 100, 90], fill=COLORS["red"])
    d.ellipse([70, 50, 150, 130], fill=COLORS["blue"])
    d.rectangle([160, 10, 190, 40], fill=COLORS["green"])
    return im


def unpackbits(data: bytes, n: int) -> bytes:
    out, i = bytearray(), 0
    while len(out) < n:
        h = data[i]
        i += 1
        if h < 128:
            out += data[i:i + h + 1]
            i += h + 1
        elif h > 128:
            out += bytes([data[i]]) * (257 - h)
            i += 1
    return bytes(out)


def test_morphology():
    m = np.zeros((20, 20), bool)
    m[8:12, 8:12] = True
    assert dilate(m, 2).sum() == 8 * 8 and erode(m, 1).sum() == 4


def test_packbits_roundtrip():
    rng = np.random.default_rng(0)
    rows = [rng.integers(0, 256, 300, dtype=np.uint8), np.zeros(300, np.uint8), np.full(1, 7, np.uint8),
            np.repeat(rng.integers(0, 3, 40, dtype=np.uint8), 9), np.array([5] * 129 + [6], np.uint8)]
    for row in rows:
        assert unpackbits(psd._packbits_row(row), row.size) == row.tobytes()


def test_decontamination_removes_bleed():
    h, w = 20, 40
    a = np.clip((np.arange(w) - 15) / 10.0, 0, 1)[None, :].repeat(h, 0)  # soft edge
    red, blue = np.array([230, 20, 20], np.float32), np.array([20, 40, 230], np.float32)
    img = (a[..., None] * red + (1 - a[..., None]) * blue).astype(np.uint8)
    fg = estimate_foreground(img, a)
    edge = (a > 0.3) & (a < 0.9)
    before = np.abs(img[edge].astype(float) - red).mean()
    after = np.abs(fg[edge].astype(float) - red).mean()
    assert after < before * 0.5, (before, after)


def read_psd_layers(data: bytes):
    """Strict, independent PSD parser (per Adobe's file-format spec) used to verify the writer."""
    import struct

    assert data[:4] == b"8BPS" and struct.unpack(">H", data[4:6])[0] == 1
    channels, height, width, depth, mode = struct.unpack(">HIIHH", data[12:26])
    assert (channels, depth, mode) == (3, 8, 3)
    pos = 26
    for _ in range(2):  # colour mode data, image resources
        pos += 4 + struct.unpack(">I", data[pos:pos + 4])[0]
    sec_len = struct.unpack(">I", data[pos:pos + 4])[0]
    sec_end = pos + 4 + sec_len
    pos += 4
    info_len = struct.unpack(">I", data[pos:pos + 4])[0]
    pos += 4
    info_end = pos + info_len
    count = struct.unpack(">h", data[pos:pos + 2])[0]
    pos += 2
    recs = []
    for _ in range(abs(count)):
        top, left, bottom, right, nch = struct.unpack(">iiiiH", data[pos:pos + 18])
        pos += 18
        chans = []
        for _ in range(nch):
            chans.append(struct.unpack(">hI", data[pos:pos + 6]))
            pos += 6
        assert data[pos:pos + 8] == b"8BIMnorm"
        opacity, clip, flags, _ = data[pos + 8:pos + 12]
        pos += 12
        extra_len = struct.unpack(">I", data[pos:pos + 4])[0]
        extra = data[pos + 4:pos + 4 + extra_len]
        pos += 4 + extra_len
        nlen = extra[8]
        recs.append({"bbox": (left, top, right, bottom), "chans": chans, "flags": flags,
                     "name": extra[9:9 + nlen].decode("latin-1")})
    layers = []
    for r in recs:
        l, t, rr, b = r["bbox"]
        w, h = rr - l, b - t
        planes = {}
        for cid, ln in r["chans"]:
            blob = data[pos:pos + ln]
            pos += ln
            assert struct.unpack(">H", blob[:2])[0] == 1
            counts = struct.unpack(">%dH" % h, blob[2:2 + 2 * h])
            q, rows = 2 + 2 * h, []
            for n in counts:
                rows.append(np.frombuffer(unpackbits(blob[q:q + n], w), np.uint8))
                q += n
            assert q == len(blob)
            planes[cid] = np.array(rows).reshape(h, w)
        rgba = np.dstack([planes[0], planes[1], planes[2], planes[-1]])
        layers.append({**r, "rgba": rgba})
    assert pos <= info_end and info_end <= sec_end
    return (width, height), layers


def test_psd_roundtrip():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    rgba = np.dstack([rng.integers(0, 255, (20, 30, 3), dtype=np.uint8), rng.integers(0, 255, (20, 30), dtype=np.uint8)])
    rgba[:, :5] = 0  # long runs exercise RLE
    l1 = psd.Layer("Bg", np.dstack([base, np.full((60, 80), 255, np.uint8)]), 0, 0)
    l2 = psd.Layer("Cutout", rgba, 10, 15, visible=False)
    data = psd.build_psd(80, 60, [l1, l2], base)
    size, layers = read_psd_layers(data)
    assert size == (80, 60) and [l["name"] for l in layers] == ["Bg", "Cutout"]
    assert [l["bbox"] for l in layers] == [(0, 0, 80, 60), (10, 15, 40, 35)]
    assert [l["flags"] for l in layers] == [0, 2]
    assert np.array_equal(layers[1]["rgba"], rgba) and np.array_equal(layers[0]["rgba"][..., :3], base)
    # the flattened preview must be readable by an unrelated reader (Pillow)
    im = Image.open(io.BytesIO(data))
    assert im.size == (80, 60) and np.array_equal(np.asarray(im.convert("RGB")), base)


def test_end_to_end_server_client():
    httpd, token = server.serve(ToyEngines(), "secret", host="127.0.0.1", port=0, block=False)
    port = httpd.server_address[1]
    c = LayerizeClient(f"http://127.0.0.1:{port}", "secret")
    assert c.health()
    bad = LayerizeClient(f"http://127.0.0.1:{port}", "nope")
    try:
        bad.job("x")
        raise AssertionError("expected auth failure")
    except Exception as exc:
        assert "401" in str(exc)

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.png")
        make_image().save(src)
        info = c.process(src)
        regs = info["regions"]
        assert len(regs) == 4, regs  # red, blue, green, whole (duplicate removed)
        assert regs[0]["bbox"] == [0, 0, 200, 160]  # biggest first = back
        job = info["job_id"]

        # text search finds the right region
        top = c.search(job, "the red box", 1)[0]
        red = next(r for r in regs if r["id"] == top["id"])
        assert abs(red["bbox"][0] - 20) <= 1 and abs(red["bbox"][1] - 20) <= 1, red

        # cutouts: tight and full-canvas
        data, pos = c.layer(job, red["id"])
        im = Image.open(io.BytesIO(data))
        assert im.mode == "RGBA" and (pos["w"], pos["h"]) == im.size and pos["x"] <= 21
        data, pos = c.layer(job, red["id"], canvas=True)
        assert Image.open(io.BytesIO(data)).size == (200, 160) and pos["x"] == 0
        saved = save_layers(c, job, [r["id"] for r in regs[1:]], os.path.join(tmp, "out"))
        assert all(os.path.exists(s["path"]) for s in saved)

        # on-demand segmentation
        new = c.segment(job, points=[[165, 15, 1]])
        assert new["source"] == "prompt" and new["bbox"][0] >= 159

        # sheet, background, exports
        assert Image.open(io.BytesIO(c.sheet(job))).size[0] > 100
        bg = np.asarray(Image.open(io.BytesIO(c.background(job, [red["id"]]))))
        assert (bg[40, 40] == 200).all()
        ids = [r["id"] for r in regs[1:]]
        data, name = c.export(job, ids, True, "psd")
        size, lays = read_psd_layers(data)
        assert name == "layers.psd" and len(lays) == 4 and size == (200, 160)
        assert lays[0]["name"] == "Background"
        zdata, zname = c.export(job, ids, True, "zip")
        z = zipfile.ZipFile(io.BytesIO(zdata))
        man = json.loads(z.read("manifest.json"))
        assert zname == "layers.zip" and "layers.psd" in z.namelist() and len(man["layers"]) == 4

        # unknown ids are a clean error, not a crash
        try:
            c.layer(job, 999)
            raise AssertionError("expected error")
        except Exception as exc:
            assert "404" in str(exc) or "400" in str(exc)

        for app in adobe.APPS:
            assert "Layerize" in adobe.APPS[app](saved) or app != "aftereffects"
    httpd.shutdown()


def test_studio_flow():
    from layerize.ui import Studio, mask_to_prompt, painted_mask

    img = np.asarray(make_image())
    with tempfile.TemporaryDirectory() as tmp:
        st = Studio(ToyEngines(), tmp)
        try:
            st.export_full()
            raise AssertionError("expected 'upload first' error")
        except ValueError as exc:
            assert "Upload" in str(exc)

        # 1. full scan
        ids = st.process(img)
        assert len(ids) == 4 and st.auto_ids() == ids
        assert all(a.shape == (160, 160, 3) for a, _ in st.gallery(ids))
        assert np.array_equal(st.full_image(), img)  # processed picture == the picture, layers inside
        psd_path, zip_path = st.export_full(True, True, "psd"), st.export_full(True, True, "zip")
        assert psd_path.endswith("full_image_with_layers.psd") and zip_path.endswith("separate_layers.zip")
        size, lays = read_psd_layers(open(psd_path, "rb").read())
        assert size == (200, 160) and len(lays) == 5
        names = zipfile.ZipFile(zip_path).namelist()
        assert "layers.psd" in names and any(n.startswith("full_canvas/") for n in names)

        # 2. manual selection: a rough rectangle drawn over the green box
        painted = np.zeros((160, 200), bool)
        painted[5:45, 155:195] = True
        box, pts = mask_to_prompt(painted)
        assert box == [155.0, 5.0, 195.0, 45.0] and 170 < pts[0][0] < 180
        rid = st.select(img, painted)
        assert st.manual_ids() == [rid]
        assert st.job.regions[rid].x0 >= 159 and st.cutout(rid).shape == (160, 200, 3)
        psd_path = st.export_manual(True)
        _, lays = read_psd_layers(open(psd_path, "rb").read())
        assert len(lays) == 2  # background + the one element
        st.remove(rid)
        assert st.manual_ids() == []
        try:
            st.export_manual()
            raise AssertionError("expected error")
        except ValueError:
            pass
        # selecting on a fresh image (no full scan) works too
        st2 = Studio(ToyEngines(), tmp)
        st2.select(img, painted)
        assert len(st2.manual_ids()) == 1 and st2.auto_ids() == []

        # editor value -> mask
        rgba = np.zeros((160, 200, 4), np.uint8)
        rgba[10:20, 10:20, 3] = 255
        assert painted_mask({"background": img, "layers": [rgba]}).sum() == 100
        try:
            painted_mask({"background": img, "layers": []})
            raise AssertionError("expected error")
        except ValueError as exc:
            assert "Draw" in str(exc)


def test_exact_export_reproduces_picture():
    from layerize.core import composite
    from layerize.pipeline import Job

    img = np.asarray(make_image())
    job = Job(img, ToyEngines(), "t")
    job.analyse()
    job.status = "ready"
    ids = list(job.regions)
    # all layers together over the untouched picture == the picture, pixel for pixel
    layers = [job.layer(i, exact=True) for i in ids]
    assert np.array_equal(composite(img, layers), img)
    data, _ = job.export(ids, False, "psd", exact=True)
    _, lays = read_psd_layers(data)
    assert len(lays) == len(ids) + 1
    assert np.array_equal(composite(img, [psd.Layer(l["name"], l["rgba"], l["bbox"][0], l["bbox"][1]) for l in lays[1:]]), img)
    # non-exact layers differ from the picture only along soft edges
    loose = [job.layer(i, exact=False) for i in ids]
    diff = np.abs(composite(img, loose).astype(int) - img.astype(int)).max(axis=2) > 8
    assert diff.mean() < 0.05


def test_make_layer_empty():
    assert make_layer(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.float32), "x") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
