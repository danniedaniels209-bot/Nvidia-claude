"""Minimal layered-PSD writer (8-bit RGB, one RGBA pixel layer per Layer, PackBits RLE).

Opens in Photoshop and After Effects (import as "Composition - Retain Layer Sizes"
to get one AE layer per PSD layer, positioned correctly).
"""
from __future__ import annotations

import struct

import numpy as np

from .core import Layer


def _packbits_row(row: np.ndarray) -> bytes:
    n = row.size
    rb = row.tobytes()
    change = np.flatnonzero(row[1:] != row[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    out = bytearray()

    def literal(b: bytes) -> None:
        for i in range(0, len(b), 128):
            chunk = b[i:i + 128]
            out.append(len(chunk) - 1)
            out.extend(chunk)

    pos = 0
    for i in np.flatnonzero((ends - starts) >= 3):
        s, e = int(starts[i]), int(ends[i])
        if s > pos:
            literal(rb[pos:s])
        remaining = e - s
        while remaining > 0:
            c = min(remaining, 128)
            if c == 1:
                literal(rb[s:s + 1])
            else:
                out.append(257 - c)
                out.append(rb[s])
            remaining -= c
        pos = e
    if pos < n:
        literal(rb[pos:])
    return bytes(out)


def _rle_channel(plane: np.ndarray) -> bytes:
    rows = [_packbits_row(plane[y]) for y in range(plane.shape[0])]
    counts = struct.pack(">%dH" % len(rows), *[len(r) for r in rows])
    return struct.pack(">H", 1) + counts + b"".join(rows)


def _pascal(name: str) -> bytes:
    b = name.encode("latin-1", "replace")[:255]
    s = bytes([len(b)]) + b
    return s + b"\0" * ((-len(s)) % 4)


def _luni(name: str) -> bytes:
    data = struct.pack(">I", len(name)) + name.encode("utf-16-be")
    data += b"\0" * ((-len(data)) % 4)
    return b"8BIM" + b"luni" + struct.pack(">I", len(data)) + data


def _layer_record(lyr: Layer, blobs: list[bytes]) -> bytes:
    rec = struct.pack(">iiii", lyr.y, lyr.x, lyr.y + lyr.h, lyr.x + lyr.w)
    rec += struct.pack(">H", 4)
    for cid, blob in zip((0, 1, 2, -1), blobs):
        rec += struct.pack(">hI", cid, len(blob))
    flags = 0 if lyr.visible else 2
    rec += b"8BIM" + b"norm" + bytes([lyr.opacity, 0, flags, 0])
    extra = struct.pack(">I", 0) + struct.pack(">I", 0) + _pascal(lyr.name) + _luni(lyr.name)
    return rec + struct.pack(">I", len(extra)) + extra


def build_psd(width: int, height: int, layers: list[Layer], composite_rgb: np.ndarray) -> bytes:
    """`layers` is bottom-to-top. `composite_rgb` is the flattened preview (H x W x 3 uint8)."""
    if width > 30000 or height > 30000:
        raise ValueError("PSD (v1) supports at most 30000 px per side")
    layers = [l for l in layers if l.w > 0 and l.h > 0]
    records, data = [], []
    for lyr in layers:
        blobs = [_rle_channel(np.ascontiguousarray(lyr.rgba[..., c])) for c in range(4)]
        records.append(_layer_record(lyr, blobs))
        data.append(b"".join(blobs))
    layer_info = struct.pack(">h", len(layers)) + b"".join(records) + b"".join(data)
    layer_info += b"\0" * (len(layer_info) % 2)
    layer_mask_section = struct.pack(">I", len(layer_info)) + layer_info + struct.pack(">I", 0)

    header = b"8BPS" + struct.pack(">H", 1) + b"\0" * 6
    header += struct.pack(">HIIHH", 3, height, width, 8, 3)
    image = struct.pack(">H", 0) + b"".join(
        np.ascontiguousarray(composite_rgb[..., c]).tobytes() for c in range(3)
    )
    return (
        header
        + struct.pack(">I", 0)  # colour mode data
        + struct.pack(">I", 0)  # image resources
        + struct.pack(">I", len(layer_mask_section)) + layer_mask_section
        + image
    )
