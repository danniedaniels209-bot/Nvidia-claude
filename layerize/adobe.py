"""ExtendScript (JSX) snippets so an agent with an Adobe automation MCP can import the downloaded layers.

`items` are dicts from client.save_layers: path, name, x, y, w, h, canvas.
Items are ordered bottom -> top. These are best-effort templates; test them once in your own setup.
"""
from __future__ import annotations

import json


def _items(items: list[dict]) -> str:
    return json.dumps([
        {"path": i["path"].replace("\\", "/"), "name": i["name"], "x": i["x"], "y": i["y"],
         "w": i["w"], "h": i["h"], "canvas": bool(i.get("canvas"))} for i in items
    ])


def aftereffects(items: list[dict]) -> str:
    return f"""(function () {{
  var comp = app.project.activeItem;
  if (!(comp instanceof CompItem)) {{ alert("Open/select a composition first"); return; }}
  var items = {_items(items)};
  app.beginUndoGroup("Layerize import");
  for (var i = 0; i < items.length; i++) {{
    var it = items[i];
    var footage = app.project.importFile(new ImportOptions(new File(it.path)));
    var layer = comp.layers.add(footage);
    layer.name = it.name;
    // full-canvas PNGs already line up when the comp is the same size as the source image
    if (!it.canvas) layer.property("Position").setValue([it.x + it.w / 2, it.y + it.h / 2]);
  }}
  app.endUndoGroup();
}})();"""


def aftereffects_psd(psd_path: str) -> str:
    return f"""(function () {{
  var io = new ImportOptions(new File({json.dumps(psd_path.replace(chr(92), '/'))}));
  io.importAs = ImportAsType.COMP;  // one comp, one AE layer per PSD layer (layer sizes retained)
  app.project.importFile(io);
}})();"""


def photoshop(items: list[dict]) -> str:
    return f"""(function () {{
  var doc = app.activeDocument;
  var items = {_items(items)};
  var prev = app.preferences.rulerUnits; app.preferences.rulerUnits = Units.PIXELS;
  for (var i = 0; i < items.length; i++) {{
    var it = items[i];
    var src = app.open(new File(it.path));
    src.selection.selectAll(); src.selection.copy(); src.close(SaveOptions.DONOTSAVECHANGES);
    app.activeDocument = doc;
    var layer = doc.paste();
    layer.name = it.name;
    var b = layer.bounds;  // [left, top, right, bottom]
    layer.translate(it.x - b[0].as("px"), it.y - b[1].as("px"));
  }}
  app.preferences.rulerUnits = prev;
}})();"""


def illustrator(items: list[dict]) -> str:
    return f"""(function () {{
  var doc = app.activeDocument;
  var items = {_items(items)};
  for (var i = 0; i < items.length; i++) {{
    var it = items[i];
    var p = doc.placedItems.add();
    p.file = new File(it.path);
    p.name = it.name;
    p.left = it.x;
    p.top = -it.y;  // Illustrator's y axis points up from the artboard top-left origin
  }}
}})();"""


APPS = {"aftereffects": aftereffects, "photoshop": photoshop, "illustrator": illustrator}
