"""The boundary with YORU, and the house rules, checked in the source.

* Documented YORU names (YORU's docs/external_api.md) may be used anywhere.
* YORU's YOLOv5 internals are used in ``adapters/_yoru.py`` only -- the list
  in docs/INITIAL_AUDIT.md section 8.  Kamikouchi-lab/YORU#43 proposes
  documenting them; move them to DOCUMENTED once it is merged.
* YORU never imports YORU-Explain.
* No argparse; no torch in the core that GUI-less tools import.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "yoru_explain"

DOCUMENTED = {
    "yoru": {"__version__"},
    "yoru.libs.plugins": {"get_detector", "list_detector_backends", "DEFAULT_CONF_THRESH",
                          "DEFAULT_IOU_THRESH"},
    "yoru.libs.user_paths": {"get_yoru_home", "get_log_dir", "get_log_file", "setup_logging",
                             "log_exception", "log_message"},
}
#: Not in YORU's external API; allowed in adapters/_yoru.py only.
INTERNAL = {
    "yoru.libs.plugins.yolov5_detector": {"V1_NMS_CONF", "load_yolov5_model", "rgb_input_requested"},
    "yoru.libs.yolov5_support": {"YOLOV5_DIR", "yolov5_importable"},
    "yoru.libs.device": {"torch_device"},
    # The bundled YOLOv5's own modules, importable inside yolov5_importable().
    "utils.augmentations": {"letterbox"},
    "utils.general": {"make_divisible", "non_max_suppression", "scale_boxes", "xywh2xyxy"},
}
BOUNDARY_MODULE = "_yoru.py"


def _imports():
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for alias in node.names:
                    yield path, node.lineno, node.module, alias.name
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    yield path, node.lineno, alias.name, None


def test_yoru_names_are_documented_or_confined_to_the_boundary_module():
    bad = []
    for path, line, module, name in _imports():
        top = module.split(".")[0]
        if top not in ("yoru", "utils", "models"):
            continue
        if name is None:
            if module != "yoru":
                bad.append(f"{path.name}:{line}: import {module}")
            continue
        if name in DOCUMENTED.get(module, set()):
            continue
        if name in INTERNAL.get(module, set()) and path.name == BOUNDARY_MODULE:
            continue
        bad.append(f"{path.name}:{line}: from {module} import {name}")
    assert not bad, "YORU internals outside adapters/_yoru.py:\n" + "\n".join(bad)


def test_yoru_does_not_import_yoru_explain():
    import yoru

    root = Path(yoru.__file__).resolve().parent
    offenders = [str(p.relative_to(root)) for p in root.rglob("*.py")
                 if "yoru_explain" in p.read_text(encoding="utf-8", errors="replace")]
    assert not offenders, offenders


def test_no_argparse():
    found = [f"{p.name}:{line}" for p, line, module, _ in _imports() if module.split(".")[0] == "argparse"]
    assert not found, found


def test_the_core_imports_without_torch_or_qt():
    code = (
        "import sys\n"
        "import yoru_explain.api, yoru_explain.core.pipeline, yoru_explain.io.runs\n"
        "import yoru_explain.methods.occlusion, yoru_explain.analysis.summary\n"
        "import yoru_explain.visualization.overlay\n"
        "print(sorted(m for m in ('torch', 'PyQt5', 'yoru') if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
