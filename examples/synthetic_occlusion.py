"""Occlusion and Grad-CAM on the synthetic detector, where the answer is known.

No weights or video needed.  The synthetic detector reads a 32x32 window per
candidate, so hiding anything outside the target's window must leave its
score exactly unchanged, and the effect must sit on the red square.  The
script prints both checks and writes the preview images.

    uv run python examples/synthetic_occlusion.py
"""

from pathlib import Path

import numpy as np

from yoru_explain import api
from yoru_explain.io.frames import imwrite
from yoru_explain.visualization.overlay import triptych

CONFIG = {
    "out_dir": "results/synthetic_example",
    "patch_size": 16,
    "stride": 8,
}


def main(config=CONFIG):
    out = Path(config["out_dir"])
    adapter = api.load_adapter("toy-grid", device="cpu")

    # A red square exactly on candidate (5, 5)'s window, a green one elsewhere, dark noise.
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 60, size=(192, 320, 3), dtype=np.uint8)
    frame[72:104, 72:104] = (0, 0, 255)
    frame[120:152, 200:232] = (0, 255, 0)
    detections = adapter.detect(frame)
    red = next(i for i, d in enumerate(detections) if d.class_name == "red")
    target = api.target_from_detection(detections[red], red, frame)
    print("target:", target.detection)

    occ = api.explain(adapter, frame, target, api.AnalysisRequest("occlusion", occlusion=api.OcclusionConfig(
        patch_size=config["patch_size"], stride=config["stride"], region="full")))
    window = (72, 72, 104, 104)
    outside = [p for p in occ.patches if not (p.patch.box[0] < window[2] and window[0] < p.patch.box[2]
                                             and p.patch.box[1] < window[3] and window[1] < p.patch.box[3])]
    print(f"occlusion: {len(occ.patches)} patches; reference score {occ.reference_score:.6f}")
    print(f"  patches outside the window with delta exactly 0: "
          f"{sum(p.delta == 0.0 for p in outside)}/{len(outside)}")
    print(f"  largest delta {np.nanmax(occ.delta_map):.4f}; unchanged-frame delta {occ.controls['noop'].delta}")

    cam = api.explain(adapter, frame, target, api.AnalysisRequest("gradcam"))
    ys, xs = np.nonzero(cam.cam > 0.5 * cam.cam.max())
    print(f"Grad-CAM ({cam.layer}): status {cam.status}, half-max region x {xs.min()}..{xs.max()}, "
          f"y {ys.min()}..{ys.max()} (the square is x 72..103, y 72..103)")

    imwrite(out / "occlusion.png", triptych(frame, occ.delta_map, "delta", target.box, label="red"))
    imwrite(out / "gradcam.png", triptych(frame, cam.cam, "cam", target.box, label="red"))
    print("previews written to", out.resolve())


if __name__ == "__main__":
    main()
