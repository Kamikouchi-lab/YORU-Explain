"""From runs to your own analysis: units, an export, a gallery, maps aligned to your orientations.

No weights or video needed: two synthetic "animals" (red squares the
synthetic detector finds), one walking right and one walking down, filmed in
two videos.  The script

1. puts each video in its own experimental unit (fly-A, fly-B),
2. runs occlusion with patches sized relative to the target,
3. exports the run (tables, maps around each box, frame crops, gallery),
4. aligns the maps to orientations "you" measured -- here the walking
   direction, as your own tracking would give it -- and saves the class
   mean (a mean of unit means) as a picture.

    uv run python examples/export_and_align.py

The squares have no head, so the aligned map only shows that the mechanics
work; with real animals the orientation is yours to measure (YORU-Explain
does not estimate it).
"""

import csv
from pathlib import Path
import shutil

import cv2
import numpy as np

from yoru_explain import api
from yoru_explain.io.frames import imwrite
from yoru_explain.visualization.overlay import colorbar, heatmap

CONFIG = {
    "out_dir": "results/export_example",
    "patch_fraction": 0.5,
    "grid": 32,
}
H, W = 192, 320


def window(gy, gx):
    """The 32x32 window the synthetic detector's candidate (gy, gx) reads."""
    x0, y0 = 16 * gx - 8, 16 * gy - 8
    return x0, y0, x0 + 32, y0 + 32


def make_video(path, place):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (W, H))
    for i in range(30):
        frame = np.zeros((H, W, 3), np.uint8)
        x0, y0, x1, y1 = window(*place(i))
        frame[y0:y1, x0:x1] = (0, 0, 255)
        out.write(frame)
    out.release()
    return path


def main(config=CONFIG):
    out = Path(config["out_dir"])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    adapter = api.load_adapter("toy-grid", device="cpu")
    project = api.Project.create(out / "project", "export example")
    walks = {"fly-A": (lambda i: (5, 4 + i // 5), 0.0),     # walking right: 0 degrees
             "fly-B": (lambda i: (2 + i // 6, 10), 90.0)}   # walking down: 90 degrees
    model = project.add_model(None, "toy-grid")
    events, heading = [], {}
    for unit, (place, angle) in walks.items():
        sid = project.add_source(make_video(out / f"{unit}.avi", place))["source_id"]
        project.set_unit(sid, unit)
        for frame in (2, 12, 22):
            with project.open_source(sid) as r:
                dets = adapter.detect(r.read(frame).image)
            red = next(i for i, d in enumerate(dets) if d.class_name == "red")
            ev = api.add_event(project, adapter, model.model_id, sid, frame, red)
            events.append(ev)
            heading[ev] = angle
    project.save()

    request = api.AnalysisRequest("occlusion", occlusion=api.OcclusionConfig(
        patch_fraction=config["patch_fraction"]), use_cache=False)
    run = api.run_analysis(project, adapter, model.model_id, events, request)
    print(f"run {run.run_id}: {run.status} {run.counts}")

    export = api.export_run(project, run.run_dir, options=api.ExportOptions(grid=config["grid"]))
    print(f"export: {export.export_dir} {export.counts}")
    with open(export.export_dir / "summary.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            print(f"  {row['level']:5} {row['class_name']:5} {row['unit_id'] or '(all units)':12} "
                  f"n_ok {row['n_ok']}  in-box mean delta {float(row['in_box_mean_delta_mean']):.4f}")

    # Your orientations: one row per event (here: the walking direction, length = box side).
    orientations = out / "orientations.csv"
    with open(export.export_dir / "events.csv", newline="", encoding="utf-8") as f, \
            open(orientations, "w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["event_id", "cx", "cy", "angle_deg", "length", "directed"])
        for e in csv.DictReader(f):
            x1, y1, x2, y2 = (float(e[k]) for k in ("x1", "y1", "x2", "y2"))
            w.writerow([e["event_id"], (x1 + x2) / 2, (y1 + y2) / 2, heading[e["event_id"]], x2 - x1, "true"])

    aligned = api.align_maps(export.export_dir, orientations, grid=config["grid"], extent=1.5,
                             out_dir=out / "aligned")
    means = aligned.mean_by_class()
    for i, name in enumerate(means.class_names):
        m = means.mean_by_class[i]
        heat, valid, vr = heatmap(m, "delta")
        big = cv2.resize(heat, (256, 256), interpolation=cv2.INTER_NEAREST)
        imwrite(out / f"aligned_mean_{name}.png", np.concatenate([big, colorbar(256, "delta", vr)]))
        print(f"aligned mean of class {name}: units per cell {int(means.n_units_by_class[i].max())}, "
              f"largest {np.nanmax(m):.4f}; forward is up in {out / f'aligned_mean_{name}.png'}")
    print("gallery:", export.export_dir / "gallery")


if __name__ == "__main__":
    main()
