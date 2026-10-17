"""One detection of any YORU model on one image: occlusion (and Grad-CAM where offered), saved as a project.

Edit CONFIG and run::

    uv run python examples/any_model_image.py

The model can be of any YORU backend -- a YOLOv5 checkpoint, YOLOv8 / YOLO11
/ YOLOv10 (OBB, segmentation and pose models too), RT-DETR, a torchvision
checkpoint written by YORU, an ONNX export.  ``backend="auto"`` picks the one
YORU's ``get_detector("auto", ...)`` would; ``black_box=True`` explains YORU's
``detect()`` only.  The identity mode is the first the model offers
(fixed-candidate, or re-detection for RT-DETR and Faster R-CNN, whose
candidates are chosen anew for every input).  The defaults use a test model
from ``tests/prepare_models.py`` and ultralytics' bus picture.
"""

from pathlib import Path

from yoru_explain import api

REPO = Path(__file__).resolve().parents[1]

CONFIG = {
    "model": str(REPO / ".cache" / "models" / "yolo11n-obb.pt"),
    "backend": "auto",
    "black_box": False,
    "image": None,                  # None: ultralytics' bus.jpg
    "detection_index": 0,
    "project_dir": "results/any_model_example_project",
    "device": "auto",
    "model_options": {"conf_thresh": 0.25, "iou_thresh": 0.45},
    "occlusion": {"patch_size": [48, 48], "stride": [24, 24], "region": "context", "context_scale": 2.0},
}


def main(config=CONFIG):
    image = config["image"]
    if image is None:
        import ultralytics

        image = str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg")
    adapter = api.load_model(config["model"], config["backend"], device=config["device"],
                             black_box=config["black_box"], **config["model_options"])
    caps = adapter.capabilities
    print(f"model: {caps.family} ({adapter.name}) on {adapter.model_info()['device']}")
    print(f"candidates: {caps.candidate_kind or 'none'}; identity modes: {list(caps.identity_modes)}; "
          f"Grad-CAM: {caps.gradients_reason}")
    for w in caps.warnings:
        print("WARNING:", w)

    project = api.Project.open_or_create(config["project_dir"])
    source = project.add_source(image)
    options = {**config["model_options"], **({"backend": adapter.model_info()["backend"]}
                                             if adapter.name == "yoru-detector" else {})}
    model = project.add_model(config["model"], adapter.name, options)
    event = api.add_event(project, adapter, model.model_id, source["source_id"], 0,
                          config["detection_index"])
    project.save()
    e = project.event(event)
    print("event:", event, e["class_name"], e["score"], "rotated:", bool(e.get("obb_angle")))

    score = api.ScoreDefinition(identity_mode=caps.identity_modes[0])
    requests = [api.AnalysisRequest("occlusion", occlusion=api.OcclusionConfig.from_dict(
        {**config["occlusion"], "score": score.to_dict()}))]
    if caps.gradients:
        requests.append(api.AnalysisRequest("gradcam"))
    for request in requests:
        outcome = api.run_analysis(project, adapter, model.model_id, [event], request)
        print(f"\n{request.method}: {outcome.status} {outcome.counts}; identity check vs YORU: "
              f"{outcome.identity_check.get('status')} (exact={outcome.identity_check.get('exact')})")
        for row in api.RunReader(outcome.run_dir).typed_results():
            for k in ("status", "error_code", "message", "original_score", "delta_score",
                      "in_box_mean_delta", "noop_delta", "gradcam_layer", "gradient_support",
                      "cam_in_box_fraction"):
                if row.get(k) is not None:
                    print(f"  {k}: {row[k]}")
        print("  folder:", outcome.run_dir)


if __name__ == "__main__":
    main()
