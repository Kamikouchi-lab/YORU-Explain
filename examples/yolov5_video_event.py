"""One YOLOv5 detection on one video frame: occlusion and Grad-CAM, saved as a project.

Edit CONFIG and run::

    uv run python examples/yolov5_video_event.py

The defaults use YORU's test video and the test-fixture fly model described in
docs/INITIAL_AUDIT.md (classes copulation / solo); frame 6450 shows a
copulating pair.  Results go to ``<project>/runs/<run_id>/``: results.csv,
perturbations.csv, maps/*.npz, previews/*.png, manifest.json, config.json.
"""

from pathlib import Path

from yoru_explain import api

REPO = Path(__file__).resolve().parents[1]

CONFIG = {
    "model": str(REPO / ".cache" / "models" / "fly_copulation_yolov5s.pt"),
    "video": str(REPO.parent / "YORU" / "test_data" / "test_video_fly_copulation.mp4"),
    "frame_index": 6450,
    "detection_index": 0,
    "project_dir": "results/yolov5_example_project",
    "device": "auto",
    # As in YORU: conf 0.25, IoU 0.45, channel order from YORU_YOLOV5_RGB (BGR by default).
    "model_options": {"conf_thresh": 0.25, "iou_thresh": 0.45, "rgb_input": None},
    "occlusion": {"patch_size": [32, 32], "stride": [16, 16], "region": "context", "context_scale": 3.0,
                  # The default: hot pink #FF69B4, in OpenCV's B,G,R order.
                  "baseline": {"kind": "constant", "value": [180, 105, 255], "sigma": 8.0},
                  "score": {"score_type": "obj_x_cls", "identity_mode": "fixed_candidate",
                            "iou_threshold": 0.5},
                  "batch_size": 1, "seed": 0},
}


def main(config=CONFIG):
    adapter = api.load_adapter("yoru-yolov5", config["model"], device=config["device"],
                               **config["model_options"])
    print("model:", adapter.model_info()["weights"], "on", adapter.model_info()["device"])
    print("Grad-CAM:", adapter.capabilities.gradients_reason)

    project = api.Project.open_or_create(config["project_dir"])
    source = project.add_source(config["video"])
    model = project.add_model(config["model"], "yoru-yolov5", config["model_options"])
    event = api.add_event(project, adapter, model.model_id, source["source_id"],
                          config["frame_index"], config["detection_index"])
    project.save()
    print("event:", event, project.event(event)["class_name"], project.event(event)["score"])

    for request in (
        api.AnalysisRequest("occlusion", occlusion=api.OcclusionConfig.from_dict(config["occlusion"])),
        api.AnalysisRequest("gradcam"),
    ):
        outcome = api.run_analysis(project, adapter, model.model_id, [event], request)
        print(f"\n{request.method}: {outcome.status} {outcome.counts}; identity check vs YORU: "
              f"{outcome.identity_check.get('status')} (exact={outcome.identity_check.get('exact')})")
        for row in api.RunReader(outcome.run_dir).typed_results():
            keys = ("status", "error_code", "message", "original_score", "delta_score", "in_box_mean_delta",
                    "out_box_mean_delta", "noop_delta", "gradient_support", "cam_in_box_fraction")
            for k in keys:
                if row.get(k) is not None:
                    print(f"  {k}: {row[k]}")
        print("  folder:", outcome.run_dir)


if __name__ == "__main__":
    main()
