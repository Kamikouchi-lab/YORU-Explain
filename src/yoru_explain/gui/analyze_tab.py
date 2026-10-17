# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Analyze tab: method settings, single and batch runs, progress, cancel, log.

The settings here become an :class:`AnalysisRequest`; the run itself is
``core.pipeline.run_analysis`` on the task thread.  Grad-CAM is offered only
when the loaded model's gradient probe passed; otherwise the reason is shown
and the method list holds occlusion alone.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import dearpygui.dearpygui as dpg

from yoru_explain.core.types import IDENTITY_MODES, SCORE_TYPES
from yoru_explain.gui import theme as th
from yoru_explain.gui import widgets
from yoru_explain.gui.widgets import LogBuffer, LogPanel, SelectTable, fmt, muted, tooltip
from yoru_explain.methods.baselines import BASELINE_KINDS, HOT_PINK

OCCLUSION = "Occlusion sensitivity (perturbation effect)"
GRADCAM = "Grad-CAM (attribution; verified models only)"
LAYER_AUTO = "auto (the feature map the candidate's score is computed from)"
REGIONS = ("context", "box", "full")
SETTINGS_W = 440


def _bgr_text(bgr) -> str:
    return ",".join(str(int(v)) for v in bgr)


class AnalyzeTab:
    def __init__(self, app):
        self.app = app
        self.state = app.state
        self.popups = app.popups
        self._last_run = None
        self._layer_names: List[str] = ["auto"]
        self._layer_labels: List[str] = [LAYER_AUTO]
        self.results = SelectTable("ana_results", ["event", "class", "status", "error", "original",
                                                   "largest drop", "in box", "outside",
                                                   "disappeared", "CAM in box", "file"],
                                   height=200, sortable=True,
                                   on_activate=lambda key: self.state.showResult.emit(*key))
        self.log = LogPanel("ana_log", app.themes)
        self._handler = LogBuffer(logging.INFO)
        logging.getLogger("yoru_explain").addHandler(self._handler)
        logging.getLogger("yoru_explain").setLevel(logging.INFO)
        st = self.state
        st.adapterChanged.connect(self._adapter_changed)
        st.tasks.busyChanged.connect(self._busy)
        st.tasks.progress.connect(self._progress)
        st.tasks.finished.connect(self._done)
        st.tasks.failed.connect(self._failed)
        st.requestImported.connect(self.set_request)

    # -- building ----------------------------------------------------------------

    def build(self) -> None:
        themes = self.app.themes
        with dpg.group(horizontal=True):
            with dpg.child_window(tag="ana_settings", width=SETTINGS_W, border=True):
                self._build_settings(themes)
            with dpg.child_window(tag="ana_right", border=False):
                dpg.add_progress_bar(tag="ana_progress", default_value=0.0, width=-1, overlay="")
                dpg.add_text("Idle.", tag="ana_status", wrap=800)
                dpg.add_spacer(height=4)
                widgets.heading("Last run", themes)
                muted("double-click a row (or select it and press View) to see it in Explore", themes)
                self.results.build()
                dpg.add_button(label="View in Explore", callback=lambda: self._open_selected())
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    t = dpg.add_text("Log")
                    dpg.bind_item_theme(t, themes.header)
                    dpg.add_button(label="Copy", callback=lambda: dpg.set_clipboard_text(self.log.text()))
                dpg.add_separator()
                self.log.build()
        self._adapter_changed()
        self._method_changed()

    def _build_settings(self, themes) -> None:
        widgets.heading("Method", themes)
        dpg.add_radio_button([OCCLUSION], tag="ana_method", default_value=OCCLUSION,
                             callback=lambda: self._method_changed())
        dpg.add_text("", tag="ana_cam_reason", wrap=SETTINGS_W - 30)
        dpg.bind_item_theme("ana_cam_reason", themes.warn)
        dpg.add_spacer(height=4)

        widgets.heading("What is explained", themes)
        with dpg.group(horizontal=True):
            dpg.add_text("score    ")
            c = dpg.add_combo(list(SCORE_TYPES), tag="ana_score", default_value=next(iter(SCORE_TYPES)),
                              width=-1, callback=lambda: self._score_help())
            tooltip(c, "Which number of the detector is explained")
        with dpg.group(horizontal=True):
            dpg.add_text("identity ")
            c = dpg.add_combo(list(IDENTITY_MODES), tag="ana_identity",
                              default_value=next(iter(IDENTITY_MODES)), width=-1,
                              callback=lambda: self._score_help())
            tooltip(c, "fixed_candidate: the same pre-NMS candidate on every input (models whose "
                       "candidates are fixed slots). redetect: run the detector end to end and match by "
                       "class and IoU; no score when the match fails.")
        with dpg.group(horizontal=True):
            dpg.add_text("match IoU")
            c = dpg.add_input_float(tag="ana_iou", default_value=0.5, min_value=0.05, max_value=1.0,
                                    min_clamped=True, max_clamped=True, step=0.05, format="%.2f",
                                    width=140)
            tooltip(c, "Re-detection: minimum IoU with the original box")
        muted("", themes, tag="ana_score_help", wrap=SETTINGS_W - 30)
        dpg.add_spacer(height=4)

        with dpg.group(tag="ana_occ"):
            widgets.heading("Occlusion", themes)
            self._int_row("patch (px)", "ana_patch", 32, 2, 512, "Square patch size in frame pixels")
            self._int_row("stride (px)", "ana_stride", 16, 1, 512)
            with dpg.group(horizontal=True):
                dpg.add_text("baseline   ")
                c = dpg.add_combo(list(BASELINE_KINDS), tag="ana_baseline", default_value="constant",
                                  width=-1, callback=lambda: self._baseline_help())
                tooltip(c, "What an occluded region is replaced with")
            muted("", themes, tag="ana_baseline_help", wrap=SETTINGS_W - 30)
            with dpg.group(horizontal=True):
                dpg.add_text("value B,G,R")
                v = dpg.add_input_text(tag="ana_value", default_value=_bgr_text(HOT_PINK), width=150,
                                       callback=lambda: self._value_typed())
                tooltip(v, "constant baseline: B,G,R. Default 180,105,255 = hot pink #FF69B4; "
                           "0,0,0 black; 114,114,114 YOLOv5 grey")
                b, g, r = HOT_PINK
                c = dpg.add_color_edit((r, g, b, 255), tag="ana_colour", no_alpha=True, no_inputs=True,
                                       no_label=True, callback=lambda: self._colour_picked())
                tooltip(c, "Pick the constant baseline colour")
            with dpg.group(horizontal=True):
                dpg.add_text("blur sigma ")
                dpg.add_input_float(tag="ana_sigma", default_value=8.0, min_value=0.5, max_value=100.0,
                                    min_clamped=True, max_clamped=True, step=0.5, format="%.2f",
                                    width=140)
            with dpg.group(horizontal=True):
                dpg.add_text("region     ")
                c = dpg.add_combo(list(REGIONS), tag="ana_region", default_value="context", width=140)
                tooltip(c, "context: box scaled by the factor; box; full frame")
            with dpg.group(horizontal=True):
                dpg.add_text("context x  ")
                dpg.add_input_float(tag="ana_context", default_value=3.0, min_value=1.0, max_value=20.0,
                                    min_clamped=True, max_clamped=True, step=0.5, format="%.2f",
                                    width=140)
            self._int_row("batch size", "ana_batch", 1, 1, 256,
                          "Images per forward pass. 1 reproduces YORU exactly; larger is faster, and "
                          "the reference score is then taken at the same batch size.")
            self._int_row("seed", "ana_seed", 0, 0, 2 ** 31 - 1, "Noise baseline seed")
            dpg.add_checkbox(label="count ambiguous matches in the map", tag="ana_ambiguous")
            dpg.add_spacer(height=4)

        with dpg.group(tag="ana_cam", show=False):
            widgets.heading("Grad-CAM", themes)
            with dpg.group(horizontal=True):
                dpg.add_text("layer    ")
                c = dpg.add_combo([LAYER_AUTO], tag="ana_layer", default_value=LAYER_AUTO, width=-1)
                tooltip(c, "auto: YOLO: the detection head's input at the candidate's level; RT-DETR: "
                           "the decoder input of the query's encoder token; Faster R-CNN: the FPN level "
                           "its proposal is pooled from; SSD: the feature map of its default box")
            with dpg.group(horizontal=True):
                dpg.add_text("precision")
                c = dpg.add_combo(["inference", "fp32"], tag="ana_precision",
                                  default_value="inference", width=140)
                tooltip(c, "inference: the same autocast as YORU (the explained score); "
                           "fp32: autocast off")
            dpg.add_spacer(height=4)

        widgets.heading("Run", themes)
        dpg.add_checkbox(label="check the adapter against YORU's detector first", tag="ana_verify",
                         default_value=True)
        dpg.add_checkbox(label="reuse identical earlier results (cache)", tag="ana_cache",
                         default_value=True)
        dpg.add_checkbox(label="write preview PNGs", tag="ana_previews", default_value=True)
        with dpg.group(horizontal=True):
            b = dpg.add_button(label="Analyze selected events", tag="ana_run_sel",
                               callback=lambda: self.run(selected_only=True))
            dpg.bind_item_theme(b, themes.primary_button)
            dpg.add_button(label="Analyze all", tag="ana_run_all",
                           callback=lambda: self.run(selected_only=False))
            dpg.add_button(label="Cancel", tag="ana_cancel", enabled=False, callback=lambda: self._cancel())
        self._baseline_help()

    @staticmethod
    def _int_row(label, tag, default, lo, hi, tip=""):
        with dpg.group(horizontal=True):
            dpg.add_text(f"{label:<11}")
            item = dpg.add_input_int(tag=tag, default_value=default, min_value=lo, max_value=hi,
                                     min_clamped=True, max_clamped=True, width=140)
            if tip:
                tooltip(item, tip)

    def layout(self, width: int, height: int) -> None:
        dpg.configure_item("ana_settings", height=height)
        right = max(300, width - SETTINGS_W - 16)
        dpg.configure_item("ana_right", width=right, height=height)
        dpg.configure_item("ana_status", wrap=right - 20)
        rest = max(240, height - 150)
        dpg.configure_item(self.results.tag, height=max(100, int(rest * 0.45)))
        dpg.configure_item(self.log.tag, height=-1)

    # -- the baseline colour -----------------------------------------------------------

    def _bgr(self):
        """The B,G,R typed in the value field, or None if it is not three 0..255 numbers."""
        try:
            bgr = tuple(int(v) for v in str(dpg.get_value("ana_value")).replace(" ", "").split(","))
        except ValueError:
            return None
        return bgr if len(bgr) == 3 and all(0 <= v <= 255 for v in bgr) else None

    def _value_typed(self):
        bgr = self._bgr()
        if bgr is not None:
            b, g, r = bgr
            dpg.set_value("ana_colour", (r, g, b, 255))

    def _colour_picked(self):
        r, g, b = (int(round(v)) for v in dpg.get_value("ana_colour")[:3])
        dpg.set_value("ana_value", _bgr_text((b, g, r)))
        dpg.set_value("ana_baseline", "constant")
        self._baseline_help()

    def colour_rgb(self):
        """The swatch's colour as (R, G, B)."""
        return tuple(int(round(v)) for v in dpg.get_value("ana_colour")[:3])

    def _baseline_help(self):
        dpg.set_value("ana_baseline_help", BASELINE_KINDS.get(dpg.get_value("ana_baseline"), ""))

    # -- the request -------------------------------------------------------------------

    def method(self) -> str:
        return "gradcam" if dpg.get_value("ana_method") == GRADCAM else "occlusion"

    def set_method(self, method: str) -> None:
        items = dpg.get_item_configuration("ana_method")["items"]
        wanted = GRADCAM if method == "gradcam" else OCCLUSION
        dpg.set_value("ana_method", wanted if wanted in items else OCCLUSION)
        self._method_changed()

    def request(self):
        from yoru_explain.core.pipeline import AnalysisRequest
        from yoru_explain.core.types import ScoreDefinition
        from yoru_explain.methods.baselines import BaselineSpec
        from yoru_explain.methods.gradcam import GradCamConfig
        from yoru_explain.methods.occlusion import OcclusionConfig

        value = tuple(int(v) for v in str(dpg.get_value("ana_value")).replace(" ", "").split(","))
        g = dpg.get_value
        occ = OcclusionConfig(
            patch_size=int(g("ana_patch")), stride=int(g("ana_stride")),
            baseline=BaselineSpec(g("ana_baseline"), value, float(g("ana_sigma"))),
            region=g("ana_region"), context_scale=float(g("ana_context")),
            score=ScoreDefinition(g("ana_score"), g("ana_identity"), round(float(g("ana_iou")), 4)),
            batch_size=int(g("ana_batch")), seed=int(g("ana_seed")),
            include_ambiguous=bool(g("ana_ambiguous")))
        layer = g("ana_layer")
        cam = GradCamConfig(layer=self._layer_names[self._layer_labels.index(layer)]
                            if layer in self._layer_labels else "auto",
                            score_type=g("ana_score"), precision=g("ana_precision"))
        return AnalysisRequest(self.method(), occlusion=occ, gradcam=cam,
                               verify_identity=bool(g("ana_verify")), use_cache=bool(g("ana_cache")),
                               write_previews=bool(g("ana_previews")))

    def set_request(self, req):
        o = req.occlusion
        s = dpg.set_value
        s("ana_patch", int(o.patch_size[0]))
        s("ana_stride", int(o.stride[0]))
        if o.baseline.kind in BASELINE_KINDS:
            s("ana_baseline", o.baseline.kind)
        s("ana_value", _bgr_text(o.baseline.value))
        self._value_typed()
        s("ana_sigma", float(o.baseline.sigma))
        s("ana_region", o.region)
        s("ana_context", float(o.context_scale))
        st = o.score.score_type if req.method == "occlusion" else req.gradcam.score_type
        if st in dpg.get_item_configuration("ana_score")["items"]:
            s("ana_score", st)
        s("ana_identity", o.score.identity_mode)
        s("ana_iou", float(o.score.iou_threshold))
        s("ana_batch", int(o.batch_size))
        s("ana_seed", int(o.seed))
        s("ana_ambiguous", bool(o.include_ambiguous))
        if req.gradcam.layer in self._layer_names:
            s("ana_layer", self._layer_labels[self._layer_names.index(req.gradcam.layer)])
        else:
            s("ana_layer", LAYER_AUTO)
        s("ana_precision", req.gradcam.precision)
        s("ana_verify", bool(req.verify_identity))
        s("ana_cache", bool(req.use_cache))
        s("ana_previews", bool(req.write_previews))
        self.set_method(req.method)
        self._baseline_help()

    # -- state ------------------------------------------------------------------------

    def _adapter_changed(self):
        a = self.state.adapter
        self._layer_names, self._layer_labels = ["auto"], [LAYER_AUTO]
        themes = self.app.themes
        if a is None:
            dpg.configure_item("ana_method", items=[OCCLUSION])
            dpg.set_value("ana_method", OCCLUSION)
            dpg.set_value("ana_cam_reason", "Load a model to see whether Grad-CAM is available for it.")
            dpg.bind_item_theme("ana_cam_reason", themes.muted)
            dpg.configure_item("ana_score", items=list(SCORE_TYPES))
            dpg.configure_item("ana_identity", items=list(IDENTITY_MODES))
        else:
            caps = a.capabilities
            if caps.gradients:
                dpg.configure_item("ana_method", items=[OCCLUSION, GRADCAM])
                dpg.set_value("ana_cam_reason", f"Grad-CAM: {caps.gradients_reason}")
                dpg.bind_item_theme("ana_cam_reason", themes.ok)
            else:
                dpg.configure_item("ana_method", items=[OCCLUSION])
                dpg.set_value("ana_method", OCCLUSION)
                dpg.set_value("ana_cam_reason", f"Grad-CAM is not offered for this model: "
                                                f"{caps.gradients_reason}")
                dpg.bind_item_theme("ana_cam_reason", themes.warn)
            for li in a.gradient_layers():
                self._layer_labels.append(li.describe())
                self._layer_names.append(li.name)
            scores = [k for k in SCORE_TYPES if k in caps.score_types]
            dpg.configure_item("ana_score", items=scores)
            if dpg.get_value("ana_score") not in scores and scores:
                dpg.set_value("ana_score", scores[0])
            modes = list(getattr(caps, "identity_modes", IDENTITY_MODES))
            dpg.configure_item("ana_identity", items=modes)
            if dpg.get_value("ana_identity") not in modes and modes:
                dpg.set_value("ana_identity", modes[0])
        dpg.configure_item("ana_layer", items=self._layer_labels)
        if dpg.get_value("ana_layer") not in self._layer_labels:
            dpg.set_value("ana_layer", LAYER_AUTO)
        self._method_changed()

    @property
    def gradcam_offered(self) -> bool:
        return GRADCAM in dpg.get_item_configuration("ana_method")["items"]

    def _identity_modes(self) -> List[str]:
        a = self.state.adapter
        if a is None:
            return list(IDENTITY_MODES)
        return list(getattr(a.capabilities, "identity_modes", IDENTITY_MODES))

    def _method_changed(self):
        occ = self.method() == "occlusion"
        dpg.configure_item("ana_occ", show=occ)
        dpg.configure_item("ana_cam", show=not occ)
        modes = self._identity_modes()
        # Grad-CAM differentiates the target's own candidate on its own frame.
        dpg.configure_item("ana_identity", enabled=occ and len(modes) > 1,
                           items=modes if occ else ["fixed_candidate"])
        dpg.configure_item("ana_iou", enabled=occ and "redetect" in modes)
        if not occ:
            dpg.set_value("ana_identity", "fixed_candidate")
        elif dpg.get_value("ana_identity") not in modes and modes:
            dpg.set_value("ana_identity", modes[0])
        self._score_help()

    def _score_help(self):
        st = dpg.get_value("ana_score")
        occ = self.method() == "occlusion"
        mode = dpg.get_value("ana_identity") if occ else "fixed_candidate"
        kind = ("Perturbation effect: score change when a region is replaced by the baseline."
                if occ else "Attribution: gradient-weighted feature map of the score (Grad-CAM).")
        a = self.state.adapter
        what = a.describe_score_type(st) if a is not None and hasattr(a, "describe_score_type") \
            else SCORE_TYPES.get(st, "")
        mode_text = IDENTITY_MODES.get(mode, "") if occ else "the target's own candidate, on its own frame"
        dpg.set_value("ana_score_help", f"{kind}\n{st}: {what}\n{mode_text}\n"
                                        "Not a biological interpretation.")

    def _busy(self, busy):
        dpg.configure_item("ana_run_sel", enabled=not busy)
        dpg.configure_item("ana_run_all", enabled=not busy)
        dpg.configure_item("ana_cancel", enabled=busy and self.state.tasks.current == "analysis")

    # -- running ------------------------------------------------------------------------

    def run(self, selected_only: bool):
        from yoru_explain.core.pipeline import run_analysis

        p, a = self.state.project, self.state.adapter
        if p is None or a is None or self.state.model_id is None:
            self.popups.message("YORU-Explain", "Open a project and load a model first.")
            return
        events = list(self.state.selected_events) if selected_only else [e["event_id"] for e in p.events]
        if not events:
            self.popups.message("YORU-Explain", "No events selected. Add detections as events in "
                                                "Explore and select them there.")
            return
        try:
            req = self.request()
        except Exception as e:
            self.popups.error("Settings", e)
            return
        tasks = self.state.tasks
        model_id = self.state.model_id

        def work():
            return run_analysis(p, a, model_id, events, req,
                                progress=lambda pr: tasks.report(pr.fraction, pr.message),
                                cancel=tasks.cancelled)

        if not tasks.submit("analysis", work):
            self.popups.message("YORU-Explain", "Another task is running.")
            return
        dpg.set_value("ana_status", f"Running {req.method} on {len(events)} event(s)...")
        dpg.set_value("ana_progress", 0.0)
        dpg.configure_item("ana_progress", overlay="0%")
        self._busy(True)

    def _cancel(self):
        self.state.tasks.cancel()
        dpg.set_value("ana_status", "Cancelling after the current batch...")

    def _progress(self, fraction, message):
        if self.state.tasks.current == "analysis":
            dpg.set_value("ana_progress", float(fraction))
            dpg.configure_item("ana_progress", overlay=f"{fraction * 100:.0f}%")
            dpg.set_value("ana_status", message)

    def _done(self, name, outcome):
        if name != "analysis":
            return
        self._last_run = outcome
        dpg.set_value("ana_progress", 1.0)
        dpg.configure_item("ana_progress", overlay="100%")
        ident = outcome.identity_check or {}
        dpg.set_value("ana_status", f"Run {outcome.run_id}: {outcome.status}; {outcome.counts}; "
                                    f"identity check {ident.get('status', '-')}"
                                    f"{' (exact)' if ident.get('exact') else ''}")
        rows, data, colors = [], [], []
        for r in outcome.results:
            rows.append([r.get("event_id"), r.get("behavior_class"), r.get("status"), r.get("error_code"),
                         fmt(r.get("original_score")), fmt(r.get("delta_score")),
                         fmt(r.get("in_box_mean_delta")), fmt(r.get("out_box_mean_delta")),
                         r.get("n_disappeared"), fmt(r.get("cam_in_box_fraction")), r.get("map_file")])
            data.append((str(outcome.run_dir), r.get("event_id")))
            colors.append(None if r.get("status") == "ok" else th.TEXT_WARN)
        self.results.set_rows(rows, data, colors)
        self.state.runs_changed()

    def _failed(self, name, text, exc):
        if name == "analysis":
            dpg.set_value("ana_status", f"Run failed: {exc}")
            self.popups.error("Analysis", exc)
            self.state.runs_changed()

    def _open_selected(self):
        sel = self.results.selected
        if sel:
            self.state.showResult.emit(*sel[0])

    def tick(self, now: float) -> None:
        for text, level in self._handler.drain():
            self.log.append(text, level)

    def detach_log(self):
        logging.getLogger("yoru_explain").removeHandler(self._handler)

    @property
    def last_run(self) -> Optional[object]:
        return self._last_run
