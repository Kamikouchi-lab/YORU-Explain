# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Analyze tab: method settings, single and batch runs, progress, cancel, log.

The settings here become an :class:`AnalysisRequest`; the run itself is
``core.pipeline.run_analyses`` on the task thread -- one run for the chosen
method, or with *Both methods* a Grad-CAM run and then an occlusion run of the
same events (``pipeline.BOTH_METHODS``), each with its own settings.  Grad-CAM
is offered only when the loaded model's gradient probe passed; otherwise the
reason is shown and the method list holds occlusion alone.

The progress bar shows the current run's fraction and the time left
(``core.eta.TimeLeft``), counted down every frame between reports.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import List, Optional

import dearpygui.dearpygui as dpg

from yoru_explain.core.eta import TimeLeft, format_duration
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
METHOD_NAMES = {"occlusion": "occlusion", "gradcam": "Grad-CAM"}
BOTH_TIP = ("Grad-CAM, then occlusion, on the same events: two runs, each with its own settings on "
            "the left (select a method to see them) and the same score. Grad-CAM takes seconds and "
            "goes first, so its maps exist even if the occlusion run is cancelled.")
PATCH_PX = "pixels"
PATCH_FRACTION = "x target size"
PATCH_UNIT_TIP = ("pixels: the same patch on every event. x target size: a square patch whose side is "
                  "that fraction of the target's size sqrt(w*h) (of its rotated box for OBB models), "
                  "rounded per event, so animals filmed at another magnification are tiled alike; the "
                  "pixels used are recorded with each result.")


def _bgr_text(bgr) -> str:
    return ",".join(str(int(v)) for v in bgr)


class AnalyzeTab:
    def __init__(self, app):
        self.app = app
        self.state = app.state
        self.popups = app.popups
        self._last_run = None
        self._last_runs: List[object] = []
        self._layer_names: List[str] = ["auto"]
        self._layer_labels: List[str] = [LAYER_AUTO]
        # Occlusion's identity mode, kept while Grad-CAM (always fixed_candidate)
        # is shown in the same field; True while the field shows occlusion's.
        self._occ_identity: Optional[str] = None
        self._identity_is_occ = True
        # The running task's methods, the run the progress is about, its time left.
        self._run_methods: List[str] = []
        self._run_index = 0
        self._started = 0.0
        self._eta = TimeLeft()
        self._shown_end: Optional[float] = None
        self._overlay = ""
        self.results = SelectTable("ana_results", ["event", "method", "class", "status", "error", "original",
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
        for signal in (st.projectChanged, st.eventsChanged, st.selectionChanged, st.adapterChanged):
            signal.connect(self._update_run_buttons)
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
                widgets.heading("Run", themes)
                muted("", themes, tag="ana_run_help", wrap=800)
                with dpg.group(horizontal=True):
                    b = dpg.add_button(label="Analyze selected events", tag="ana_run_sel",
                                       callback=lambda: self.run(selected_only=True))
                    dpg.bind_item_theme(b, themes.primary_button)
                    dpg.add_button(label="Analyze all events", tag="ana_run_all",
                                   callback=lambda: self.run(selected_only=False))
                    dpg.add_button(label="Cancel", tag="ana_cancel", enabled=False,
                                   callback=lambda: self._cancel())
                with dpg.group(horizontal=True):
                    b = dpg.add_button(label="Both methods: selected events", tag="ana_both_sel",
                                       callback=lambda: self.run(selected_only=True, both=True))
                    tooltip(b, BOTH_TIP)
                    b = dpg.add_button(label="Both methods: all events", tag="ana_both_all",
                                       callback=lambda: self.run(selected_only=False, both=True))
                    tooltip(b, BOTH_TIP)
                dpg.add_progress_bar(tag="ana_progress", default_value=0.0, width=-1, overlay="")
                dpg.add_text("Idle.", tag="ana_status", wrap=800)
                dpg.add_spacer(height=4)
                widgets.heading("Last run", themes)
                muted("Double-click a row to see its map on the frame (2 Pick events). Every run is "
                      "listed in 4 Results.", themes, tag="ana_results_help", wrap=800)
                self.results.build()
                dpg.add_button(label="Show on the frame", callback=lambda: self._open_selected())
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    t = dpg.add_text("Log")
                    dpg.bind_item_theme(t, themes.header)
                    dpg.add_button(label="Copy", callback=lambda: dpg.set_clipboard_text(self.log.text()))
                dpg.add_separator()
                self.log.build()
        self._adapter_changed()
        self._method_changed()
        self._update_run_buttons()

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
            with dpg.group(horizontal=True):
                dpg.add_text("patch in ")
                c = dpg.add_radio_button([PATCH_PX, PATCH_FRACTION], tag="ana_patch_unit",
                                         default_value=PATCH_PX, horizontal=True,
                                         callback=lambda: self._show_rows())
                tooltip(c, PATCH_UNIT_TIP)
            with dpg.group(tag="ana_px_rows"):
                self._int_row("patch (px)", "ana_patch", 32, 2, 512, "Square patch size in frame pixels")
                self._int_row("stride (px)", "ana_stride", 16, 1, 512)
            with dpg.group(tag="ana_frac_rows", show=False):
                self._float_row("patch", "ana_patch_frac", 0.25, 0.01, 4.0,
                                "Patch side as a fraction of the target's size sqrt(w*h)")
                self._float_row("stride", "ana_stride_frac", 0.125, 0.005, 4.0,
                                "Step as a fraction of the target's size (half the patch by default)")
            with dpg.group(horizontal=True):
                dpg.add_text("baseline   ")
                c = dpg.add_combo(list(BASELINE_KINDS), tag="ana_baseline", default_value="constant",
                                  width=-1, callback=lambda: self._baseline_help())
                tooltip(c, "What an occluded region is replaced with")
            muted("", themes, tag="ana_baseline_help", wrap=SETTINGS_W - 30)
            with dpg.group(horizontal=True, tag="ana_value_row"):
                dpg.add_text("value B,G,R")
                v = dpg.add_input_text(tag="ana_value", default_value=_bgr_text(HOT_PINK), width=150,
                                       callback=lambda: self._value_typed())
                tooltip(v, "constant baseline: B,G,R. Default 180,105,255 = hot pink #FF69B4; "
                           "0,0,0 black; 114,114,114 YOLOv5 grey")
                b, g, r = HOT_PINK
                c = dpg.add_color_edit((r, g, b, 255), tag="ana_colour", no_alpha=True, no_inputs=True,
                                       no_label=True, callback=lambda: self._colour_picked())
                tooltip(c, "Pick the constant baseline colour")
            with dpg.group(horizontal=True, tag="ana_sigma_row"):
                dpg.add_text("blur sigma ")
                dpg.add_input_float(tag="ana_sigma", default_value=8.0, min_value=0.5, max_value=100.0,
                                    min_clamped=True, max_clamped=True, step=0.5, format="%.2f",
                                    width=140)
            with dpg.group(tag="ana_seed_row"):
                self._int_row("seed", "ana_seed", 0, 0, 2 ** 31 - 1, "Noise baseline seed")
            with dpg.group(horizontal=True):
                dpg.add_text("region     ")
                c = dpg.add_combo(list(REGIONS), tag="ana_region", default_value="context", width=140,
                                  callback=lambda: self._show_rows())
                tooltip(c, "Where patches are placed. context: the box scaled by the factor; box: the "
                           "box only; full: the whole frame")
            with dpg.group(horizontal=True, tag="ana_context_row"):
                dpg.add_text("context x  ")
                dpg.add_input_float(tag="ana_context", default_value=3.0, min_value=1.0, max_value=20.0,
                                    min_clamped=True, max_clamped=True, step=0.5, format="%.2f",
                                    width=140)
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

        with dpg.collapsing_header(label="Advanced", tag="ana_advanced", default_open=False):
            with dpg.group(tag="ana_occ_adv"):
                self._int_row("batch size", "ana_batch", 1, 1, 256,
                              "Images per forward pass. 1 reproduces YORU exactly; larger is faster, "
                              "and the reference score is then taken at the same batch size.")
                dpg.add_checkbox(label="count ambiguous matches in the map", tag="ana_ambiguous")
            dpg.add_checkbox(label="check the adapter against YORU's detector first", tag="ana_verify",
                             default_value=True)
            dpg.add_checkbox(label="reuse identical earlier results (cache)", tag="ana_cache",
                             default_value=True)
            dpg.add_checkbox(label="write preview PNGs", tag="ana_previews", default_value=True)
        self._baseline_help()

    @staticmethod
    def _int_row(label, tag, default, lo, hi, tip=""):
        with dpg.group(horizontal=True):
            dpg.add_text(f"{label:<11}")
            item = dpg.add_input_int(tag=tag, default_value=default, min_value=lo, max_value=hi,
                                     min_clamped=True, max_clamped=True, width=140)
            if tip:
                tooltip(item, tip)

    @staticmethod
    def _float_row(label, tag, default, lo, hi, tip=""):
        with dpg.group(horizontal=True):
            dpg.add_text(f"{label:<11}")
            item = dpg.add_input_float(tag=tag, default_value=default, min_value=lo, max_value=hi,
                                       min_clamped=True, max_clamped=True, step=0.025, format="%.3f",
                                       width=140)
            dpg.add_text("x target size")
            if tip:
                tooltip(item, tip)

    def layout(self, width: int, height: int) -> None:
        dpg.configure_item("ana_settings", height=height)
        right = max(300, width - SETTINGS_W - 16)
        dpg.configure_item("ana_right", width=right, height=height)
        for tag in ("ana_status", "ana_run_help", "ana_results_help"):
            dpg.configure_item(tag, wrap=right - 20)
        rest = max(240, height - 258)
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
        self._show_rows()

    def _show_rows(self):
        """Show only the settings the chosen baseline and region use."""
        kind = dpg.get_value("ana_baseline")
        dpg.configure_item("ana_value_row", show=kind == "constant")
        dpg.configure_item("ana_sigma_row", show=kind == "blur")
        dpg.configure_item("ana_seed_row", show=kind == "noise")
        dpg.configure_item("ana_context_row", show=dpg.get_value("ana_region") == "context")
        relative = dpg.get_value("ana_patch_unit") == PATCH_FRACTION
        dpg.configure_item("ana_px_rows", show=not relative)
        dpg.configure_item("ana_frac_rows", show=relative)

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
        relative = g("ana_patch_unit") == PATCH_FRACTION
        occ = OcclusionConfig(
            patch_size=int(g("ana_patch")), stride=int(g("ana_stride")),
            patch_fraction=round(float(g("ana_patch_frac")), 4) if relative else None,
            stride_fraction=round(float(g("ana_stride_frac")), 4) if relative else None,
            baseline=BaselineSpec(g("ana_baseline"), value, float(g("ana_sigma"))),
            region=g("ana_region"), context_scale=float(g("ana_context")),
            score=ScoreDefinition(g("ana_score"), self._occlusion_identity(),
                                  round(float(g("ana_iou")), 4)),
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
        s("ana_patch_unit", PATCH_FRACTION if o.relative else PATCH_PX)
        if o.relative:
            s("ana_patch_frac", float(o.patch_fraction))
            s("ana_stride_frac", float(o.stride_fraction if o.stride_fraction is not None
                                       else o.patch_fraction / 2))
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
        self._occ_identity = o.score.identity_mode
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

    def _occlusion_identity(self) -> str:
        """Occlusion's identity mode: the field's while it shows occlusion's, else the one kept."""
        if self._identity_is_occ:
            return dpg.get_value("ana_identity")
        modes = self._identity_modes()
        if self._occ_identity in modes:
            return self._occ_identity
        return "fixed_candidate" if "fixed_candidate" in modes or not modes else modes[0]

    def _method_changed(self):
        occ = self.method() == "occlusion"
        if self._identity_is_occ:
            # Kept for the occlusion run of Both methods while Grad-CAM is shown.
            self._occ_identity = dpg.get_value("ana_identity")
        dpg.configure_item("ana_occ", show=occ)
        dpg.configure_item("ana_occ_adv", show=occ)
        dpg.configure_item("ana_cam", show=not occ)
        modes = self._identity_modes()
        # Grad-CAM differentiates the target's own candidate on its own frame.
        dpg.configure_item("ana_identity", enabled=occ and len(modes) > 1,
                           items=modes if occ else ["fixed_candidate"])
        dpg.configure_item("ana_iou", enabled=occ and "redetect" in modes)
        if not occ:
            dpg.set_value("ana_identity", "fixed_candidate")
        elif self._occ_identity in modes:
            dpg.set_value("ana_identity", self._occ_identity)
        elif modes:
            dpg.set_value("ana_identity", modes[0])
        self._identity_is_occ = occ
        self._score_help()
        self._update_run_buttons()

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
        self._update_run_buttons()
        dpg.configure_item("ana_cancel", enabled=busy and self.state.tasks.current == "analysis")

    def _update_run_buttons(self):
        """Say how many events each button runs, and what is missing before one can run."""
        if not dpg.does_item_exist("ana_run_sel"):
            return
        p, st = self.state.project, self.state
        n_sel = len(st.selected_events)
        n_all = len(p.events) if p is not None else 0
        ready = p is not None and st.adapter is not None and not st.tasks.busy

        def events(n):
            return f"{n} event{'' if n == 1 else 's'}"

        dpg.configure_item("ana_run_sel", enabled=ready and n_sel > 0,
                           label=f"Analyze {events(n_sel)} (selected)" if n_sel else
                           "Analyze selected events")
        dpg.configure_item("ana_run_all", enabled=ready and n_all > 0,
                           label=f"Analyze all {events(n_all)}" if n_all else "Analyze all events")
        both = ready and self.gradcam_offered
        dpg.configure_item("ana_both_sel", enabled=both and n_sel > 0,
                           label=f"Both methods: {events(n_sel)} (selected)" if n_sel else
                           "Both methods: selected events")
        dpg.configure_item("ana_both_all", enabled=both and n_all > 0,
                           label=f"Both methods: all {events(n_all)}" if n_all else
                           "Both methods: all events")
        method = "Grad-CAM" if self.method() == "gradcam" else "occlusion"
        if p is None:
            text = "Create or load a project in 1 Project first."
        elif st.adapter is None:
            text = "Load a model in 1 Project first."
        elif n_all == 0:
            text = "No events yet: pick detections in 2 Pick events (click a box, Add selected as event)."
        elif n_sel == 0:
            text = (f"Runs {method} with the settings on the left. Select the events to analyse in "
                    f"2 Pick events, or analyse all {events(n_all)}.")
        else:
            shown = ", ".join(st.selected_events[:6]) + (" ..." if n_sel > 6 else "")
            text = f"Runs {method} with the settings on the left on the selected {shown}."
        if p is not None and st.adapter is not None and n_all:
            text += ("\nBoth methods: Grad-CAM, then occlusion, as two runs, each with its own settings "
                     "(select a method on the left to see them)." if self.gradcam_offered else
                     "\nBoth methods needs Grad-CAM, which this model does not offer.")
        dpg.set_value("ana_run_help", text)

    # -- running ------------------------------------------------------------------------

    def run(self, selected_only: bool, both: bool = False):
        """Analyse the selected (or all) events with the chosen method, or *both* methods."""
        from yoru_explain.core.pipeline import BOTH_METHODS, run_analyses

        p, a = self.state.project, self.state.adapter
        if p is None or a is None or self.state.model_id is None:
            self.popups.message("YORU-Explain", "Create or load a project and load a model in "
                                                "1 Project first.")
            return
        events = list(self.state.selected_events) if selected_only else [e["event_id"] for e in p.events]
        if not events:
            self.popups.message("YORU-Explain", "No events selected. Pick detections as events in "
                                                "2 Pick events and select them there.")
            return
        if both and not self.gradcam_offered:
            self.popups.message("YORU-Explain", "Both methods needs Grad-CAM, which this model does not "
                                                f"offer: {a.capabilities.gradients_reason}")
            return
        try:
            req = self.request()
            requests = [replace(req, method=m) for m in BOTH_METHODS] if both else [req]
        except Exception as e:
            self.popups.error("Settings", e)
            return
        tasks = self.state.tasks
        model_id = self.state.model_id

        def work():
            return run_analyses(p, a, model_id, events, requests,
                                progress=lambda pr: tasks.report(pr.fraction, pr.message, pr),
                                cancel=tasks.cancelled)

        if not tasks.submit("analysis", work):
            self.popups.message("YORU-Explain", "Another task is running.")
            return
        self._run_methods = [r.method for r in requests]
        self._run_index = 0
        self._started = time.monotonic()
        self._eta.restart(self._started)
        self._shown_end = None
        names = ", then ".join(METHOD_NAMES[m] for m in self._run_methods)
        dpg.set_value("ana_status", f"Running {names} on {len(events)} event(s)...")
        dpg.set_value("ana_progress", 0.0)
        self._show_time_left()
        self._busy(True)

    def _cancel(self):
        self.state.tasks.cancel()
        later = self._run_methods[self._run_index + 1:]
        dpg.set_value("ana_status", "Cancelling after the current batch..." + (
            f" ({', '.join(METHOD_NAMES[m] for m in later)} will not run)" if later else ""))

    def _progress(self, fraction, message, detail=None):
        if self.state.tasks.current != "analysis":
            return
        run_index = int(getattr(detail, "run_index", 0) or 0)
        if run_index != self._run_index:
            # The next method's run: its own fraction, its own pace.
            self._run_index = run_index
            self._eta.restart()
            self._shown_end = None
        self._eta.update(fraction)
        dpg.set_value("ana_progress", float(fraction))
        n = len(self._run_methods)
        prefix = (f"Run {run_index + 1} of {n} ({METHOD_NAMES[self._run_methods[run_index]]}): "
                  if n > 1 and run_index < n else "")
        dpg.set_value("ana_status", prefix + message)
        self._show_time_left()

    def _time_left(self, now: float) -> Optional[float]:
        """The estimate as shown: one that rises only a little is not shown.

        Otherwise the text would flicker between two numbers whenever the
        countdown and a new report straddle a rounding step.  The error this
        allows is at most 0.25 s or a tenth of the time left.
        """
        left = self._eta.remaining(now)
        if left is None:
            self._shown_end = None
            return None
        end, shown = now + left, self._shown_end
        if shown is None or end < shown or end - shown > max(0.25, 0.1 * left):
            self._shown_end = shown = end
        return max(0.0, shown - now)

    def time_left_text(self, now: Optional[float] = None) -> str:
        """The progress bar's text: the run, how far it is, and the time left."""
        now = time.monotonic() if now is None else now
        f = self._eta.fraction
        n = len(self._run_methods)
        i = min(self._run_index, max(0, n - 1))
        parts = [f"{i + 1}/{n} {METHOD_NAMES[self._run_methods[i]]}  {f * 100:.0f}%" if n > 1
                 else f"{f * 100:.0f}%"]
        left = self._time_left(now)
        if f >= 0.999:
            parts.append("finishing...")
        elif left is None:
            parts.append("estimating the time left...")
        elif left < 1.0:
            parts.append("less than a second left")
        else:
            parts.append(f"about {format_duration(left)} left")
        later = self._run_methods[i + 1:]
        if later:
            parts.append("next: " + ", ".join(METHOD_NAMES[m] for m in later))
        return "  |  ".join(parts)

    def _show_time_left(self, text: Optional[str] = None) -> None:
        text = self.time_left_text() if text is None else text
        if text != self._overlay:
            self._overlay = text
            dpg.configure_item("ana_progress", overlay=text)

    def _done(self, name, outcome):
        if name != "analysis":
            return
        outcomes = list(outcome) if isinstance(outcome, (list, tuple)) else [outcome]
        self._last_runs = outcomes
        self._last_run = outcomes[-1] if outcomes else None
        took = format_duration(time.monotonic() - self._started)
        finished = bool(outcomes) and all(o.status == "completed" for o in outcomes) \
            and len(outcomes) == len(self._run_methods)
        dpg.set_value("ana_progress", 1.0)
        self._show_time_left(f"100%  |  {'done in' if finished else 'stopped after'} {took}")
        lines = []
        for o in outcomes:
            ident = o.identity_check or {}
            lines.append(f"Run {o.run_id}: {o.status}; {o.counts}; identity check "
                         f"{ident.get('status', '-')}{' (exact)' if ident.get('exact') else ''}.")
        skipped = self._run_methods[len(outcomes):]
        if skipped and outcomes:
            lines.append(f"Not run: {', '.join(METHOD_NAMES[m] for m in skipped)} "
                         f"(the run before was {outcomes[-1].status}).")
        lines.append("Double-click a row below to see its map.")
        dpg.set_value("ana_status", "\n".join(lines))
        rows, data, colors = [], [], []
        for o in outcomes:
            for r in o.results:
                method = r.get("method")
                rows.append([r.get("event_id"), METHOD_NAMES.get(method, method), r.get("behavior_class"),
                             r.get("status"), r.get("error_code"),
                             fmt(r.get("original_score")), fmt(r.get("delta_score")),
                             fmt(r.get("in_box_mean_delta")), fmt(r.get("out_box_mean_delta")),
                             r.get("n_disappeared"), fmt(r.get("cam_in_box_fraction")), r.get("map_file")])
                data.append((str(o.run_dir), r.get("event_id")))
                colors.append(None if r.get("status") == "ok" else th.TEXT_WARN)
        self.results.set_rows(rows, data, colors)
        self.state.runs_changed()

    def _failed(self, name, text, exc):
        if name == "analysis":
            dpg.set_value("ana_status", f"Run failed: {exc}")
            self._show_time_left(f"stopped after {format_duration(time.monotonic() - self._started)}")
            self.popups.error("Analysis", exc)
            self.state.runs_changed()

    def _open_selected(self):
        sel = self.results.selected
        if sel:
            self.state.showResult.emit(*sel[0])

    def tick(self, now: float) -> None:
        for text, level in self._handler.drain():
            self.log.append(text, level)
        if self.state.tasks.busy and self.state.tasks.current == "analysis":
            # The time left counts down between the task's reports.
            self._show_time_left()

    def detach_log(self):
        logging.getLogger("yoru_explain").removeHandler(self._handler)

    @property
    def last_run(self) -> Optional[object]:
        """The last run of the last task (with Both methods: the occlusion run, if it ran)."""
        return self._last_run

    @property
    def last_runs(self) -> List[object]:
        """Every run of the last task, in the order they ran."""
        return list(self._last_runs)
