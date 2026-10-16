# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Analyze tab: method settings, single and batch runs, progress, cancel, log.

The settings here become an :class:`AnalysisRequest`; the run itself is
``core.pipeline.run_analysis`` on the task thread.  Grad-CAM can only be
chosen when the loaded model's gradient probe passed; otherwise the reason
is shown and the option stays off.
"""

from __future__ import annotations

import logging

from PyQt5 import QtCore, QtWidgets

from yoru_explain.core.types import IDENTITY_MODES, SCORE_TYPES
from yoru_explain.gui.qtutil import QtLogHandler, error_box, fill_table, fmt, form_row, table
from yoru_explain.gui.state import AppState
from yoru_explain.methods.baselines import BASELINE_KINDS, HOT_PINK


class AnalyzeTab(QtWidgets.QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        root = QtWidgets.QHBoxLayout(self)

        # -- settings ------------------------------------------------------
        settings = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(settings)
        mbox = QtWidgets.QGroupBox("Method")
        ml = QtWidgets.QVBoxLayout(mbox)
        self.m_occ = QtWidgets.QRadioButton("Occlusion sensitivity (perturbation effect)")
        self.m_cam = QtWidgets.QRadioButton("Grad-CAM (attribution; verified models only)")
        self.m_occ.setChecked(True)
        self.cam_reason = QtWidgets.QLabel("")
        self.cam_reason.setWordWrap(True)
        self.cam_reason.setStyleSheet("color: #a05000")
        for w in (self.m_occ, self.m_cam, self.cam_reason):
            ml.addWidget(w)
        sl.addWidget(mbox)

        self.score_box = QtWidgets.QGroupBox("What is explained")
        fl = QtWidgets.QFormLayout(self.score_box)
        self.score_type = QtWidgets.QComboBox()
        for k, v in SCORE_TYPES.items():
            self.score_type.addItem(k, k)
            self.score_type.setItemData(self.score_type.count() - 1, v, QtCore.Qt.ToolTipRole)
        self.identity = QtWidgets.QComboBox()
        for k, v in IDENTITY_MODES.items():
            self.identity.addItem(k, k)
            self.identity.setItemData(self.identity.count() - 1, v, QtCore.Qt.ToolTipRole)
        self.iou = QtWidgets.QDoubleSpinBox()
        self.iou.setRange(0.05, 1.0)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(0.5)
        form_row(fl, "score", self.score_type, "Which number of the detector is explained")
        form_row(fl, "identity", self.identity,
                 "fixed_candidate: the same pre-NMS candidate on every input. redetect: run NMS and "
                 "match by class and IoU; no score when the match fails.")
        form_row(fl, "match IoU", self.iou, "Re-detection: minimum IoU with the original box")
        self.score_help = QtWidgets.QLabel("")
        self.score_help.setWordWrap(True)
        fl.addRow(self.score_help)
        sl.addWidget(self.score_box)

        self.occ_box = QtWidgets.QGroupBox("Occlusion")
        ol = QtWidgets.QFormLayout(self.occ_box)
        self.patch = QtWidgets.QSpinBox()
        self.patch.setRange(2, 512)
        self.patch.setValue(32)
        self.stride = QtWidgets.QSpinBox()
        self.stride.setRange(1, 512)
        self.stride.setValue(16)
        self.baseline = QtWidgets.QComboBox()
        for k, v in BASELINE_KINDS.items():
            self.baseline.addItem(k, k)
            self.baseline.setItemData(self.baseline.count() - 1, v, QtCore.Qt.ToolTipRole)
        self.value = QtWidgets.QLineEdit(",".join(str(v) for v in HOT_PINK))
        self.value.setToolTip("constant baseline: B,G,R. Default 180,105,255 = hot pink #FF69B4; "
                              "0,0,0 black; 114,114,114 YOLOv5 grey")
        self.colour = QtWidgets.QPushButton("Colour...")
        self.colour.setToolTip("Pick the constant baseline colour")
        value_box = QtWidgets.QWidget()
        value_row = QtWidgets.QHBoxLayout(value_box)
        value_row.setContentsMargins(0, 0, 0, 0)
        value_row.addWidget(self.value, 1)
        value_row.addWidget(self.colour)
        self.sigma = QtWidgets.QDoubleSpinBox()
        self.sigma.setRange(0.5, 100)
        self.sigma.setValue(8.0)
        self.region = QtWidgets.QComboBox()
        self.region.addItems(["context", "box", "full"])
        self.context = QtWidgets.QDoubleSpinBox()
        self.context.setRange(1.0, 20.0)
        self.context.setValue(3.0)
        self.batch = QtWidgets.QSpinBox()
        self.batch.setRange(1, 256)
        self.batch.setValue(1)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        self.ambiguous = QtWidgets.QCheckBox("count ambiguous matches in the map")
        form_row(ol, "patch (px)", self.patch, "Square patch size in frame pixels")
        form_row(ol, "stride (px)", self.stride)
        form_row(ol, "baseline", self.baseline, "What an occluded region is replaced with")
        form_row(ol, "value (B,G,R)", value_box)
        self.colour.clicked.connect(self._pick_colour)
        self.value.textChanged.connect(self._show_colour)
        self._show_colour()
        form_row(ol, "blur sigma", self.sigma)
        form_row(ol, "region", self.region, "context: box scaled by the factor; box; full frame")
        form_row(ol, "context x", self.context)
        form_row(ol, "batch size", self.batch,
                 "Images per forward pass. 1 reproduces YORU exactly; larger is faster, and the "
                 "reference score is then taken at the same batch size.")
        form_row(ol, "seed", self.seed, "Noise baseline seed")
        ol.addRow(self.ambiguous)
        sl.addWidget(self.occ_box)

        self.cam_box = QtWidgets.QGroupBox("Grad-CAM")
        cl = QtWidgets.QFormLayout(self.cam_box)
        self.layer = QtWidgets.QComboBox()
        self.precision = QtWidgets.QComboBox()
        self.precision.addItems(["inference", "fp32"])
        form_row(cl, "layer", self.layer, "auto: the Detect input at the candidate's level")
        form_row(cl, "precision", self.precision,
                 "inference: the same autocast as YORU (the explained score); fp32: autocast off")
        sl.addWidget(self.cam_box)

        obox = QtWidgets.QGroupBox("Run")
        rl = QtWidgets.QVBoxLayout(obox)
        self.verify = QtWidgets.QCheckBox("check the adapter against YORU's detector first")
        self.verify.setChecked(True)
        self.cache = QtWidgets.QCheckBox("reuse identical earlier results (cache)")
        self.cache.setChecked(True)
        self.previews = QtWidgets.QCheckBox("write preview PNGs")
        self.previews.setChecked(True)
        for w in (self.verify, self.cache, self.previews):
            rl.addWidget(w)
        brow = QtWidgets.QHBoxLayout()
        self.b_sel = QtWidgets.QPushButton("Analyze selected events")
        self.b_all = QtWidgets.QPushButton("Analyze all events")
        self.b_cancel = QtWidgets.QPushButton("Cancel")
        self.b_cancel.setEnabled(False)
        for b in (self.b_sel, self.b_all, self.b_cancel):
            brow.addWidget(b)
        rl.addLayout(brow)
        sl.addWidget(obox)
        sl.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(settings)
        scroll.setMinimumWidth(380)
        root.addWidget(scroll)

        # -- progress, results, log ------------------------------------------
        right = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(right)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.status = QtWidgets.QLabel("Idle.")
        self.status.setWordWrap(True)
        vl.addWidget(self.progress)
        vl.addWidget(self.status)
        vl.addWidget(QtWidgets.QLabel("<b>Last run</b> (double-click a row to view it in Explore)"))
        self.results = table(["event", "class", "status", "error", "original", "largest drop",
                              "in box", "outside", "disappeared", "CAM in box", "file"])
        vl.addWidget(self.results, 2)
        vl.addWidget(QtWidgets.QLabel("<b>Log</b>"))
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        vl.addWidget(self.log, 2)
        root.addWidget(right, 1)

        self._handler = QtLogHandler(logging.INFO)
        self._handler.emitter.record.connect(self._log_line)
        logging.getLogger("yoru_explain").addHandler(self._handler)
        logging.getLogger("yoru_explain").setLevel(logging.INFO)
        self._last_run = None

        self.m_occ.toggled.connect(self._method_changed)
        self.score_type.currentIndexChanged.connect(self._score_help)
        self.identity.currentIndexChanged.connect(self._score_help)
        self.b_sel.clicked.connect(lambda: self.run(selected_only=True))
        self.b_all.clicked.connect(lambda: self.run(selected_only=False))
        self.b_cancel.clicked.connect(self._cancel)
        self.results.itemDoubleClicked.connect(self._open_result)
        state.adapterChanged.connect(self._adapter_changed)
        state.tasks.busyChanged.connect(self._busy)
        state.tasks.progress.connect(self._progress)
        state.tasks.finished.connect(self._done)
        state.tasks.failed.connect(self._failed)
        state.requestImported.connect(self.set_request)
        self._adapter_changed()
        self._method_changed()
        self._score_help()

    # -- the baseline colour -------------------------------------------------

    def _bgr(self):
        """The B,G,R typed in the value field, or None if it is not three 0..255 numbers."""
        try:
            bgr = tuple(int(v) for v in self.value.text().replace(" ", "").split(","))
        except ValueError:
            return None
        return bgr if len(bgr) == 3 and all(0 <= v <= 255 for v in bgr) else None

    def _show_colour(self):
        bgr = self._bgr()
        if bgr is None:
            self.colour.setStyleSheet("")
            return
        b, g, r = bgr
        text = "#000000" if 0.299 * r + 0.587 * g + 0.114 * b > 140 else "#ffffff"
        self.colour.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); color: {text};")

    def _pick_colour(self):
        from PyQt5 import QtGui

        b, g, r = self._bgr() or HOT_PINK
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(r, g, b), self, "Baseline colour")
        if c.isValid():
            self.value.setText(f"{c.blue()},{c.green()},{c.red()}")
            self.baseline.setCurrentIndex(self.baseline.findData("constant"))

    # -- the request ---------------------------------------------------------

    def request(self):
        from yoru_explain.core.pipeline import AnalysisRequest
        from yoru_explain.core.types import ScoreDefinition
        from yoru_explain.methods.baselines import BaselineSpec
        from yoru_explain.methods.gradcam import GradCamConfig
        from yoru_explain.methods.occlusion import OcclusionConfig

        value = tuple(int(v) for v in self.value.text().replace(" ", "").split(","))
        occ = OcclusionConfig(
            patch_size=self.patch.value(), stride=self.stride.value(),
            baseline=BaselineSpec(self.baseline.currentData(), value, self.sigma.value()),
            region=self.region.currentText(), context_scale=self.context.value(),
            score=ScoreDefinition(self.score_type.currentData(), self.identity.currentData(),
                                  round(self.iou.value(), 4)),
            batch_size=self.batch.value(), seed=self.seed.value(),
            include_ambiguous=self.ambiguous.isChecked())
        cam = GradCamConfig(layer=self.layer.currentData() or "auto",
                            score_type=self.score_type.currentData(),
                            precision=self.precision.currentText())
        return AnalysisRequest("occlusion" if self.m_occ.isChecked() else "gradcam", occlusion=occ,
                               gradcam=cam, verify_identity=self.verify.isChecked(),
                               use_cache=self.cache.isChecked(), write_previews=self.previews.isChecked())

    def set_request(self, req):
        o = req.occlusion
        self.m_occ.setChecked(req.method == "occlusion")
        self.m_cam.setChecked(req.method == "gradcam")
        self.patch.setValue(o.patch_size[0])
        self.stride.setValue(o.stride[0])
        self.baseline.setCurrentIndex(max(0, self.baseline.findData(o.baseline.kind)))
        self.value.setText(",".join(str(v) for v in o.baseline.value))
        self.sigma.setValue(o.baseline.sigma)
        self.region.setCurrentText(o.region)
        self.context.setValue(o.context_scale)
        st = o.score.score_type if req.method == "occlusion" else req.gradcam.score_type
        self.score_type.setCurrentIndex(max(0, self.score_type.findData(st)))
        self.identity.setCurrentIndex(max(0, self.identity.findData(o.score.identity_mode)))
        self.iou.setValue(o.score.iou_threshold)
        self.batch.setValue(o.batch_size)
        self.seed.setValue(o.seed)
        self.ambiguous.setChecked(o.include_ambiguous)
        i = self.layer.findData(req.gradcam.layer)
        self.layer.setCurrentIndex(max(0, i))
        self.precision.setCurrentText(req.gradcam.precision)
        self.verify.setChecked(req.verify_identity)
        self.cache.setChecked(req.use_cache)
        self.previews.setChecked(req.write_previews)

    # -- state ---------------------------------------------------------------

    def _adapter_changed(self):
        a = self.state.adapter
        self.layer.clear()
        self.layer.addItem("auto (Detect input at the candidate's level)", "auto")
        if a is None:
            self.m_cam.setEnabled(False)
            self.cam_reason.setText("Load a model to see whether Grad-CAM is available for it.")
            return
        caps = a.capabilities
        self.m_cam.setEnabled(bool(caps.gradients))
        if not caps.gradients:
            self.m_occ.setChecked(True)
            self.cam_reason.setStyleSheet("color: #a05000")
            self.cam_reason.setText(f"Grad-CAM is not offered for this model: {caps.gradients_reason}")
        else:
            self.cam_reason.setStyleSheet("color: #2a6a2a")
            self.cam_reason.setText(f"Grad-CAM: {caps.gradients_reason}")
        for li in a.gradient_layers():
            self.layer.addItem(li.describe(), li.name)
        for i in range(self.score_type.count()):
            item = self.score_type.model().item(i)
            item.setEnabled(self.score_type.itemData(i) in caps.score_types)

    def _method_changed(self):
        occ = self.m_occ.isChecked()
        self.occ_box.setVisible(occ)
        self.cam_box.setVisible(not occ)
        self.identity.setEnabled(occ)
        self.iou.setEnabled(occ)
        if not occ:
            self.identity.setCurrentIndex(self.identity.findData("fixed_candidate"))
        self._score_help()

    def _score_help(self):
        st = self.score_type.currentData()
        mode = self.identity.currentData() if self.m_occ.isChecked() else "fixed_candidate"
        kind = ("Perturbation effect: score change when a region is replaced by the baseline."
                if self.m_occ.isChecked() else
                "Attribution: gradient-weighted feature map of the score (Grad-CAM).")
        self.score_help.setText(f"<small>{kind}<br>{SCORE_TYPES.get(st, '')}<br>"
                                f"{IDENTITY_MODES.get(mode, '')}<br>Not a biological interpretation.</small>")

    def _busy(self, busy):
        self.b_sel.setEnabled(not busy)
        self.b_all.setEnabled(not busy)
        self.b_cancel.setEnabled(busy and self.state.tasks.current == "analysis")

    # -- running -------------------------------------------------------------

    def run(self, selected_only: bool):
        from yoru_explain.core.pipeline import run_analysis

        p, a = self.state.project, self.state.adapter
        if p is None or a is None or self.state.model_id is None:
            QtWidgets.QMessageBox.information(self, "YORU-Explain", "Open a project and load a model first.")
            return
        events = list(self.state.selected_events) if selected_only else [e["event_id"] for e in p.events]
        if not events:
            QtWidgets.QMessageBox.information(self, "YORU-Explain",
                                              "No events. Add detections as events in Explore.")
            return
        try:
            req = self.request()
        except Exception as e:
            error_box(self, "Settings", e)
            return
        tasks = self.state.tasks
        model_id = self.state.model_id

        def work():
            return run_analysis(p, a, model_id, events, req,
                                progress=lambda pr: tasks.report(pr.fraction, pr.message),
                                cancel=tasks.cancelled)

        if not tasks.submit("analysis", work):
            QtWidgets.QMessageBox.information(self, "YORU-Explain", "Another task is running.")
            return
        self.status.setText(f"Running {req.method} on {len(events)} event(s)...")
        self.progress.setValue(0)
        self._busy(True)

    def _cancel(self):
        self.state.tasks.cancel()
        self.status.setText("Cancelling after the current batch...")

    def _progress(self, fraction, message):
        if self.state.tasks.current == "analysis":
            self.progress.setValue(int(fraction * 1000))
            self.status.setText(message)

    def _done(self, name, outcome):
        if name != "analysis":
            return
        self._last_run = outcome
        self.progress.setValue(1000)
        ident = outcome.identity_check or {}
        self.status.setText(f"Run {outcome.run_id}: {outcome.status}; {outcome.counts}; identity check "
                            f"{ident.get('status', '-')}{' (exact)' if ident.get('exact') else ''}")
        rows, data = [], []
        for r in outcome.results:
            rows.append([r.get("event_id"), r.get("behavior_class"), r.get("status"), r.get("error_code"),
                         fmt(r.get("original_score")), fmt(r.get("delta_score")),
                         fmt(r.get("in_box_mean_delta")), fmt(r.get("out_box_mean_delta")),
                         r.get("n_disappeared"), fmt(r.get("cam_in_box_fraction")), r.get("map_file")])
            data.append((str(outcome.run_dir), r.get("event_id")))
        fill_table(self.results, rows, data=data)
        self.state.runs_changed()

    def _failed(self, name, text, exc):
        if name == "analysis":
            self.status.setText(f"Run failed: {exc}")
            error_box(self, "Analysis", exc)
            self.state.runs_changed()

    def _open_result(self, item):
        data = self.results.item(item.row(), 0).data(QtCore.Qt.UserRole)
        if data:
            self.state.showResult.emit(*data)

    def _log_line(self, text, level):
        import html

        color = "#b00000" if level >= logging.ERROR else ("#a05000" if level >= logging.WARNING else None)
        if color:
            self.log.appendHtml(f'<span style="color:{color}">{html.escape(text)}</span>')
        else:
            self.log.appendPlainText(text)

    def detach_log(self):
        logging.getLogger("yoru_explain").removeHandler(self._handler)
