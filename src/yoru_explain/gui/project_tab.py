# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Project tab: the project folder, its videos/images, its models."""

from __future__ import annotations

from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from yoru_explain.gui.qtutil import error_box, fill_table, selected_data, table
from yoru_explain.gui.state import AppState, project_name, short

VIDEO_FILTER = "Videos and images (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.mpg *.mpeg *.png *.jpg *.jpeg *.bmp *.tif *.tiff)"


class ProjectTab(QtWidgets.QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.project_label = QtWidgets.QLabel(project_name(None))
        self.project_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        b_new = QtWidgets.QPushButton("New project...")
        b_open = QtWidgets.QPushButton("Open project...")
        self.b_save = QtWidgets.QPushButton("Save")
        b_new.clicked.connect(self.new_project)
        b_open.clicked.connect(self.open_project)
        self.b_save.clicked.connect(self.save)
        top.addWidget(QtWidgets.QLabel("<b>Project:</b>"))
        top.addWidget(self.project_label, 1)
        for b in (b_new, b_open, self.b_save):
            top.addWidget(b)
        layout.addLayout(top)

        # -- sources -----------------------------------------------------
        src_box = QtWidgets.QGroupBox("Videos and images")
        sl = QtWidgets.QVBoxLayout(src_box)
        self.sources = table(["id", "file", "kind", "size", "fps", "frames", "flips", "seek check",
                              "detections CSV"])
        sl.addWidget(self.sources)
        row = QtWidgets.QHBoxLayout()
        self.b_add_src = QtWidgets.QPushButton("Add video / image...")
        self.v_flip = QtWidgets.QCheckBox("v_flip")
        self.h_flip = QtWidgets.QCheckBox("h_flip")
        for w in (self.v_flip, self.h_flip):
            w.setToolTip("Flip frames as YORU did before detecting. YORU's CSV files do not record this.")
        self.b_seek = QtWidgets.QPushButton("Check seeking")
        self.b_seek.setToolTip("Compare seeked frames with frames decoded in order on 5 samples")
        self.b_csv = QtWidgets.QPushButton("Attach YORU detections CSV...")
        self.b_csv_events = QtWidgets.QPushButton("Events from CSV...")
        self.b_csv_events.setToolTip("Add the attached CSV's detections as events (filtered by class). "
                                     "A run finds each box again on its frame or reports it.")
        self.b_verify = QtWidgets.QPushButton("Verify files")
        self.b_add_src.clicked.connect(self.add_source)
        self.b_seek.clicked.connect(self.check_seeking)
        self.b_csv.clicked.connect(self.attach_csv)
        self.b_csv_events.clicked.connect(self.events_from_csv)
        self.b_verify.clicked.connect(self.verify)
        for w in (self.b_add_src, self.v_flip, self.h_flip, self.b_seek, self.b_csv, self.b_csv_events,
                  self.b_verify):
            row.addWidget(w)
        row.addStretch(1)
        sl.addLayout(row)
        layout.addWidget(src_box, 1)

        # -- models ------------------------------------------------------
        mdl_box = QtWidgets.QGroupBox("Models")
        ml = QtWidgets.QVBoxLayout(mdl_box)
        self.models = table(["id", "file", "sha256", "adapter", "options"])
        ml.addWidget(self.models)
        row = QtWidgets.QHBoxLayout()
        self.b_add_model = QtWidgets.QPushButton("Add YOLOv5 model...")
        self.b_add_toy = QtWidgets.QPushButton("Add synthetic test model")
        self.b_add_toy.setToolTip("A detector of red/green squares with known answers, for validating "
                                  "the pipeline. Not a model of animals.")
        self.conf = QtWidgets.QDoubleSpinBox()
        self.conf.setRange(0.01, 0.99)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.25)
        self.conf.setToolTip("Confidence threshold. Above 0.25 it is applied after NMS, as YORU does.")
        self.iou = QtWidgets.QDoubleSpinBox()
        self.iou.setRange(0.05, 0.95)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(0.45)
        self.channel = QtWidgets.QComboBox()
        self.channel.addItems(["YORU setting (YORU_YOLOV5_RGB)", "BGR (YORU v1)", "RGB"])
        self.b_add_model.clicked.connect(self.add_model)
        self.b_add_toy.clicked.connect(self.add_toy)
        for w in (self.b_add_model, self.b_add_toy, QtWidgets.QLabel("conf"), self.conf,
                  QtWidgets.QLabel("IoU"), self.iou, QtWidgets.QLabel("channels"), self.channel):
            row.addWidget(w)
        row.addStretch(1)
        ml.addLayout(row)
        row2 = QtWidgets.QHBoxLayout()
        self.b_load = QtWidgets.QPushButton("Load selected model")
        self.b_load.clicked.connect(self.load_model)
        self.model_status = QtWidgets.QLabel("No model loaded.")
        self.model_status.setWordWrap(True)
        row2.addWidget(self.b_load)
        row2.addWidget(self.model_status, 1)
        ml.addLayout(row2)
        layout.addWidget(mdl_box, 1)

        state.projectChanged.connect(self.refresh)
        state.adapterChanged.connect(self.refresh_model_status)
        state.tasks.busyChanged.connect(self._busy)
        state.tasks.finished.connect(self._task_done)
        state.tasks.failed.connect(self._task_failed)
        self.refresh()

    # -- project ---------------------------------------------------------

    def new_project(self):
        from yoru_explain.io.project import Project

        d = QtWidgets.QFileDialog.getExistingDirectory(self, "New project folder (empty or new)")
        if not d:
            return
        try:
            self.state.set_project(Project.create(d))
        except Exception as e:
            error_box(self, "New project", e)

    def open_project(self):
        from yoru_explain.io.project import Project

        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Open project folder")
        if not d:
            return
        try:
            self.state.set_project(Project.open(d))
        except Exception as e:
            error_box(self, "Open project", e)

    def save(self):
        try:
            self.state.save_project()
        except Exception as e:
            error_box(self, "Save project", e)

    def _need_project(self):
        if self.state.project is None:
            QtWidgets.QMessageBox.information(self, "YORU-Explain", "Create or open a project first.")
            return None
        return self.state.project

    # -- sources ---------------------------------------------------------

    def add_source(self):
        p = self._need_project()
        if p is None:
            return
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add videos or images", "", VIDEO_FILTER)
        try:
            for f in files:
                p.add_source(f, v_flip=self.v_flip.isChecked(), h_flip=self.h_flip.isChecked())
            p.save()
        except Exception as e:
            error_box(self, "Add source", e)
        self.refresh()

    def check_seeking(self):
        p = self._need_project()
        sel = selected_data(self.sources)
        if p is None or not sel:
            return

        def work(source_id):
            with p.open_source(source_id) as r:
                return source_id, r.check_seeking()

        if not self.state.tasks.submit("check_seeking", work, sel[0]):
            QtWidgets.QMessageBox.information(self, "YORU-Explain", "Another task is running.")

    def attach_csv(self):
        p = self._need_project()
        sel = selected_data(self.sources)
        if p is None or not sel:
            QtWidgets.QMessageBox.information(self, "YORU-Explain", "Select a video first.")
            return
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "YORU detections CSV", "", "CSV (*.csv)")
        if not f:
            return
        try:
            from yoru_explain.io.detections_csv import load_detections_csv

            layout, frames = load_detections_csv(f)
            s = p.source(sel[0])
            entry = {"path": str(Path(f).resolve()), "layout": layout,
                     "frames": len(frames), "detections": sum(len(v) for v in frames.values())}
            s.setdefault("detections_files", []).append(entry)
            p.save()
        except Exception as e:
            error_box(self, "Attach detections", e)
        self.refresh()

    def events_from_csv(self):
        from yoru_explain import api

        p = self._need_project()
        sel = selected_data(self.sources)
        if p is None or not sel:
            QtWidgets.QMessageBox.information(self, "YORU-Explain", "Select a video first.")
            return
        s = p.source(sel[0])
        files = s.get("detections_files", [])
        if not files:
            QtWidgets.QMessageBox.information(self, "YORU-Explain", "Attach a detections CSV first.")
            return
        classes, ok = QtWidgets.QInputDialog.getText(
            self, "Events from CSV", "Classes to keep (comma-separated; empty = all):")
        if not ok:
            return
        limit, ok = QtWidgets.QInputDialog.getInt(self, "Events from CSV", "At most this many events:",
                                                  100, 1, 1_000_000)
        if not ok:
            return
        names = [c.strip() for c in classes.split(",") if c.strip()] or None
        try:
            added = []
            for entry in files:
                added += api.add_events_from_detections_csv(
                    p, s["source_id"], entry["path"], model_id=self.state.model_id or "",
                    class_names=names, limit=limit - len(added))
                if len(added) >= limit:
                    break
            p.save()
        except Exception as e:
            error_box(self, "Events from CSV", e)
            return
        self.state.eventsChanged.emit()
        QtWidgets.QMessageBox.information(self, "Events from CSV", f"{len(added)} event(s) added.")

    def verify(self):
        p = self._need_project()
        if p is None:
            return
        lines = []
        for s in p.sources:
            lines.append(f"{s['source_id']}: {p.verify_source(s['source_id'])['status']}  {s['path']}")
        from yoru_explain.core.hashing import sha256_file

        for m in p.models:
            if m.sha256 == "synthetic":
                lines.append(f"{m.model_id}: synthetic")
                continue
            ok = Path(m.path).is_file() and sha256_file(m.path) == m.sha256
            lines.append(f"{m.model_id}: {'ok' if ok else 'missing or changed'}  {m.path}")
        QtWidgets.QMessageBox.information(self, "Files", "\n".join(lines) or "Nothing to verify.")

    # -- models ----------------------------------------------------------

    def _options(self):
        rgb = {0: None, 1: False, 2: True}[self.channel.currentIndex()]
        return {"conf_thresh": round(self.conf.value(), 4), "iou_thresh": round(self.iou.value(), 4),
                "rgb_input": rgb}

    def add_model(self):
        p = self._need_project()
        if p is None:
            return
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "YOLOv5 checkpoint", "", "PyTorch (*.pt)")
        if not f:
            return
        try:
            p.add_model(f, "yoru-yolov5", self._options())
            p.save()
        except Exception as e:
            error_box(self, "Add model", e)
        self.refresh()

    def add_toy(self):
        p = self._need_project()
        if p is None:
            return
        p.add_model(None, "toy-grid", {})
        p.save()
        self.refresh()

    def load_model(self):
        p = self._need_project()
        sel = selected_data(self.models)
        if p is None:
            return
        if not sel:
            if len(p.models) == 1:
                sel = [p.models[0].model_id]
            else:
                QtWidgets.QMessageBox.information(self, "YORU-Explain", "Select a model first.")
                return
        record = p.model(sel[0])
        device = self.state.device

        def work():
            from yoru_explain import api

            weights = None if record.sha256 == "synthetic" else record.path
            dev = device if record.adapter != "toy-grid" else ("cpu" if device == "auto" else device)
            return record.model_id, api.load_adapter(record.adapter, weights, device=dev,
                                                     **dict(record.options))

        if not self.state.tasks.submit("load_model", work):
            QtWidgets.QMessageBox.information(self, "YORU-Explain", "Another task is running.")
            return
        self.model_status.setText(f"Loading {record.model_id}...")

    def refresh_model_status(self):
        a = self.state.adapter
        if a is None:
            self.model_status.setText("No model loaded.")
            return
        info = a.model_info()
        caps = a.capabilities
        grad = "Grad-CAM available" if caps.gradients else f"Grad-CAM unavailable ({caps.gradients_reason})"
        self.model_status.setText(
            f"<b>{self.state.model_id}</b> loaded: {a.name} on {info.get('device')}, "
            f"{info.get('channel_order', '')}; classes {info.get('class_names')}; {grad}; "
            f"batch-invariant: {caps.batch_invariant} (max diff {caps.batch_max_abs_diff})")

    # -- refresh ---------------------------------------------------------

    def refresh(self):
        p = self.state.project
        self.project_label.setText(project_name(p))
        if p is None:
            fill_table(self.sources, [])
            fill_table(self.models, [])
            return
        rows, ids = [], []
        for s in p.sources:
            seek = s.get("seek_check") or {}
            csvs = ", ".join(Path(c["path"]).name for c in s.get("detections_files", []))
            rows.append([s["source_id"], short(s["path"]), s["kind"], f"{s['width']}x{s['height']}",
                         f"{s['fps']:.3f}" if s.get("fps") else "", s.get("frame_count"),
                         ("v" if s["v_flip"] else "") + ("h" if s["h_flip"] else "") or "-",
                         seek.get("status", ""), csvs])
            ids.append(s["source_id"])
        fill_table(self.sources, rows, data=ids)
        mrows, mids = [], []
        for m in p.models:
            mrows.append([m.model_id, short(m.path), m.sha256[:12], m.adapter, dict(m.options)])
            mids.append(m.model_id)
        fill_table(self.models, mrows, data=mids)

    def _busy(self, busy):
        for b in (self.b_load, self.b_seek):
            b.setEnabled(not busy)

    def _task_done(self, name, result):
        if name == "load_model":
            model_id, adapter = result
            self.state.set_adapter(adapter, model_id)
        elif name == "check_seeking" and self.state.project is not None:
            source_id, check = result
            self.state.project.source(source_id)["seek_check"] = check
            self.state.project.save()
            self.refresh()
            if check.get("status") == "failed":
                QtWidgets.QMessageBox.warning(
                    self, "Seeking", f"Seeking is not exact for this video (frames {check['mismatched']}). "
                    "YORU-Explain reads analysed frames in order and verifies seeks by presentation "
                    "time, so results are not affected; plain seeking would show wrong frames.")

    def _task_failed(self, name, text, exc):
        if name in ("load_model", "check_seeking"):
            if name == "load_model":
                self.model_status.setText("Loading failed.")
            error_box(self, name.replace("_", " ").capitalize(), exc)
