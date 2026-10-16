# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Explore tab: browse frames, pick a detection, look at its results.

Frames are decoded exactly (in order, or by a seek verified against
presentation times) on a worker thread, and the loaded model runs on each
shown frame, so the boxes are the detector's own.  Clicking a box selects
it; "Add as event" stores it -- with its pre-NMS candidate and the frame's
hash -- for analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PyQt5 import QtCore, QtWidgets

from yoru_explain.gui.image_view import VIEW_MODES, FrameView, compose, pick_box
from yoru_explain.gui.qtutil import error_box, fill_table, fmt, selected_data, table
from yoru_explain.gui.state import AppState
from yoru_explain.gui.worker import FrameRequest

DISCLAIMER = ("Perturbation effects and attributions describe the model's score for this detection. "
              "They are not a biological interpretation.")


class ExploreTab(QtWidgets.QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.frame = None            # LoadedFrame on display
        self.detections: List = []
        self._det_adapter = None     # the adapter that produced self.detections
        self.selected: Optional[int] = None
        self._indexed: Dict[str, dict] = {}
        self._imported: Dict[str, dict] = {}
        self._pending_select = None  # (event target) to select once its frame arrives
        self._result = None          # (reader, row, map dict)

        root = QtWidgets.QHBoxLayout(self)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(split)

        # -- left: frame ---------------------------------------------------
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        nav = QtWidgets.QHBoxLayout()
        self.source = QtWidgets.QComboBox()
        self.index = QtWidgets.QSpinBox()
        self.index.setRange(0, 0)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        b_prev = QtWidgets.QPushButton("<")
        b_next = QtWidgets.QPushButton(">")
        b_prev.setFixedWidth(28)
        b_next.setFixedWidth(28)
        nav.addWidget(QtWidgets.QLabel("Source"))
        nav.addWidget(self.source, 2)
        nav.addWidget(QtWidgets.QLabel("Frame"))
        nav.addWidget(b_prev)
        nav.addWidget(self.index)
        nav.addWidget(b_next)
        nav.addWidget(self.slider, 3)
        ll.addLayout(nav)
        self.view = FrameView()
        ll.addWidget(self.view, 1)
        mode_row = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(VIEW_MODES)
        self.alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha.setRange(10, 90)
        self.alpha.setValue(50)
        self.alpha.setFixedWidth(120)
        self.frame_info = QtWidgets.QLabel("")
        mode_row.addWidget(QtWidgets.QLabel("View"))
        mode_row.addWidget(self.mode)
        mode_row.addWidget(QtWidgets.QLabel("overlay alpha"))
        mode_row.addWidget(self.alpha)
        mode_row.addWidget(self.frame_info, 1)
        ll.addLayout(mode_row)
        split.addWidget(left)

        # -- right: detections, events, results -----------------------------
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.addWidget(QtWidgets.QLabel("<b>Detections on this frame</b> (click a box or a row)"))
        self.det_table = table(["#", "class", "score", "box (x1, y1, x2, y2)", "candidate"])
        self.det_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        rl.addWidget(self.det_table, 2)
        row = QtWidgets.QHBoxLayout()
        self.b_add = QtWidgets.QPushButton("Add selected as event")
        self.b_add_all = QtWidgets.QPushButton("Add all on frame")
        self.b_add_csv = QtWidgets.QPushButton("Add CSV boxes on frame")
        self.b_add_csv.setToolTip("The dashed boxes of an attached YORU CSV, as events; a run finds "
                                  "each again with the loaded model (same class, IoU >= 0.9)")
        self.note = QtWidgets.QLineEdit()
        self.note.setPlaceholderText("note (optional)")
        row.addWidget(self.b_add)
        row.addWidget(self.b_add_all)
        row.addWidget(self.b_add_csv)
        row.addWidget(self.note, 1)
        rl.addLayout(row)
        rl.addWidget(QtWidgets.QLabel("<b>Events</b> (selection is what Analyze runs on)"))
        self.events = table(["event", "source", "frame", "class", "score", "candidate", "origin", "note"])
        rl.addWidget(self.events, 3)
        b_remove = QtWidgets.QPushButton("Remove selected events")
        rl.addWidget(b_remove)
        rl.addWidget(QtWidgets.QLabel("<b>Results of the selected event</b>"))
        self.results = QtWidgets.QComboBox()
        rl.addWidget(self.results)
        self.info = QtWidgets.QTextBrowser()
        self.info.setMinimumHeight(140)
        rl.addWidget(self.info, 2)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._request_frame)

        self.source.currentIndexChanged.connect(self._source_changed)
        self.index.valueChanged.connect(self._index_changed)
        self.slider.valueChanged.connect(self.index.setValue)
        b_prev.clicked.connect(lambda: self.index.setValue(self.index.value() - 1))
        b_next.clicked.connect(lambda: self.index.setValue(self.index.value() + 1))
        self.view.clicked.connect(self._clicked)
        self.det_table.itemSelectionChanged.connect(self._det_row_selected)
        self.b_add.clicked.connect(lambda: self.add_events(all_on_frame=False))
        self.b_add_all.clicked.connect(lambda: self.add_events(all_on_frame=True))
        self.b_add_csv.clicked.connect(self.add_csv_events)
        b_remove.clicked.connect(self.remove_events)
        self.events.itemSelectionChanged.connect(self._events_selected)
        self.events.itemDoubleClicked.connect(lambda _i: self.goto_selected_event())
        self.results.currentIndexChanged.connect(self._result_chosen)
        self.mode.currentIndexChanged.connect(self.redraw)
        self.alpha.valueChanged.connect(self.redraw)

        state.projectChanged.connect(self.refresh_sources)
        state.eventsChanged.connect(self.refresh_events)
        state.runsChanged.connect(self._events_selected)
        state.adapterChanged.connect(self._request_frame)
        state.frames.frameReady.connect(self._frame_ready)
        state.frames.indexReady.connect(self._index_ready)
        state.showResult.connect(self.show_result)
        self.refresh_sources()

    # -- sources and frames ----------------------------------------------

    def _current_source(self):
        p = self.state.project
        sid = self.source.currentData()
        if p is None or sid is None:
            return None
        try:
            return p.source(sid)
        except Exception:
            return None

    def refresh_sources(self):
        self.source.blockSignals(True)
        self.source.clear()
        p = self.state.project
        if p is not None:
            for s in p.sources:
                self.source.addItem(f"{s['source_id']}  {Path(s['path']).name}", s["source_id"])
        self.source.blockSignals(False)
        self._indexed.clear()
        self._imported.clear()
        self._source_changed()

    def _source_changed(self):
        s = self._current_source()
        n = max(0, int(s.get("frame_count") or 1) - 1) if s else 0
        for w in (self.index, self.slider):
            w.blockSignals(True)
            w.setRange(0, n)
            w.setValue(0)
            w.blockSignals(False)
        self.frame = None
        self.detections = []
        self.view.set_image(None)
        if s is None:
            return
        if s["kind"] == "video" and s["source_id"] not in self._indexed:
            self._indexed[s["source_id"]] = {"status": "building"}
            self.state.frames.build_index(self._req(s, 0))
        self._load_imported(s)
        self._request_frame()

    def _load_imported(self, s):
        from yoru_explain.io.detections_csv import load_detections_csv

        sid = s["source_id"]
        if sid in self._imported:
            return
        frames: dict = {}
        for entry in s.get("detections_files", []):
            try:
                _, f = load_detections_csv(entry["path"])
                for k, v in f.items():
                    frames.setdefault(k, []).extend(v)
            except Exception as e:
                error_box(self, "Detections CSV", e)
        self._imported[sid] = frames

    def _req(self, s, index):
        return FrameRequest(s["source_id"], s["path"], s["v_flip"], s["h_flip"], int(index),
                            detect=self.state.adapter is not None)

    def _index_changed(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self._debounce.start()

    def _request_frame(self):
        s = self._current_source()
        if s is None:
            return
        self.frame_info.setText(f"loading frame {self.index.value()}...")
        self.state.frames.request(self._req(s, self.index.value()))

    def _index_ready(self, source_id, summary):
        self._indexed[source_id] = summary
        s = self._current_source()
        if s is not None and s["source_id"] == source_id and summary.get("status") == "ok":
            # The decoded frame count, not the container's estimate.
            last = int(summary["frames"]) - 1
            for w in (self.index, self.slider):
                w.blockSignals(True)
                w.setMaximum(last)
                w.blockSignals(False)
        self._update_frame_info()

    def _frame_ready(self, result):
        s = self._current_source()
        req = result.request
        if s is None or req.source_id != s["source_id"] or req.index != self.index.value():
            return  # an older request
        if result.error:
            self.frame_info.setText(f"frame {req.index}: {result.exception}")
            return
        self.frame = result.frame
        self.detections = list(result.detections)
        self._det_adapter = result.adapter
        self.selected = None
        if self._pending_select is not None:
            self.selected = self._match_pending()
            self._pending_select = None
        self._fill_detections()
        self._update_frame_info()
        self.redraw()

    def _update_frame_info(self):
        f = self.frame
        s = self._current_source()
        if f is None or s is None:
            return
        idx = self._indexed.get(s["source_id"], {})
        ts = f"{f.timestamp_ms / 1000:.3f} s" if f.timestamp_ms is not None else "-"
        model = self.state.model_id or "no model loaded"
        self.frame_info.setText(f"frame {f.frame_index}  t={ts}  access={f.access}  "
                                f"time index: {idx.get('status', '-')}  model: {model}")

    # -- detections -------------------------------------------------------

    def _fill_detections(self):
        rows = []
        for i, d in enumerate(self.detections):
            rows.append([i, d.class_name, f"{d.score:.4f}",
                         ", ".join(f"{v:.1f}" for v in d.box),
                         d.candidate.describe() if d.candidate else "-"])
        self.det_table.blockSignals(True)
        fill_table(self.det_table, rows, data=list(range(len(rows))))
        if self.selected is not None:
            self.det_table.selectRow(self.selected)
        self.det_table.blockSignals(False)

    def _clicked(self, x, y):
        i = pick_box([d.box for d in self.detections], x, y)
        if i is not None:
            self.selected = i
            self.det_table.selectRow(i)
            self.redraw()

    def _det_row_selected(self):
        sel = selected_data(self.det_table)
        if sel:
            self.selected = sel[0]
            self.redraw()

    def add_events(self, all_on_frame: bool):
        from yoru_explain import api

        p, a, f, s = self.state.project, self.state.adapter, self.frame, self._current_source()
        if p is None or a is None or f is None or s is None:
            QtWidgets.QMessageBox.information(self, "YORU-Explain",
                                              "Open a project, load a model and show a frame first.")
            return
        if self._det_adapter is not a:
            # Detections of a model that is no longer the loaded one: refresh first.
            self._request_frame()
            QtWidgets.QMessageBox.information(self, "YORU-Explain",
                                              "The model changed; the detections are being refreshed. "
                                              "Select the box again.")
            return
        idx = list(range(len(self.detections))) if all_on_frame else (
            [self.selected] if self.selected is not None else [])
        if not idx:
            QtWidgets.QMessageBox.information(self, "YORU-Explain", "Select a detection first.")
            return
        try:
            for i in idx:
                t = api.target_from_detection(
                    self.detections[i], i, f.image, source_id=s["source_id"], path=s["path"],
                    kind=s["kind"], frame_index=f.frame_index, timestamp_ms=f.timestamp_ms,
                    v_flip=s["v_flip"], h_flip=s["h_flip"], model_id=self.state.model_id or "",
                    access=f.access, frame_sha1=f.sha1)
                p.add_event(t, self.note.text().strip())
            p.save()
        except Exception as e:
            error_box(self, "Add event", e)
        self.state.eventsChanged.emit()

    def add_csv_events(self):
        from yoru_explain import api

        p, f, s = self.state.project, self.frame, self._current_source()
        imported = self._imported.get(s["source_id"], {}).get(f.frame_index, []) if (f and s) else []
        if p is None or not imported:
            QtWidgets.QMessageBox.information(self, "YORU-Explain", "No CSV boxes on this frame.")
            return
        try:
            for x in imported:
                t = api.target_from_detection(
                    x.detection, -1, f.image, source_id=s["source_id"], path=s["path"], kind=s["kind"],
                    frame_index=f.frame_index, timestamp_ms=f.timestamp_ms, v_flip=s["v_flip"],
                    h_flip=s["h_flip"], model_id=self.state.model_id or "", access=f.access,
                    frame_sha1=f.sha1, origin="csv_import")
                p.add_event(t, self.note.text().strip())
            p.save()
        except Exception as e:
            error_box(self, "Add CSV events", e)
        self.state.eventsChanged.emit()

    # -- events -----------------------------------------------------------

    def refresh_events(self):
        p = self.state.project
        rows, ids = [], []
        if p is not None:
            for e in p.events:
                cand = e.get("candidate_index", "")
                rows.append([e["event_id"], e["source_id"], e["frame_index"], e["class_name"],
                             fmt(e["score"]), cand, e.get("origin", ""), e.get("note", "")])
                ids.append(e["event_id"])
        self.events.blockSignals(True)
        fill_table(self.events, rows, data=ids)
        self.events.blockSignals(False)
        self._events_selected()

    def remove_events(self):
        p = self.state.project
        ids = selected_data(self.events)
        if p is None or not ids:
            return
        if QtWidgets.QMessageBox.question(self, "Remove events", f"Remove {len(ids)} event(s)? "
                                          "Existing run results stay on disk.") != QtWidgets.QMessageBox.Yes:
            return
        p.remove_events(ids)
        p.save()
        self.state.eventsChanged.emit()

    def _events_selected(self):
        ids = selected_data(self.events)
        self.state.selected_events = ids
        self.results.blockSignals(True)
        self.results.clear()
        if len(ids) == 1:
            for reader, row in self.state.results_for(ids[0]):
                label = f"{reader.run_id}  {row.get('method')}  {row.get('status')}"
                if row.get("error_code"):
                    label += f" ({row['error_code']})"
                self.results.addItem(label, (str(reader.dir), row.get("event_id")))
        self.results.blockSignals(False)
        self._result_chosen()

    def goto_selected_event(self):
        ids = selected_data(self.events)
        p = self.state.project
        if p is None or len(ids) != 1:
            return
        t = p.target(ids[0])
        i = self.source.findData(t.frame.source_id)
        if i >= 0 and i != self.source.currentIndex():
            self.source.setCurrentIndex(i)
        self._pending_select = t
        self.index.setValue(t.frame.frame_index)
        self._debounce.start()

    def _match_pending(self):
        from yoru_explain.core.geometry import iou

        t = self._pending_select
        if t is None or self.frame is None or t.frame.frame_index != self.frame.frame_index:
            return None
        best = None
        for i, d in enumerate(self.detections):
            if d.class_id != t.detection.class_id:
                continue
            v = iou(d.box, t.box)
            if best is None or v > best[0]:
                best = (v, i)
        return best[1] if best and best[0] >= 0.5 else None

    # -- results ------------------------------------------------------------

    def show_result(self, run_dir: str, event_id: str):
        for r in range(self.events.rowCount()):
            if self.events.item(r, 0).data(QtCore.Qt.UserRole) == event_id:
                self.events.selectRow(r)
                break
        self.goto_selected_event()
        for i in range(self.results.count()):
            if self.results.itemData(i) == (run_dir, event_id):
                self.results.setCurrentIndex(i)
                break

    def _result_chosen(self):
        from yoru_explain.io.runs import RunReader

        data = self.results.currentData()
        self._result = None
        if data:
            run_dir, event_id = data
            try:
                reader = RunReader(run_dir)
                row = reader.result(event_id)
                maps = reader.load_map(row["map_file"]) if row and row.get("map_file") else None
                self._result = (reader, row, maps)
            except Exception as e:
                error_box(self, "Result", e)
        self._show_info()
        self.redraw()

    def _show_info(self):
        if self._result is None:
            self.info.setHtml(f"<i>No result selected.</i><p><small>{DISCLAIMER}</small></p>")
            return
        reader, row, maps = self._result
        kind = {"perturbation_effect": "Perturbation effect (occlusion)",
                "attribution": "Attribution (Grad-CAM)"}.get(row.get("method_kind"), row.get("method"))
        lines = [f"<b>{kind}</b> — run {reader.run_id}",
                 f"status: <b>{row.get('status')}</b> {row.get('error_code') or ''}",
                 f"explained: {row.get('message')}",
                 f"score type: {row.get('score_type')}, identity: {row.get('identity_mode')}",
                 f"original score: {fmt(row.get('original_score'), 6)}; reference at batch size: "
                 f"{fmt(row.get('reference_score'), 6)}"]
        if row.get("method") == "occlusion":
            lines += [f"baseline: {row.get('baseline')}",
                      f"largest drop: {fmt(row.get('delta_score'))} (patch {row.get('max_delta_patch_id')}, "
                      f"score {fmt(row.get('perturbed_score'))}); smallest delta {fmt(row.get('delta_min'))}",
                      f"mean delta in box {fmt(row.get('in_box_mean_delta'))}, outside "
                      f"{fmt(row.get('out_box_mean_delta'))}",
                      f"patches {row.get('n_perturbations')}: counted {row.get('n_counted')}, "
                      f"disappeared {row.get('n_disappeared')}, class changed {row.get('n_class_changed')}, "
                      f"unmatched {row.get('n_unmatched')}, ambiguous {row.get('n_ambiguous')}, "
                      f"candidate dropped by NMS {row.get('n_candidate_dropped')}",
                      f"controls: unchanged-frame delta {fmt(row.get('noop_delta'))}; whole region "
                      f"replaced: {row.get('full_region_status')} {fmt(row.get('full_region_score'))}"]
        else:
            lines += [f"layer {row.get('gradcam_layer')} (stride {row.get('gradcam_stride')}), gradient "
                      f"at {row.get('gradient_support')} cell(s)",
                      f"CAM max {fmt(row.get('cam_max'))}, share inside box "
                      f"{fmt(row.get('cam_in_box_fraction'))}"]
        lines.append(f"<small>{DISCLAIMER}</small>")
        self.info.setHtml("<br>".join(lines))

    # -- drawing --------------------------------------------------------------

    def redraw(self):
        f = self.frame
        if f is None:
            return
        values, kind = None, "delta"
        mode = self.mode.currentText()
        if self._result is not None and self._result[2] is not None:
            _, row, maps = self._result
            event_frame = int(float(row.get("frame_index") or -1))
            same = event_frame == f.frame_index and str(row.get("frame_sha1", "")) in ("", f.sha1)
            if same:
                if "delta_map" in maps:
                    values, kind = maps["delta_map"], "delta"
                elif "cam" in maps:
                    values, kind = maps["cam"], "cam"
        image, panels = compose(f.image, values, kind, mode, self.alpha.value() / 100)
        self.view.set_image(image, frame_width=f.image.shape[1], panels=panels)
        boxes = [d.box for d in self.detections]
        labels = [f"{i}: {d.class_name} {d.score:.2f}" for i, d in enumerate(self.detections)]
        imported = []
        s = self._current_source()
        if s is not None:
            imported = [x.detection.box for x in self._imported.get(s["source_id"], {}).get(f.frame_index, [])]
        self.view.set_boxes(boxes, selected=self.selected, labels=labels, dashed=imported)
