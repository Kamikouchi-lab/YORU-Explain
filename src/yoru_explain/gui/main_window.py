# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The main window: five tabs over one shared state, and the render loop.

The window shell -- a viewport fitted to the screen, a Japanese-capable font,
the Window menu, the remembered size -- is YORU's ``GuiSession``; the tabs
are YORU-Explain's own, built from DearPyGui and YORU's theme.

DearPyGui callbacks run on this thread, from the loop in :func:`run`
(manual callback management), so they touch the tabs' state without locks.
A callback that raises is reported in a popup and logged; it never ends the
loop.  Long work runs on the worker threads, which the loop polls.
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Optional

import dearpygui.dearpygui as dpg

from yoru_explain import __version__
from yoru_explain.gui.analyze_tab import AnalyzeTab
from yoru_explain.gui.explore_tab import ExploreTab
from yoru_explain.gui.project_tab import ProjectTab
from yoru_explain.gui.settings_tab import SettingsTab
from yoru_explain.gui.state import AppState, project_name
from yoru_explain.gui.statistics_tab import StatisticsTab
from yoru_explain.gui.theme import apply_explain_theme
from yoru_explain.gui.widgets import Popups

logger = logging.getLogger("yoru_explain.gui")

WINDOW = "explain_main"
TAB_BAR = "explain_tabs"
TABS = (("project", "Project"), ("explore", "Explore"), ("analyze", "Analyze"),
        ("statistics", "Statistics"), ("settings", "Settings"))
#: Tasks the tabs report themselves; any other failure is shown by the window.
TAB_TASKS = ("analysis", "load_model", "check_seeking")
#: How long quitting waits for a cancelled task to reach its next check.
QUIT_WAIT_S = 60.0


class ExplainApp:
    def __init__(self, state: Optional[AppState] = None):
        self.state = state or AppState()
        self.session = None
        self.themes = None
        self.popups: Optional[Popups] = None
        self._status_text = ""
        self._layout_key = None
        self._quit_at: Optional[float] = None
        self._quit_deadline = 0.0
        self.closed = False

    # -- building ------------------------------------------------------------

    def build(self) -> None:
        """Create the window.  Needs a DearPyGui context, not a viewport."""
        self.themes = apply_explain_theme()
        self.popups = Popups(self.themes)
        self.state.report_error = self.report_error
        self.project_tab = ProjectTab(self)
        self.explore_tab = ExploreTab(self)
        self.analyze_tab = AnalyzeTab(self)
        self.statistics_tab = StatisticsTab(self)
        self.settings_tab = SettingsTab(self, self.analyze_tab)
        self.tabs = {"project": self.project_tab, "explore": self.explore_tab,
                     "analyze": self.analyze_tab, "statistics": self.statistics_tab,
                     "settings": self.settings_tab}
        with dpg.texture_registry() as registry:
            pass
        with dpg.window(tag=WINDOW, no_title_bar=True, no_move=True, no_resize=True,
                        no_collapse=True, no_scrollbar=True, no_bring_to_front_on_focus=True,
                        width=1400, height=900):
            dpg.bind_item_theme(WINDOW, self.themes.window)
            with dpg.tab_bar(tag=TAB_BAR):
                for key, label in TABS:
                    with dpg.tab(label=label, tag=f"tab_{key}"):
                        with dpg.child_window(tag=f"page_{key}", border=False):
                            if key == "explore":
                                self.explore_tab.build(registry)
                            else:
                                self.tabs[key].build()
            dpg.add_separator()
            dpg.add_text("", tag="explain_status")
            dpg.bind_item_theme("explain_status", self.themes.muted)
        with dpg.handler_registry():
            dpg.add_key_press_handler(callback=lambda s, key: self._on_key(key))
        self.state.showResult.connect(lambda *_: self.show_tab("explore"))
        self.state.tasks.failed.connect(self._unhandled)
        self._status()
        self.layout()

    def build_menus(self) -> None:
        """The viewport menus next to GuiSession's Window menu."""
        with dpg.menu(label="Project"):
            dpg.add_menu_item(label="New project...", callback=lambda: self.project_tab.new_project())
            dpg.add_menu_item(label="Open project...", callback=lambda: self.project_tab.open_project())
            dpg.add_menu_item(label="Save", callback=lambda: self.project_tab.save())
        with dpg.menu(label="Help"):
            dpg.add_menu_item(label="Open log folder", callback=lambda: self.settings_tab._open_logs())
            dpg.add_menu_item(label=f"YORU-Explain {__version__}", enabled=False)

    def layout(self) -> None:
        if self.session is not None:
            x0, top, width, height = self.session.content_region()
        else:
            x0, top, width, height = 0, 0, 1400, 900
        dpg.set_item_pos(WINDOW, [x0, top])
        dpg.configure_item(WINDOW, width=width, height=height)
        # Window padding (12, 10), the tab bar, and the status line below.
        page_w, page_h = max(400, width - 24), max(300, height - 92)
        for key, tab in self.tabs.items():
            dpg.configure_item(f"page_{key}", width=page_w, height=page_h)
            tab.layout(page_w - 4, page_h - 4)

    # -- tabs ------------------------------------------------------------------

    def current_tab(self) -> str:
        value = dpg.get_value(TAB_BAR)
        for key, _ in TABS:
            tag = f"tab_{key}"
            if value == tag or value == dpg.get_alias_id(tag):
                return key
        return TABS[0][0]   # before the first frame: the first tab is the one shown

    def show_tab(self, key: str) -> None:
        dpg.set_value(TAB_BAR, f"tab_{key}")

    def _on_key(self, key) -> None:
        if self.current_tab() == "explore":
            self.explore_tab.on_key(key)

    # -- errors and status -----------------------------------------------------

    def report_error(self, context: str, exc: BaseException) -> None:
        """Log (with traceback) to ``yoru_explain.log`` and show it in a popup."""
        self.popups.error(context, exc)

    def _unhandled(self, name, text, exc):
        if name not in TAB_TASKS:
            self.popups.error(name, exc)

    def _status(self) -> None:
        a = self.state.adapter
        model = (f"{self.state.model_id} ({a.name} on {a.model_info().get('device')})"
                 if a is not None else "no model")
        busy = f"  |  running: {self.state.tasks.current}" if self.state.tasks.busy else ""
        quitting = "  |  stopping the running task before quitting..." if self._quit_at else ""
        text = f"Project: {project_name(self.state.project)}  |  Model: {model}{busy}{quitting}"
        if text != self._status_text:
            self._status_text = text
            dpg.set_value("explain_status", text)

    @property
    def status_text(self) -> str:
        return self._status_text

    # -- the loop --------------------------------------------------------------

    def tick(self) -> None:
        now = time.monotonic()
        if self.session is not None:
            # A resize, or a new text size from the Window menu (the menu bar
            # grows with it), re-lays out the pages.
            key = (dpg.get_viewport_client_width(), dpg.get_viewport_client_height(),
                   self.session.text_scale)
            if key != self._layout_key:
                self._layout_key = key
                self.layout()
        self.popups.tick()
        self.state.poll()
        for key in ("explore", "analyze"):
            try:
                self.tabs[key].tick(now)
            except Exception as exc:
                self.report_error(f"{key} tab failed", exc)
        self._status()
        if self._quit_at is not None:
            busy = self.state.tasks.busy
            if not busy or now > self._quit_deadline:
                if busy:
                    logger.warning("quit with task %r still running", self.state.tasks.current)
                self.closed = True
                dpg.stop_dearpygui()

    def request_close(self) -> None:
        """The window's close button: ask first while a task runs."""
        if self._quit_at is not None:
            return
        if not self.state.tasks.busy:
            self.quit()
            return
        self.popups.confirm("YORU-Explain", "A task is running. Cancel it and quit?",
                            self._cancel_and_quit, yes="Cancel and quit", no="Keep running")

    def _cancel_and_quit(self) -> None:
        # Cancellation is checked between batches and events; let the task get
        # there, so its run folder is closed with a manifest.
        self.state.tasks.cancel()
        self.quit(wait_s=QUIT_WAIT_S)

    def quit(self, wait_s: float = 0.0) -> None:
        self._quit_at = time.monotonic()
        self._quit_deadline = self._quit_at + wait_s

    def shutdown(self) -> None:
        try:
            self.state.save_project()
        except Exception:
            pass
        self.analyze_tab.detach_log()
        self.state.shutdown()


def run_callbacks(app: ExplainApp, jobs) -> None:
    """``dpg.run_callbacks``, except that a callback that raises is reported, not fatal."""
    for job in jobs or ():
        fn = job[0]
        if fn is None:
            continue
        try:
            n = len(inspect.signature(fn).parameters)
            fn(*job[1:1 + n])
        except Exception as exc:
            app.report_error("Unexpected error", exc)


def run(project_dir: Optional[str] = None, *, exit_after_ms: Optional[int] = None,
        screenshot: Optional[str] = None, tab: Optional[str] = None) -> int:
    """Open the window and run until it is closed.

    *project_dir* opens (or creates) that project first; *tab* shows that tab.
    *exit_after_ms* closes the window after that long, and *screenshot* saves
    the window to that PNG just before (smoke tests).
    """
    from yoru.gui_layout import GuiSession

    from yoru_explain.core.logs import setup_logging

    setup_logging()
    app = ExplainApp()
    session = GuiSession("yoru_explain", f"YORU-Explain {__version__}", width=1500, height=920,
                         min_width=1100, min_height=700)
    session.begin()
    dpg.configure_app(manual_callback_management=True)
    try:
        app.build()
        session.add_layout_menu(extra_builder=app.build_menus)
        session.finish(fill_window=WINDOW, on_resize=app.layout)
        app.session = session
        app.layout()
        # The close button asks first while a task runs (request_close).
        dpg.configure_viewport(0, disable_close=True)
        dpg.set_exit_callback(lambda: app.request_close())
        if project_dir:
            from yoru_explain.io.project import Project

            app.state.set_project(Project.open_or_create(project_dir))
        if tab:
            app.show_tab(tab)
        start = time.monotonic()
        shot = None
        while dpg.is_dearpygui_running():
            run_callbacks(app, dpg.get_callback_queue())
            app.tick()
            dpg.render_dearpygui_frame()
            if exit_after_ms is not None:
                elapsed_ms = (time.monotonic() - start) * 1000.0
                if screenshot and shot is None and elapsed_ms >= 0.7 * exit_after_ms:
                    shot = screenshot
                    dpg.output_frame_buffer(file=screenshot)
                if elapsed_ms >= exit_after_ms:
                    break
    finally:
        try:
            app.shutdown()
            session.save()
        finally:
            dpg.destroy_context()
    logger.info("YORU-Explain closed")
    return 0

