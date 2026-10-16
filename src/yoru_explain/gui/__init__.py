# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The DearPyGui GUI.  It only displays and dispatches; analysis lives in the core.

The window shell (screen-fitted viewport, Japanese-capable font, the Window
menu) and the base theme are YORU's documented GUI primitives
(``yoru.gui_layout.GuiSession``, ``yoru.gui_base.apply_default_theme``), as in
YORU-Tracker; the screens are YORU-Explain's own.

Every DearPyGui callback runs on the render thread, from the loop in
:func:`yoru_explain.gui.main_window.run` (manual callback management), so
the tabs touch items without locks.  Long work runs on the threads of
:mod:`yoru_explain.gui.worker`, which the loop polls once per frame.

Unlike PyQt5, DearPyGui and tkinter can be imported before torch on Windows
(checked with DearPyGui 1.11.1 and torch 2.11), so no import order is imposed.
"""
