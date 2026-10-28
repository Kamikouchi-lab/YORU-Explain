# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""YORU's theme with YORU-Explain's accent.

The base is YORU's own DearPyGui theme (``yoru.gui_base.apply_default_theme``),
so the three applications look like siblings.  YORU-Tracker puts a teal
accent on top; YORU-Explain's is a pink taken from the default occlusion
baseline (hot pink, #FF69B4), so its windows are never mistaken for either.
"""

from __future__ import annotations

from dataclasses import dataclass

import dearpygui.dearpygui as dpg

#: Hot pink, the default occlusion baseline.
ACCENT = (255, 105, 180)
ACCENT_DIM = (132, 46, 98)
ACCENT_HOVER = (170, 62, 126)
ACCENT_ACTIVE = (108, 36, 80)
TEXT_HEADER = (255, 160, 210)
TEXT_MUTED = (150, 165, 190)
TEXT_WARN = (255, 190, 110)
TEXT_ERROR = (255, 120, 120)
TEXT_OK = (130, 220, 160)

#: Boxes on the frame (RGB): detections, the selected one, an attached CSV's.
BOX = (255, 220, 0)
BOX_SELECTED = (255, 60, 60)
BOX_IMPORTED = (120, 200, 255)


@dataclass(frozen=True)
class Themes:
    window: int
    primary_button: int
    header: int
    muted: int
    warn: int
    error: int
    ok: int
    box: int
    box_selected: int
    box_imported: int


def _text_theme(color):
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, color, category=dpg.mvThemeCat_Core)
    return theme


def _line_theme(color, weight):
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, weight, category=dpg.mvThemeCat_Plots)
    return theme


def apply_explain_theme() -> Themes:
    from yoru.gui_base import apply_default_theme

    apply_default_theme()
    core = dpg.mvThemeCat_Core

    with dpg.theme() as window:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Button, ACCENT_DIM, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, ACCENT_HOVER, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, ACCENT_ACTIVE, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, ACCENT, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, ACCENT, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (255, 150, 205), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (190, 70, 140), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_Header, (88, 40, 82), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, ACCENT_HOVER, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, ACCENT_DIM, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, ACCENT_DIM, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, ACCENT_HOVER, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, (40, 38, 70), category=core)
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, (28, 28, 30), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotPadding, 4, 4, category=dpg.mvThemeCat_Plots)
        # Disabled buttons look disabled.
        with dpg.theme_component(dpg.mvButton, enabled_state=False):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (52, 52, 66), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (52, 52, 66), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (52, 52, 66), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (130, 135, 150), category=core)

    with dpg.theme() as primary:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (196, 64, 138), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (226, 86, 162), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (160, 50, 112), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 248, 252), category=core)
        with dpg.theme_component(dpg.mvButton, enabled_state=False):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (70, 52, 70), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (70, 52, 70), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (70, 52, 70), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (150, 140, 150), category=core)

    return Themes(
        window=window,
        primary_button=primary,
        header=_text_theme(TEXT_HEADER),
        muted=_text_theme(TEXT_MUTED),
        warn=_text_theme(TEXT_WARN),
        error=_text_theme(TEXT_ERROR),
        ok=_text_theme(TEXT_OK),
        box=_line_theme(BOX, 1.6),
        box_selected=_line_theme(BOX_SELECTED, 3.0),
        box_imported=_line_theme(BOX_IMPORTED, 1.2),
    )
