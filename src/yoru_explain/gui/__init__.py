# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The PyQt5 GUI.  It only displays and dispatches; analysis lives in the core.

torch is imported here, before any PyQt5 module, on purpose.  On Windows,
importing PyQt5 loads its own copy of the MSVC runtime
(``PyQt5/Qt5/bin/MSVCP140.dll``, older than torch's), and a torch imported
afterwards fails with WinError 1114 in ``c10.dll`` (measured with PyQt5-Qt5
5.15.2 and torch 2.11).  Every GUI module is under this package, so importing
any of them goes through this line first.
"""

import torch  # noqa: F401  (must precede PyQt5; see above)
