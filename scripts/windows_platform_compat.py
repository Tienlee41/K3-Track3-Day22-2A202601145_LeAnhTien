"""Avoid Python 3.13's WMI platform probe before importing PyTorch on Windows."""

from __future__ import annotations

import os
import platform
import socket
import sys


def prime_platform_cache() -> None:
    """Populate ``platform.uname`` without WMI, which can hang on some laptops."""
    if sys.platform != "win32" or sys.version_info < (3, 13):
        return

    win = sys.getwindowsversion()
    machine = os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64")
    version = ".".join(str(part) for part in win[:3])
    platform._uname_cache = platform.uname_result(  # type: ignore[attr-defined]
        "Windows",
        socket.gethostname(),
        str(win.major),
        version,
        machine,
    )


prime_platform_cache()
