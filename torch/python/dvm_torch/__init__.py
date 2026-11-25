# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""dvm-torch: eager (graph-less) operator fusion for native PyTorch.

Importing this module is side-effect free. Overrides are installed only when
``TORCH_DVM_FUSION`` is set at import time or ``enable()`` is called, so an
installed-but-unused package is indistinguishable from stock PyTorch.
"""

import atexit
import os

import torch  # noqa: F401  (loads the ATen shared libraries before the extension)

from . import _C  # noqa: F401  (loads the extension)
from . import _sync

__all__ = [
    "enable",
    "disable",
    "is_enabled",
    "stats",
    "reset_stats",
    "flush",
    "Explorer",
]

__version__ = "0.1.0"


def enable(mode=None, **kwargs):
    """Turn fusion on and install CUDA overrides and Python synchronization hooks.

    Args:
        mode: ``"override"``, the only supported mode.
        **kwargs: forwarded to :func:`_C.enable`, e.g. ``level=1``,
            ``disable_ops="where,sum"``, ``guards=False``.
    """
    was_enabled = is_enabled()
    _sync.install()
    try:
        return _C.enable(mode=mode or "", **kwargs)
    except Exception:
        if not was_enabled:
            _C.disable()
            _sync.uninstall()
        raise


def disable():
    """Flush, then restore dispatcher entries and Python synchronization APIs."""
    result = _C.disable()
    _sync.uninstall()
    return result


def is_enabled():
    """True when fusion is currently active."""
    return _C.is_enabled()


def stats():
    """Per-process counters: emitted ops, flush reasons, graph size histogram."""
    return _C.stats()


def reset_stats():
    """Clear the counters without touching the installed overrides."""
    return _C.reset_stats()


def flush():
    """Materialise the pending graph now."""
    return _C.flush()


class Explorer:
    """Debug driver: runs a callable with fusion forced on, then reports stats."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._was_enabled = False

    def __enter__(self):
        self._was_enabled = is_enabled()
        _C.reset_stats()
        enable(**self._kwargs)
        return self

    def __exit__(self, exc_type, exc, tb):
        flush()
        self.report = stats()
        if not self._was_enabled:
            disable()
        return False


@atexit.register
def _flush_on_exit():
    # Never leave a half-built graph behind at interpreter shutdown.
    try:
        if is_enabled():
            flush()
    except Exception:
        pass


def _auto_enable_from_env():
    text = os.environ.get("TORCH_DVM_FUSION", "")
    if text and text.strip().lower() not in ("false", "0", "off", "no", ""):
        try:
            _sync.install()
            _C.enable_from_env()
            if not is_enabled():
                _sync.uninstall()
        except Exception as exc:  # pragma: no cover - only on bad env input
            _C.disable()
            _sync.uninstall()
            raise RuntimeError(
                "dvm-torch: invalid TORCH_DVM_FUSION={!r}: {}".format(text, exc)
            ) from exc


_auto_enable_from_env()
