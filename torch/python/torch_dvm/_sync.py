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
"""Reversible Python hooks for synchronization outside the dispatcher."""

import functools
import threading

import torch

from . import _C

_patches = []
_installed = False
_local = threading.local()
_lock = threading.RLock()

_STREAM_METHODS = (
    "wait_event", "wait_stream", "record_event", "synchronize", "query",
    "__enter__", "__exit__",
)
_EVENT_METHODS = ("record", "wait", "synchronize", "query")


def _before(original):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if (not _installed or getattr(_local, "inside", False)
                or not _C.is_enabled()):
            return original(*args, **kwargs)
        _local.inside = True
        try:
            # Capture entry has already flushed. Its body runs through native
            # kernels, and must not submit deferred work to the captured stream.
            capturing = (torch.cuda.is_initialized()
                         and torch.cuda.is_current_stream_capturing())
            if not capturing:
                # Submit before query as well: the native query then observes
                # the actual queue, without introducing a GPU synchronization.
                _C.flush()
            return original(*args, **kwargs)
        finally:
            _local.inside = False

    return wrapped


def _replace(owner, name, replacement):
    original = getattr(owner, name)
    inherited = name not in vars(owner)
    setattr(owner, name, replacement)
    _patches.append((owner, name, original, replacement, inherited))


def _patch_methods(owner, names):
    for name in names:
        if hasattr(owner, name):
            _replace(owner, name, _before(getattr(owner, name)))


def _subclass(base, methods):
    # The generic Stream/Event types are immutable. Keep their native storage
    # and C API compatibility by subclassing, and retain isinstance checks for
    # objects created before installation.
    class CompatibleType(type(base)):
        def __instancecheck__(cls, instance):
            return isinstance(instance, base)

        def __subclasscheck__(cls, subclass):
            return issubclass(subclass, base)

    namespace = {
        "__module__": base.__module__,
        "__doc__": base.__doc__,
        "__wrapped__": base,
    }
    for name in methods:
        if hasattr(base, name):
            namespace[name] = _before(getattr(base, name))
    return CompatibleType(base.__name__, (base,), namespace)


def _patch_generic_types():
    if not hasattr(torch, "Stream") or not hasattr(torch, "Event"):
        return
    stream_base, event_base = torch.Stream, torch.Event
    stream_type = _subclass(stream_base, _STREAM_METHODS)
    event_type = _subclass(event_base, _EVENT_METHODS)

    @functools.wraps(stream_base.record_event)
    def record_event(self, event=None):
        if not _installed:
            return stream_base.record_event(self, event)
        if event is None:
            event = event_type(device=self.device)
        event.record(self)
        return event

    stream_type.record_event = _before(record_event)
    _replace(torch, "Stream", stream_type)
    _replace(torch, "Event", event_type)

    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None or not hasattr(accelerator, "current_stream"):
        return
    original_current_stream = accelerator.current_stream

    @functools.wraps(original_current_stream)
    def current_stream(*args, **kwargs):
        stream = original_current_stream(*args, **kwargs)
        if not _installed or stream.device.type != "cuda":
            return stream
        return stream_type(stream_id=stream.stream_id,
                           device_index=stream.device_index,
                           device_type=stream.device_type)

    _replace(accelerator, "current_stream", current_stream)


def install():
    """Install once, without initializing CUDA or changing dispatcher state."""
    global _installed
    with _lock:
        if _installed:
            return
        try:
            _patch_methods(torch.cuda.Stream, _STREAM_METHODS)
            _patch_methods(torch.cuda.Event, _EVENT_METHODS)
            # Flush before leaving the producing stream so workspace allocation
            # uses the same stream as the kernel launch.
            _patch_methods(torch.cuda, ("synchronize", "set_stream"))
            _patch_methods(torch.cuda.CUDAGraph, ("capture_begin", "replay"))
            accelerator = getattr(torch, "accelerator", None)
            if accelerator is not None:
                _patch_methods(accelerator, ("synchronize", "set_stream"))
            _patch_generic_types()
            _installed = True
        except Exception:
            uninstall()
            raise


def uninstall():
    """Restore our replacements while preserving wrappers installed later."""
    global _installed
    with _lock:
        _installed = False
        while _patches:
            owner, name, original, replacement, inherited = _patches.pop()
            if getattr(owner, name) is replacement:
                if inherited:
                    delattr(owner, name)
                else:
                    setattr(owner, name, original)
