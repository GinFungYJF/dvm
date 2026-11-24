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
"""Synchronization must observe deferred work even without another ATen call."""

import functools
import inspect

import pytest
import torch

import dvm_torch
from conftest import CUDA, assert_same, fused


def _classes(api):
    if api == "cuda":
        return torch.cuda.Stream, torch.cuda.Event
    return torch.Stream, torch.Event


@CUDA
@pytest.mark.parametrize("api", ["cuda", "generic"])
@pytest.mark.parametrize("boundary", [
    "record", "wait", "event_query", "event_sync", "wait_event",
    "wait_stream", "stream_query", "stream_sync", "record_event",
    "device_sync", "accelerator_sync",
])
def test_boundary_submits_without_another_operator(api, boundary):
    a = torch.ones(4096, device="cuda")
    torch.cuda.synchronize()
    # CUDA objects created before enable must pick up the installed methods.
    old_stream = torch.cuda.Stream()
    old_event = torch.cuda.Event()
    with fused():
        Stream, Event = _classes(api)
        stream = old_stream if api == "cuda" else Stream(device="cuda")
        event = old_event if api == "cuda" else Event(device="cuda")
        other = Stream(device="cuda")
        event.record(other)
        with stream:
            dvm_torch.reset_stats()
            out = a * 2 + 1
            assert dvm_torch.stats()["flushes"] == 0
            calls = {
                "record": lambda: event.record(stream),
                "wait": lambda: event.wait(stream),
                "event_query": event.query,
                "event_sync": event.synchronize,
                "wait_event": lambda: stream.wait_event(event),
                "wait_stream": lambda: stream.wait_stream(other),
                "stream_query": stream.query,
                "stream_sync": stream.synchronize,
                "record_event": stream.record_event,
                "device_sync": torch.cuda.synchronize,
                "accelerator_sync": torch.accelerator.synchronize,
            }
            result = calls[boundary]()
            # Check before context exit or any tensor read can flush the graph.
            assert dvm_torch.stats()["flushes"] == 1
            if boundary.endswith("query"):
                assert isinstance(result, bool)
            if boundary == "record_event":
                assert isinstance(result, Event)
            stream.synchronize()
    assert_same(out, torch.full_like(a, 3))


@CUDA
@pytest.mark.parametrize("api", ["cuda", "generic"])
@pytest.mark.parametrize("boundary", ["event", "event_wait", "wait_stream", "record_event"])
def test_two_nondefault_streams_match_eager(api, boundary):
    a = torch.arange(65536, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()

    def run(check_pending):
        Stream, Event = _classes(api)
        producer, consumer = Stream(device="cuda"), Stream(device="cuda")
        event = Event() if api == "cuda" else Event(device="cuda")
        with producer:
            dvm_torch.reset_stats()
            pending = a * 2 + 1
            if boundary == "wait_stream":
                consumer.wait_stream(producer)
            else:
                if boundary == "record_event":
                    event = producer.record_event()
                else:
                    event.record(producer)
                if boundary == "event_wait":
                    event.wait(consumer)
                else:
                    consumer.wait_event(event)
            if check_pending:
                stats = dvm_torch.stats()
                assert stats["flushes"] == 1
        with consumer:
            result = pending.clone()
        consumer.synchronize()
        return result

    dvm_torch.disable()
    expected = run(False)
    with fused():
        actual = run(True)
    assert_same(actual, expected)


@CUDA
def test_accelerator_current_stream_and_recorded_event_are_wrapped():
    a = torch.ones(1024, device="cuda")
    with fused():
        stream = torch.accelerator.current_stream()
        assert type(stream) is torch.Stream
        event = stream.record_event()
        assert type(event) is torch.Event
        for boundary in (stream.query, event.query):
            dvm_torch.reset_stats()
            out = a * 2 + 1
            boundary()
            assert dvm_torch.stats()["flushes"] == 1
    assert_same(out, torch.full_like(a, 3))


@CUDA
@pytest.mark.parametrize("api", ["cuda", "generic"])
def test_capture_begin_excludes_previous_pending_work(api):
    a = torch.ones(4096, device="cuda")
    torch.cuda.synchronize()
    with fused():
        Stream, _ = _classes(api)
        stream = Stream(device="cuda")
        graph = torch.cuda.CUDAGraph()
        with stream:
            dvm_torch.reset_stats()
            pending = a * 2 + 1
            assert dvm_torch.stats()["flushes"] == 0
            graph.capture_begin()
            try:
                assert dvm_torch.stats()["flushes"] == 1
                out = pending * 4
            finally:
                graph.capture_end()
            assert dvm_torch.stats()["emitted"] == 2
            # Replaying must not recalculate the pre-capture value from a.
            a.fill_(8)
            graph.replay()
            graph.replay()
            stream.synchronize()
            assert dvm_torch.stats()["emitted"] == 2
    assert_same(out, torch.full_like(a, 12))


@CUDA
def test_cuda_graph_context_and_replay_flush_pending_input_updates():
    a = torch.ones(4096, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with fused():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = a * 4
        a.add_(2)
        graph.replay()
        torch.cuda.synchronize()
    assert_same(out, torch.full_like(a, 12))


def test_hooks_are_idempotent_and_restore_original_objects():
    dvm_torch.disable()
    owners = [
        (torch.cuda.Stream, "query"), (torch.cuda.Event, "record"),
        (torch.cuda, "synchronize"), (torch.cuda, "set_stream"),
        (torch.cuda.CUDAGraph, "capture_begin"),
        (torch.accelerator, "synchronize"), (torch.accelerator, "current_stream"),
        (torch, "Stream"), (torch, "Event"),
    ]
    originals = [getattr(owner, name) for owner, name in owners]
    stream_attributes = set(vars(torch.cuda.Stream))
    try:
        dvm_torch.enable()
        replacements = [getattr(owner, name) for owner, name in owners]
        assert all(a is not b for a, b in zip(originals, replacements))
        assert inspect.signature(torch.cuda.Event.record) == inspect.signature(originals[1])
        assert torch.cuda.Event.record.__doc__ == originals[1].__doc__
        dvm_torch.enable()
        assert all(getattr(owner, name) is replacement
                   for (owner, name), replacement in zip(owners, replacements))
    finally:
        dvm_torch.disable()
    assert all(getattr(owner, name) is original
               for (owner, name), original in zip(owners, originals))
    assert set(vars(torch.cuda.Stream)) == stream_attributes


def test_failed_enable_restores_hooks():
    dvm_torch.disable()
    original = torch.cuda.Event.record
    with pytest.raises(RuntimeError, match="unknown mode"):
        dvm_torch.enable(mode="invalid")
    assert not dvm_torch.is_enabled()
    assert torch.cuda.Event.record is original


@CUDA
def test_generic_objects_survive_disable_and_reenable():
    dvm_torch.disable()
    Stream, Event = torch.Stream, torch.Event
    old_stream, old_event = Stream(device="cuda"), Event(device="cuda")
    with fused():
        assert isinstance(old_stream, torch.Stream)
        assert isinstance(old_event, torch.Event)
        stream = torch.Stream(device="cuda")
    assert isinstance(stream.record_event(), Event)
    a = torch.ones(1024, device="cuda")
    torch.cuda.synchronize()
    with fused():
        with stream:
            dvm_torch.reset_stats()
            out = a * 2 + 1
            stream.synchronize()
            assert dvm_torch.stats()["flushes"] == 1
    assert_same(out, torch.full_like(a, 3))


@CUDA
def test_disable_preserves_a_later_wrapper(monkeypatch):
    original = torch.cuda.synchronize
    dvm_torch.enable()
    installed = torch.cuda.synchronize

    @functools.wraps(installed)
    def later(*args, **kwargs):
        return installed(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", later)
    try:
        dvm_torch.disable()
        assert torch.cuda.synchronize is later
        torch.cuda.synchronize()
    finally:
        # Restore the original, including the fixture's saved value.
        monkeypatch.undo()
        torch.cuda.synchronize = original
