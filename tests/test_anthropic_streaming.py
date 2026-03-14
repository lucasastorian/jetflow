from types import SimpleNamespace

import pytest

from jetflow.clients.anthropic.async_ import AsyncAnthropicClient
from jetflow.clients.anthropic.sync import AnthropicClient
from jetflow.models.events import MessageEnd, ThoughtDelta, ThoughtEnd, ThoughtStart


def _content_block_start(block_type: str, **kwargs):
    return SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type=block_type, **kwargs))


def _thinking_delta(text: str):
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="thinking_delta", thinking=text))


def _signature_delta(signature: str):
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="signature_delta", signature=signature))


def _content_block_stop():
    return SimpleNamespace(type="content_block_stop")


def _collect_sync_events(events):
    client = object.__new__(AnthropicClient)
    return list(client._stream_completion_events(iter(events), logger=None))


async def _collect_async_events(events):
    client = object.__new__(AsyncAnthropicClient)

    async def _stream():
        for event in events:
            yield event

    collected = []
    async for event in client._stream_completion_events(_stream(), logger=None):
        collected.append(event)
    return collected


def _assert_streamed_thought_ids(events):
    thought_starts = [e for e in events if isinstance(e, ThoughtStart)]
    thought_deltas = [e for e in events if isinstance(e, ThoughtDelta)]
    thought_ends = [e for e in events if isinstance(e, ThoughtEnd)]
    message_end = next(e for e in events if isinstance(e, MessageEnd))

    assert len(thought_starts) == 1
    assert len(thought_deltas) == 1
    assert len(thought_ends) == 1

    # Stream IDs should stay stable through start/delta/end even if the provider
    # signature arrives later via signature_delta.
    assert thought_starts[0].id == thought_deltas[0].id == thought_ends[0].id
    assert thought_starts[0].id.startswith("thought_")
    assert thought_ends[0].thought == "plan first"

    # Final message thought block should keep provider signature for replay.
    assert message_end.message.thoughts[0].id == "sig-final"
    assert message_end.message.thoughts[0].summaries == ["plan first"]


def test_sync_stream_thought_ids_stay_consistent_when_signature_arrives_late():
    events = [
        _content_block_start("thinking", signature=""),
        _thinking_delta("plan first"),
        _signature_delta("sig-final"),
        _content_block_stop(),
    ]
    _assert_streamed_thought_ids(_collect_sync_events(events))


@pytest.mark.asyncio
async def test_async_stream_thought_ids_stay_consistent_when_signature_arrives_late():
    events = [
        _content_block_start("thinking", signature=""),
        _thinking_delta("plan first"),
        _signature_delta("sig-final"),
        _content_block_stop(),
    ]
    _assert_streamed_thought_ids(await _collect_async_events(events))
