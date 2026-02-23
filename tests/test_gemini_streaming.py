import base64
from types import SimpleNamespace

import pytest

from jetflow.clients.gemini.async_ import AsyncGeminiClient
from jetflow.clients.gemini.sync import GeminiClient
from jetflow.models.events import ContentDelta, MessageEnd, ThoughtDelta, ThoughtEnd, ThoughtStart


def _part(*, text=None, thought=False, thought_signature=None, function_call=None):
    return SimpleNamespace(
        text=text,
        thought=thought,
        thought_signature=thought_signature,
        function_call=function_call,
    )


def _chunk(parts):
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        finish_reason=None,
        grounding_metadata=None,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=None)


def _response(parts):
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        finish_reason=None,
        grounding_metadata=None,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=None)


def _collect_sync_events(chunks):
    client = object.__new__(GeminiClient)
    return list(client._stream_events(iter(chunks), logger=None))


async def _collect_async_events(chunks):
    client = object.__new__(AsyncGeminiClient)

    async def _stream():
        for chunk in chunks:
            yield chunk

    events = []
    async for event in client._stream_events(_stream(), logger=None):
        events.append(event)
    return events


def _assert_thought_streaming_contract(events):
    content_deltas = [e.delta for e in events if isinstance(e, ContentDelta)]
    thought_deltas = [e.delta for e in events if isinstance(e, ThoughtDelta)]
    thought_starts = [e for e in events if isinstance(e, ThoughtStart)]
    thought_ends = [e for e in events if isinstance(e, ThoughtEnd)]

    assert content_deltas == ["Answer."]
    assert thought_deltas == ["Plan ", "then act."]
    assert len(thought_starts) == 1
    assert len(thought_ends) == 1
    assert thought_starts[0].id == thought_ends[0].id
    assert thought_ends[0].thought == "Plan then act."

    message_end = next(e for e in events if isinstance(e, MessageEnd))
    assert message_end.message.content == "Answer."
    assert len(message_end.message.thoughts) == 1
    assert message_end.message.thoughts[0].summaries == ["Plan then act."]
    assert message_end.message.thoughts[0].id == base64.b64encode(b"sig").decode("ascii")


def test_sync_stream_keeps_thoughts_out_of_content_deltas():
    chunks = [
        _chunk([_part(text="Plan ", thought=True)]),
        _chunk([_part(text="then act.", thought=True, thought_signature=b"sig")]),
        _chunk([_part(text="Answer.", thought=False)]),
    ]
    _assert_thought_streaming_contract(_collect_sync_events(chunks))


@pytest.mark.asyncio
async def test_async_stream_keeps_thoughts_out_of_content_deltas():
    chunks = [
        _chunk([_part(text="Plan ", thought=True)]),
        _chunk([_part(text="then act.", thought=True, thought_signature=b"sig")]),
        _chunk([_part(text="Answer.", thought=False)]),
    ]
    events = await _collect_async_events(chunks)
    _assert_thought_streaming_contract(events)


def test_sync_stream_handles_signature_only_thought_tail():
    chunks = [
        _chunk([_part(text="Thinking...", thought=True)]),
        _chunk([_part(text=None, thought=True, thought_signature=b"tail-sig")]),
    ]
    events = _collect_sync_events(chunks)

    assert [e.delta for e in events if isinstance(e, ContentDelta)] == []
    thought_ends = [e for e in events if isinstance(e, ThoughtEnd)]
    assert len(thought_ends) == 1
    assert thought_ends[0].thought == "Thinking..."

    message_end = next(e for e in events if isinstance(e, MessageEnd))
    assert message_end.message.thoughts[0].id == base64.b64encode(b"tail-sig").decode("ascii")


@pytest.mark.asyncio
async def test_async_stream_handles_signature_only_thought_tail():
    chunks = [
        _chunk([_part(text="Thinking...", thought=True)]),
        _chunk([_part(text=None, thought=True, thought_signature=b"tail-sig")]),
    ]
    events = await _collect_async_events(chunks)

    assert [e.delta for e in events if isinstance(e, ContentDelta)] == []
    thought_ends = [e for e in events if isinstance(e, ThoughtEnd)]
    assert len(thought_ends) == 1
    assert thought_ends[0].thought == "Thinking..."

    message_end = next(e for e in events if isinstance(e, MessageEnd))
    assert message_end.message.thoughts[0].id == base64.b64encode(b"tail-sig").decode("ascii")


def test_sync_parse_response_merges_thought_parts_and_keeps_signature():
    response = _response([
        _part(text="Plan ", thought=True),
        _part(text="then act.", thought=True, thought_signature=b"sig"),
        _part(text="Answer.", thought=False),
    ])
    client = object.__new__(GeminiClient)
    message = client._parse_response(response, logger=None)

    assert message.content == "Answer."
    assert len(message.thoughts) == 1
    assert message.thoughts[0].summaries == ["Plan then act."]
    assert message.thoughts[0].id == base64.b64encode(b"sig").decode("ascii")


def test_sync_parse_response_handles_signature_only_thought_tail():
    response = _response([
        _part(text="Thinking...", thought=True),
        _part(text=None, thought=True, thought_signature=b"tail-sig"),
    ])
    client = object.__new__(GeminiClient)
    message = client._parse_response(response, logger=None)

    assert len(message.thoughts) == 1
    assert message.thoughts[0].summaries == ["Thinking..."]
    assert message.thoughts[0].id == base64.b64encode(b"tail-sig").decode("ascii")
