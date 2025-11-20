# Citations

Track sources and references from action results, surface them in real-time as the LLM mentions them, and show users which sources informed the final answer.

---

## Quick start

```python
from jetflow import Agent, action, ActionResult, ContentDelta, MessageEnd
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(description="Search query")

@action(SearchParams)
def search(params: SearchParams, citation_start: int = 1) -> ActionResult:
    """Search and return results with citation metadata"""
    return ActionResult(
        content=f"Tesla Q4 revenue: $25.2B <{citation_start}>\nFull year: $96.8B <{citation_start + 1}>",
        citations={
            citation_start: {"source": "Q4 Earnings", "url": "https://..."},
            citation_start + 1: {"source": "Annual Report", "url": "https://..."}
        },
        summary="Found 2 results"
    )

agent = Agent(client=client, actions=[search])

with agent.stream("What was Tesla's Q4 revenue?") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            print(event.delta, end="", flush=True)

            # Citations appear as tags are detected
            if event.citations:
                for cid, meta in event.citations.items():
                    print(f"\n[{cid}] {meta['source']}")

        elif isinstance(event, MessageEnd):
            # Final message only includes citations actually used
            if event.message.citations:
                print("\n\nSources:")
                for cid, meta in event.message.citations.items():
                    print(f"  [{cid}] {meta['source']}")
```

**Output:**
```
Tesla's Q4 revenue was $25.2B
[1] Q4 Earnings
, bringing full year to $96.8B
[2] Annual Report
.

Sources:
  [1] Q4 Earnings
  [2] Annual Report
```

---

## How it works

### 1. Actions return citations

Actions use `citation_start` to get the next available ID and return citations with their content:

```python
@action(schema)
def search(params, citation_start: int = 1) -> ActionResult:
    citations = {
        citation_start: {"source": "Doc A", "page": 5},
        citation_start + 1: {"source": "Doc B", "page": 12}
    }

    content = f"Result 1 <{citation_start}>\nResult 2 <{citation_start + 1}>"

    return ActionResult(content=content, citations=citations)
```

**Key points:**
- Use `citation_start` parameter to get the next available ID
- Return citation tags like `<1>`, `<2>` in your content
- Map each ID to metadata (source, URL, page, date, etc.)
- IDs auto-increment across the conversation

### 2. Agent tracks citations

The agent's `CitationManager` stores all citations and tracks which ones appear in responses:

```python
# After actions execute
agent.citation_manager.citations
# {1: {"source": "Doc A", ...}, 2: {"source": "Doc B", ...}}

# Check what's used in final response
agent.citation_manager.get_used_citations(content)
# {1: {"source": "Doc A", ...}}  (if only <1> appeared)
```

### 3. Streaming detects citations in real-time

As content streams, the agent checks the full buffer for citation tags:

```python
with agent.stream("query") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            # As soon as <1> appears in the stream, citations is populated
            if event.citations:
                # {"1": {"source": "Doc A", "page": 5}}
                show_citation_tooltip(event.citations)
```

**How detection works:**
- Agent accumulates full content buffer as deltas arrive
- Checks buffer for citation tags like `<1>`, `<2>`, `<1, 2>`
- Tracks which IDs have been seen to avoid duplicates
- Attaches metadata to `ContentDelta` when new tags appear

This handles split tags correctly (e.g., `"Appl"` + `"e <1>"` becomes `"Apple <1>"`).

### 4. Final response filters citations

The final message only includes citations the assistant actually referenced:

```python
for event in events:
    if isinstance(event, ActionExecuted):
        # All citations from this action
        event.message.citations  # {1: {...}, 2: {...}, 3: {...}}

    elif isinstance(event, MessageEnd):
        # Only citations mentioned in assistant's response
        event.message.citations  # {1: {...}, 3: {...}}
```

---

## Citation tag formats

Supports flexible XML-style tags:

```python
"Result <1>"           # Single citation
"Result <1, 2, 3>"     # Multiple citations
"Result <c1>"          # With 'c' prefix
"Result < 1 >"         # With spaces
```

All extract correctly using `CitationExtractor.extract_ids(text)`.

---

## Non-streaming usage

Citations work with `agent.run()` too—just check the final response:

```python
response = agent.run("Search for Tesla revenue")

# Final message citations
if response.messages[-1].citations:
    for cid, meta in response.messages[-1].citations.items():
        print(f"[{cid}] {meta['source']}")

# Or use the citation manager directly
for cid, meta in agent.citation_manager.citations.items():
    print(f"[{cid}] {meta}")
```

---

## Citation metadata structure

You control the metadata dict—common patterns:

```python
# Web search
{
    "source": "Article title",
    "url": "https://example.com",
    "date": "2024-01-15",
    "author": "Jane Doe"
}

# PDF document
{
    "source": "Q4 Earnings Report",
    "filename": "tesla_q4_2024.pdf",
    "page": 15,
    "section": "Financial Results"
}

# Database record
{
    "source": "Customers table",
    "record_id": "cust-12345",
    "timestamp": "2024-01-15T10:30:00Z",
    "query": "SELECT revenue FROM ..."
}

# API response
{
    "source": "Stripe API",
    "endpoint": "/v1/charges",
    "request_id": "req_abc123",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

Store whatever your UI needs to render source links, tooltips, or audit trails.

---

## Multi-action citation flow

When multiple actions return citations, IDs continue incrementing:

```python
# First action
@action(schema)
def search_web(params, citation_start: int = 1):
    return ActionResult(
        content="Result <1>",
        citations={1: {"source": "Web"}}
    )

# Second action
@action(schema)
def search_docs(params, citation_start: int = 1):
    # citation_start will be 2 (next available)
    return ActionResult(
        content="Result <2>",
        citations={2: {"source": "Docs"}}
    )
```

The agent automatically passes the next ID to each action via `citation_start`.

---

## Stateful actions with citations

For class-based actions, `citation_start` is passed to `__call__`:

```python
class SearchAction(BaseAction):
    def __init__(self):
        self.cache = {}

    def __call__(self, action: Action, citation_start: int = 1) -> ActionResult:
        results = self._search(action.body["query"])

        citations = {}
        content_parts = []
        cid = citation_start

        for result in results:
            content_parts.append(f"{result.text} <{cid}>")
            citations[cid] = {
                "source": result.source,
                "url": result.url
            }
            cid += 1

        return ActionResult(
            content="\n".join(content_parts),
            citations=citations
        )
```

---

## Citation manager API

Access the citation manager directly when needed:

```python
# Get next available ID
next_id = agent.citation_manager.get_next_id()  # 1

# Add citations manually (rare)
agent.citation_manager.add_citations({
    10: {"source": "Manual entry"}
})

# Look up a citation
meta = agent.citation_manager.get_citation(1)

# Get citations used in text
used = agent.citation_manager.get_used_citations("Check <1> and <3>")
# {1: {...}, 3: {...}}

# Reset between conversations
agent.citation_manager.reset()
```

---

## Best practices

### Return rich metadata

Include everything your UI might need:

```python
citations = {
    1: {
        "source": "Tesla Q4 2024 Earnings",
        "url": "https://ir.tesla.com/q4-2024",
        "page": 3,
        "section": "Revenue",
        "date": "2024-01-24",
        "highlight": "Revenue increased 122% YoY"  # Exact quote
    }
}
```

### Show citations in real-time

Use `ContentDelta.citations` for instant feedback:

```python
if isinstance(event, ContentDelta):
    text_display.append(event.delta)

    if event.citations:
        for cid, meta in event.citations.items():
            sidebar.add_citation(cid, meta)
            # Highlight citation tag in text
            highlight_citation_tag(cid)
```

### Separate source types

Use a `type` field to render differently:

```python
citations = {
    1: {"type": "web", "url": "...", "title": "..."},
    2: {"type": "pdf", "filename": "...", "page": 5},
    3: {"type": "database", "table": "...", "record_id": "..."}
}

# In UI
if meta["type"] == "web":
    render_link(meta["url"])
elif meta["type"] == "pdf":
    render_pdf_reference(meta["filename"], meta["page"])
```

### Track unused citations

Compare action citations vs final citations to see what the LLM ignored:

```python
action_citations = set()
final_citations = set()

for event in events:
    if isinstance(event, ActionExecuted) and event.message.citations:
        action_citations.update(event.message.citations.keys())

    elif isinstance(event, MessageEnd) and event.message.citations:
        final_citations.update(event.message.citations.keys())

unused = action_citations - final_citations
print(f"LLM ignored citations: {unused}")
```

---

## Common patterns

### Wikipedia-style footnotes

```python
content_buffer = ""
footnotes = {}

with agent.stream("query") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            content_buffer += event.delta

            if event.citations:
                footnotes.update(event.citations)

# Render
print(content_buffer)
print("\nReferences:")
for cid, meta in footnotes.items():
    print(f"[{cid}] {meta['source']} - {meta['url']}")
```

### Interactive tooltips

```python
# React/JS example
if (event.citations) {
    Object.entries(event.citations).forEach(([cid, meta]) => {
        // Find citation tag in rendered text
        const tag = document.querySelector(`[data-citation="${cid}"]`);

        // Attach tooltip
        tippy(tag, {
            content: `
                <strong>${meta.source}</strong><br>
                <a href="${meta.url}">${meta.url}</a>
            `,
            allowHTML: true
        });
    });
}
```

### Audit trail

```python
audit_log = []

for event in events:
    if isinstance(event, ActionExecuted):
        if event.message.citations:
            audit_log.append({
                "action": event.message.action_id,
                "citations_returned": list(event.message.citations.keys()),
                "timestamp": datetime.now()
            })

    elif isinstance(event, MessageEnd):
        if event.message.citations:
            audit_log.append({
                "citations_used": list(event.message.citations.keys()),
                "timestamp": datetime.now()
            })

# Later: prove which sources informed the response
```

---

## Troubleshooting

**Citations not appearing in ContentDelta?**

- Make sure actions return citations with matching IDs in content tags
- Check that tags use supported format: `<1>`, `<2>`, `<1, 2>`
- Verify `CitationManager` has the citations before streaming starts

**Citation IDs restart from 1?**

- Call `agent.reset()` before each conversation to clear state
- Don't create a new agent instance per turn—reuse the same agent

**Getting int vs str type errors?**

- `ContentDelta.citations` uses string keys (`"1"`, `"2"`) for JSON compatibility
- `ActionResult.citations` and `MessageEnd.citations` use int keys (`1`, `2`)
- Convert as needed: `{str(k): v for k, v in citations.items()}`

---

Citations turn actions into trusted, verifiable responses. Use them whenever your agent pulls information from external sources—search results, databases, PDFs, APIs—so users can validate the answer and trace its provenance.
