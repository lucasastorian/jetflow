# Grok custom_field Support Test

This test verifies that the Grok client properly handles actions decorated with `@action(custom_field=...)`.

## Problem

OpenAI's Responses API supports a special `custom` tool type that accepts raw string input instead of JSON:

```python
@action(schema=PythonExec, custom_field="code")
class E2BPythonExec(BaseAction):
    ...
```

This generates a tool schema like:
```json
{
  "type": "custom",
  "name": "e2b_python_exec",
  "description": "..."
}
```

**However, Grok (xAI) doesn't support the `custom` tool type.** Grok only accepts:
- `function`
- `web_search`
- `x_search`
- `file_search`
- `code_interpreter`
- `mcp`

## Solution

The Grok clients override the tool building logic to always use standard `function` format, even for actions with `custom_field`:

```python
# Instead of {"type": "custom"}, Grok uses:
{
  "type": "function",
  "name": "e2b_python_exec",
  "description": "...",
  "parameters": {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"]
  }
}
```

## Test Coverage

The test file `test_grok_custom_field.py` verifies:

1. **Schema Format** - Tools are converted to standard function format
2. **Sync Client** - Grok sync client works with custom_field actions
3. **Sync Streaming** - Streaming works with custom_field actions
4. **Async Client** - Async Grok client works with custom_field actions
5. **Async Streaming** - Async streaming works with custom_field actions

## Running the Tests

```bash
# Set required environment variables
export XAI_API_KEY="your-grok-api-key"
export E2B_API_KEY="your-e2b-api-key"

# Install dependencies
pip install jetflow[e2b]

# Run the test
python tests/test_grok_custom_field.py
```

## Expected Output

```
======================================================================
Grok Client - custom_field Support Tests
======================================================================

=== Test: Grok Tool Schema Format ===
✅ Tool schema correctly formatted as standard function
   Tool type: function
   Tool name: e2b_python_exec
   Has parameters: True

=== Test: Grok Sync with custom_field ===
✅ Grok sync handled custom_field action
   Iterations: 2
   Tool calls: 1
   Result preview: The factorial of 15 is 1,307,674,368,000...

...

======================================================================
Summary
======================================================================
✅ PASS - Tool Schema Format
✅ PASS - Grok Sync - custom_field
✅ PASS - Grok Sync Streaming
✅ PASS - Grok Async - custom_field
✅ PASS - Grok Async Streaming

5 passed, 0 failed, 0 skipped
```

## Implementation Files

- `jetflow/clients/grok/utils.py` - Grok-specific param builder
- `jetflow/clients/grok/async_.py` - Async client with overridden methods
- `jetflow/clients/grok/sync.py` - Sync client with overridden methods
