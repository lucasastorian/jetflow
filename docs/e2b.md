# E2B Code Execution

Run Python in a sandboxed cloud environment. Access S3/GCS data, generate charts, persist state across calls.

## Setup

```bash
pip install jetflow[e2b]
export E2B_API_KEY=e2b_...  # Get key at e2b.dev
```

## Basic Usage

```python
from jetflow import Agent
from jetflow.clients.openai import OpenAIClient
from jetflow.actions.e2b_python_exec import E2BPythonExec

agent = Agent(
    client=OpenAIClient(model="gpt-4o"),
    actions=[E2BPythonExec()],
    system_prompt="You are a data analyst. Use Python to analyze data and create visualizations."
)

resp = agent.run("Generate a correlation matrix for a sample dataset and visualize it")
```

## Data Analysis with S3 Storage

Mount your S3 bucket so the agent can analyze real data:

```python
from jetflow import Agent
from jetflow.clients.openai import OpenAIClient
from jetflow.actions.e2b_python_exec import E2BPythonExec, S3Storage

agent = Agent(
    client=OpenAIClient(model="gpt-4o"),
    actions=[E2BPythonExec(
        storage=S3Storage(
            bucket="market-data",
            access_key_id="AKIA...",
            secret_access_key="...",
            region="us-east-1"
        ),
        embeddable_charts=True
    )],
    system_prompt="You are a quantitative analyst. Data is mounted at /home/user/bucket/"
)

resp = agent.run("Load returns.parquet and plot risk-adjusted performance by sector")
```

## Persistent Sessions

Keep state across multiple runs:

```python
exec = E2BPythonExec(session_id="analysis-001", persistent=True)
agent = Agent(client=client, actions=[exec])

# First run: load data
agent.run("Load the CSV into a DataFrame called 'df'")

# Second run: variables persist
agent.run("Now calculate summary statistics on df")
```

## Storage Options

### S3
```python
from jetflow.actions.e2b_python_exec import S3Storage

S3Storage(
    bucket="my-bucket",
    access_key_id="AKIA...",
    secret_access_key="...",
    region="us-east-1",
    mount_path="/home/user/data"  # Default: /home/user/bucket
)
```

### GCS
```python
from jetflow.actions.e2b_python_exec import GCSStorage

GCSStorage(
    bucket="my-bucket",
    service_account_key='{"type": "service_account", ...}',
)
```

### Cloudflare R2
```python
from jetflow.actions.e2b_python_exec import R2Storage

R2Storage(
    bucket="my-bucket",
    account_id="abc123",
    access_key_id="...",
    secret_access_key="...",
)
```

!!! note "Custom Template Required"
    Storage mounting requires a custom E2B template with FUSE tools (s3fs, gcsfuse) installed. See [E2B Template Docs](https://e2b.dev/docs/template/quickstart).

## Chart Extraction

Matplotlib charts are automatically captured:

```python
resp = agent.run("Create a time series chart of monthly returns")

for chart in resp.charts:
    print(chart.type)   # "line"
    print(chart.base64) # PNG data for embedding
```

## Widget Extraction

Extract HTML content (tearsheets, reports) from the sandbox as embeddable widgets:

```python
from jetflow import Agent, action
from jetflow.clients.openai import OpenAIClient
from jetflow.actions.e2b_python_exec import E2BPythonExec, ExtractWidget
from pydantic import BaseModel

class Done(BaseModel):
    summary: str

@action(schema=Done, exit=True)
def done(params: Done) -> str:
    return params.summary

exec = E2BPythonExec(persistent=True, session_id="reports")
widget_extractor = ExtractWidget(python_exec=exec)

agent = Agent(
    client=OpenAIClient(model="gpt-4o"),
    actions=[exec, widget_extractor, done],
    system_prompt="""Generate HTML reports and save to /tmp/.
    After saving, use ExtractWidget to extract the file as a widget.
    Provide a unique id for each widget.""",
    require_action=True
)

resp = agent.run("Create a performance tearsheet and extract it as a widget")

# Access widget from message metadata
for msg in resp.messages:
    if hasattr(msg, 'metadata') and msg.metadata and 'widget' in msg.metadata:
        widget = msg.metadata['widget']
        print(widget['id'])       # "performance-tearsheet"
        print(widget['type'])     # "html"
        print(widget['content'])  # HTML content
```

The LLM workflow is:
1. Generate HTML content in Python
2. Save to file: `with open('/tmp/report.html', 'w') as f: f.write(html)`
3. Call `ExtractWidget(id="my-widget", file_path="/tmp/report.html")`
4. Widget content is returned in metadata for UI rendering

## File Operations

```python
exec = E2BPythonExec(persistent=True, session_id="my-session")
exec.__start__()

# Write data in
exec.write_file("/home/user/data.csv", csv_content)

# Read results out
content = exec.read_file("/home/user/output.csv")

# List files
files = exec.list_files("/home/user")

exec.stop()
```

## Import/Export DataFrames

```python
import pandas as pd

exec = E2BPythonExec(persistent=True, session_id="my-session")
exec.__start__()

# Push DataFrame into sandbox
exec.import_dataframe("market_data", my_dataframe)

# Run agent
agent = Agent(client=client, actions=[exec])
agent.run("Analyze market_data")

# Pull DataFrame out
result_df = exec.extract_dataframe("result")

exec.stop()
```

## Configuration Reference

```python
E2BPythonExec(
    session_id: str = None,       # For persistent sessions
    persistent: bool = False,     # Pause instead of terminate
    timeout: int = 300,           # Sandbox timeout (seconds)
    api_key: str = None,          # Override E2B_API_KEY
    embeddable_charts: bool = False,  # Return charts as HTML
    template: str = None,         # Custom Docker image
    storage: BaseStorage = None,  # S3/GCS/R2 mount
)
```
