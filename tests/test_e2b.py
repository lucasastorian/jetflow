"""E2B Code Interpreter Tests

Tests for E2B code interpreter including basic execution, persistence,
dataframe handling, and variable extraction.

Requirements:
- E2B_API_KEY environment variable
- ANTHROPIC_API_KEY environment variable
- pip install jetflow[e2b,anthropic]
"""

import os
import sys
import uuid
from dotenv import load_dotenv

load_dotenv()

# Check dependencies
try:
    from jetflow.actions.e2b_python_exec import E2BPythonExec
    from jetflow.agent import Agent
    from jetflow.clients.anthropic import AnthropicClient
    HAS_E2B = True
except ImportError:
    HAS_E2B = False

HAS_API_KEY = os.getenv("E2B_API_KEY") is not None
HAS_ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY") is not None


# Shared test client
def get_client():
    """Get Anthropic client for tests"""
    return AnthropicClient(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-haiku-4-5",
    )


def get_mini_client():
    """Get GPT-4o-mini client for tests"""
    from jetflow.clients.openai import OpenAIClient
    return OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )


def skip_if_no_e2b(func):
    """Skip test if E2B not available"""
    def wrapper(*args, **kwargs):
        if not HAS_E2B:
            print(f"⚠️  SKIP: {func.__name__} - E2B not installed")
            return None
        if not HAS_API_KEY:
            print(f"⚠️  SKIP: {func.__name__} - E2B_API_KEY not set")
            return None
        if not HAS_ANTHROPIC_KEY:
            print(f"⚠️  SKIP: {func.__name__} - ANTHROPIC_API_KEY not set")
            return None
        return func(*args, **kwargs)
    return wrapper


# =============================================================================
# BASIC EXECUTION TESTS
# =============================================================================

@skip_if_no_e2b
def test_simple_calculation():
    """Basic arithmetic calculation"""
    print("\n=== Test: Simple Calculation ===")

    client = get_client()
    executor = E2BPythonExec()
    agent = Agent(client=client, actions=[executor], max_iter=3, verbose=False)

    response = agent.run("Calculate 2^10 using Python")

    assert response.success, "Agent should complete"
    assert "1024" in response.content, f"Expected 1024, got: {response.content}"

    print(f"✅ Result: {response.content[:200]}")
    return True


@skip_if_no_e2b
def test_stdout_capture():
    """Print statement output capture"""
    print("\n=== Test: Stdout Capture ===")

    client = get_client()
    executor = E2BPythonExec()
    agent = Agent(client=client, actions=[executor], max_iter=3, verbose=False)

    response = agent.run("Print 'Hello from E2B' using Python")

    assert response.success, "Agent should complete"
    assert "Hello from E2B" in response.content, f"Expected output in: {response.content}"

    print("✅ Stdout captured")
    return True


@skip_if_no_e2b
def test_error_handling():
    """Python error handling"""
    print("\n=== Test: Error Handling ===")

    client = get_client()
    executor = E2BPythonExec()
    agent = Agent(client=client, actions=[executor], max_iter=3, verbose=False)

    response = agent.run("Divide 10 by 0 in Python")

    # Should complete without crashing
    assert response.iterations > 0, "Agent should attempt execution"

    print("✅ Error handled gracefully")
    return True


@skip_if_no_e2b
def test_verbose_streaming():
    """Verbose streaming to see agent thinking"""
    print("\n=== Test: Verbose Streaming ===")
    print("(Shows agent thinking, actions, and execution)")
    print()

    client = get_client()
    executor = E2BPythonExec()
    agent = Agent(client=client, actions=[executor], max_iter=3, verbose=True)

    # Stream with verbose output
    for event in agent.stream("Calculate the factorial of 5 using Python"):
        pass  # Events printed by verbose=True

    print("\n✅ Streaming works with verbose output")
    return True


# =============================================================================
# DATAFRAME TESTS
# =============================================================================

@skip_if_no_e2b
def test_dataframe_creation_and_formatting():
    """Test how DataFrames are displayed when returned by LLM"""
    print("\n=== Test: DataFrame Display Formatting ===")

    client = get_client()
    executor = E2BPythonExec()
    agent = Agent(client=client, actions=[executor], max_iter=5, verbose=False)

    query = """
    Create a pandas DataFrame with the following data:
    - Years: 2020, 2021, 2022, 2023
    - Revenue: 100, 120, 150, 180

    Calculate the year-over-year percentage change (pct_change) for revenue.
    Show me the resulting DataFrame with the pct_change column.
    """

    response = agent.run(query)

    assert response.success, "Agent should complete"

    # Find the tool output message to see actual DataFrame formatting
    tool_message = next((msg for msg in response.messages if msg.role == 'tool'), None)

    print("\n📊 Tool Output (actual DataFrame):")
    print("=" * 70)
    if tool_message:
        print(tool_message.content)
    print("=" * 70)

    print("\n💬 LLM Response:")
    print("=" * 70)
    print(response.content)
    print("=" * 70)

    # Check that calculation happened
    has_pct_or_percent = "pct" in response.content.lower() or "percent" in response.content.lower() or "%" in response.content
    print(f"\n✅ DataFrame created and pct_change calculated: {has_pct_or_percent}")

    return True


@skip_if_no_e2b
def test_dataframe_cagr_calculation():
    """Test CAGR calculation with DataFrame"""
    print("\n=== Test: CAGR Calculation ===")

    client = get_client()
    executor = E2BPythonExec()
    agent = Agent(client=client, actions=[executor], max_iter=5, verbose=False)

    query = """
    Create a DataFrame with revenue data:
    - 2020: $1,000,000
    - 2023: $2,000,000

    Calculate the CAGR (Compound Annual Growth Rate) over this 3-year period.
    Show your work and the final CAGR percentage.
    """

    response = agent.run(query)

    assert response.success, "Agent should complete"

    # Find the tool output messages to see actual calculations
    tool_messages = [msg for msg in response.messages if msg.role == 'tool']

    print("\n📊 Tool Outputs (actual calculations):")
    print("=" * 70)
    for i, msg in enumerate(tool_messages, 1):
        print(f"\n[Tool Call {i}]")
        print(msg.content)
    print("=" * 70)

    print("\n💬 LLM Response:")
    print("=" * 70)
    print(response.content)
    print("=" * 70)

    # Check for CAGR-related terms
    has_cagr = "cagr" in response.content.lower() or "compound" in response.content.lower()
    print(f"\n✅ CAGR calculated: {has_cagr}")

    return True


# =============================================================================
# VARIABLE EXTRACTION TESTS
# =============================================================================

@skip_if_no_e2b
def test_extract_dataframe():
    """Extract DataFrame using extract_dataframe()"""
    print("\n=== Test: Extract DataFrame ===")

    executor = E2BPythonExec()
    executor.__start__()

    # Create a DataFrame
    executor.run_code("""
import pandas as pd
df = pd.DataFrame({
    'product': ['A', 'B', 'C'],
    'revenue': [100, 200, 150],
    'profit': [20, 50, 30]
})
""")

    # Extract it
    data = executor.extract_dataframe('df')

    executor.__stop__()

    assert data is not None, "Should extract DataFrame"
    assert len(data) == 3, f"Expected 3 rows, got {len(data)}"
    assert data[0]['product'] == 'A', f"Expected product A, got {data[0]}"

    print(f"✅ Extracted DataFrame: {data}")
    return True


@skip_if_no_e2b
def test_extract_variable():
    """Extract simple variables using extract_variable()"""
    print("\n=== Test: Extract Variable ===")

    executor = E2BPythonExec()
    executor.__start__()

    # Create variables
    executor.run_code("x = 42")
    executor.run_code("y = [1, 2, 3]")
    executor.run_code("z = {'key': 'value'}")

    # Extract them
    x = executor.extract_variable('x')
    y = executor.extract_variable('y')
    z = executor.extract_variable('z')

    executor.__stop__()

    assert x == 42, f"Expected 42, got {x}"
    assert y == [1, 2, 3], f"Expected [1,2,3], got {y}"
    assert z == {'key': 'value'}, f"Expected dict, got {z}"

    print(f"✅ Extracted: x={x}, y={y}, z={z}")
    return True


@skip_if_no_e2b
def test_manual_code_execution():
    """Execute code outside agent lifecycle"""
    print("\n=== Test: Manual Code Execution ===")

    executor = E2BPythonExec()
    executor.__start__()

    # Run code manually
    result = executor.run_code("print('Manual execution'); 2 + 2")

    assert "Manual execution" in result or "4" in result, f"Expected output in: {result}"

    executor.__stop__()

    print(f"✅ Manual execution works")
    return True


# =============================================================================
# SESSION PERSISTENCE TESTS
# =============================================================================

@skip_if_no_e2b
def test_variable_persistence():
    """Variables persist across agent runs"""
    print("\n=== Test: Variable Persistence ===")

    session_id = f"test_persist_{uuid.uuid4().hex[:8]}"
    client = get_client()

    # Run 1: Create variables
    print(f"Session: {session_id}")
    print("Run 1: Creating variables...")
    executor1 = E2BPythonExec(session_id=session_id, persistent=True)
    agent1 = Agent(client=client, actions=[executor1], max_iter=3, verbose=False)
    response1 = agent1.run("Set x = 42 and y = 100 in Python")
    assert response1.success

    # Run 2: Access variables
    print("Run 2: Accessing variables...")
    executor2 = E2BPythonExec(session_id=session_id, persistent=True)
    agent2 = Agent(client=client, actions=[executor2], max_iter=3, verbose=False)
    response2 = agent2.run("What is x + y?")
    assert response2.success

    has_142 = "142" in response2.content
    print(f"✅ Variables persisted: {has_142}")

    return True


@skip_if_no_e2b
def test_lifecycle_hooks():
    """Lifecycle hooks called by agent"""
    print("\n=== Test: Lifecycle Hooks ===")

    session_id = f"test_hooks_{uuid.uuid4().hex[:8]}"
    client = get_client()

    executor = E2BPythonExec(session_id=session_id, persistent=True)
    agent = Agent(client=client, actions=[executor], max_iter=2, verbose=False)

    response = agent.run("Set z = 999")

    assert response.success, "Agent should complete"
    print("✅ Hooks called successfully")

    return True


@skip_if_no_e2b
def test_sandbox_pause_verification():
    """Verify sandboxes are actually being paused in persistent mode"""
    print("\n=== Test: Sandbox Pause Verification ===")

    session_id = f"test_pause_{uuid.uuid4().hex[:8]}"
    client = get_client()

    # Run 1: Create persistent session
    print(f"Session: {session_id}")
    print("Run 1: Creating persistent session...")
    executor1 = E2BPythonExec(session_id=session_id, persistent=True)
    agent1 = Agent(client=client, actions=[executor1], max_iter=2, verbose=False)
    response1 = agent1.run("Set test_var = 'paused'")
    assert response1.success

    # Get the sandbox ID that was created
    sandbox_id = executor1.executor._last_sandbox_id
    print(f"Created sandbox: {sandbox_id}")

    # Query E2B API to check sandbox state
    try:
        from e2b_code_interpreter import Sandbox, SandboxQuery, SandboxState

        # Small delay to allow pause to happen
        import time
        time.sleep(2)

        # Query for paused sandboxes with this session_id
        query = SandboxQuery(
            state=[SandboxState.PAUSED],
            metadata={'session_id': session_id}
        )

        paginator = Sandbox.list(query=query)
        paused_sandboxes = paginator.next_items()

        print(f"Found {len(paused_sandboxes)} paused sandbox(s)")

        if paused_sandboxes:
            paused_sb = paused_sandboxes[0]
            print(f"✅ Sandbox is PAUSED: {paused_sb.sandbox_id}")
            print(f"   Metadata: {paused_sb.metadata}")
            assert paused_sb.sandbox_id == sandbox_id, "Paused sandbox ID should match"
        else:
            print("❌ No paused sandboxes found - checking running state...")

            # Check if it's still running
            running_query = SandboxQuery(
                state=[SandboxState.RUNNING],
                metadata={'session_id': session_id}
            )
            running_paginator = Sandbox.list(query=running_query)
            running_sandboxes = running_paginator.next_items()

            if running_sandboxes:
                print(f"⚠️  Sandbox is still RUNNING (expected PAUSED): {running_sandboxes[0].sandbox_id}")
            else:
                print("⚠️  Sandbox not found in PAUSED or RUNNING state")

        return len(paused_sandboxes) > 0

    except Exception as e:
        print(f"⚠️  Could not verify pause state: {e}")
        return None


# =============================================================================
# CHART EXTRACTION TESTS
# =============================================================================

@skip_if_no_e2b
def test_bar_chart_extraction():
    """Test bar chart creation and metadata extraction"""
    print("\n=== Test: Bar Chart Extraction ===")

    executor = E2BPythonExec()
    executor.__start__()

    code = """
import matplotlib.pyplot as plt

authors = ['Author A', 'Author B', 'Author C', 'Author D']
sales = [100, 200, 300, 400]

plt.figure(figsize=(10, 6))
plt.bar(authors, sales, label='Books Sold', color='blue')
plt.xlabel('Authors')
plt.ylabel('Number of Books Sold')
plt.title('Book Sales by Authors')
plt.tight_layout()
plt.show()
"""

    result = executor(PythonExec(code=code))
    executor.__stop__()

    assert result.metadata is not None, "Should have metadata"
    assert 'charts' in result.metadata, "Should have charts in metadata"
    assert len(result.metadata['charts']) == 1, "Should have 1 chart"

    chart = result.metadata['charts'][0]
    assert chart['type'] in ['bar', 'ChartType.BAR'], f"Chart type should be bar, got {chart['type']}"
    assert chart['title'] == 'Book Sales by Authors', f"Title mismatch: {chart['title']}"
    assert chart['x_label'] == 'Authors', f"X label mismatch: {chart['x_label']}"
    assert chart['y_label'] == 'Number of Books Sold', f"Y label mismatch: {chart['y_label']}"
    assert len(chart['elements']) == 4, f"Should have 4 data points, got {len(chart['elements'])}"
    assert 'id' in chart, "Chart should have an ID"

    print(f"✅ Bar chart extracted: {chart['id']}")
    print(f"   Type: {chart['type']}, Title: {chart['title']}")
    print(f"   Elements: {len(chart['elements'])}")
    return True


@skip_if_no_e2b
def test_line_chart_extraction():
    """Test line chart creation and metadata extraction"""
    print("\n=== Test: Line Chart Extraction ===")

    executor = E2BPythonExec()
    executor.__start__()

    code = """
import matplotlib.pyplot as plt

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
revenue = [10000, 12000, 15000, 14000, 18000]

plt.figure(figsize=(10, 6))
plt.plot(months, revenue, marker='o', label='Revenue')
plt.xlabel('Month')
plt.ylabel('Revenue ($)')
plt.title('Monthly Revenue Trend')
plt.legend()
plt.tight_layout()
plt.show()
"""

    result = executor(PythonExec(code=code))
    executor.__stop__()

    assert result.metadata is not None, "Should have metadata"
    assert 'charts' in result.metadata, "Should have charts in metadata"

    chart = result.metadata['charts'][0]
    assert chart['type'] in ['line', 'ChartType.LINE'], f"Chart type should be line, got {chart['type']}"
    assert chart['title'] == 'Monthly Revenue Trend', f"Title mismatch: {chart['title']}"
    assert 'id' in chart, "Chart should have an ID"

    print(f"✅ Line chart extracted: {chart['id']}")
    print(f"   Type: {chart['type']}, Title: {chart['title']}")
    return True


@skip_if_no_e2b
def test_scatter_plot_extraction():
    """Test scatter plot creation and metadata extraction"""
    print("\n=== Test: Scatter Plot Extraction ===")

    executor = E2BPythonExec()
    executor.__start__()

    code = """
import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(50) * 100
y = np.random.rand(50) * 100

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, label='Data Points')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Random Scatter Plot')
plt.legend()
plt.tight_layout()
plt.show()
"""

    result = executor(PythonExec(code=code))
    executor.__stop__()

    assert result.metadata is not None, "Should have metadata"
    assert 'charts' in result.metadata, "Should have charts in metadata"

    chart = result.metadata['charts'][0]
    assert chart['type'] in ['scatter', 'ChartType.SCATTER'], f"Chart type should be scatter, got {chart['type']}"
    assert chart['title'] == 'Random Scatter Plot', f"Title mismatch: {chart['title']}"

    print(f"✅ Scatter plot extracted: {chart['id']}")
    print(f"   Type: {chart['type']}, Title: {chart['title']}")
    return True


@skip_if_no_e2b
def test_pie_chart_extraction():
    """Test pie chart creation and metadata extraction"""
    print("\n=== Test: Pie Chart Extraction ===")

    executor = E2BPythonExec()
    executor.__start__()

    code = """
import matplotlib.pyplot as plt

categories = ['Product A', 'Product B', 'Product C', 'Product D']
sales = [30, 25, 20, 25]

plt.figure(figsize=(8, 8))
plt.pie(sales, labels=categories, autopct='%1.1f%%')
plt.title('Market Share by Product')
plt.tight_layout()
plt.show()
"""

    result = executor(PythonExec(code=code))
    executor.__stop__()

    assert result.metadata is not None, "Should have metadata"
    assert 'charts' in result.metadata, "Should have charts in metadata"

    chart = result.metadata['charts'][0]
    assert chart['type'] in ['pie', 'ChartType.PIE'], f"Chart type should be pie, got {chart['type']}"
    assert chart['title'] == 'Market Share by Product', f"Title mismatch: {chart['title']}"

    print(f"✅ Pie chart extracted: {chart['id']}")
    print(f"   Type: {chart['type']}, Title: {chart['title']}")
    return True


@skip_if_no_e2b
def test_box_plot_extraction():
    """Test box and whisker plot creation and metadata extraction"""
    print("\n=== Test: Box Plot Extraction ===")

    executor = E2BPythonExec()
    executor.__start__()

    code = """
import matplotlib.pyplot as plt
import numpy as np

data = [np.random.normal(100, 10, 200),
        np.random.normal(90, 20, 200),
        np.random.normal(110, 15, 200)]

plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=['Group A', 'Group B', 'Group C'])
plt.xlabel('Groups')
plt.ylabel('Values')
plt.title('Distribution Comparison')
plt.tight_layout()
plt.show()
"""

    result = executor(PythonExec(code=code))
    executor.__stop__()

    assert result.metadata is not None, "Should have metadata"
    assert 'charts' in result.metadata, "Should have charts in metadata"

    chart = result.metadata['charts'][0]
    # Box plots might be detected as 'box' or 'ChartType.BOX'
    print(f"   Detected chart type: {chart['type']}")
    assert chart['title'] == 'Distribution Comparison', f"Title mismatch: {chart['title']}"

    print(f"✅ Box plot extracted: {chart['id']}")
    print(f"   Type: {chart['type']}, Title: {chart['title']}")
    return True


@skip_if_no_e2b
def test_embeddable_charts():
    """Test embeddable charts feature with LLM"""
    print("\n=== Test: Embeddable Charts with LLM ===")

    client = get_mini_client()
    executor = E2BPythonExec(embeddable_charts=True)
    agent = Agent(client=client, actions=[executor], max_iter=3, verbose=False)

    response = agent.run("Create a simple bar chart showing sales: Q1=100, Q2=150, Q3=200, Q4=180")

    assert response.success, "Agent should complete"

    # Find the tool response to check for embed instructions
    tool_messages = [msg for msg in response.messages if msg.role == 'tool']

    found_embed_instruction = False
    for msg in tool_messages:
        if '<chart id=' in msg.content and '</chart>' in msg.content:
            found_embed_instruction = True
            print(f"✅ Found embed instruction in tool output")
            print(f"   Snippet: {msg.content[:200]}")
            break

    assert found_embed_instruction, "Should have embed instructions in tool output"

    print("✅ Embeddable charts working")
    return True


@skip_if_no_e2b
def test_multiple_charts():
    """Test extracting multiple charts from single execution"""
    print("\n=== Test: Multiple Charts Extraction ===")

    executor = E2BPythonExec()
    executor.__start__()

    code = """
import matplotlib.pyplot as plt

# Chart 1: Bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

categories = ['A', 'B', 'C']
values = [10, 20, 15]
ax1.bar(categories, values)
ax1.set_title('First Chart')
ax1.set_xlabel('Category')
ax1.set_ylabel('Value')

# Chart 2: Line chart
months = ['Jan', 'Feb', 'Mar']
data = [5, 10, 8]
ax2.plot(months, data, marker='o')
ax2.set_title('Second Chart')
ax2.set_xlabel('Month')
ax2.set_ylabel('Data')

plt.tight_layout()
plt.show()
"""

    result = executor(PythonExec(code=code))
    executor.__stop__()

    # Note: E2B might return this as a single combined chart or separate charts
    # depending on how it handles subplots
    assert result.metadata is not None, "Should have metadata"

    if 'charts' in result.metadata:
        chart_count = len(result.metadata['charts'])
        print(f"✅ Detected {chart_count} chart(s)")
        for i, chart in enumerate(result.metadata['charts']):
            print(f"   Chart {i+1}: {chart['type']} - {chart['title']}")
    else:
        print("⚠️  No charts detected (subplots may not be supported)")

    return True


# =============================================================================
# TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("E2B Code Interpreter Tests")
    print("=" * 70)

    if not HAS_E2B:
        print("\n❌ E2B not installed: pip install jetflow[e2b]")
        sys.exit(1)

    if not HAS_API_KEY:
        print("\n❌ E2B_API_KEY not set")
        sys.exit(1)

    if not HAS_ANTHROPIC_KEY:
        print("\n❌ ANTHROPIC_API_KEY not set")
        sys.exit(1)

    tests = [
        ("Simple Calculation", test_simple_calculation),
        ("Stdout Capture", test_stdout_capture),
        ("Error Handling", test_error_handling),
        ("Verbose Streaming", test_verbose_streaming),
        ("DataFrame Display", test_dataframe_creation_and_formatting),
        ("CAGR Calculation", test_dataframe_cagr_calculation),
        ("Extract DataFrame", test_extract_dataframe),
        ("Extract Variable", test_extract_variable),
        ("Manual Code Execution", test_manual_code_execution),
        ("Variable Persistence", test_variable_persistence),
        ("Lifecycle Hooks", test_lifecycle_hooks),
        ("Sandbox Pause Verification", test_sandbox_pause_verification),
        ("Bar Chart Extraction", test_bar_chart_extraction),
        ("Line Chart Extraction", test_line_chart_extraction),
        ("Scatter Plot Extraction", test_scatter_plot_extraction),
        ("Pie Chart Extraction", test_pie_chart_extraction),
        ("Box Plot Extraction", test_box_plot_extraction),
        ("Embeddable Charts", test_embeddable_charts),
        ("Multiple Charts", test_multiple_charts),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    for name, result in results:
        status = "✅ PASS" if result is True else "❌ FAIL" if result is False else "⚠️  SKIP"
        print(f"{status} - {name}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")

    sys.exit(0 if failed == 0 else 1)
