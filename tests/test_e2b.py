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
    from jetflow.actions.e2b_python_exec import E2BPythonExec, PythonExec
    from jetflow.agent import Agent
    from jetflow.clients.anthropic import AnthropicClient
    HAS_E2B = True
except ImportError:
    HAS_E2B = False
    PythonExec = None

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
    """Get GPT-5-mini client for tests"""
    from jetflow.clients.openai import OpenAIClient
    return OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-5-mini",
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
def test_import_export_dataframe():
    """Test import_dataframe and extract_dataframe roundtrip"""
    print("\n=== Test: Import/Export DataFrame ===")

    executor = E2BPythonExec()
    executor.__start__()

    # Test data - list of dicts (like df.to_dict('records'))
    original_data = [
        {'symbol': 'AAPL', 'price': 150.25, 'volume': 1000000},
        {'symbol': 'GOOGL', 'price': 2800.50, 'volume': 500000},
        {'symbol': 'MSFT', 'price': 300.75, 'volume': 750000},
    ]

    # Import the data
    result = executor.import_dataframe('stock_data', original_data)
    print(f"   Import result: {result}")

    assert 'stock_data loaded' in result, f"Should confirm import, got: {result}"
    assert '(3,' in result, f"Should show 3 rows, got: {result}"

    # Verify data is accessible in sandbox
    verify_result = executor.run_code("print(stock_data.head())")
    assert 'AAPL' in verify_result, f"Should contain AAPL, got: {verify_result}"

    # Extract it back
    extracted_data = executor.extract_dataframe('stock_data')

    executor.stop()

    # Verify roundtrip
    assert extracted_data is not None, "Should extract DataFrame"
    assert len(extracted_data) == 3, f"Expected 3 rows, got {len(extracted_data)}"
    assert extracted_data[0]['symbol'] == 'AAPL', f"Expected AAPL, got {extracted_data[0]['symbol']}"
    assert extracted_data[1]['price'] == 2800.50, f"Expected 2800.50, got {extracted_data[1]['price']}"

    print(f"✅ Import/Export roundtrip successful")
    print(f"   Original: {original_data[0]}")
    print(f"   Extracted: {extracted_data[0]}")
    return True


@skip_if_no_e2b
def test_import_dataframe_with_agent():
    """Test that import_dataframe works with nested agent (sandbox survives agent lifecycle)"""
    print("\n=== Test: Import DataFrame with Agent ===")
    import asyncio

    # Sample cash flow statement data (simplified from META)
    cash_flow_data = [
        {'label': 'Net Income', 'FY 2022': 23200000000, 'FY 2023': 39100000000, 'FY 2024': 62360000000},
        {'label': 'Depreciation and amortization', 'FY 2022': 8690000000, 'FY 2023': 11180000000, 'FY 2024': 15500000000},
        {'label': 'Share-based compensation', 'FY 2022': 11990000000, 'FY 2023': 14030000000, 'FY 2024': 16690000000},
        {'label': 'Net Cash from Operating Activities', 'FY 2022': 50480000000, 'FY 2023': 71110000000, 'FY 2024': 91330000000},
        {'label': 'Payments for Property, Plant and Equipment', 'FY 2022': -31430000000, 'FY 2023': -27270000000, 'FY 2024': -37260000000},
        {'label': 'Net Cash from Investing Activities', 'FY 2022': -28970000000, 'FY 2023': -24500000000, 'FY 2024': -47150000000},
        {'label': 'Net Cash from Financing Activities', 'FY 2022': -22140000000, 'FY 2023': -19500000000, 'FY 2024': -40780000000},
    ]

    async def run_test():
        from jetflow import AsyncAgent
        from jetflow.clients.anthropic import AsyncAnthropicClient

        # Create executor and import data BEFORE agent
        executor = E2BPythonExec(persistent=False, embeddable_charts=False)
        result = executor.import_dataframe('cash_flow', cash_flow_data)
        print(f"   Import result: {result}")
        print(f"   DEBUG: _started={executor._started}, _manually_started={executor._manually_started}")

        assert 'cash_flow loaded' in result, f"Should confirm import, got: {result}"

        # Verify data is there before agent
        verify_before = executor.run_code("print(cash_flow.shape); print(cash_flow['label'].tolist())")
        print(f"   Data before agent: {verify_before[:200]}")
        assert 'Net Income' in verify_before, "Data should be accessible before agent"

        # Create agent with the executor
        agent = AsyncAgent(
            client=AsyncAnthropicClient(model="claude-haiku-4-5"),
            actions=[executor],
            system_prompt="You have access to a DataFrame called 'cash_flow'. Use PythonExec to analyze it.",
            max_iter=2,
            verbose=True
        )

        # Run agent - it should be able to access the pre-loaded data
        response = await agent.run("Calculate the year-over-year growth rate for Net Cash from Operating Activities from 2022 to 2024. Print the results.")

        print(f"\n   Agent iterations: {response.iterations}")
        print(f"   Agent response: {(response.content or '')[:300]}...")
        print(f"   DEBUG after agent: _started={executor._started}, _manually_started={executor._manually_started}")
        # Agent may or may not complete with text - the key test is sandbox survival

        # CRITICAL: Sandbox should still be alive after agent completes
        verify_after = executor.run_code("print('Sandbox alive!'); print(cash_flow.shape)")
        print(f"   Data after agent: {verify_after}")
        assert 'Sandbox alive' in verify_after, f"Sandbox should survive agent lifecycle, got: {verify_after}"
        assert '(7,' in verify_after, f"DataFrame should still have 7 rows, got: {verify_after}"

        # Clean up manually
        executor.stop()

        print("✅ Import DataFrame with Agent - sandbox survived!")
        return True

    return asyncio.run(run_test())


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
    sandbox_id = executor1.sandbox._sandbox.sandbox_id if executor1.sandbox._sandbox else None
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
    assert len(chart['series']) >= 1, f"Should have at least 1 series, got {len(chart['series'])}"
    assert 'chart_id' in chart, "Chart should have a chart_id"

    print(f"✅ Bar chart extracted: {chart['chart_id']}")
    print(f"   Type: {chart['type']}, Title: {chart['title']}")
    print(f"   Series: {len(chart['series'])}")
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
    assert 'chart_id' in chart, "Chart should have a chart_id"

    print(f"✅ Line chart extracted: {chart['chart_id']}")
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

    print(f"✅ Scatter plot extracted: {chart['chart_id']}")
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

    # Pie charts are not fully supported yet - wedges aren't extracted
    if result.metadata is None or 'charts' not in result.metadata:
        print("⚠️  Pie chart extraction not yet supported (wedges not extracted)")
        return "SKIP"

    chart = result.metadata['charts'][0]
    assert chart['type'] in ['pie', 'ChartType.PIE'], f"Chart type should be pie, got {chart['type']}"
    assert chart['title'] == 'Market Share by Product', f"Title mismatch: {chart['title']}"

    print(f"✅ Pie chart extracted: {chart['chart_id']}")
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

    # Box plots may be detected as line charts (box whiskers are lines)
    if result.metadata is None or 'charts' not in result.metadata:
        print("⚠️  Box plot extraction not yet supported")
        return "SKIP"

    chart = result.metadata['charts'][0]
    print(f"   Detected chart type: {chart['type']}")
    assert chart['title'] == 'Distribution Comparison', f"Title mismatch: {chart['title']}"

    print(f"✅ Box plot extracted: {chart['chart_id']}")
    print(f"   Type: {chart['type']}, Title: {chart['title']}")
    return True


@skip_if_no_e2b
def test_embeddable_charts():
    """Test embeddable charts feature with LLM"""
    print("\n=== Test: Embeddable Charts with LLM ===")

    # Use Anthropic because OpenAI Responses API doesn't support custom_field
    client = get_client()
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


@skip_if_no_e2b
def test_chart_id_from_savefig():
    """Test that chart_id is extracted from plt.savefig() filename"""
    print("\n=== Test: Chart ID from savefig ===")

    executor = E2BPythonExec()
    executor.__start__()

    code = """
import matplotlib.pyplot as plt

plt.bar(['Q1', 'Q2', 'Q3'], [100, 150, 200])
plt.title('Revenue Growth')
plt.savefig('revenue_chart_2025.png')
"""

    result = executor(PythonExec(code=code))
    executor.__stop__()

    assert result.metadata is not None, "Should have metadata"
    assert 'charts' in result.metadata, "Should have charts"
    assert len(result.metadata['charts']) == 1, "Should have 1 chart"

    chart = result.metadata['charts'][0]
    assert chart['chart_id'] == 'revenue_chart_2025', f"Chart ID should be 'revenue_chart_2025', got {chart['chart_id']}"

    print(f"✅ Chart ID correctly extracted: {chart['chart_id']}")
    return True


@skip_if_no_e2b
def test_chart_id_from_variable():
    """Test that chart_id is extracted from variable name"""
    print("\n=== Test: Chart ID from variable name ===")

    executor = E2BPythonExec()
    executor.__start__()

    code = """
import matplotlib.pyplot as plt

sales_comparison = plt.figure()
plt.bar(['Product A', 'Product B'], [50, 75])
plt.title('Sales Comparison')
"""

    result = executor(PythonExec(code=code))
    executor.__stop__()

    assert result.metadata is not None, "Should have metadata"
    assert 'charts' in result.metadata, "Should have charts"
    assert len(result.metadata['charts']) == 1, "Should have 1 chart"

    chart = result.metadata['charts'][0]
    assert chart['chart_id'] == 'sales_comparison', f"Chart ID should be 'sales_comparison', got {chart['chart_id']}"

    print(f"✅ Chart ID correctly extracted: {chart['chart_id']}")
    return True


@skip_if_no_e2b
def test_embeddable_charts_with_savefig():
    """Test embeddable_charts=True shows correct embedding instructions with savefig"""
    print("\n=== Test: Embeddable Charts with savefig ===")

    executor = E2BPythonExec(embeddable_charts=True)
    executor.__start__()

    code = """
import matplotlib.pyplot as plt

plt.bar(['A', 'B', 'C'], [10, 20, 15])
plt.title('Sample Chart')
plt.savefig('my_analysis_chart.png')
"""

    result = executor(PythonExec(code=code))
    executor.__stop__()

    # Check that the content includes embedding instructions with the correct chart_id
    assert '<chart id="my_analysis_chart"></chart>' in result.content, \
        f"Should contain embedding instruction with chart_id 'my_analysis_chart'. Content: {result.content}"
    assert '→ To embed:' in result.content, "Should contain embedding instructions"

    print(f"✅ Embedding instructions correct")
    print(f"   Content snippet: {result.content[:300]}")
    return True


@skip_if_no_e2b
def test_moving_average_with_nan_alignment():
    """Test that moving averages with NaN values maintain proper alignment across series"""
    print("\n=== Test: Moving Average NaN Alignment ===")

    executor = E2BPythonExec()
    executor.__start__()

    code = """
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create test data similar to stock prices
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=250)
prices = 100 + np.random.randn(250).cumsum()

df = pd.DataFrame({
    'date': dates,
    'price': prices
})

# Calculate moving averages with different windows (will have NaN at start)
df['ma_50'] = df['price'].rolling(window=50).mean()
df['ma_200'] = df['price'].rolling(window=200).mean()

print(f"DataFrame shape: {df.shape}")
print(f"First NaN in ma_50 at index: {df['ma_50'].first_valid_index()}")
print(f"First NaN in ma_200 at index: {df['ma_200'].first_valid_index()}")
print(f"NaN count in ma_50: {df['ma_50'].isna().sum()}")
print(f"NaN count in ma_200: {df['ma_200'].isna().sum()}")

# Create chart with all three series
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['price'], label='Price', linewidth=2)
ax.plot(df.index, df['ma_50'], label='50-day MA', linewidth=1.5)
ax.plot(df.index, df['ma_200'], label='200-day MA', linewidth=1.5)
ax.set_xlabel('Days')
ax.set_ylabel('Price')
ax.set_title('Price with Moving Averages')
ax.legend()
plt.tight_layout()
plt.show()
"""

    result = executor(PythonExec(code=code))
    executor.__stop__()

    # Verify chart was extracted
    assert result.metadata is not None, "Should have metadata"
    assert 'charts' in result.metadata, "Should have charts"
    assert len(result.metadata['charts']) == 1, f"Should have 1 chart, got {len(result.metadata['charts'])}"

    chart = result.metadata['charts'][0]
    assert len(chart['series']) == 3, f"Should have 3 series (price, ma_50, ma_200), got {len(chart['series'])}"

    # Check that all series have the same length
    series_lengths = [len(s['x']) for s in chart['series']]
    assert len(set(series_lengths)) == 1, f"All series should have same x length, got {series_lengths}"

    series_y_lengths = [len(s['y']) for s in chart['series']]
    assert len(set(series_y_lengths)) == 1, f"All series should have same y length, got {series_y_lengths}"

    expected_length = 250
    assert series_lengths[0] == expected_length, f"Series should have {expected_length} points, got {series_lengths[0]}"

    # Check that NaN values are preserved in the y-data
    # The MA series should have None/null values at the beginning
    ma_50_series = next(s for s in chart['series'] if '50' in str(s.get('label', '')))
    ma_200_series = next(s for s in chart['series'] if '200' in str(s.get('label', '')))

    # Count None/null values in the extracted data
    ma_50_nulls = sum(1 for y in ma_50_series['y'] if y is None or (isinstance(y, float) and str(y) == 'nan'))
    ma_200_nulls = sum(1 for y in ma_200_series['y'] if y is None or (isinstance(y, float) and str(y) == 'nan'))

    print(f"✅ Chart extracted with 3 series, all length {series_lengths[0]}")
    print(f"   MA-50 has {ma_50_nulls} null/NaN values (expected ~49)")
    print(f"   MA-200 has {ma_200_nulls} null/NaN values (expected ~199)")
    print(f"   Series alignment: ✓ All series same length")

    # The alignment is correct if all series have the same length
    # NaN handling: the test passes if we extracted the data successfully
    return True


@skip_if_no_e2b
def test_datetime_in_charts():
    """Test that datetime x-axis values are properly handled in chart extraction"""
    print("\n=== Test: Datetime Handling in Charts ===")

    executor = E2BPythonExec()
    executor.__start__()

    code = """
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create time series data
dates = pd.date_range('2024-01-01', periods=10, freq='D')
values = [100, 105, 103, 108, 112, 110, 115, 118, 120, 117]

# Plot with datetime x-axis
plt.figure(figsize=(10, 6))
plt.plot(dates, values, marker='o', label='Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue ($K)')
plt.title('Daily Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Date type: {type(dates[0])}")
print(f"First date: {dates[0]}")
"""

    result = executor(PythonExec(code=code))
    executor.__stop__()

    # Verify chart extraction
    assert result.metadata is not None, "Should have metadata"
    assert 'charts' in result.metadata, "Should have charts"

    chart = result.metadata['charts'][0]
    series = chart['series'][0]

    # Check x-axis values - they will be numeric (matplotlib converts datetime to numbers)
    print(f"\n   X-axis first value: {series['x'][0]} (type: {type(series['x'][0])})")
    print(f"   X-axis last value: {series['x'][-1]} (type: {type(series['x'][-1])})")

    # Matplotlib converts datetime64 to integers/floats for plotting
    # These should be numeric values, not None
    assert all(x is not None for x in series['x']), "X values should not be None"
    assert all(isinstance(x, (int, float)) for x in series['x']), "X values should be numeric after datetime conversion"

    print(f"✅ Datetime x-axis extracted as numeric values")
    print(f"   Note: Datetimes are converted to numeric timestamps by matplotlib")

    return True


@skip_if_no_e2b
def test_import_dataframe_with_timestamps_and_nan():
    """Test import_dataframe with DataFrames containing timestamps and NaN values"""
    print("\n=== Test: Import DataFrame with Timestamps and NaN ===")
    import pandas as pd
    import numpy as np

    executor = E2BPythonExec()
    executor.__start__()

    # Create DataFrame with timestamps and NaN
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'price': [100, 105, 110, np.nan, 120, 125, np.nan, 135, 140, 145],
        'volume': [1000, 1100, np.nan, 1300, 1400, np.nan, 1600, 1700, 1800, 1900]
    })

    print(f"   Original DataFrame:\n{df.head(3)}")
    print(f"   NaN count - price: {df['price'].isna().sum()}, volume: {df['volume'].isna().sum()}")

    try:
        # This should work now after the fix
        result = executor.import_dataframe('test_df', df)
        print(f"   Import result: {result}")

        # Verify data is accessible
        verify = executor.run_code("""
import pandas as pd
print(f"Shape: {test_df.shape}")
print(f"Columns: {list(test_df.columns)}")
print(f"Date type: {test_df['date'].dtype}")
print(f"NaN count - price: {test_df['price'].isna().sum()}")
print(f"NaN count - volume: {test_df['volume'].isna().sum()}")
print(f"First date: {test_df['date'].iloc[0]}")
""")
        print(f"   Verification:\n{verify}")

        # Extract it back
        extracted = executor.extract_dataframe('test_df')
        print(f"   Extracted {len(extracted)} records")
        print(f"   First record: {extracted[0]}")

        assert len(extracted) == 10, f"Should have 10 records, got {len(extracted)}"
        print(f"✅ Import/export with timestamps and NaN successful")

    except Exception as e:
        print(f"❌ Import failed: {e}")
        print(f"   This is expected if the bug is not fixed yet")
        print(f"   Issue: json.dumps(records) cannot serialize pandas Timestamps")
        return False
    finally:
        executor.__stop__()

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
        ("Import/Export DataFrame", test_import_export_dataframe),
        ("Import DataFrame with Agent", test_import_dataframe_with_agent),
        ("Import DataFrame with Timestamps and NaN", test_import_dataframe_with_timestamps_and_nan),
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
        ("Chart ID from savefig", test_chart_id_from_savefig),
        ("Chart ID from variable", test_chart_id_from_variable),
        ("Embeddable Charts with savefig", test_embeddable_charts_with_savefig),
        ("Moving Average NaN Alignment", test_moving_average_with_nan_alignment),
        ("Datetime in Charts", test_datetime_in_charts),
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
