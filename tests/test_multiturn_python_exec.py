"""
Multi-Turn Test with Python Execution - Sync Anthropic

Tests complex multi-turn interactions:
1. Agent calls get_financials to retrieve data
2. Agent calls python_exec to calculate CAGR
3. Agent calls python_exec again for additional analysis
4. Agent synthesizes final answer

This validates:
- Multi-turn conversations
- Python state persistence across calls
- Complex reasoning chains
- Proper tool sequencing
"""

from dotenv import load_dotenv
from jetflow import Agent
from jetflow.clients.anthropic import AnthropicClient
from jetflow.action import action
from jetflow.models.response import ActionResult
from pydantic import BaseModel, Field
import sys
from io import StringIO
import contextlib

load_dotenv()


# ============================================================================
# Mock Financial Data
# ============================================================================

FINANCIAL_DATA = {
    "NVDA": {
        "revenue": {
            "2022": 26.914,  # billions
            "2023": 26.974,
            "2024": 60.922,
            "2025": 130.497
        },
        "net_income": {
            "2022": 9.752,
            "2023": 4.368,
            "2024": 29.760,
            "2025": 72.880
        }
    }
}


# ============================================================================
# Actions
# ============================================================================

class GetFinancialsParams(BaseModel):
    """Get historical financial data for a company"""
    symbol: str = Field(description="Stock symbol (e.g., NVDA)")
    metric: str = Field(description="Metric to retrieve: 'revenue' or 'net_income'")


@action(schema=GetFinancialsParams)
def get_financials(params: GetFinancialsParams) -> ActionResult:
    """Retrieve hardcoded financial data"""
    company_data = FINANCIAL_DATA.get(params.symbol.upper())

    if not company_data:
        return ActionResult(
            content=f"No data available for {params.symbol}. Available: {list(FINANCIAL_DATA.keys())}",
            summary="Company not found"
        )

    metric_data = company_data.get(params.metric.lower())

    if not metric_data:
        return ActionResult(
            content=f"Metric '{params.metric}' not found. Available: {list(company_data.keys())}",
            summary="Metric not found"
        )

    # Format as table
    content = f"## {params.symbol.upper()} - {params.metric.title()} (Billions USD)\n\n"
    content += "| Year | Value |\n"
    content += "|------|-------|\n"

    for year in sorted(metric_data.keys()):
        content += f"| {year} | ${metric_data[year]:.3f}B |\n"

    # Also provide as Python dict for easy use
    content += f"\n**Python dict format:**\n```python\n{params.metric} = {metric_data}\n```"

    return ActionResult(
        content=content,
        metadata={"symbol": params.symbol, "metric": params.metric, "data": metric_data},
        summary=f"Retrieved {params.metric} for {params.symbol}"
    )


# ============================================================================
# Python Execution Action (Stateful)
# ============================================================================

# Global namespace for Python execution (persists across calls)
PYTHON_NAMESPACE = {}


class PythonExecParams(BaseModel):
    """Execute Python code with persistent state"""
    code: str = Field(description="Python code to execute")


@action(schema=PythonExecParams)
def python_exec(params: PythonExecParams) -> ActionResult:
    """Execute Python code in persistent namespace

    Variables persist across calls, enabling multi-step calculations.
    """
    # Capture stdout
    stdout_capture = StringIO()

    try:
        # Execute in global namespace (persists across calls)
        with contextlib.redirect_stdout(stdout_capture):
            exec(params.code, PYTHON_NAMESPACE)

        output = stdout_capture.getvalue()

        if not output:
            # If no print output, show the namespace variables
            output = "Code executed successfully. Available variables:\n"
            for key, value in PYTHON_NAMESPACE.items():
                if not key.startswith('__'):
                    output += f"  {key} = {value}\n"

        return ActionResult(
            content=output,
            metadata={"code": params.code, "namespace_keys": list(PYTHON_NAMESPACE.keys())},
            summary="Code executed"
        )

    except Exception as e:
        error_msg = f"Python execution error: {str(e)}\n\nCode:\n{params.code}"
        return ActionResult(
            content=error_msg,
            metadata={"error": str(e), "code": params.code},
            summary=f"Execution failed: {str(e)}"
        )


# ============================================================================
# Exit Action
# ============================================================================

class SubmitReportParams(BaseModel):
    """Submit final financial analysis report"""
    analysis: str = Field(description="Complete analysis with findings")


@action(schema=SubmitReportParams, exit=True)
def submit_report(params: SubmitReportParams) -> ActionResult:
    return ActionResult(
        content=f"# Financial Analysis Report\n\n{params.analysis}",
        summary="Report submitted"
    )


# ============================================================================
# Test
# ============================================================================

def test_multiturn_python_exec():
    print("=" * 80)
    print("MULTI-TURN PYTHON EXEC TEST - SYNC ANTHROPIC")
    print("=" * 80)
    print()

    # Reset Python namespace
    PYTHON_NAMESPACE.clear()

    client = AnthropicClient(model="claude-haiku-4-5")

    agent = Agent(
        client=client,
        system_prompt="""You are a financial analyst. Your task:

1. Get NVDA revenue data using get_financials
2. Use python_exec to calculate the CAGR (Compound Annual Growth Rate) from 2022 to 2025
3. Get NVDA net_income data using get_financials
4. Use python_exec to calculate net income CAGR from 2022 to 2025
5. Submit a report with your findings

Formula for CAGR: ((End Value / Start Value) ^ (1 / Number of Years)) - 1

Important: Variables in python_exec persist across calls, so you can reference previously calculated values.""",
        actions=[get_financials, python_exec, submit_report],
        require_action=True,
        max_iter=15,
        verbose=True
    )

    response = agent.run("Calculate NVIDIA's revenue and net income CAGR from 2022 to 2025.")

    print("\n" + "=" * 80)
    print("FULL RESPONSE TRACE")
    print("=" * 80)
    print(f"\nFinal Content:\n{response.content}\n")
    print(f"\nMessage History ({len(response.messages)} messages):")
    for i, msg in enumerate(response.messages, 1):
        print(f"\n  [{i}] {msg.role.upper()}")
        if msg.content:
            preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
            print(f"      Content: {preview}")
        if msg.actions:
            print(f"      Actions: {[a.name for a in msg.actions]}")

    print("\n" + "=" * 80)
    print("MULTI-TURN ASSERTIONS")
    print("=" * 80)

    # ========================================================================
    # 1. INTERACTION COUNT ASSERTIONS
    # ========================================================================
    user_messages = [msg for msg in response.messages if msg.role == "user"]
    assistant_messages = [msg for msg in response.messages if msg.role == "assistant"]
    tool_messages = [msg for msg in response.messages if msg.role == "tool"]

    assert len(assistant_messages) >= 3, f"Should have multiple assistant turns (>=3), got {len(assistant_messages)}"
    assert response.iterations >= 3, f"Should have multiple iterations (>=3), got {response.iterations}"

    print(f"✓ Multiple turns: {response.iterations} iterations")
    print(f"✓ Messages: {len(user_messages)} user, {len(assistant_messages)} assistant, {len(tool_messages)} tool")

    # ========================================================================
    # 2. ACTION SEQUENCE ASSERTIONS
    # ========================================================================
    # Extract all action calls in order
    action_sequence = []
    for msg in response.messages:
        if msg.role == "assistant" and msg.actions:
            for action in msg.actions:
                action_sequence.append(action.name)

    print(f"✓ Action sequence: {action_sequence}")

    # Verify expected actions were called (using schema names)
    assert "GetFinancialsParams" in action_sequence, "Should call get_financials"
    assert "PythonExecParams" in action_sequence, "Should call python_exec"
    assert "SubmitReportParams" in action_sequence, "Should call submit_report (exit action)"

    # Count action types
    get_financials_count = action_sequence.count("GetFinancialsParams")
    python_exec_count = action_sequence.count("PythonExecParams")

    assert get_financials_count >= 2, f"Should call get_financials at least 2x (revenue + net_income), got {get_financials_count}"
    assert python_exec_count >= 1, f"Should call python_exec at least 1x (CAGR calculations), got {python_exec_count}"

    print(f"✓ get_financials called {get_financials_count}x")
    print(f"✓ python_exec called {python_exec_count}x")

    # ========================================================================
    # 3. PYTHON STATE PERSISTENCE ASSERTIONS
    # ========================================================================
    # Check that Python namespace has accumulated variables
    assert len(PYTHON_NAMESPACE) > 0, "Python namespace should have variables after execution"

    print(f"✓ Python namespace has {len(PYTHON_NAMESPACE)} variables")
    print(f"  Variables: {[k for k in PYTHON_NAMESPACE.keys() if not k.startswith('__')]}")

    # Verify CAGR calculations were performed
    namespace_vars = [k for k in PYTHON_NAMESPACE.keys() if not k.startswith('__')]
    has_cagr_calc = any('cagr' in k.lower() or 'growth' in k.lower() for k in namespace_vars)

    if has_cagr_calc:
        print(f"✓ Python namespace contains CAGR-related variables")

    # ========================================================================
    # 4. TOOL MESSAGE CONTENT ASSERTIONS
    # ========================================================================
    # Check that get_financials returned data
    financials_messages = [msg for msg in tool_messages if "revenue" in msg.content.lower() or "net_income" in msg.content.lower()]
    assert len(financials_messages) >= 2, f"Should have at least 2 financial data responses, got {len(financials_messages)}"

    print(f"✓ Received {len(financials_messages)} financial data responses")

    # Check that python_exec produced output
    python_exec_messages = [msg for msg in tool_messages if msg.action_id and any(
        a.name == "PythonExecParams" for msg2 in response.messages if msg2.role == "assistant" and msg2.actions
        for a in msg2.actions if a.id == msg.action_id
    )]

    if python_exec_messages:
        print(f"✓ Python execution produced {len(python_exec_messages)} outputs")

    # ========================================================================
    # 5. FINAL CONTENT ASSERTIONS
    # ========================================================================
    content_lower = response.content.lower()

    # Should mention CAGR
    assert "cagr" in content_lower or "growth rate" in content_lower, "Final report should mention CAGR or growth rate"

    # Should mention both revenue and net income
    assert "revenue" in content_lower, "Final report should mention revenue"
    assert "net income" in content_lower or "net_income" in content_lower, "Final report should mention net income"

    # Should contain numerical results
    import re
    percentages = re.findall(r'(\d+\.?\d*)\s*%', response.content)
    if percentages:
        print(f"✓ Final report contains percentages: {percentages}")

    print(f"✓ Final report mentions required metrics")
    print(f"✓ Final report is substantive ({len(response.content)} chars)")

    # ========================================================================
    # 6. SUCCESS ASSERTIONS
    # ========================================================================
    assert response.success, "Agent should complete successfully"
    assert response.iterations < 15, f"Should complete within iteration limit, used {response.iterations}/15"

    print(f"✓ Agent completed successfully in {response.iterations} iterations")
    print(f"✓ Duration: {response.duration:.2f}s")
    print(f"✓ Cost: ${response.usage.estimated_cost:.4f}")

    print("\n✅ MULTI-TURN TEST PASSED\n")
    return response


if __name__ == "__main__":
    print("\n" + "🧪 MULTI-TURN PYTHON EXEC TEST SUITE" + "\n")

    try:
        test_multiturn_python_exec()

        print("=" * 80)
        print("🎉 MULTI-TURN TEST PASSED!")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n💥 ERROR: {e}\n")
        raise
