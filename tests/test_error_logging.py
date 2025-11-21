"""Test that action errors are properly logged"""

from jetflow import Agent, action
from jetflow.clients.legacy_openai import LegacyOpenAIClient
from pydantic import BaseModel


class DivideSchema(BaseModel):
    """Divide two numbers"""
    a: float
    b: float


@action(DivideSchema)
def divide(params: DivideSchema) -> str:
    """Action that will fail when dividing by zero"""
    result = params.a / params.b
    return f"{params.a} / {params.b} = {result}"


def test_error_logging():
    """Test that validation and execution errors are logged"""

    agent = Agent(
        client=LegacyOpenAIClient(model="gpt-5-mini"),
        actions=[divide],
        verbose=True
    )

    print("\n" + "="*80)
    print("TEST: Error logging for division by zero")
    print("="*80 + "\n")

    # This should cause a division by zero error and log it
    response = agent.run("What is 10 divided by 0?")

    # Debug: print all messages
    print(f"\nTotal messages: {len(response.messages)}")
    for i, msg in enumerate(response.messages):
        print(f"  Message {i}: role={msg.role}, error={getattr(msg, 'error', None)}, content={msg.content[:100] if msg.content else None}")

    # Check that response contains error in history
    has_error = any(getattr(msg, 'error', False) for msg in response.messages)
    print(f"\n✓ Error logged: {has_error}")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_error_logging()
