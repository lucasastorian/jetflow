"""
Test Runner - Runs all client tests

Tests core functionality across all client types:
1. Sync Anthropic (claude-haiku-4-5) - streaming & non-streaming
2. Async Anthropic (claude-haiku-4-5) - streaming & non-streaming
3. Sync OpenAI (gpt-5-mini) - streaming & non-streaming
4. Async OpenAI (gpt-5-mini) - streaming & non-streaming
5. Sync Legacy OpenAI (gpt-4o-mini) - streaming & non-streaming
6. Async Legacy OpenAI (gpt-4o-mini) - streaming & non-streaming
7. Action Types - validates sync/async function/class actions
8. Citations - validates citation system
9. Multi-turn Python Exec - validates stateful multi-turn interactions
"""

import subprocess
import sys
import time

TESTS = [
    ("All Clients (unified)", "tests/test_clients.py"),
    ("Action Types", "tests/test_action_types.py"),
    ("Citations", "tests/test_citations.py"),
    ("Multi-turn Python Exec", "tests/test_multiturn_python_exec.py"),
]

def run_test(name, script):
    """Run a single test script"""
    print("\n" + "=" * 80)
    print(f"RUNNING: {name}")
    print("=" * 80)

    start = time.time()
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    duration = time.time() - start

    if result.returncode == 0:
        print(f"✅ {name} PASSED ({duration:.2f}s)")
        return True
    else:
        print(f"❌ {name} FAILED ({duration:.2f}s)")
        print("\nSTDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        return False


if __name__ == "__main__":
    print("\n" + "🧪 RUNNING ALL JETFLOW TESTS" + "\n")

    results = {}
    total_start = time.time()

    for name, script in TESTS:
        results[name] = run_test(name, script)

    total_duration = time.time() - total_start

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")

    print("\n" + "=" * 80)
    print(f"Total: {passed} passed, {failed} failed ({total_duration:.2f}s)")
    print("=" * 80)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 ALL TESTS PASSED!\n")
        sys.exit(0)
