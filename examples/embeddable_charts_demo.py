"""
Embeddable Charts Demo

Demonstrates the embeddable_charts feature that adds XML embed instructions
to tool output, allowing LLMs to reference charts in their responses.
"""

from jetflow.actions.e2b_python_exec import E2BPythonExec, PythonExec

print("=" * 70)
print("Embeddable Charts Demo")
print("=" * 70)

# Example 1: Default behavior (embeddable_charts=False)
print("\n=== Example 1: Default (No Embed Instructions) ===")

executor1 = E2BPythonExec()
executor1.__start__()

result1 = executor1(PythonExec(code="""
import matplotlib.pyplot as plt

plt.bar(['Q1', 'Q2', 'Q3', 'Q4'], [100, 120, 150, 140])
plt.xlabel('Quarter')
plt.ylabel('Revenue ($M)')
plt.title('Quarterly Revenue')
plt.show()
"""))

print("\nTool Output:")
print(result1.content)

print("\nMetadata:")
print(f"  Charts: {len(result1.metadata['charts'])}")
print(f"  Chart ID: {result1.metadata['charts'][0]['id']}")

executor1.__stop__()

# Example 2: With embeddable_charts=True
print("\n\n=== Example 2: Embeddable Charts (With XML Instructions) ===")

executor2 = E2BPythonExec(embeddable_charts=True)
executor2.__start__()

result2 = executor2(PythonExec(code="""
import matplotlib.pyplot as plt

categories = ['Product A', 'Product B', 'Product C']
sales = [45, 30, 25]

plt.pie(sales, labels=categories, autopct='%1.1f%%')
plt.title('Market Share by Product')
plt.show()
"""))

print("\nTool Output:")
print(result2.content)

print("\nMetadata:")
print(f"  Charts: {len(result2.metadata['charts'])}")
print(f"  Chart ID: {result2.metadata['charts'][0]['id']}")

executor2.__stop__()

# Example 3: Multiple charts
print("\n\n=== Example 3: Multiple Embeddable Charts ===")

executor3 = E2BPythonExec(embeddable_charts=True)
executor3.__start__()

result3 = executor3(PythonExec(code="""
import matplotlib.pyplot as plt

# Chart 1: Revenue
plt.figure(figsize=(8, 5))
plt.bar(['Jan', 'Feb', 'Mar'], [100, 120, 115])
plt.title('Monthly Revenue')
plt.ylabel('Revenue ($K)')
plt.show()

# Chart 2: Growth
plt.figure(figsize=(8, 5))
plt.plot(['Jan', 'Feb', 'Mar'], [5.2, 4.8, 6.1], marker='o', color='green')
plt.title('Growth Rate')
plt.ylabel('Growth (%)')
plt.show()
"""))

print("\nTool Output:")
print(result3.content)

print("\nMetadata:")
for i, chart in enumerate(result3.metadata['charts'], 1):
    print(f"  Chart {i}: {chart['id']} ({chart['type']}) - {chart['title']}")

executor3.__stop__()

# Example 4: Usage with LLM
print("\n\n=== Example 4: LLM Usage Pattern ===")

print("""
When embeddable_charts=True, the LLM sees:

Tool Output:
  "Created bar chart: **Quarterly Revenue**
   Embed with: `<chart id="chart-quarterly-revenue-abc123"></chart>`"

The LLM can then respond:

  "Here's the quarterly revenue analysis:

   <chart id="chart-quarterly-revenue-abc123"></chart>

   As you can see, Q3 had the highest revenue at $150M..."

Your frontend can then:
1. Parse the <chart id="..."></chart> tags
2. Look up the chart data in metadata by ID
3. Render interactive charts in place of the XML tags
""")

print("\n" + "=" * 70)
print("Key Points")
print("=" * 70)
print("""
✅ embeddable_charts=False (default):
   - Charts in metadata only
   - Tool output has no embed instructions
   - Cleaner output for non-embedding use cases

✅ embeddable_charts=True:
   - Charts in metadata AND embed instructions in content
   - LLM can reference charts via XML tags
   - Enables rich chart-embedded responses
   - Format: <chart id="unique-chart-id"></chart>

✅ Chart IDs are:
   - Generated from title (slugified) + hash
   - Unique and stable
   - Example: "chart-quarterly-revenue-abc123"

✅ Use cases:
   - Dashboard builders
   - Report generators
   - Data analysis chatbots
   - Any scenario where LLM should reference visualizations
""")
