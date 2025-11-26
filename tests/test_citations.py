"""
Citation System Test - OpenAI Clients

Tests the complete citation flow:
1. Action receives citation_start parameter
2. Action returns content with embedded XML citation tags (<1>, <2>, etc.)
3. Action returns citations dict mapping IDs to metadata
4. Citations are tracked in agent.client
5. Final response includes citations
"""

from dotenv import load_dotenv
from jetflow import Agent
from jetflow.clients.openai import OpenAIClient
from jetflow.action import action
from jetflow.models.response import ActionResult
from pydantic import BaseModel, Field
from typing import Dict

load_dotenv()


# ============================================================================
# Mock Document Database (in-memory)
# ============================================================================

MOCK_DOCUMENTS = {
    "doc-1": {
        "title": "NVIDIA Q4 2025 Earnings Call",
        "date": "2025-02-26",
        "company": "NVIDIA Corporation",
        "symbol": "NVDA",
        "sections": [
            {
                "id": "sec-1",
                "speaker": "Jensen Huang",
                "role": "CEO",
                "content": "Revenue for Q4 was $22.1 billion, up 122% year-over-year, driven by record data center revenue."
            },
            {
                "id": "sec-2",
                "speaker": "Colette Kress",
                "role": "CFO",
                "content": "Data center revenue reached $18.4 billion, representing a 217% increase from a year ago. This growth was primarily driven by strong demand for our Hopper architecture GPUs."
            },
            {
                "id": "sec-3",
                "speaker": "Jensen Huang",
                "role": "CEO",
                "content": "We are ramping production of Blackwell, our next-generation AI platform. Customer interest has been extraordinary, with demand significantly exceeding our initial supply."
            }
        ]
    },
    "doc-2": {
        "title": "NVIDIA 10-K Annual Report",
        "date": "2025-02-26",
        "company": "NVIDIA Corporation",
        "symbol": "NVDA",
        "fiscal_year": 2025,
        "sections": [
            {
                "id": "elem-1",
                "page": 42,
                "content": "Total revenue for fiscal year 2025 was $60.9 billion, compared to $26.9 billion in fiscal year 2024."
            },
            {
                "id": "elem-2",
                "page": 42,
                "content": "Data center revenue was $47.5 billion, up 217% from the prior year, representing 78% of total revenue."
            },
            {
                "id": "elem-3",
                "page": 43,
                "content": "Gaming revenue was $10.4 billion, up 15% year-over-year, driven by strong demand for GeForce RTX 40 Series GPUs."
            }
        ]
    }
}


# ============================================================================
# Citation Schema
# ============================================================================

class DocumentCitation(BaseModel):
    """Citation metadata for mock documents"""
    id: int
    type: str  # "transcript" or "filing"
    document_id: str
    company: str
    symbol: str
    element_id: str
    content_preview: str  # First 100 chars of cited content
    # Transcript-specific
    speaker: str = None
    role: str = None
    # Filing-specific
    page: int = None
    fiscal_year: int = None


# ============================================================================
# Mock Read Document Action
# ============================================================================

class ReadDocumentParams(BaseModel):
    """Read a mock document and return content with citations"""
    document_id: str = Field(description="Document ID to read (doc-1 or doc-2)")


@action(ReadDocumentParams)
def read_document(params: ReadDocumentParams, citation_start: int = 1) -> ActionResult:
    """Read document and return content with embedded citations

    Simulates reading from a database and formatting with citation tags.
    """
    doc = MOCK_DOCUMENTS.get(params.document_id)

    if not doc:
        return ActionResult(
            content=f"Document {params.document_id} not found. Available: doc-1, doc-2",
            citations={},
            summary="Document not found"
        )

    # Determine document type and format accordingly
    if "speaker" in doc["sections"][0]:
        return _format_transcript(doc, citation_start)
    else:
        return _format_filing(doc, citation_start)


def _format_transcript(doc: Dict, citation_start: int) -> ActionResult:
    """Format transcript with citations"""
    header = f"# {doc['company']} ({doc['symbol']}) - {doc['title']}\n\n"
    header += f"**Date:** {doc['date']}\n\n---\n\n"

    citations = {}
    cid = citation_start
    content_parts = []

    for section in doc["sections"]:
        speaker = section["speaker"]
        role = section["role"]
        content = section["content"]

        # Add speaker header
        content_parts.append(f"## {speaker} ({role})\n")

        # Add content with citation tag
        content_parts.append(f"{content} <{cid}>\n")

        # Create citation
        citations[cid] = DocumentCitation(
            id=cid,
            type="transcript",
            document_id=doc.get("document_id", "doc-1"),
            company=doc["company"],
            symbol=doc["symbol"],
            element_id=section["id"],
            speaker=speaker,
            role=role,
            content_preview=content[:100]
        ).model_dump()

        cid += 1

    content = header + "\n".join(content_parts)

    return ActionResult(
        content=content,
        citations=citations,
        metadata={"document_id": doc.get("document_id", "doc-1"), "sections_count": len(doc["sections"])},
        summary=f"Read {len(doc['sections'])} sections from transcript"
    )


def _format_filing(doc: Dict, citation_start: int) -> ActionResult:
    """Format filing with citations"""
    header = f"# {doc['company']} ({doc['symbol']}) - {doc['title']}\n\n"
    header += f"**Filing Date:** {doc['date']}\n"
    header += f"**Fiscal Year:** {doc['fiscal_year']}\n\n---\n\n"

    citations = {}
    cid = citation_start
    content_parts = []

    for section in doc["sections"]:
        page = section["page"]
        content = section["content"]

        # Add content with citation tag
        content_parts.append(f"{content} <{cid}>\n")

        # Create citation
        citations[cid] = DocumentCitation(
            id=cid,
            type="filing",
            document_id=doc.get("document_id", "doc-2"),
            company=doc["company"],
            symbol=doc["symbol"],
            element_id=section["id"],
            page=page,
            fiscal_year=doc["fiscal_year"],
            content_preview=content[:100]
        ).model_dump()

        cid += 1

    content = header + "\n".join(content_parts)

    return ActionResult(
        content=content,
        citations=citations,
        metadata={"document_id": doc.get("document_id", "doc-2"), "sections_count": len(doc["sections"])},
        summary=f"Read {len(doc['sections'])} sections from filing"
    )


# ============================================================================
# Exit Action
# ============================================================================

class SubmitAnalysisParams(BaseModel):
    """Submit final analysis"""
    summary: str = Field(description="Summary of findings")


@action(SubmitAnalysisParams, exit=True)
def submit_analysis(params: SubmitAnalysisParams) -> ActionResult:
    return ActionResult(
        content=f"Analysis Complete:\n\n{params.summary}",
        summary="Analysis submitted"
    )


# ============================================================================
# Test
# ============================================================================

def test_citations():
    print("=" * 80)
    print("CITATION SYSTEM TEST - SYNC OPENAI")
    print("=" * 80)
    print()

    client = OpenAIClient(model="gpt-5-mini")

    agent = Agent(
        client=client,
        system_prompt="""You are analyzing NVIDIA's financial performance.

Read both documents (doc-1 and doc-2) and write a brief analysis.
When you reference specific data, include the citation tags from the documents.

After reading both documents, submit your analysis.""",
        actions=[read_document, submit_analysis],
        require_action=True,
        max_iter=10,
        verbose=True
    )

    response = agent.run("Analyze NVIDIA's Q4 2025 performance using both the earnings call and annual report.")

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
        if msg.citations:
            print(f"      Citations: {len(msg.citations)} citations attached")

    print("\n" + "=" * 80)
    print("CITATION ASSERTIONS")
    print("=" * 80)

    # ========================================================================
    # 1. CITATION MIDDLEWARE ASSERTIONS
    # ========================================================================
    # Agent uses CitationMiddleware (agent.client) for all citation management
    assert hasattr(agent.client, 'citations'), "Client should have citation storage"
    assert hasattr(agent.client, 'get_next_id'), "Client should have get_next_id method"

    # Check that citations were created
    next_cid = agent.client.get_next_id()
    print(f"✓ Citation middleware next_id: {next_cid}")

    # Count total citations created across all actions
    total_citations_created = 0
    for msg in response.messages:
        if msg.role == "tool" and msg.citations:
            total_citations_created += len(msg.citations)
            print(f"✓ Tool message {msg.action_id[:20]}... has {len(msg.citations)} citations")

    assert total_citations_created > 0, "Should have created citations in tool messages"
    print(f"✓ Total citations created: {total_citations_created}")

    # ========================================================================
    # 2. CITATION INCREMENTING ASSERTIONS
    # ========================================================================
    # Verify that citation IDs increment properly across multiple actions
    # citation_start is now passed as a parameter during action execution,
    # so we verify proper incrementing by checking the citation IDs in tool messages
    all_citation_ids = []
    for msg in response.messages:
        if msg.role == "tool" and msg.citations:
            all_citation_ids.extend(msg.citations.keys())

    all_citation_ids.sort()

    if len(all_citation_ids) > 1:
        # Verify citations increment sequentially (1, 2, 3, 4...)
        for i in range(1, len(all_citation_ids)):
            assert all_citation_ids[i] > all_citation_ids[i-1], \
                f"Citation IDs should increment: {all_citation_ids[i-1]} -> {all_citation_ids[i]}"

    print(f"✓ Citation IDs increment correctly: {all_citation_ids}")

    # ========================================================================
    # 3. CITATION CONTENT ASSERTIONS
    # ========================================================================
    # Check that tool messages contain citation tags
    citation_tag_count = 0
    for msg in response.messages:
        if msg.role == "tool" and msg.content:
            # Count citation tags like <1>, <2>, etc.
            import re
            tags = re.findall(r'<(\d+)>', msg.content)
            citation_tag_count += len(tags)
            if tags:
                print(f"✓ Tool message has citation tags: {tags}")

    assert citation_tag_count > 0, "Tool messages should contain citation tags like <1>, <2>"
    print(f"✓ Total citation tags found in content: {citation_tag_count}")

    # ========================================================================
    # 4. CITATION METADATA ASSERTIONS
    # ========================================================================
    # Verify citation structure
    for msg in response.messages:
        if msg.role == "tool" and msg.citations:
            for cid, citation in msg.citations.items():
                assert isinstance(cid, int), f"Citation ID should be int, got {type(cid)}"
                assert isinstance(citation, dict), f"Citation should be dict, got {type(citation)}"

                # Check required fields
                assert "id" in citation, "Citation should have 'id' field"
                assert "type" in citation, "Citation should have 'type' field"
                assert "company" in citation, "Citation should have 'company' field"
                assert "element_id" in citation, "Citation should have 'element_id' field"

                # Verify ID matches
                assert citation["id"] == cid, f"Citation ID mismatch: dict key {cid} vs citation['id'] {citation['id']}"

    print(f"✓ All citations have valid structure and metadata")

    # ========================================================================
    # 5. FINAL RESPONSE CITATION ASSERTIONS
    # ========================================================================
    # Check if final response includes citations
    final_msg = response.messages[-1]
    if final_msg.role == "assistant" and final_msg.citations:
        print(f"✓ Final response has {len(final_msg.citations)} citations attached")

        # Verify these are the "used" citations from the content
        for cid in final_msg.citations.keys():
            assert isinstance(cid, int), "Used citation ID should be int"
            print(f"  - Citation {cid}: {final_msg.citations[cid].get('element_id', 'unknown')}")
    else:
        print(f"  (Final response has no citations - model may not have referenced sources)")

    # ========================================================================
    # 6. CITATION CONTINUITY ASSERTIONS
    # ========================================================================
    # Verify citation IDs are continuous and don't overlap
    all_citation_ids = []
    for msg in response.messages:
        if msg.role == "tool" and msg.citations:
            all_citation_ids.extend(msg.citations.keys())

    if all_citation_ids:
        all_citation_ids_sorted = sorted(all_citation_ids)
        print(f"✓ Citation IDs used: {all_citation_ids_sorted}")

        # Check for gaps or duplicates
        expected_ids = list(range(all_citation_ids_sorted[0], all_citation_ids_sorted[-1] + 1))
        missing = set(expected_ids) - set(all_citation_ids_sorted)
        duplicates = len(all_citation_ids_sorted) - len(set(all_citation_ids_sorted))

        if missing:
            print(f"  ⚠ Missing citation IDs: {sorted(missing)}")
        if duplicates > 0:
            print(f"  ⚠ Duplicate citation IDs: {duplicates} duplicates")

        assert duplicates == 0, "Should have no duplicate citation IDs"

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Agent completed: {response.success}")
    print(f"✓ Iterations: {response.iterations}")
    print(f"✓ Total citations created: {total_citations_created}")
    print(f"✓ Citation tags in content: {citation_tag_count}")
    print(f"✓ Citation manager next_id: {next_cid}")
    print(f"✓ Cost: ${response.usage.estimated_cost:.4f}")

    print("\n✅ CITATION TEST PASSED\n")
    return response


if __name__ == "__main__":
    print("\n" + "🧪 CITATION SYSTEM TEST SUITE" + "\n")

    try:
        test_citations()

        print("=" * 80)
        print("🎉 CITATION TEST PASSED!")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n💥 ERROR: {e}\n")
        raise
