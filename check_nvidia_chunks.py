import sys
sys.path.insert(0, '.')
from src.rag.retriever import SECRetriever

retriever = SECRetriever()

results, context = retriever.search_and_format(
    query  = "NVIDIA AI chip supply chain risks export controls",
    ticker = "NVDA",
    top_k  = 5
)

print("RETRIEVED CHUNKS:")
print("="*60)
for i, r in enumerate(results, 1):
    print(f"\nChunk {i}: {r['company']} "
          f"({r['form_type']} {r['year']}) "
          f"[score: {r['score']}]")
    print(f"Section: {r['section']}")
    print(f"Text: {r['text'][:400]}")
    print("-"*40)

print("\nFULL CONTEXT SENT TO LLM:")
print("="*60)
print(context[:2000])