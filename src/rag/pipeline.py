"""
Complete RAG pipeline combining retriever and generator.
src/rag/pipeline.py
"""

from src.rag.retriever import SECRetriever
from src.rag.generator import SECGenerator


class SECRagPipeline:
    """
    End-to-end RAG pipeline for SEC financial filings.
    """

    def __init__(self):
        self.retriever = SECRetriever()
        self.generator = SECGenerator()

    def ask(
        self,
        question: str,
        ticker:   str  = None,
        year:     str  = None,
        sector:   str  = None,
        top_k:    int  = 8
    ) -> dict:
        """
        Ask a question about SEC filings.

        Returns:
            answer:   Generated answer text
            sources:  List of source documents used
            question: Original question
        """
        # Step 1: Retrieve relevant chunks
        results = self.retriever.search(
            query  = question,
            top_k  = top_k,
            ticker = ticker,
            year   = year,
            sector = sector
        )

        if not results:
            return {
                "question": question,
                "answer":   "No relevant information found "
                            "in the SEC filings database.",
                "sources":  [],
                "success":  False
            }

        # Step 2: Format context
        context = self.retriever.format_context(results)

        # Step 3: Generate answer
        response = self.generator.generate(
            question = question,
            context  = context
        )

        # Step 4: Format sources
        sources = [
            {
                "company":   r["company"],
                "form_type": r["form_type"],
                "year":      r["year"],
                "score":     r["score"],
                "preview":   r["text"][:200]
            }
            for r in results
        ]

        return {
            "question": question,
            "answer":   response.get("answer", ""),
            "sources":  sources,
            "success":  response.get("success", False),
            "model":    response.get("model", ""),
        }

    def print_answer(self, result: dict):
        """Pretty print a pipeline result."""
        print(f"\n{'='*60}")
        print(f"Q: {result['question']}")
        print(f"{'─'*60}")
        print(f"A: {result['answer']}")
        print(f"{'─'*60}")
        print(f"Sources used:")
        for i, s in enumerate(result['sources'], 1):
            print(f"  {i}. {s['company']} "
                  f"({s['form_type']} {s['year']}) "
                  f"[relevance: {s['score']}]")
        print(f"{'='*60}")


if __name__ == "__main__":
    pipeline = SECRagPipeline()

    questions = [
        {"q": "What are NVIDIA main risks in 2025?",
         "ticker": "NVDA"},
        {"q": "How has Microsoft Azure revenue grown?",
         "ticker": "MSFT"},
        {"q": "What do energy companies say about "
              "climate risk?",
         "sector": "Energy"},
        {"q": "Compare JPMorgan and Goldman Sachs "
              "risk management approaches"},
    ]

    for item in questions:
        result = pipeline.ask(
            question = item["q"],
            ticker   = item.get("ticker"),
            sector   = item.get("sector"),
        )
        pipeline.print_answer(result)