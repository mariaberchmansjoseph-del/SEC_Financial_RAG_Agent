"""
RAG Generator — generates answers from retrieved context.
Uses Groq (free) or OpenAI depending on available key.
src/rag/generator.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are a financial analyst assistant
specialising in SEC filings analysis.

Your role:
- Answer questions using ONLY the provided context
- Always cite your sources (company, filing type, year)
- Be precise with financial figures when present
- If context does not contain the answer say so clearly
- Never hallucinate financial data
- Keep answers concise but complete (3-5 sentences)"""


class SECGenerator:

    def __init__(self):
        if GROQ_KEY:
            from groq import Groq
            self.client  = Groq(api_key=GROQ_KEY)
            self.backend = "groq"
            
            self.model   = "openai/gpt-oss-20b"
            print(f"✅ Generator: Groq connected "
                  f"(llama-3.1-8b-instant)")
        elif OPENAI_KEY:
            from openai import OpenAI
            self.client  = OpenAI(api_key=OPENAI_KEY)
            self.backend = "openai"
            self.model   = "gpt-4o-mini"
            print(f"✅ Generator: OpenAI connected (gpt-4o-mini)")
        else:
            self.client  = None
            self.backend = "fallback"
            self.model   = "fallback"
            print("⚠️  Generator: No API key — using fallback")
            print("    Add GROQ_API_KEY or OPENAI_API_KEY to .env")

    def generate(self, question: str, context: str) -> dict:
        if self.backend == "fallback" or self.client is None:
            return self._fallback(question, context)

        prompt = f"""Context from SEC filings:

{context}

Question: {question}

Answer based strictly on the context above.
Cite which company and filing year you are drawing from."""

        try:
            response = self.client.chat.completions.create(
                model    = self.model,
                messages = [
                    {"role": "system",
                     "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": prompt}
                ],
                temperature = 0.1,
                max_tokens  = 400
            )
            answer     = response.choices[0].message.content
            tokens     = response.usage.total_tokens

            return {
                "answer":      answer,
                "tokens_used": tokens,
                "model":       self.model,
                "success":     True
            }

        except Exception as e:
            error = str(e)
            if "429" in error or "credits" in error.lower():
                print(f"  ⚠️  API limit hit — switching to fallback")
                return self._fallback(question, context)
            return {
                "answer":  f"Error: {error[:100]}",
                "success": False
            }

    def _fallback(self, question: str, context: str) -> dict:
        """Keyword-based answer when no LLM available."""
        sentences = [s.strip() for s in context.split(".")
                     if len(s.strip()) > 40]
        q_words   = set(question.lower().split())

        scored = []
        for sent in sentences:
            overlap = len(q_words & set(sent.lower().split()))
            scored.append((overlap, sent))

        scored.sort(reverse=True)
        top = [s for _, s in scored[:3] if s]

        answer = (
            "Based on retrieved SEC filings:\n\n"
            + " ".join(top)
            + "\n\n[Note: Add GROQ_API_KEY to .env "
            "for AI-generated answers — free at groq.com]"
        )
        return {"answer": answer, "success": True,
                "model": "fallback"}