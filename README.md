\# SEC Financial Intelligence Agent



A production-grade multi-agent RAG system for querying

SEC financial filings across 20 major companies.



\## Overview



This system allows users to ask natural language questions

about company strategy, financials, and risk factors,

sourced directly from official SEC filings (10-K, 10-Q, 8-K).



\*\*Example questions:\*\*

\- "What does NVIDIA say about AI chip demand in their 2025 10-K?"

\- "What are JPMorgan's top 3 risks identified for 2025?"

\- "Compare Microsoft and Google's AI investment strategies"

\- "How has ExxonMobil's climate risk language changed since 2022?"



\## Architecture



SEC EDGAR API → Document Processing → Azure AI Search

↓

User Question → Multi-Agent Orchestrator → GPT-4o

↓

Grounded Answer + Citations





\## Companies Covered



| Sector | Companies |

|--------|-----------|

| Financial | JPMorgan, Goldman Sachs, Citigroup, BofA, BNY Mellon, Morgan Stanley |

| Technology | Microsoft, Google, NVIDIA, Meta, Apple, Amazon |

| Energy | ExxonMobil, Chevron, ConocoPhillips |

| Global Banks | HSBC, Visa, Mastercard |



\## Filing Coverage



\- \*\*Types:\*\* 10-K (annual), 10-Q (quarterly), 8-K (events)

\- \*\*Years:\*\* 2022 – 2025

\- \*\*Total:\*\* \~400 documents, \~100,000 pages



\## Tech Stack



| Component | Technology |

|-----------|-----------|

| Data Collection | SEC EDGAR API (free, official) |

| Document Parsing | Azure Document Intelligence |

| Vector Store | Azure AI Search (hybrid search) |

| LLM | Azure OpenAI GPT-4o |

| Agent Framework | Azure AI Foundry Agent Service |

| Evaluation | RAGAS + Azure AI Evaluation |

| Serving | FastAPI + Streamlit |

| CI/CD | GitHub Actions |

| Tracking | MLflow |



\## Agents



\- \*\*Orchestrator\*\* — routes questions to specialised agents

\- \*\*Filing Analyst\*\* — deep dive into single company filings

\- \*\*Financial Analyst\*\* — metric extraction and calculations

\- \*\*Comparison Analyst\*\* — cross-company analysis

\- \*\*Risk Analyst\*\* — risk factor extraction and tracking



\## Setup



```bash

git clone https://github.com/mariaberchmansjoseph-del/SEC\_Financial\_RAG\_Agent

cd SEC\_Financial\_RAG\_Agent

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

cp .env.example .env

\# Fill in your Azure credentials in .env

python scripts/download\_filings.py

```



\## Results



\*To be updated after training and evaluation\*



| Metric | Score |

|--------|-------|

| Groundedness | TBD |

| Relevance | TBD |

| Citation Rate | TBD |

| P95 Latency | TBD |



\## Project Status



\- \[x] Project structure

\- \[x] Company configuration

\- \[ ] SEC filing download

\- \[ ] Document parsing

\- \[ ] Chunking pipeline

\- \[ ] Azure AI Search indexing

\- \[ ] Multi-agent implementation

\- \[ ] Evaluation framework

\- \[ ] FastAPI serving

\- \[ ] Streamlit demo

\- \[ ] GitHub Actions CI/CD



\## Dataset



All data sourced from SEC EDGAR (edgar.sec.gov).

SEC filings are public domain — no license restrictions.

Data is not committed to this repository.

Run `python scripts/download\_filings.py` to collect it.



\## Author

Maria Berchmans Joseph

