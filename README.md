# Corrective RAG — LangGraph Pipeline

A progressive series of Jupyter notebooks building a **Corrective RAG** system using **LangGraph**, **LangChain**, **FAISS**, and **Tavily web search**. Each notebook adds a layer of intelligence on top of the previous one.

---

## Project Structure

```
corrective-rag/
├── documents/              # PDF knowledge base (book1.pdf, book2.pdf, book3.pdf)
├── venv/                   # Python virtual environment (not tracked)
├── .env                    # API keys (not tracked)
├── 1_basic_rag.ipynb
├── 2_retrieval_refinement.ipynb
├── 3_retrieval_evaluator.ipynb
├── 4_web_search_refinement.ipynb
├── 5_query_rewrite.ipynb
├── 6_ambiguous.ipynb
└── requirements.txt
```

---

## Notebook Progression

| # | Notebook | What it adds |
|---|----------|--------------|
| 1 | `1_basic_rag.ipynb` | Naive RAG: load PDFs → chunk → FAISS → retrieve → generate |
| 2 | `2_retrieval_refinement.ipynb` | Sentence-level filtering of retrieved context before generation |
| 3 | `3_retrieval_evaluator.ipynb` | LLM-based per-doc scoring; routes CORRECT vs INCORRECT |
| 4 | `4_web_search_refinement.ipynb` | Falls back to Tavily web search when retrieval is INCORRECT |
| 5 | `5_query_rewrite.ipynb` | Rewrites user query into a better web search query before search |
| 6 | `6_ambiguous.ipynb` | **Full pipeline** — adds AMBIGUOUS verdict: merges internal + web docs |

---

## Notebook 6 — Ambiguous Handler (Full Pipeline)

### Problem
Binary CORRECT / INCORRECT routing fails when retrieved chunks are *partially* relevant — not good enough to answer alone, but not completely off-topic either.

### Solution: Three-Way Verdict

```
retrieved docs
     │
     ▼
eval_each_doc  ──── scores each chunk [0.0 – 1.0]
     │
     ├── any score > 0.7  →  CORRECT   → refine (internal only) → generate
     ├── all scores < 0.3 →  INCORRECT → rewrite_query → web_search → refine → generate
     └── otherwise        →  AMBIGUOUS → rewrite_query → web_search → refine (internal + web) → generate
```

### LangGraph State

```python
class State(TypedDict):
    question: str
    docs: List[Document]        # raw retrieved docs
    good_docs: List[Document]   # docs above LOWER_TH (0.3)
    verdict: str                # CORRECT / INCORRECT / AMBIGUOUS
    reason: str                 # explanation of verdict
    strips: List[str]           # sentences from context
    kept_strips: List[str]      # sentences kept after LLM filter
    refined_context: str        # final clean context
    web_query: str              # rewritten query for web
    web_docs: List[Document]    # Tavily results as Documents
    answer: str                 # final LLM answer
```

### Pipeline Nodes

| Node | Role |
|------|------|
| `retrieve` | FAISS similarity search (k=4) |
| `eval_each_doc` | Scores each chunk via `gpt-4o-mini` structured output; sets verdict |
| `rewrite_query` | Rewrites question into 6–14 word web search query |
| `web_search` | Tavily search (5 results) → `List[Document]` |
| `refine` | Decomposes context into sentences, LLM-filters irrelevant ones |
| `generate` | Answers from `refined_context` only |

### Routing Logic

```python
def route_after_eval(state):
    if state["verdict"] == "CORRECT":
        return "refine"       # skip web search
    else:
        return "rewrite_query"  # INCORRECT or AMBIGUOUS → web
```

For **AMBIGUOUS**, the `refine` node merges `good_docs + web_docs` before filtering — giving the LLM the best of both sources.

### Thresholds

```python
UPPER_TH = 0.7   # chunk must score above this to call it CORRECT
LOWER_TH = 0.3   # chunk must score below this to be discarded entirely
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/corrective-rag.git
cd corrective-rag
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### 3. Add PDFs

Place your PDF files in `documents/`:
```
documents/book1.pdf
documents/book2.pdf
documents/book3.pdf
```

### 4. Run

Open any notebook and run all cells. Start from `1_basic_rag.ipynb` to follow the progression.

---

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful graph orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — chains, prompts, document loaders
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [OpenAI](https://platform.openai.com/) — `gpt-4o-mini` for generation/evaluation, `text-embedding-3-large` for embeddings
- [Tavily](https://tavily.com/) — real-time web search API
- [Pydantic](https://docs.pydantic.dev/) — structured LLM outputs

---

## Key Design Decisions

- **Score-based evaluation** (float 0–1) instead of binary keep/drop gives a nuanced third state (AMBIGUOUS).
- **Sentence-level refinement** strips irrelevant noise from retrieved context before generation.
- **Query rewriting** before web search improves Tavily result quality (especially for recency-aware queries).
- All LLM calls use **structured output** (Pydantic models) for reliable JSON parsing.