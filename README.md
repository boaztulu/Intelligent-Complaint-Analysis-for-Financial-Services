<h1 align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOGJiNThmZWUzZDc4NjE0ODZmYTQzOGJiNzE3Y2NjNWZjMjM5NjU0NiZjdD1n/3o7aD5tv1ogNBtDhDi/giphy.gif" width="80"/>
  <br>
  <span style="color:#008080;">CrediTrust&nbsp;Complaint&nbsp;RAG&nbsp;Chatbot</span>
</h1>

<p align="center">
A ⚡️&nbsp;<b>Retrieval‑Augmented Generation (RAG)</b> pipeline that turns raw CFPB complaint narratives into <i>actionable, evidence‑backed answers</i>&nbsp;💡
</p>

&nbsp;

## 🎯 What’s Inside?
| Stage | 📄 Task | 🚀 Output |
|-------|---------|-----------|
| **1** | *EDA & Pre‑processing* | Clean <code>filtered_complaints.csv</code> + snazzy graphs 📊 |
| **2** | *Chunk → Embed → Index* | FAISS vector DB <sub>(blazing‑fast search)</sub> 🔎 |
| **3** | *RAG Core Logic* | Plug‑n‑play retriever + LLM prompt 🤖 |
| **4** | *Gradio Chat UI* | Ask <i>“Why are BNPL users furious?”</i> and get sourced answers in seconds ✨ |

---

## 📁 Repo Map

```text
.
├── data/
│   ├── complaints.zip             ← raw CFPB CSVs (compressed)
│   └── filtered_complaints.csv    ← ✅ output from Task 1
├── vector_store/                  ← ✅ FAISS index + metadata
├── scripts/
│   ├── task1_eda_preprocessing.py
│   ├── task2_chunk_embed_index.py
│   ├── task3_rag.py               ← backend pipeline
│   └── rag_openrouter.py          ← alt LLM endpoint demo
├── notebook/
│   ├── task1_eda_preprocessing.ipynb
│   ├── task2_chunk_embed_index.ipynb
│   ├── task3_rag.ipynb
│   └── rag_openrouter.ipynb
├── app.py                         ← 🎨 Gradio chat interface
├── requirements.txt               ← all deps pinned
└── README.md                      ← you are here 💜


## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-org/creditrust-complaint-rag.git
cd creditrust-complaint-rag
```

git clone https://github.com/your-org/creditrust-complaint-rag.git
cd creditrust-complaint-rag
pip install -r requirements.txt

## Environment Variables

B# --- OpenAI (Task‑3A) ---
export OPENAI_API_KEY="sk-<your_openai_secret>"
# --- OpenRouter / DeepSeek (Task‑3B & UI) ---
export OPENROUTER_API_KEY="sk-or-<your_openrouter_secret>"
# (optional) DeepSeek embeddings
export DEEPSEEK_API_KEY="<your_deepseek_key>"
