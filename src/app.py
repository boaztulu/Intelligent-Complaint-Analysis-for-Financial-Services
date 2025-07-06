import os
from pathlib import Path

import gradio as gr
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

# ─── Configuration ─────────────────────────────────────────────────────────

# FAISS vector store directory (from Task 2)
VSTORE_DIR = Path("vector_store")

# Embedding model name (must match what you used in Task 2)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
os.environ["OPENAI_API_KEY"] = "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"

# OpenRouter / Deepseek setup (Task 4 uses Deepseek free model via OpenRouter)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL    = "deepseek/deepseek-r1:free"

# Prompt template for RAG
PROMPT_TEMPLATE = (
    "You are a financial analyst assistant for CrediTrust.\n"
    "Use only the following excerpts from customer complaints to answer the question.\n"
    "If the context doesn't contain the answer, say you don't have enough information.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\nAnswer:"
)

# Number of chunks to retrieve
TOP_K = 5


# ─── Core Functions ─────────────────────────────────────────────────────────

def load_vectorstore(vs_dir: Path, embedding_model: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.load_local(
        folder_path=str(vs_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )


def retrieve_chunks(vectorstore: FAISS, question: str, k: int):
    return vectorstore.similarity_search(question, k=k)


def build_prompt(chunks, question: str, template: str) -> str:
    context = "\n\n".join(doc.page_content for doc in chunks)
    return template.format(context=context, question=question)


def generate_answer(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("Set OPENROUTER_API_KEY environment variable to your sk-or-… key")
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
    resp = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ─── Build Gradio Interface ─────────────────────────────────────────────────

# Load vectorstore once at startup
vectorstore = load_vectorstore(VSTORE_DIR, EMBED_MODEL)

def qa_fn(question: str):
    # 1) retrieve and build prompt
    chunks = retrieve_chunks(vectorstore, question, TOP_K)
    prompt = build_prompt(chunks, question, PROMPT_TEMPLATE)

    # 2) generate
    answer = generate_answer(prompt)

    # 3) format sources
    sources_md = "#### Sources\n"
    for doc in chunks:
        meta = doc.metadata
        snippet = doc.page_content.replace("\n", " ")
        sources_md += (
            f"- **Product**: {meta['product']}, **ID**: {meta['complaint_id']}\n\n"
            f"  > {snippet[:200]}…\n\n"
        )
    return answer, sources_md


# Create the Gradio app
with gr.Blocks() as demo:
    gr.Markdown("# CrediTrust Complaint Chatbot")
    gr.Markdown(
        "Ask questions about customer complaints across Credit Cards, BNPL, "
        "Savings, Personal Loans, and Money Transfers."
    )

    with gr.Row():
        inp = gr.Textbox(
            label="Your question",
            placeholder="e.g. Why are people unhappy with BNPL?",
            lines=2
        )
        ask_btn = gr.Button("Ask")
        clear_btn = gr.Button("Clear")

    answer_out = gr.Textbox(label="Answer", interactive=False, lines=5)
    sources_out = gr.Markdown("#### Sources will appear here")

    ask_btn.click(fn=qa_fn, inputs=inp, outputs=[answer_out, sources_out])
    clear_btn.click(lambda: ("", "#### Sources will appear here"), [], [inp, answer_out, sources_out])

    gr.Markdown("*Powered by Deepseek via OpenRouter & FAISS*")

if __name__ == "__main__":
    demo.launch()
