import argparse
from pathlib import Path

from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI


def load_vectorstore(vs_dir: Path, embedding_model: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.load_local(
        folder_path=str(vs_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )


def retrieve_chunks(
    vectorstore: FAISS,
    question: str,
    k: int
):
    return vectorstore.similarity_search(question, k=k)


def build_prompt(
    chunks,
    question: str,
    template: str
) -> str:
    context = "\n\n".join(doc.page_content for doc in chunks)
    return template.format(context=context, question=question)


def generate_openrouter(
    prompt: str,
    base_url: str,
    api_key: str,
    model_name: str
) -> str:
    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def main():
    p = argparse.ArgumentParser(
        description="RAG via OpenRouter.ai (Deepseek free model)"
    )
    p.add_argument(
        "--vectorstore-dir",
        type=Path,
        default="vector_store",
        help="Path to FAISS index directory"
    )
    p.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model used by FAISS"
    )
    p.add_argument(
        "--openrouter-base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="OpenRouter API base URL"
    )
    p.add_argument(
        "--openrouter-api-key",
        type=str,
        required=True,
        help="Your sk-or-… OpenRouter API key"
    )
    p.add_argument(
        "--openrouter-model",
        type=str,
        default="deepseek/deepseek-r1:free",
        help="Model on OpenRouter (e.g. deepseek/deepseek-r1:free)"
    )
    p.add_argument(
        "--prompt-template",
        type=str,
        default=(
            "You are a financial analyst assistant for CrediTrust.\n"
            "Use only the following excerpts from customer complaints to answer the question.\n"
            "If the context doesn't contain the answer, say you don't have enough information.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\nAnswer:"
        ),
        help="Prompt template with {context} and {question}"
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve"
    )
    p.add_argument(
        "--question",
        type=str,
        required=True,
        help="The user question about complaints"
    )
    args = p.parse_args()

    # 1) Load FAISS vector store
    vs = load_vectorstore(args.vectorstore_dir, args.embedding_model)

    # 2) Retrieve top-k chunks
    chunks = retrieve_chunks(vs, args.question, args.k)

    # 3) Build prompt
    prompt = build_prompt(chunks, args.question, args.prompt_template)

    # 4) Call OpenRouter
    answer = generate_openrouter(
        prompt=prompt,
        base_url=args.openrouter_base_url,
        api_key=args.openrouter_api_key,
        model_name=args.openrouter_model
    )

    # 5) Output
    print("\n=== ANSWER ===\n")
    print(answer)
    print("\n=== SOURCES ===\n")
    for doc in chunks:
        m = doc.metadata
        snippet = doc.page_content.replace("\n", " ")[:200]
        print(f"- [id={m['complaint_id']}, product={m['product']}] “{snippet}…”")


if __name__ == "__main__":
    main()
