import argparse
from pathlib import Path

import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Substring pattern for our five products (must match your cleaned CSV)
PATTERN = (
    r"credit card|personal loan|buy now pay later|"
    r"savings account|money transfer"
)


def run_chunk_embed_index(
    input_csv: Path,
    output_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
):
    # 1) Load cleaned complaints
    df = pd.read_csv(input_csv)
    df = df.reset_index().rename(columns={"index": "complaint_id"})

    # 2) Chunk each narrative
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator=True,
    )
    docs = []
    for _, row in df.iterrows():
        text = row["clean_narrative"]
        if not text or not isinstance(text, str):
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "complaint_id": int(row["complaint_id"]),
                        "product": row["Product"],
                        "chunk_index": i,
                    },
                )
            )
    print(f"→ Created {len(docs)} chunks from {len(df)} complaints.")

    # 3) Embed & index with FAISS
    embedder = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.from_documents(docs, embedder)

    # 4) Persist vector store
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(output_dir))
    print(f"Vector store saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task 2: Chunk, embed, and index complaint narratives"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default="data/filtered_complaints.csv",
        help="Cleaned & filtered complaints CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="vector_store",
        help="Directory for FAISS index & metadata",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Max characters per text chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Overlap characters between chunks",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF embedding model",
    )
    args = parser.parse_args()

    run_chunk_embed_index(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
    )
