import argparse
import json
from pathlib import Path
from typing import List

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "llamaindex" / "chunks.jsonl"
DEFAULT_PERSIST = PROJECT_ROOT / "data" / "llamaindex" / "storage"


def load_chunks(jsonl_path: Path) -> List[Document]:
    docs: List[Document] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec["text"]
            meta = rec.get("meta", {})
            section_path = meta.get("section_path", [])
            title = " / ".join(section_path) if section_path else meta.get("doc", "")
            docs.append(
                Document(
                    text=text,
                    metadata={
                        "title": title,
                        **meta,
                    },
                )
            )
    return docs


def main():
    parser = argparse.ArgumentParser(description="Build and persist a LlamaIndex vector index from chunks.jsonl")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS), help="Path to chunks.jsonl")
    parser.add_argument("--persist", default=str(DEFAULT_PERSIST), help="Directory to persist the index")
    parser.add_argument(
        "--embed-model",
        default="BAAI/bge-m3",
        help="HuggingFace embedding model (multilingual recommended: BAAI/bge-m3)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    args = parser.parse_args()

    chunks_path = Path(args.chunks).expanduser().resolve()
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks not found: {chunks_path}")

    persist_dir = Path(args.persist).expanduser().resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks from: {chunks_path}")
    documents = load_chunks(chunks_path)
    print(f"Loaded documents: {len(documents)}")

    print(f"Loading embedding model: {args.embed_model}")
    embed_model = HuggingFaceEmbedding(model_name=args.embed_model, cache_folder=str(PROJECT_ROOT / ".cache"))
    # Optional: if future versions support batch size adjust, set here.

    print("Building VectorStoreIndex...")
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    print(f"Persisting index to: {persist_dir}")
    storage_context = index.storage_context
    storage_context.persist(persist_dir=str(persist_dir))
    print("Done.")


if __name__ == "__main__":
    main()
