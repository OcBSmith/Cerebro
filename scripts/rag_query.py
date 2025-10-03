import argparse
import re
from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSIST = PROJECT_ROOT / "data" / "llamaindex" / "storage"


def main():
    parser = argparse.ArgumentParser(description="Run a Spanish RAG query against a persisted LlamaIndex store")
    parser.add_argument("--persist", default=str(DEFAULT_PERSIST), help="Persist directory of the index")
    parser.add_argument("--query", required=False, help="Query in Spanish")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k neighbors for retrieval")
    parser.add_argument("--embed-model", default="BAAI/bge-m3", help="Embedding model (should match the one used to build the index)")
    parser.add_argument("--llm-model", default="google/gemma-2-2b-it", help="HF model id for generation (Gemma IT recommended)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--interactive", action="store_true", help="Interactive mode (REPL) if no --query is provided")
    args = parser.parse_args()

    persist_dir = Path(args.persist).expanduser().resolve()
    if not persist_dir.exists():
        raise FileNotFoundError(f"Persist dir not found: {persist_dir}")

    # Embeddings (must match build-time model family for best results)
    embed_model = HuggingFaceEmbedding(
        model_name=args.embed_model,
        cache_folder=str(PROJECT_ROOT / ".cache"),
    )

    # LLM (Gemma) via HuggingFace
    llm = HuggingFaceLLM(
        model_name=args.llm_model,
        generate_kwargs={
            "temperature": args.temperature,
            "do_sample": args.temperature > 0,
        },
        tokenizer_name=args.llm_model,
        # Device / quantization are auto-handled by transformers accelerate config; adjust via env if needed
    )

    # Inject into global Settings for query engine
    Settings.embed_model = embed_model
    Settings.llm = llm

    # Load storage and index (0.14 API)
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage_context)

    # Build query engine
    query_engine = index.as_query_engine(similarity_top_k=args.top_k, response_mode="compact")

    # Spanish system instruction to ensure Spanish answers
    system_prompt = (
        "Eres un asistente experto en ORCA. RESPONDE EXCLUSIVAMENTE EN ESPAÑOL. "
        "Si la pregunta no está en español, tradúcela y responde en español. "
        "Usa únicamente la información de los pasajes recuperados; si falta información, dilo explícitamente. "
        "Sé conciso y técnico cuando proceda."
    )

    def run_one(q: str):
        resp = query_engine.query(f"[Instrucción del sistema]: {system_prompt}\n\n[Pregunta]: {q}")
        # Extract text and tidy whitespace
        text = getattr(resp, "response", str(resp))
        text = text.strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        print("=== Respuesta ===")
        print(text)
        print()
        print("=== Pasajes (Top-{}): ===".format(args.top_k))
        for i, node in enumerate(resp.source_nodes, 1):
            meta = node.metadata or {}
            title = meta.get("title", "")
            section = meta.get("section_path", meta.get("section", ""))
            header = title if title else (" / ".join(section) if isinstance(section, list) else str(section))
            header = header.strip() if header else ""
            print(f"[{i}] {header}")
            snippet = node.get_text().strip()
            if len(snippet) > 600:
                snippet = snippet[:600].rstrip() + " …"
            print(snippet)
            print("-")

    if args.query:
        run_one(args.query)
    else:
        if not args.interactive:
            raise SystemExit("Provide --query or use --interactive mode")
        print("Modo interactivo. Escribe tu pregunta (o 'salir' para terminar):")
        while True:
            try:
                q = input("> ").strip()
            except EOFError:
                break
            if not q or q.lower() in {"salir", "exit", "quit"}:
                break
            run_one(q)


if __name__ == "__main__":
    main()
