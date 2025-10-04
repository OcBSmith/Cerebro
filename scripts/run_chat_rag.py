import os
import sys
from pathlib import Path

# Import the chat app main
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))  # ensure scripts/ on path
from chat_app import main as chat_main  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_args():
    # Allow overrides via env vars; otherwise use sensible defaults
    model_id = os.getenv("CHAT_MODEL_ID", "google/gemma-2-2b-it")
    embed_model = os.getenv("EMBED_MODEL_ID", "BAAI/bge-m3")
    persist = os.getenv("LLAMAINDEX_PERSIST", str(PROJECT_ROOT / "data" / "llamaindex" / "storage"))
    host = os.getenv("CHAT_HOST", "127.0.0.1")
    port = os.getenv("CHAT_PORT", "7860")

    args = [
        "--model-id", model_id,
        "--rag",
        "--persist", persist,
        "--embed-model", embed_model,
        "--host", host,
        "--port", str(port),
    ]
    # Offline toggle if present
    if os.getenv("HF_OFFLINE", "0") in ("1", "true", "True"):
        args.append("--offline")
    # Local models dir for cache/offline
    models_dir = os.getenv("MODELS_DIR")
    if models_dir:
        args.extend(["--models-dir", models_dir])
    return args


def main():
    args = build_args()
    # Rewire argv and call the underlying CLI main
    sys.argv = ["chat_app.py"] + args
    chat_main()


if __name__ == "__main__":
    main()
