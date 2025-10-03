import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# RAG (optional)
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSIST = PROJECT_ROOT / "data" / "llamaindex" / "storage"


def set_offline_mode(models_dir: Optional[Path], offline: bool):
    if models_dir:
        models_dir = Path(models_dir).resolve()
        os.environ.setdefault("HF_HOME", str(models_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(models_dir))
        os.environ.setdefault("HF_HUB_CACHE", str(models_dir))
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"


def load_qwen(model_id: str):
    local_path = Path(model_id) if os.path.isdir(model_id) else None
    src = str(local_path) if local_path else model_id
    tokenizer = AutoTokenizer.from_pretrained(src, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        src,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return tokenizer, model


def format_prompt(system_prompt: str, history: List[Tuple[str, str]], user_msg: str, tokenizer) -> List[dict]:
    # Build chat markup for models supporting chat templates
    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for h_user, h_assistant in history:
        if h_user:
            messages.append({"role": "user", "content": h_user})
        if h_assistant:
            messages.append({"role": "assistant", "content": h_assistant})
    messages.append({"role": "user", "content": user_msg})

    if hasattr(tokenizer, "apply_chat_template"):
        # return string prompt ready for generation
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt_text
    else:
        # Fallback: simple concatenation
        sys = f"[SYSTEM]\n{system_prompt}\n\n" if system_prompt else ""
        chat_lines = []
        for u, a in history:
            if u:
                chat_lines.append(f"[USER] {u}")
            if a:
                chat_lines.append(f"[ASSISTANT] {a}")
        chat_lines.append(f"[USER] {user_msg}\n[ASSISTANT]")
        return sys + "\n".join(chat_lines)


def build_system_prompt(spanish_only: bool, rag_context: Optional[str]) -> str:
    parts = []
    if spanish_only:
        parts.append(
            "Eres un asistente técnico. RESPONDE SIEMPRE EN ESPAÑOL. "
            "Si la pregunta no está en español, tradúcela y responde en español."
        )
    else:
        parts.append("Eres un asistente técnico.")
    if rag_context:
        parts.append(
            "Usa únicamente la información del siguiente contexto cuando contestes; "
            "si falta información, dilo explícitamente.\n\n[Contexto]\n" + rag_context
        )
    return "\n\n".join(parts)


def generate(
    message: str,
    history: List[Tuple[str, str]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    rag_enabled: bool,
    retriever,
) -> str:
    # Build RAG context if enabled
    rag_context = None
    if rag_enabled and retriever is not None and message.strip():
        nodes = retriever.retrieve(message)
        snippets = []
        for n in nodes[:5]:
            text = n.get_text().strip()
            if len(text) > 600:
                text = text[:600].rstrip() + " …"
            meta = n.metadata or {}
            title = meta.get("title") or ""
            if title:
                snippets.append(f"[{title}]\n{text}")
            else:
                snippets.append(text)
        rag_context = "\n---\n".join(snippets)

    system_prompt = build_system_prompt(spanish_only=True, rag_context=rag_context)
    prompt = format_prompt(system_prompt, history, message, tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": max(temperature, 1e-6) if temperature > 0 else 1.0,
        "top_p": top_p,
        "top_k": top_k,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Simple cleanups
    output_text = output_text.strip()
    return output_text


def create_interface(tokenizer, model, rag_enabled: bool, retriever):
    def _respond(message, history):
        return generate(
            message=message,
            history=history or [],
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            rag_enabled=rag_enabled,
            retriever=retriever,
        )

    chat = gr.ChatInterface(
        fn=_respond,
        title="Chat RAG (Qwen 0.5B Instruct)",
        description=(
            "Modelo: Qwen/Qwen2.5-0.5B-Instruct. Responde en español. "
            + ("RAG activado: usa el índice LlamaIndex." if rag_enabled else "RAG desactivado: chat base.")
        ),
        examples=[
            "¿Qué es DLPNO-CCSD(T) en ORCA?",
            "Dame un input mínimo para DLPNO-CCSD(T)",
            "Diferencias entre DLPNO y LPNO en ORCA 6.1",
        ],
    )
    return chat


def main():
    parser = argparse.ArgumentParser(description="Gradio chat app con Qwen 0.5B y RAG opcional")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model id")
    parser.add_argument("--rag", action="store_true", help="Activar RAG con índice LlamaIndex")
    parser.add_argument("--persist", default=str(DEFAULT_PERSIST), help="Directorio del índice LlamaIndex")
    parser.add_argument("--embed-model", default="BAAI/bge-m3", help="Modelo de embeddings si RAG está activo")
    parser.add_argument("--models-dir", default=None, help="Directorio local de modelos/caché HF para modo offline")
    parser.add_argument("--offline", action="store_true", help="Forzar modo offline (HF_HUB_OFFLINE=1)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # Offline/cache
    set_offline_mode(Path(args.models_dir) if args.models_dir else None, args.offline)

    # Cargar modelo
    print(f"Cargando modelo: {args.model_id}")
    tokenizer, model = load_qwen(args.model_id)

    # RAG opcional
    retriever = None
    if args.rag:
        persist_dir = Path(args.persist).expanduser().resolve()
        if not persist_dir.exists():
            raise FileNotFoundError(f"Persist dir no encontrado: {persist_dir}")
        print(f"Cargando índice LlamaIndex desde: {persist_dir}")
        # Inyectar embeddings (debe coincidir con build-time). Permitir ruta local
        embed_src = str(Path(args.embed_model)) if os.path.isdir(args.embed_model) else args.embed_model
        embed_model = HuggingFaceEmbedding(
            model_name=embed_src,
            cache_folder=str(PROJECT_ROOT / ".cache"),
        )
        Settings.embed_model = embed_model
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(storage_context)
        retriever = index.as_retriever(similarity_top_k=8)

    # UI
    app = create_interface(tokenizer, model, rag_enabled=args.rag, retriever=retriever)
    app.queue().launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
