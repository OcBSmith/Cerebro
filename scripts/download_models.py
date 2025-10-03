import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "BAAI/bge-m3",
]

def main():
    parser = argparse.ArgumentParser(description="Descargar modelos a una carpeta local para modo offline")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Lista de HF model IDs")
    parser.add_argument("--out", required=True, help="Directorio destino (cache local)")
    args = parser.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    for mid in args.models:
        print(f"Descargando: {mid}")
        snapshot_download(repo_id=mid, local_dir=out / mid.replace("/", "__"), local_dir_use_symlinks=False)
        print(f"OK: {mid}")

    print(f"Hecho. Modelos en: {out}")

if __name__ == "__main__":
    main()
