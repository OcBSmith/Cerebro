import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MD_OUT = PROJECT_ROOT / "output" / "md_out"


def concat_md(doc_folder: Path, out_file: Path) -> None:
    parts = sorted(doc_folder.glob("*.md"))
    if not parts:
        raise FileNotFoundError(f"No hay .md en {doc_folder}")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as fout:
        for p in parts:
            title = f"\n\n---\n# {p.stem}\n---\n\n"
            fout.write(title)
            fout.write(p.read_text(encoding="utf-8"))
    print(f"Generado: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Concatena todos los .md de output/md_out/<doc>/ en un único archivo")
    parser.add_argument("doc", help="Nombre de la carpeta en output/md_out (p. ej., orca_manual_6_1_0)")
    parser.add_argument("--out", "-o", help="Ruta de salida opcional (por defecto output/md_out/<doc>_full.md)")
    args = parser.parse_args()

    doc_dir = MD_OUT / args.doc
    if not doc_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {doc_dir}")

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = MD_OUT / f"{args.doc}_full.md"

    concat_md(doc_dir, out_path)


if __name__ == "__main__":
    main()
