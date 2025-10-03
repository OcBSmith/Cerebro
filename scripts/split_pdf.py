import argparse
from pathlib import Path

try:
    from pypdf import PdfReader, PdfWriter
except Exception:
    raise SystemExit("Falta 'pypdf'. Instala con: .\\.venv\\Scripts\\python -m pip install pypdf")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_IN = PROJECT_ROOT / "data" / "pdf_in"
SPLITS_ROOT = PROJECT_ROOT / "output" / "pdf_splits"


def split_pdf(input_pdf: Path, pages_per_chunk: int, outdir: Path) -> None:
    reader = PdfReader(str(input_pdf))
    total = len(reader.pages)
    outdir.mkdir(parents=True, exist_ok=True)

    part = 1
    for start in range(0, total, pages_per_chunk):
        end = min(start + pages_per_chunk, total)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])
        out_path = outdir / f"{input_pdf.stem}_part{part:03d}.pdf"
        with open(out_path, "wb") as f:
            writer.write(f)
        print(f"Creado: {out_path} (páginas {start+1}-{end} de {total})")
        part += 1


def main():
    parser = argparse.ArgumentParser(description="Divide un PDF en trozos por número de páginas")
    parser.add_argument("input", help="Ruta al PDF de entrada (por defecto busca en data/pdf_in si das solo el nombre)")
    parser.add_argument("--pages", "-p", type=int, default=25, help="Páginas por trozo (default: 25)")
    parser.add_argument("--outdir", "-o", help="Carpeta de salida (default: output/pdf_splits/<nombre_pdf>/)")
    args = parser.parse_args()

    # Resolver entrada: si es nombre simple, mirar en data/pdf_in
    in_candidate = Path(args.input)
    if not in_candidate.suffix:
        in_candidate = in_candidate.with_suffix('.pdf')
    if not in_candidate.is_absolute():
        in_path = (DATA_IN / in_candidate).resolve()
    else:
        in_path = in_candidate.resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {in_path}")

    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        outdir = SPLITS_ROOT / in_path.stem

    split_pdf(in_path, args.pages, outdir)


if __name__ == "__main__":
    main()
