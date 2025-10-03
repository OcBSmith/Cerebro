from pathlib import Path
from typing import List

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_ROOT = PROJECT_ROOT / "output" / "pdf_splits"  # cada subcarpeta tiene trozos *_partNNN.pdf
MD_OUT = PROJECT_ROOT / "output" / "md_out"


def find_split_folders(base: Path) -> List[Path]:
    if not base.exists():
        return []
    return [p for p in base.iterdir() if p.is_dir()]


def list_pdfs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.pdf") if p.is_file()])


def main() -> None:
    MD_OUT.mkdir(parents=True, exist_ok=True)
    split_folders = find_split_folders(SPLITS_ROOT)

    if not split_folders:
        print(f"No se encontraron subcarpetas en: {SPLITS_ROOT}. Primero usa scripts/split_pdf.py")
        return

    pdf_opts = PdfPipelineOptions()
    pdf_opts.ocr_options = RapidOcrOptions()
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
        }
    )

    total_ok = 0
    total_files = 0

    for fold in split_folders:
        pdfs = list_pdfs(fold)
        if not pdfs:
            continue
        # carpeta de salida para este documento
        doc_out = MD_OUT / fold.name
        doc_out.mkdir(parents=True, exist_ok=True)

        for pdf in pdfs:
            total_files += 1
            out_md = doc_out / (pdf.stem + ".md")
            try:
                print(f"[CONVIERTIENDO] {fold.name} / {pdf.name} -> {out_md.relative_to(PROJECT_ROOT)}")
                result = converter.convert(str(pdf))
                out_md.write_text(result.document.export_to_markdown(), encoding="utf-8")
                total_ok += 1
            except Exception as e:
                print(f"[ERROR] {fold.name} / {pdf.name}: {e}")

    print(f"Listo. Convertidos: {total_ok}/{total_files}. Salidas en: {MD_OUT}")


if __name__ == "__main__":
    main()
