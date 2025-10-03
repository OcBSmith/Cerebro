from pathlib import Path
from typing import List

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_IN = PROJECT_ROOT / "data" / "pdf_in"
MD_OUT = PROJECT_ROOT / "output" / "md_out"


def list_pdfs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.pdf") if p.is_file()])


def main() -> None:
    DATA_IN.mkdir(parents=True, exist_ok=True)
    MD_OUT.mkdir(parents=True, exist_ok=True)

    pdfs = list_pdfs(DATA_IN)
    if not pdfs:
        print(f"No se encontraron PDFs en: {DATA_IN}")
        return

    pdf_opts = PdfPipelineOptions()
    pdf_opts.ocr_options = RapidOcrOptions()  # ONNX Runtime GPU si está disponible
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
        }
    )

    ok = 0
    for pdf in pdfs:
        out_md = MD_OUT / (pdf.stem + ".md")
        try:
            print(f"[CONVIERTIENDO] {pdf.name} -> {out_md.relative_to(PROJECT_ROOT)}")
            result = converter.convert(str(pdf))
            out_md.write_text(result.document.export_to_markdown(), encoding="utf-8")
            ok += 1
        except Exception as e:
            print(f"[ERROR] {pdf.name}: {e}")

    print(f"Listo. Convertidos: {ok}/{len(pdfs)}. Salida en: {MD_OUT}")


if __name__ == "__main__":
    main()
