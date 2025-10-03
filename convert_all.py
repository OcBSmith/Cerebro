from pathlib import Path
from typing import List

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PipelineOptions, RapidOcrOptions

ROOT = Path(__file__).parent
PDF_IN = ROOT / "pdf_in"
MD_OUT = ROOT / "md_out"


def list_pdfs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.pdf") if p.is_file()])


def main() -> None:
    PDF_IN.mkdir(parents=True, exist_ok=True)
    MD_OUT.mkdir(parents=True, exist_ok=True)

    pdfs = list_pdfs(PDF_IN)
    if not pdfs:
        print(f"No se encontraron PDFs en: {PDF_IN}")
        return

    opts = PipelineOptions()
    opts.do_ocr = True
    opts.ocr_options = RapidOcrOptions()  # ONNX Runtime GPU si está disponible
    converter = DocumentConverter(pipeline_options=opts)

    ok = 0
    for pdf in pdfs:
        out_md = MD_OUT / (pdf.stem + ".md")
        try:
            print(f"[CONVIERTIENDO] {pdf.name} -> {out_md.name}")
            result = converter.convert(str(pdf))
            out_md.write_text(result.document.export_to_markdown(), encoding="utf-8")
            ok += 1
        except Exception as e:
            print(f"[ERROR] {pdf.name}: {e}")

    print(f"Listo. Convertidos: {ok}/{len(pdfs)}. Salida en: {MD_OUT}")


if __name__ == "__main__":
    main()
