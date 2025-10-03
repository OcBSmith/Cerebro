import argparse
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "md_out"


def convert_pdf(
    input_path: Path,
    output_path: Path,
    full_page_ocr: bool = False,
    ocr_backend: str = "rapidocr",
    tess_lang: str = "eng",
) -> None:
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = True
    if ocr_backend == "rapidocr":
        pdf_opts.ocr_options = RapidOcrOptions(force_full_page_ocr=full_page_ocr)
    elif ocr_backend == "tesseract_cli":
        pdf_opts.ocr_options = TesseractCliOcrOptions(
            force_full_page_ocr=full_page_ocr,
            lang=tess_lang,
        )
    else:
        raise ValueError(f"Unsupported ocr backend: {ocr_backend}")

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
        }
    )
    result = converter.convert(str(input_path))

    md = result.document.export_to_markdown()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"Wrote: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown with Docling + RapidOCR (GPU if available)")
    parser.add_argument("input", help="Path to input PDF")
    parser.add_argument("--out", "-o", help="Path to output Markdown. Defaults to <output/md_out>/<input_name>.md")
    parser.add_argument("--full-ocr", action="store_true", help="Force full-page OCR (RapidOCR)")
    parser.add_argument(
        "--ocr-backend",
        choices=["rapidocr", "tesseract_cli"],
        default="rapidocr",
        help="OCR backend to use (default: rapidocr)",
    )
    parser.add_argument(
        "--tess-lang",
        default="eng",
        help="Tesseract language code(s), e.g., 'eng', 'spa', 'eng+spa' (only for tesseract_cli)",
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")
    if in_path.suffix.lower() != ".pdf":
        raise ValueError("Input must be a PDF file")

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = DEFAULT_OUT_DIR / (in_path.stem + ".md")

    convert_pdf(
        in_path,
        out_path,
        full_page_ocr=args.full_ocr,
        ocr_backend=args.ocr_backend,
        tess_lang=args.tess_lang,
    )


if __name__ == "__main__":
    main()
