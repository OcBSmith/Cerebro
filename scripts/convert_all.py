import argparse
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
    parser = argparse.ArgumentParser(description="Convertir PDFs a Markdown con Docling (ajustes seguros)")
    parser.add_argument("--in", dest="inp", default=str(DATA_IN), help="Carpeta de entrada con PDFs")
    parser.add_argument("--out", dest="out", default=str(MD_OUT), help="Carpeta de salida para .md")
    # OCR y lenguaje
    parser.add_argument("--ocr", action="store_true", help="Forzar OCR (si procede)")
    parser.add_argument("--no-ocr", action="store_true", help="Desactivar OCR si el PDF tiene texto embebido")
    parser.add_argument("--lang", default=None, help="Idioma OCR, p.ej. 'es' o 'en'")
    # Limpieza estructural
    parser.add_argument("--remove-headers", action="store_true", help="Intentar eliminar cabeceras/pies")
    parser.add_argument("--merge-hyphens", action="store_true", help="Unir palabras cortadas por guion al final de línea")
    parser.add_argument("--keep-headings", action="store_true", help="Conservar encabezados detectados")
    parser.add_argument("--keep-lists", action="store_true", help="Conservar listas detectadas")
    # Tablas
    parser.add_argument("--tables-as-markdown", action="store_true", help="Exportar tablas como Markdown si es posible")
    parser.add_argument("--skip-tables", action="store_true", help="Intentar omitir tablas (p.ej. TOC ruidoso)")
    # Salida enriquecida
    parser.add_argument("--export-html", action="store_true", help="Exportar a HTML (conserva <img> y estructura si la versión lo soporta)")
    parser.add_argument("--assets-dir", default=None, help="Directorio para guardar assets (imágenes) si la versión lo permite")
    parser.add_argument("--keep-captions", action="store_true", help="Volcar captions/figuras al texto cuando sea posible")

    args = parser.parse_args()

    in_dir = Path(args.inp).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list_pdfs(in_dir)
    if not pdfs:
        print(f"No se encontraron PDFs en: {in_dir}")
        return

    pdf_opts = PdfPipelineOptions()
    ocr_opts = RapidOcrOptions()

    # Aplicación segura de opciones (solo si existen en la versión instalada)
    def set_safe(obj, attr, value):
        if hasattr(obj, attr):
            setattr(obj, attr, value)
            return True
        return False

    # OCR enable/disable y lenguaje
    if args.no_ocr:
        set_safe(ocr_opts, "enable", False)
    elif args.ocr:
        set_safe(ocr_opts, "enable", True)
    if args.lang:
        # Algunos backends usan 'lang' o 'language'
        if not set_safe(ocr_opts, "lang", args.lang):
            set_safe(ocr_opts, "language", args.lang)

    # Limpieza estructural
    if args.remove_headers:
        set_safe(pdf_opts, "remove_headers_footers", True)
    if args.merge_hyphens:
        set_safe(pdf_opts, "merge_hyphenated_words", True)
    if args.keep_headings:
        set_safe(pdf_opts, "keep_headings", True)
    if args.keep_lists:
        set_safe(pdf_opts, "keep_lists", True)

    # Tablas
    if args.tables_as_markdown:
        set_safe(pdf_opts, "tables_as_markdown", True)
    if args.skip_tables:
        # Algunas versiones usan flags tipo 'extract_tables' o 'enable_tables'
        if not set_safe(pdf_opts, "extract_tables", False):
            set_safe(pdf_opts, "enable_tables", False)

    # Asignar OCR al pipeline
    set_safe(pdf_opts, "ocr_options", ocr_opts)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
        }
    )

    # Log de opciones efectivas
    print("[Docling] Opciones aplicadas (según disponibilidad en tu versión):")
    for name in [
        "remove_headers_footers",
        "merge_hyphenated_words",
        "keep_headings",
        "keep_lists",
        "tables_as_markdown",
        "extract_tables",
        "enable_tables",
    ]:
        if hasattr(pdf_opts, name):
            print(f" - pdf_opts.{name} = {getattr(pdf_opts, name)}")
    for name in ["enable", "lang", "language"]:
        if hasattr(ocr_opts, name):
            print(f" - ocr_opts.{name} = {getattr(ocr_opts, name)}")

    ok = 0
    for pdf in pdfs:
        out_md = out_dir / (pdf.stem + (".html" if args.export_html else ".md"))
        try:
            rel = out_md.relative_to(PROJECT_ROOT)
        except Exception:
            rel = out_md
        try:
            print(f"[CONVIERTIENDO] {pdf.name} -> {rel}")
            result = converter.convert(str(pdf))
            doc = result.document
            # Guardar assets si existe API
            if args.assets_dir:
                assets_dir = Path(args.assets_dir).expanduser().resolve() / pdf.stem
                assets_dir.mkdir(parents=True, exist_ok=True)
                # Intenta métodos comunes para exportar assets
                for m in ("export_assets", "save_assets", "write_assets"):
                    if hasattr(doc, m):
                        try:
                            getattr(doc, m)(str(assets_dir))  # type: ignore
                            print(f"[ASSETS] Guardados en: {assets_dir}")
                            break
                        except Exception as _:
                            pass
            # Exportar contenido
            if args.export_html and hasattr(doc, "export_to_html"):
                html = doc.export_to_html()
                out_md.write_text(html, encoding="utf-8")
            else:
                text = doc.export_to_markdown()
                # Intento de conservar captions si la versión lo expone
                if args.keep_captions:
                    # Algunos documentos exponen figuras via doc.figures o doc.images
                    caps = []
                    for attr in ("figures", "images"):
                        if hasattr(doc, attr):
                            try:
                                items = getattr(doc, attr)
                                for it in items or []:
                                    cap = None
                                    for key in ("caption", "alt", "title"):
                                        if isinstance(it, dict) and key in it and it[key]:
                                            cap = str(it[key])
                                            break
                                        if hasattr(it, key) and getattr(it, key):
                                            cap = str(getattr(it, key))
                                            break
                                    if cap:
                                        caps.append(f"Figure: {cap}")
                            except Exception:
                                pass
                    if caps:
                        text = text + "\n\n" + "\n".join(caps)
                out_md.write_text(text, encoding="utf-8")
            ok += 1
        except Exception as e:
            print(f"[ERROR] {pdf.name}: {e}")

    print(f"Listo. Convertidos: {ok}/{len(pdfs)}. Salida en: {out_dir}")


if __name__ == "__main__":
    main()
