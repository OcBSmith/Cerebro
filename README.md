# Cerebro

Guía rápida para preparar documentos, construir el índice RAG y lanzar el chat.

## Requisitos
- Python 3.10+
- Entorno: `.venv` activado
- Modelos en caché: `models_cache/` (opcional si trabajas offline)

## 1) Dividir PDFs grandes en trozos
Usa `scripts/split_pdf.py` para cortar un PDF en partes de N páginas.

Ejemplo (20 páginas por parte):
```powershell
\.venv\Scripts\python .\scripts\split_pdf.py "orca_manual_6_1_0.pdf" --pages 20 --outdir ".\output\pdf_splits\orca_manual_6_1_0"
```
Notas:
- El script antepone `data/pdf_in/` al nombre; pasa solo el nombre del archivo.
- Salida: `output/pdf_splits/orca_manual_6_1_0/*.pdf`

## 2) Convertir PDFs a Markdown (Docling con flags de calidad)
`scripts/convert_all.py` admite flags seguros (solo aplica los que soporte tu versión de Docling):

Recomendado para manuales con TOC/tablas ruidosas:
```powershell
\.venv\Scripts\python .\scripts\convert_all.py --in ".\output\pdf_splits\orca_manual_6_1_0" --out ".\output\md_out\orca_manual_6_1_0" \
  --remove-headers --merge-hyphens --keep-headings --keep-lists --skip-tables --lang "en" --ocr
```
Alternativa con tablas en Markdown:
```powershell
\.venv\Scripts\python .\scripts\convert_all.py --in ".\output\pdf_splits\orca_manual_6_1_0" --out ".\output\md_out\orca_manual_6_1_0" \
  --remove-headers --merge-hyphens --keep-headings --keep-lists --tables-as-markdown --lang "en" --ocr
```

### 2.5) Generar chunks desde HTML/MD (con captions)
Construye `data/llamaindex/chunks.jsonl` a partir de `output/md_out/` e incluye mini‑chunks de figuras con captions y ruta del asset.

```powershell
\.venv\Scripts\python .\scripts\make_chunks_from_docs.py \
  --src ".\output\md_out" \
  --assets ".\output\assets" \
  --out ".\data\llamaindex\chunks.jsonl" \
  --max-chars 2500 --overlap 400
```

- Hace backup automático de `chunks.jsonl` existente como `chunks.jsonl.bak`.
- En los nuevos registros verás `meta.source_path` y, para figuras, `meta.asset_path` y `meta.type = "figure"`.

## 3) Limpiar chunks (sin tocar el Markdown)
Para mejorar inputs ORCA y eliminar artefactos, limpia `chunks.jsonl` in-place:
```powershell
\.venv\Scripts\python .\scripts\tidy_chunks_inplace.py --chunks ".\data\llamaindex\chunks.jsonl"
```
Imprime: `Processed N chunks. Modified M.`

### Notas de calidad (chunks generados)
- **`meta.source_path`**: presente en los nuevos registros (ej. `output/md_out/.../parte.html`).
- **Figuras (captions)**: hay mini‑chunks con `meta.type = "figure"` y el texto comienza por `Figure: ...`.
- **`asset_path`**: en muchos casos aparece `null` porque el HTML no incluye `<img src>` dentro de `<figure>`; solo hay `<figcaption>`. Aun así, la semántica de la figura (caption) ya es indexable.
- **Mejoras opcionales**: si quieres enlazar el binario de la imagen en metadatos, podemos ampliar el recolector para:
  - buscar `<img>` adyacentes (fuera de `<figure>`), o
  - mapear rutas relativas del HTML a `output/assets/` por nombre de archivo.

## 4) Reconstruir el índice (e5-small-v2)
IMPORTANT: El modelo de embeddings del índice debe coincidir con el usado en el chat.

```powershell
\.venv\Scripts\python .\scripts\build_llamaindex_index.py --persist ".\data\llamaindex\storage" --embed-model "intfloat/e5-small-v2"
```
Verifica el log: `Loading embedding model: intfloat/e5-small-v2`.

## 5) Lanzar el chat
```powershell
$env:CHAT_MODEL_ID="microsoft/Phi-3.5-mini-instruct"
$env:EMBED_MODEL_ID="intfloat/e5-small-v2"
$env:MODELS_DIR="$PWD\models_cache"
$env:HF_OFFLINE="1"
$env:CHAT_PORT="7861"
\.venv\Scripts\python .\scripts\run_chat_rag.py
```
Abre: http://127.0.0.1:%CHAT_PORT%

## Solución de problemas
- **Error dimensiones (384 vs 1024):** índice construido con `BAAI/bge-m3` (1024) y chat usando `e5-small-v2` (384). Reconstruye con el `--embed-model` correcto (paso 4).
- **`manifest.json 404`:** inocuo en Gradio; ignóralo.
- **Gemma “System role not supported”:** el código pliega el system prompt automáticamente; usa concatenación si falla la plantilla.
- **Inputs ORCA en una línea:** el postprocesado y `tidy_chunks_inplace.py` fuerzan multi‑línea; si ves un caso que no parte, captura el texto y ajustamos reglas.

## Directorios
- `data/pdf_in/`: PDFs fuente
- `output/pdf_splits/<doc>/`: PDFs divididos

## Conversión recomendada (ejecutada)
Para máxima preservación (tablas, estructura, imágenes y captions) sobre los splits ya creados:

```powershell
\.venv\Scripts\python .\scripts\convert_all.py --in ".\output\pdf_splits\orca_manual_6_1_0" --out ".\output\md_out\orca_manual_6_1_0" \
  --remove-headers --merge-hyphens --keep-headings --keep-lists --tables-as-markdown --lang "en" --ocr \
  --export-html --assets-dir ".\output\assets" --keep-captions
```

### Qué hace cada flag
- **--remove-headers**: intenta eliminar cabeceras/pies repetidos.
- **--merge-hyphens**: une palabras cortadas por guion al final de línea.
- **--keep-headings/--keep-lists**: conserva jerarquía y listas para mejor chunking/recall.
- **--tables-as-markdown**: vierte tablas vectoriales como Markdown (si no las necesitas, usar `--skip-tables`).
- **--lang "en" --ocr**: activa OCR con idioma (útil en páginas escaneadas).
- **--export-html**: exporta a HTML manteniendo `<img>` y estructura.
- **--assets-dir**: guarda imágenes en una carpeta y las referencia desde la salida.
- **--keep-captions**: añade captions/alt de figuras al texto (aportan al RAG).

## Siguientes pasos
- Reconstruir índice con `e5-small-v2` y lanzar el chat (ver secciones 4 y 5).
- Si alguna tabla queda como imagen, se conserva el asset; podemos aplicar extracción específica en esas páginas si es necesario.
- `output/md_out/`: Markdown generado
- `data/llamaindex/chunks.jsonl`: fragmentos para el índice
- `data/llamaindex/storage/`: índice persistido

## Modelos recomendados (ligeros)
- Generación: `microsoft/Phi-3.5-mini-instruct` o `Qwen/Qwen2.5-1.5B-Instruct`
- Embeddings: `intfloat/e5-small-v2` (Apache-2.0)