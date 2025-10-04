import argparse
import json
import os
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional

# -------- Helpers for HTML to text (stdlib only) ---------
class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []
        self._in_script_style = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._in_script_style = True
        if tag in ("p", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5"):
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._in_script_style = False
        if tag in ("p", "li", "tr"):
            self._chunks.append("\n")

    def handle_data(self, data):
        if self._in_script_style:
            return
        self._chunks.append(data)

    def get_text(self) -> str:
        s = "".join(self._chunks)
        # Normalize spaces and newlines
        s = re.sub(r"\r\n?|\u00A0", "\n", s)
        s = re.sub(r"[ \t\f\v]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


# --------- Figure harvesting (very light, stdlib) ----------
class _FigureHarvester(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_figure = False
        self.in_figcaption = False
        self.current_src: Optional[str] = None
        self.current_caption_chunks: List[str] = []
        self.items: List[dict] = []

    def handle_starttag(self, tag, attrs):
        if tag == "figure":
            self.in_figure = True
            self.current_src = None
            self.current_caption_chunks = []
        if self.in_figure and tag == "img":
            for k, v in attrs:
                if k == "src":
                    self.current_src = v
        if self.in_figure and tag == "figcaption":
            self.in_figcaption = True
        if self.in_figure and tag in ("br",):
            self.current_caption_chunks.append("\n")

    def handle_endtag(self, tag):
        if tag == "figcaption":
            self.in_figcaption = False
        if tag == "figure":
            cap = "".join(self.current_caption_chunks).strip()
            self.items.append({"src": self.current_src, "caption": cap})
            self.in_figure = False
            self.current_src = None
            self.current_caption_chunks = []

    def handle_data(self, data):
        if self.in_figure and self.in_figcaption:
            self.current_caption_chunks.append(data)


def harvest_figures(html: str) -> List[dict]:
    h = _FigureHarvester()
    h.feed(html)
    # remove empties
    return [it for it in h.items if (it.get("caption") or it.get("src"))]


# --------- Chunking ----------
@dataclass
class ChunkerCfg:
    max_chars: int = 2500
    overlap: int = 400


def split_into_chunks(text: str, cfg: ChunkerCfg) -> Iterable[str]:
    """Greedy char-based chunking with overlap; preserves paragraph boundaries when possible."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    buf: List[str] = []
    cur = 0
    for p in paras:
        if cur + len(p) + 2 <= cfg.max_chars or not buf:
            buf.append(p)
            cur += len(p) + 2
        else:
            yield "\n\n".join(buf)
            # overlap by taking tail of last buffer near overlap size
            joined = "\n\n".join(buf)
            tail = joined[-cfg.overlap :] if cfg.overlap < len(joined) else joined
            buf = [tail, p]
            cur = len(tail) + 2 + len(p)
    if buf:
        yield "\n\n".join(buf)


# --------- Main pipeline ----------

def process_file(path: Path, assets_root: Optional[Path], cfg: ChunkerCfg, out, base_doc: str):
    text: str = ""
    figs: List[dict] = []
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".html":
        text = html_to_text(raw)
        figs = harvest_figures(raw)
    else:
        # Markdown: take as-is
        text = raw
    # Write text chunks
    rel = path.as_posix()
    idx = 0
    for chunk in split_into_chunks(text, cfg):
        rec = {
            "id": f"{path.stem}_{idx:05d}",
            "text": chunk.strip(),
            "meta": {
                "doc": base_doc,
                "source_path": rel,
            },
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        idx += 1
    # Write figure items as mini-chunks (caption + asset path)
    for item in figs:
        cap = (item.get("caption") or "").strip()
        src = item.get("src")
        asset_path = None
        if src and assets_root is not None:
            # try to resolve relative to provided assets root
            asset_path = (assets_root / src).as_posix()
        fig_text = (f"Figure: {cap}" if cap else "Figure").strip()
        if src:
            fig_text += f"\nAsset: {src}"
        rec = {
            "id": f"{path.stem}_fig_{idx:05d}",
            "text": fig_text,
            "meta": {
                "doc": base_doc,
                "source_path": rel,
                "asset_path": asset_path or src,
                "type": "figure",
            },
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        idx += 1


def main():
    ap = argparse.ArgumentParser(description="Build chunks.jsonl from converted MD/HTML (with captions)")
    ap.add_argument("--src", default=str(Path("output/md_out")), help="Root folder with converted docs")
    ap.add_argument("--out", default=str(Path("data/llamaindex/chunks.jsonl")), help="Output chunks.jsonl path")
    ap.add_argument("--assets", default=str(Path("output/assets")), help="Root of exported assets (images)")
    ap.add_argument("--max-chars", type=int, default=2500)
    ap.add_argument("--overlap", type=int, default=400)
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out_path = Path(args.out).resolve()
    assets_root = Path(args.assets).resolve() if args.assets else None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = ChunkerCfg(max_chars=args.max_chars, overlap=args.overlap)

    # Backup existing chunks
    if out_path.exists():
        bkp = out_path.with_suffix(".jsonl.bak")
        try:
            out_path.replace(bkp)
            print(f"[BACKUP] {bkp}")
        except Exception:
            pass

    # Walk src and process .html/.md
    files: List[Path] = []
    for p in src.rglob("*"):
        if p.suffix.lower() in (".html", ".md") and p.is_file():
            files.append(p)
    files.sort()

    total_files = 0
    total_chunks = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for f in files:
            base_doc = f.parent.name if f.parent != src else f.stem
            process_file(f, assets_root, cfg, fout, base_doc=base_doc)
            total_files += 1

    print(f"[DONE] Processed {total_files} files -> {out_path}")


if __name__ == "__main__":
    main()
