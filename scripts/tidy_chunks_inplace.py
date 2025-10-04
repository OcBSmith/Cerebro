import argparse
import json
import re
from datetime import datetime
from pathlib import Path

ARTIFACT_LINE_PATTERNS = [
    re.compile(r"^\s*<!--\s*image\s*-->\s*$", re.I),
    re.compile(r"^\s*<!--\s*formula-not-decoded\s*-->\s*$", re.I),
    re.compile(r"^(continues on next page|continued from previous page)\b.*$", re.I),
]

# Replace weird symbols and excessive spaces
REPLACEMENTS = [
    (re.compile(r"[\u2423\u21AA\u2192\u200B\u00A0]+"), " "),  # visible space/arrow/zwsp/nbsp
    (re.compile(r"\s{2,}"), " "),
]

# Heuristic ORCA splitter: introduce newlines before common markers
ORCA_BREAK_REGEX = re.compile(
    r"\s+(%[A-Za-z][A-Za-z0-9_]*|end\b|\*\s+xyz\b|VeryTightSCF|TightSCF|RIJCOSX|PAL\d*\b|PAL\b|def2\-[A-Za-z0-9\-]+|def2/[A-Za-z0-9\-]+|NROOTS|nroots|DT0L|DTol)\b",
    re.I,
)


def looks_like_orca_flat(s: str) -> bool:
    s0 = s.strip()
    if not s0:
        return False
    has_bang = s0.startswith("!")
    has_pct = "%" in s0
    has_geom = "* xyz" in s0.lower() or s0.lower().startswith("*xyz")
    long_single_line = ("\n" not in s0) and (len(s0) > 120)
    return (has_bang or has_pct or has_geom) and long_single_line


def reformat_orca_text(s: str) -> str:
    s = s.strip()
    # ensure newline before % sections, end, geometry marker, and flags
    s = ORCA_BREAK_REGEX.sub(lambda m: "\n" + m.group(1), s)
    # Geometry markers on their own lines
    s = s.replace(" * xyz", "\n* xyz")
    s = re.sub(r"\s+end\b", "\nend", s, flags=re.I)
    # If starts with '!' and first segment is long, break after it
    if s.startswith("!") and "\n" not in s.split("\n", 1)[0]:
        s = re.sub(r"^!(.+?)\s+", r"!\1\n", s, count=1)
    # Put closing '*' on its own line (best effort)
    s = s.replace(" * ", "\n* ")
    # Collapse multiple blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def clean_text(text: str) -> str:
    # Remove artifact lines
    lines = text.splitlines()
    kept = [ln for ln in lines if not any(p.search(ln) for p in ARTIFACT_LINE_PATTERNS)]
    s = "\n".join(kept)
    # Replace odd characters and normalize spaces
    for pat, repl in REPLACEMENTS:
        s = pat.sub(repl, s)
    # If we detect a flattened ORCA input, reformat it
    if looks_like_orca_flat(s):
        s = reformat_orca_text(s)
    return s.strip()


def process_chunks(chunks_path: Path, backup: bool = True) -> None:
    tmp_out = chunks_path.with_suffix(".jsonl.tmp")
    if backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunks_path.rename(chunks_path.with_suffix(f".jsonl.bak_{ts}"))
    in_f = chunks_path.with_suffix(f".jsonl.bak_{ts}") if backup else chunks_path

    total = 0
    changed = 0
    with in_f.open("r", encoding="utf-8") as fin, tmp_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = rec.get("text", "")
            cleaned = clean_text(text)
            if cleaned != text:
                changed += 1
                rec["text"] = cleaned
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_out.replace(chunks_path)
    print(f"Processed {total} chunks. Modified {changed}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tidy LlamaIndex chunks.jsonl in place")
    ap.add_argument("--chunks", default=str(Path("data/llamaindex/chunks.jsonl")), help="Path to chunks.jsonl")
    ap.add_argument("--no-backup", action="store_true", help="Do not create a .bak timestamped backup")
    args = ap.parse_args()

    path = Path(args.chunks).resolve()
    if not path.exists():
        raise SystemExit(f"Not found: {path}")
    process_chunks(path, backup=not args.no_backup)
