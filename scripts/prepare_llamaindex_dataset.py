import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "output" / "md_out" / "orca_manual_6_1_0_full.md"
DEFAULT_OUT = PROJECT_ROOT / "data" / "llamaindex" / "chunks.jsonl"


@dataclass
class Chunk:
    id: str
    text: str
    section_path: List[str]
    offset_char: int
    chunk_index: int


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def iter_blocks(md_text: str) -> Iterable[Tuple[str, int]]:
    """Yield paragraph blocks with their start offset in the full text.
    Splits by blank lines while preserving headings and code fences as separate blocks.
    """
    lines = md_text.splitlines(keepends=True)
    block: List[str] = []
    offset = 0
    block_start = 0
    in_code = False

    def flush():
        nonlocal block, block_start
        if block:
            yield ("".join(block), block_start)
            block = []

    i = 0
    while i < len(lines):
        line = lines[i]
        # Track code fences
        if line.strip().startswith("```"):
            if not in_code:
                # starting code block: flush current block
                yield from flush()
                in_code = True
                block_start = offset
                block.append(line)
            else:
                in_code = False
                block.append(line)
                yield from flush()
            offset += len(line)
            i += 1
            continue

        if in_code:
            block.append(line)
            offset += len(line)
            i += 1
            continue

        if line.strip() == "":
            # blank line -> block boundary
            offset += len(line)
            i += 1
            yield from flush()
            block_start = offset
            continue

        # headings should be their own block
        if HEADING_RE.match(line):
            yield from flush()
            block_start = offset
            block = [line]
            offset += len(line)
            i += 1
            yield from flush()
            block_start = offset
            continue

        # normal content line
        if not block:
            block_start = offset
        block.append(line)
        offset += len(line)
        i += 1

    # final flush
    if block:
        yield ("".join(block), block_start)


def build_chunks(md_text: str, target_chars: int, overlap_chars: int) -> List[Chunk]:
    """Greedy chunking: try to end on block boundaries, prefer to keep headings with following text.
    Tracks current heading stack to populate section_path metadata.
    """
    chunks: List[Chunk] = []
    current: List[str] = []
    current_len = 0
    current_offset = 0
    chunk_idx = 0
    section_stack: List[Tuple[int, str]] = []  # (level, title)

    def section_path() -> List[str]:
        return [t for _, t in section_stack]

    for block, start_off in iter_blocks(md_text):
        m = HEADING_RE.match(block.strip())
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            # Update heading stack
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, title))
            # Headings as separators: if current chunk large, emit
            if current_len >= target_chars * 0.8:
                chunks.append(
                    Chunk(
                        id=f"orca_{chunk_idx:05d}",
                        text="".join(current).strip(),
                        section_path=section_path(),
                        offset_char=current_offset,
                        chunk_index=chunk_idx,
                    )
                )
                chunk_idx += 1
                # start new chunk with heading
                current = [block]
                current_len = len(block)
                current_offset = start_off
            else:
                # keep heading in current (or start new)
                if not current:
                    current_offset = start_off
                current.append(block)
                current_len += len(block)
            continue

        # Non-heading block
        if not current:
            current_offset = start_off
        # If adding this block exceeds target, emit current chunk and start a new one with overlap
        if current_len + len(block) > target_chars and current:
            chunks.append(
                Chunk(
                    id=f"orca_{chunk_idx:05d}",
                    text="".join(current).strip(),
                    section_path=section_path(),
                    offset_char=current_offset,
                    chunk_index=chunk_idx,
                )
            )
            chunk_idx += 1
            # Overlap: take tail of current text
            tail = "".join(current)
            overlap_text = tail[-overlap_chars :]
            current = [overlap_text, block]
            current_len = len(overlap_text) + len(block)
            current_offset = max(0, start_off - len(overlap_text))
        else:
            current.append(block)
            current_len += len(block)

    if current:
        chunks.append(
            Chunk(
                id=f"orca_{chunk_idx:05d}",
                text="".join(current).strip(),
                section_path=section_path(),
                offset_char=current_offset,
                chunk_index=chunk_idx,
            )
        )

    return chunks


def write_jsonl(chunks: List[Chunk], out_path: Path, doc_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            rec = {
                "id": c.id,
                "text": c.text,
                "meta": {
                    "doc": doc_name,
                    "section_path": c.section_path,
                    "offset_char": c.offset_char,
                    "chunk_index": c.chunk_index,
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare LlamaIndex chunks from a Markdown file")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to input Markdown")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL path")
    parser.add_argument("--doc-name", default="orca_manual_6_1_0", help="Document name metadata")
    parser.add_argument("--target-chars", type=int, default=6000, help="Target characters per chunk (~1024 tokens)")
    parser.add_argument("--overlap", type=int, default=600, help="Character overlap between chunks (~10%)")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    md_text = in_path.read_text(encoding="utf-8")
    chunks = build_chunks(md_text, target_chars=args.target_chars, overlap_chars=args.overlap)
    write_jsonl(chunks, Path(args.out).expanduser().resolve(), args.doc_name)
    print(f"Wrote {len(chunks)} chunks -> {args.out}")


if __name__ == "__main__":
    main()
