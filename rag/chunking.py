"""Generate embedding-ready body chunks from LLM-filtered Xiaohongshu cards."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

try:
    from rag.schema import ChunkRecord, ChunkingStats, KnowledgeCard
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from rag.schema import ChunkRecord, ChunkingStats, KnowledgeCard


SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{2,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        default="rag/data/xiaohongshu_knowledge_cards_llm.jsonl",
        help="Input JSONL file containing LLM-filtered knowledge cards.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="rag/data/xiaohongshu_chunks.jsonl",
        help="Output JSONL file for embedding-ready chunks.",
    )
    parser.add_argument(
        "--output-stats",
        default="rag/data/xiaohongshu_chunks.stats.json",
        help="Output JSON file for chunking statistics.",
    )
    parser.add_argument(
        "--target-chars",
        type=int,
        default=280,
        help="Target character length for each body chunk.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=360,
        help="Hard maximum character length for each body chunk.",
    )
    parser.add_argument(
        "--overlap-sentences",
        type=int,
        default=0,
        help="How many trailing sentence units to overlap between chunks.",
    )
    parser.add_argument(
        "--limit-cards",
        type=int,
        default=None,
        help="Optional card limit for quick verification runs.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    normalized = str(text or "")
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("\u200b", " ")
    normalized = normalized.replace("\ufeff", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(WHITESPACE_RE.sub(" ", line).strip() for line in normalized.splitlines())
    normalized = MULTI_NEWLINE_RE.sub("\n", normalized)
    return normalized.strip()


def dedupe_keep_order(items: Iterable[str], *, limit: int | None = None) -> list[str]:
    output: list[str] = []
    for item in items:
        value = normalize_text(item)
        if not value or value in output:
            continue
        output.append(value)
        if limit is not None and len(output) >= limit:
            break
    return output


def engagement_score(card: KnowledgeCard) -> int:
    engagement = card.engagement
    return engagement.collects * 3 + engagement.likes + engagement.comments * 2


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    pieces = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(normalized) if piece and piece.strip()]
    return pieces or [normalized]


def split_long_unit(text: str, *, max_chars: int, overlap_chars: int = 40) -> list[str]:
    cleaned = normalize_text(text)
    if len(cleaned) <= max_chars:
        return [cleaned]

    step = max(1, max_chars - overlap_chars)
    units: list[str] = []
    start = 0
    while start < len(cleaned):
        window = cleaned[start : start + max_chars].strip()
        if window:
            units.append(window)
        if start + max_chars >= len(cleaned):
            break
        start += step
    return units


def build_body_chunks(
    text: str,
    *,
    target_chars: int,
    max_chars: int,
    overlap_sentences: int,
) -> list[str]:
    sentence_units: list[str] = []
    for sentence in split_sentences(text):
        sentence_units.extend(split_long_unit(sentence, max_chars=max_chars))

    if not sentence_units:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(sentence_units):
        current_units: list[str] = []
        current_len = 0
        index = start

        while index < len(sentence_units):
            unit = sentence_units[index]
            proposed_len = current_len + len(unit) + (1 if current_units else 0)
            if current_units and proposed_len > max_chars:
                break
            current_units.append(unit)
            current_len = proposed_len
            index += 1
            if current_len >= target_chars:
                break

        if not current_units:
            current_units = [sentence_units[start]]
            index = start + 1

        chunk = normalize_text(" ".join(current_units))
        if chunk:
            chunks.append(chunk)

        if index >= len(sentence_units):
            break

        start = max(start + 1, index - overlap_sentences)

    return chunks


def build_header(card: KnowledgeCard) -> str:
    lines = [f"标题: {normalize_text(card.title)}"]
    if card.regions:
        lines.append(f"区域: {'，'.join(dedupe_keep_order(card.regions, limit=6))}")
    if card.keywords:
        lines.append(f"关键词: {'，'.join(dedupe_keep_order(card.keywords, limit=8))}")
    if card.poi_names:
        lines.append(f"POI: {'，'.join(dedupe_keep_order(card.poi_names, limit=12))}")
    if card.route_hints:
        lines.append(f"路线提示: {' | '.join(dedupe_keep_order(card.route_hints, limit=4))}")
    return "\n".join(lines)


def build_embedding_text(card: KnowledgeCard, chunk_text: str) -> str:
    header_text = build_header(card)
    return f"{header_text}\n正文片段: {normalize_text(chunk_text)}"


def iter_cards(path: Path, limit_cards: int | None) -> Iterable[KnowledgeCard]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit_cards is not None and index >= limit_cards:
                break
            if not line.strip():
                continue
            yield KnowledgeCard.model_validate_json(line)


def write_jsonl(path: Path, records: Iterable[ChunkRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_stats(path: Path, stats: ChunkingStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats.model_dump(), handle, ensure_ascii=False, indent=2)


def chunk_card(
    card: KnowledgeCard,
    *,
    target_chars: int,
    max_chars: int,
    overlap_sentences: int,
) -> list[ChunkRecord]:
    body_chunks = build_body_chunks(
        card.text,
        target_chars=target_chars,
        max_chars=max_chars,
        overlap_sentences=overlap_sentences,
    )

    if not body_chunks:
        return []

    header_text = build_header(card)
    score = engagement_score(card)
    records: list[ChunkRecord] = []
    for chunk_index, chunk_text in enumerate(body_chunks):
        record = ChunkRecord(
            chunk_id=f"{card.note_id}:body:{chunk_index:03d}",
            note_id=card.note_id,
            chunk_type="body",
            chunk_index=chunk_index,
            title=card.title,
            source_url=card.source_url,
            chunk_text=chunk_text,
            header_text=header_text,
            embedding_text=f"{header_text}\n正文片段: {normalize_text(chunk_text)}",
            poi_names=dedupe_keep_order(card.poi_names, limit=12),
            route_hints=dedupe_keep_order(card.route_hints, limit=4),
            engagement=card.engagement,
            engagement_score=score,
            regions=dedupe_keep_order(card.regions, limit=6),
            keywords=dedupe_keep_order(card.keywords, limit=8),
            parent_text_length=len(normalize_text(card.text)),
            chunk_text_length=len(chunk_text),
        )
        records.append(record)
    return records


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl)

    all_records: list[ChunkRecord] = []
    cards_processed = 0
    max_chunk_chars = 0
    total_chunk_chars = 0

    for card in iter_cards(input_path, args.limit_cards):
        cards_processed += 1
        records = chunk_card(
            card,
            target_chars=args.target_chars,
            max_chars=args.max_chars,
            overlap_sentences=args.overlap_sentences,
        )
        all_records.extend(records)
        for record in records:
            total_chunk_chars += record.chunk_text_length
            max_chunk_chars = max(max_chunk_chars, record.chunk_text_length)

    chunks_written = write_jsonl(Path(args.output_jsonl), all_records)
    stats = ChunkingStats(
        cards_processed=cards_processed,
        chunks_written=chunks_written,
        average_chunk_chars=(total_chunk_chars / chunks_written) if chunks_written else 0.0,
        max_chunk_chars=max_chunk_chars,
    )
    write_stats(Path(args.output_stats), stats)

    print(
        json.dumps(
            {
                "output_jsonl": args.output_jsonl,
                "output_stats": args.output_stats,
                **stats.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
