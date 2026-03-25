"""Use an LLM to filter and enrich Xiaohongshu knowledge cards."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

try:
    from rag.schema import KnowledgeCard
except ModuleNotFoundError:  # Allows python rag/...py from repo root.
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from rag.schema import KnowledgeCard


load_dotenv()

DEFAULT_INPUT_JSONL = "rag/data/xiaohongshu_knowledge_cards.jsonl"
DEFAULT_OUTPUT_JSONL = "rag/data/xiaohongshu_knowledge_cards_llm.jsonl"
DEFAULT_REJECTED_JSONL = "rag/data/xiaohongshu_knowledge_cards_llm.rejected.jsonl"
DEFAULT_STATS_JSON = "rag/data/xiaohongshu_knowledge_cards_llm.stats.json"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"

SYSTEM_PROMPT = """
你是一个严格的数据清洗器，负责把小红书笔记转换成适合 citywalk RAG 的结构化卡片。

规则：
1. 先判断这条笔记是否与 citywalk 攻略直接相关。
2. 如果不相关，is_citywalk_relevant 设为 false，并给出一句简短原因。
3. 如果相关，提取干净的 poi_names、route_hints、regions、keywords。
4. poi_names 只能是地点、街区、街道、景点名称，不要放地铁口、广告词、情绪句。
5. route_hints 只能是路线顺序、步行提示、区域串联建议，不要放泛泛夸赞。
6. regions 只放城区、片区、街区。
7. keywords 放适合检索的关键词。
8. 只输出 JSON，不要输出 Markdown，不要输出解释。

输出 JSON 结构：
{
  "is_citywalk_relevant": true,
  "relevance_reason": "一句话原因",
  "poi_names": ["POI1"],
  "route_hints": ["路线提示"],
  "regions": ["区域"],
  "keywords": ["关键词"]
}
""".strip()

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
TRANSIT_FRAGMENT_RE = re.compile(r"(?:地铁|号线|[A-Za-z]口|出口|入口|公交|站台)")
SENTENCE_PUNCT_RE = re.compile(r"[，。！？；;：:]")
MULTISPACE_RE = re.compile(r"\s+")

GENERIC_POI_NAMES = {
    "citywalk",
    "city walk",
    "路线",
    "推荐路线",
    "旅游攻略",
    "citywalk路线",
    "武汉citywalk",
    "散步路线",
}
MARKETING_PHRASES = (
    "周末去哪儿",
    "拍照很出片",
    "特别适合",
    "氛围感",
    "值得一去",
    "太好逛",
    "好吃好逛",
)
ROUTE_HINT_SIGNALS = (
    "→",
    "->",
    "路线",
    "步行",
    "从",
    "到",
    "沿着",
    "入口",
    "出口",
    "全程",
    "分钟",
    "公里",
    "先去",
    "再去",
)
GENERIC_REGION_TERMS = {"武汉", "湖北", "中国", "旅游", "citywalk"}


class ExtractionResult(BaseModel):
    is_citywalk_relevant: bool
    relevance_reason: str
    poi_names: list[str] = Field(default_factory=list)
    route_hints: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class ProcessingStats(BaseModel):
    total_seen: int = 0
    processed: int = 0
    kept: int = 0
    rejected: int = 0
    parse_failures: int = 0
    skipped_existing: int = 0


@dataclass(slots=True)
class CardProcessingResult:
    status: str
    card: KnowledgeCard | None
    rejected_record: dict[str, Any] | None


Extractor = Callable[..., ExtractionResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--output-rejected-jsonl", default=DEFAULT_REJECTED_JSONL)
    parser.add_argument("--output-stats", default=DEFAULT_STATS_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _model_validate(model_cls: Any, payload: Any):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _normalize_text(value: str) -> str:
    normalized = MULTISPACE_RE.sub(" ", str(value or "").strip())
    return normalized.strip(" \t\r\n-~|/\\，。！？；;：:()（）[]【】\"'“”")


def _dedupe_keep_order(items: list[str], *, limit: int | None = None) -> list[str]:
    output: list[str] = []
    for raw in items:
        item = _normalize_text(raw)
        if not item or item in output:
            continue
        output.append(item)
        if limit is not None and len(output) >= limit:
            break
    return output


def cleanup_poi_names(items: list[str], *, limit: int = 12) -> list[str]:
    cleaned: list[str] = []
    for item in _dedupe_keep_order(items):
        lowered = item.lower()
        if len(item) < 2 or len(item) > 24:
            continue
        if item in GENERIC_POI_NAMES or lowered in GENERIC_POI_NAMES:
            continue
        if TRANSIT_FRAGMENT_RE.search(item):
            continue
        if SENTENCE_PUNCT_RE.search(item):
            continue
        cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


def cleanup_route_hints(items: list[str], *, limit: int = 8) -> list[str]:
    cleaned: list[str] = []
    for item in _dedupe_keep_order(items):
        if len(item) < 4 or len(item) > 100:
            continue
        has_route_signal = any(signal in item for signal in ROUTE_HINT_SIGNALS)
        if not has_route_signal and any(phrase in item for phrase in MARKETING_PHRASES):
            continue
        cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


def cleanup_regions(items: list[str], *, limit: int = 6) -> list[str]:
    cleaned: list[str] = []
    for item in _dedupe_keep_order(items):
        lowered = item.lower()
        if len(item) < 2 or len(item) > 20:
            continue
        if lowered in GENERIC_REGION_TERMS:
            continue
        if SENTENCE_PUNCT_RE.search(item):
            continue
        cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


def cleanup_keywords(items: list[str], *, limit: int = 10) -> list[str]:
    cleaned: list[str] = []
    for item in _dedupe_keep_order(items):
        if len(item) < 2 or len(item) > 24:
            continue
        cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


def parse_response_content(content: str) -> ExtractionResult:
    text = MARKDOWN_FENCE_RE.sub("", (content or "").strip())
    match = JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError("LLM response did not contain a JSON object")
    payload = json.loads(match.group(0))
    return _model_validate(ExtractionResult, payload)


def build_client():
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _build_messages(card: KnowledgeCard) -> list[dict[str, str]]:
    user_prompt = f"title: {card.title}\ntext: {card.text}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _extract_text_from_response(response: Any) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
        return "\n".join(part for part in parts if part)
    return str(content)


def extract_with_llm(
    card: KnowledgeCard,
    *,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
    retries: int = 2,
) -> ExtractionResult:
    llm_client = client or build_client()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = llm_client.chat.completions.create(
                model=model,
                temperature=0,
                messages=_build_messages(card),
            )
            return parse_response_content(_extract_text_from_response(response))
        except Exception as error:  # pragma: no cover - network path
            last_error = error
            if attempt >= retries:
                break
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"LLM extraction failed for note {card.note_id}: {last_error}")


def _serialize_json_line(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _build_rejected_record(card: KnowledgeCard, *, reason: str, status: str) -> dict[str, Any]:
    return {
        "note_id": card.note_id,
        "title": card.title,
        "source_url": card.source_url,
        "reason": reason,
        "status": status,
        "record": _model_dump(card),
    }


def process_card(card: KnowledgeCard, extraction: ExtractionResult) -> CardProcessingResult:
    if not extraction.is_citywalk_relevant:
        return CardProcessingResult(
            status="rejected",
            card=None,
            rejected_record=_build_rejected_record(
                card,
                reason=extraction.relevance_reason,
                status="irrelevant",
            ),
        )

    rewritten = KnowledgeCard(
        note_id=card.note_id,
        title=card.title,
        text=card.text,
        source_url=card.source_url,
        poi_names=cleanup_poi_names(extraction.poi_names),
        route_hints=cleanup_route_hints(extraction.route_hints),
        engagement=card.engagement,
        regions=cleanup_regions(extraction.regions),
        keywords=cleanup_keywords(extraction.keywords),
    )
    return CardProcessingResult(status="kept", card=rewritten, rejected_record=None)


def _iter_input_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield line


def _load_existing_note_ids(*paths: Path) -> set[str]:
    note_ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in _iter_input_lines(path):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            note_id = payload.get("note_id")
            if isinstance(note_id, str) and note_id:
                note_ids.add(note_id)
    return note_ids


def process_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    rejected_path: Path,
    stats_path: Path,
    extractor: Extractor = extract_with_llm,
    limit: int | None = None,
    skip_existing: bool = False,
    model: str = DEFAULT_MODEL,
) -> ProcessingStats:
    input_path = Path(input_path)
    output_path = Path(output_path)
    rejected_path = Path(rejected_path)
    stats_path = Path(stats_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    existing_note_ids = _load_existing_note_ids(output_path, rejected_path) if skip_existing else set()
    output_mode = "a" if skip_existing and output_path.exists() else "w"
    rejected_mode = "a" if skip_existing and rejected_path.exists() else "w"
    stats = ProcessingStats()

    with output_path.open(output_mode, encoding="utf-8") as output_handle, rejected_path.open(rejected_mode, encoding="utf-8") as rejected_handle:
        for raw_line in _iter_input_lines(input_path):
            if limit is not None and stats.total_seen >= limit:
                break
            stats.total_seen += 1
            payload = json.loads(raw_line)
            card = _model_validate(KnowledgeCard, payload)

            if card.note_id in existing_note_ids:
                stats.skipped_existing += 1
                continue

            try:
                extraction = extractor(card, model=model)
            except Exception as error:
                stats.parse_failures += 1
                rejected_handle.write(
                    _serialize_json_line(
                        _build_rejected_record(card, reason=str(error), status="error")
                    )
                )
                existing_note_ids.add(card.note_id)
                continue

            stats.processed += 1
            result = process_card(card, extraction)
            if result.status == "kept" and result.card is not None:
                output_handle.write(_serialize_json_line(_model_dump(result.card)))
                stats.kept += 1
            elif result.rejected_record is not None:
                rejected_handle.write(_serialize_json_line(result.rejected_record))
                stats.rejected += 1
            existing_note_ids.add(card.note_id)

    stats_path.write_text(
        json.dumps(_model_dump(stats), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def main() -> None:
    args = parse_args()
    process_jsonl(
        input_path=Path(args.input_jsonl),
        output_path=Path(args.output_jsonl),
        rejected_path=Path(args.output_rejected_jsonl),
        stats_path=Path(args.output_stats),
        limit=args.limit,
        skip_existing=args.skip_existing,
        model=args.model,
    )


if __name__ == "__main__":
    main()
