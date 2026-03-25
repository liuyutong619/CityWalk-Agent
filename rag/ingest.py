"""Clean Xiaohongshu raw exports into retrieval-ready knowledge cards."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from glob import glob
from pathlib import Path
from typing import Any

try:
    from rag.schema import Engagement, IngestionStats, KnowledgeCard
except ModuleNotFoundError:  # Allows `python rag/ingest.py` from repo root.
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from rag.schema import Engagement, IngestionStats, KnowledgeCard


AUTH_GATE_PHRASES = (
    "为保护账号安全",
    "扫码验证身份",
    "小红书APP",
    "请使用已登录该账号的",
)

ROUTE_SEPARATORS = ("->", "-->", "→", "➡", "➜", "➔", "⟶")
ROUTE_PART_SPLIT_RE = re.compile(r"\s*(?:→|->|-->|➡|➜|➔|⟶|-|—|/)\s*")
BRACKET_POI_RE = re.compile(r"【([^】]{1,32})】")
NUMBERED_POI_RE = re.compile(
    r"(?:^|[\n。；;])\s*(?:\d+[.、]?|[0-9]️⃣)\s*(?:[^\w\s\u4e00-\u9fff]*\s*)?([^\n📍⏰🎫✨✅❌💡🚇▫️]{2,30})"
)
LABEL_POI_RE = re.compile(r"(?:地点|定位|导航|地址)[:：]\s*[\"“]?([^，。；;\n]+)")
QUOTED_NAME_RE = re.compile(r"[“\"]([^”\"]{2,20})[”\"]")
GO_TO_RE = re.compile(r"(?:^|[，。；;、\s])去(?!哪)([^，。！？!?\n]{2,24})")
HASHTAG_RE = re.compile(r"#([^\s#@]+)")
MENTION_RE = re.compile(r"@([^\s@]+)")
TRANSIT_SPLIT_RE = re.compile(r"\s*(?:地铁|(?:\d+(?:/\d+)?)号线|[A-Z]口)")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\u2600-\u27BF"
    "\u2B00-\u2BFF"
    "]+",
    flags=re.UNICODE,
)

ROUTE_HINT_KEYWORDS = (
    "路线",
    "步行",
    "骑行",
    "徒步",
    "citywalk",
    "漫步",
    "顺着",
    "沿着",
    "导航",
    "入口",
    "出口",
    "地铁",
    "公交",
    "码头",
    "全程",
    "分钟",
    "公里",
    "打车",
)

GENERIC_POI_TERMS = {
    "citywalk",
    "city walk",
    "攻略",
    "路线",
    "散步",
    "漫步",
    "打卡",
    "周末去哪儿",
    "旅游攻略",
    "武汉旅游攻略",
    "武汉citywalk",
}

LOCATION_SUFFIXES = (
    "街",
    "路",
    "巷",
    "里",
    "门",
    "楼",
    "桥",
    "寺",
    "山",
    "湖",
    "园",
    "公园",
    "景区",
    "绿道",
    "江滩",
    "江边",
    "码头",
    "村",
    "广场",
    "站",
    "塔",
    "馆",
    "中心",
    "遗址",
    "小卖部",
    "花圃",
    "溪涧",
    "古村",
    "栈道",
    "天街",
    "大堤",
)

ACTION_SPLIT_RE = re.compile(
    r"(?:喝|看|坐|吹|骑|逛|许|感受|体验|散步|拍|数|抄|买|打卡|徒步|吃|发呆|露营|野餐|出片|推荐|偶遇|下山|上山|直达)"
)
LABEL_STRIP_RE = re.compile(r"^(?:游玩路线|路线图|路线|打卡路线|行程|顺序|地点|定位|导航|地址)[:：]?\s*")
STRUCTURAL_SPLIT_RE = re.compile(r"\s*(?:地铁|公交|自驾|门票|开放|交通|游玩tips|周边联动|穿搭建议|个人体验|地图|Tips)[:：]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default="rag/raw_data/*.json",
        help="Glob pattern for raw crawl files. .rag.json files are ignored by default.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="rag/data/xiaohongshu_knowledge_cards.jsonl",
        help="Output JSONL path for cleaned knowledge cards.",
    )
    parser.add_argument(
        "--output-stats",
        default="rag/data/xiaohongshu_knowledge_cards.stats.json",
        help="Output JSON path for run statistics.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional limit for quick verification runs.",
    )
    return parser.parse_args()


def normalize_text(text: Any, *, preserve_newlines: bool = False) -> str:
    if text is None:
        return ""

    normalized = str(text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("\u200b", " ")
    normalized = normalized.replace("\ufeff", " ")
    normalized = normalized.replace("️", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    for separator in ROUTE_SEPARATORS:
        normalized = normalized.replace(separator, "→")

    normalized = HASHTAG_RE.sub(lambda match: f" {match.group(1)} ", normalized)
    normalized = MENTION_RE.sub(" ", normalized)
    normalized = EMOJI_RE.sub(" ", normalized)

    if preserve_newlines:
        normalized = "\n".join(WHITESPACE_RE.sub(" ", line).strip() for line in normalized.splitlines())
        normalized = MULTI_NEWLINE_RE.sub("\n\n", normalized)
        return normalized.strip()

    normalized = normalized.replace("\n", " ")
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value

    text = str(value).strip().replace(",", "")
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def iter_note_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("notes"), list):
        return [item for item in payload["notes"] if isinstance(item, dict)]
    if isinstance(payload.get("records"), list):
        return [item for item in payload["records"] if isinstance(item, dict)]
    return []


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[。！？!?；;\n])", text)
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def dedupe_keep_order(items: list[str], *, limit: int | None = None) -> list[str]:
    output: list[str] = []
    for item in items:
        value = item.strip()
        if not value or value in output:
            continue
        output.append(value)
        if limit is not None and len(output) >= limit:
            break
    return output


def cleanup_candidate_name(text: str) -> str:
    candidate = normalize_text(text)
    candidate = LABEL_STRIP_RE.sub("", candidate)
    candidate = re.sub(r"^(?:\d+[.、]?|[0-9]️⃣)\s*", "", candidate)
    candidate = TRANSIT_SPLIT_RE.split(candidate, maxsplit=1)[0]
    candidate = STRUCTURAL_SPLIT_RE.split(candidate, maxsplit=1)[0]
    candidate = ACTION_SPLIT_RE.split(candidate, maxsplit=1)[0]
    candidate = candidate.strip(" -~:：·,.，。！？!?/\\()（）[]【】")
    return candidate


def is_location_like(text: str) -> bool:
    lowered = text.lower()
    if lowered in GENERIC_POI_TERMS:
        return False
    if len(text) < 2 or len(text) > 20:
        return False
    if any(marker in text for marker in ("攻略", "推荐", "朋友", "舒服", "好玩", "好逛", "路线")):
        return False
    if any(text.endswith(suffix) for suffix in LOCATION_SUFFIXES):
        return True
    return len(text) <= 8


def is_generic_poi(text: str) -> bool:
    return not is_location_like(text)


def split_route_parts(text: str) -> list[str]:
    cleaned = LABEL_STRIP_RE.sub("", normalize_text(text))
    parts = [cleanup_candidate_name(part) for part in ROUTE_PART_SPLIT_RE.split(cleaned)]
    return [part for part in parts if is_location_like(part)]


def looks_like_route_sequence(text: str) -> bool:
    compact = normalize_text(text)
    if "→" in compact:
        return True
    hyphen_count = compact.count("-") + compact.count("—")
    if hyphen_count >= 2:
        return True
    if any(label in compact for label in ("游玩路线", "路线图", "打卡路线", "推荐路线")) and hyphen_count >= 1:
        return True
    return False


def extract_route_sequences(body: str) -> list[str]:
    sequences: list[str] = []
    for candidate in split_sentences(body):
        compact = normalize_text(candidate)
        if not looks_like_route_sequence(compact):
            continue
        parts = split_route_parts(compact)
        if len(parts) >= 2:
            sequences.append(" → ".join(parts))
    return dedupe_keep_order(sequences, limit=5)


def extract_labeled_pois(body: str) -> list[str]:
    results: list[str] = []
    for match in LABEL_POI_RE.finditer(body):
        candidate = cleanup_candidate_name(match.group(1))
        if is_location_like(candidate):
            results.append(candidate)

    for line in body.splitlines():
        compact = normalize_text(line)
        if "地铁" not in compact:
            continue
        for match in QUOTED_NAME_RE.finditer(compact):
            candidate = cleanup_candidate_name(match.group(1))
            if is_location_like(candidate):
                results.append(candidate)
    return results


def extract_go_to_pois(body: str) -> list[str]:
    results: list[str] = []
    for match in GO_TO_RE.finditer(body):
        candidate = cleanup_candidate_name(match.group(1))
        if is_location_like(candidate):
            results.append(candidate)
    return results


def extract_poi_names(title: str, body: str) -> list[str]:
    candidates: list[str] = []
    analysis_text = "\n".join(part for part in (title, body) if part)

    for match in BRACKET_POI_RE.finditer(analysis_text):
        candidates.append(cleanup_candidate_name(match.group(1)))

    for sequence in extract_route_sequences(body):
        candidates.extend(cleanup_candidate_name(part) for part in sequence.split("→"))

    for match in NUMBERED_POI_RE.finditer(body):
        candidates.append(cleanup_candidate_name(match.group(1)))

    candidates.extend(extract_labeled_pois(body))
    candidates.extend(extract_go_to_pois(body))

    cleaned = [value for value in candidates if value and not is_generic_poi(value)]
    return dedupe_keep_order(cleaned, limit=12)


def extract_route_hints(body: str) -> list[str]:
    hints: list[str] = []
    hints.extend(extract_route_sequences(body))

    for sentence in split_sentences(body):
        compact = normalize_text(sentence)
        if len(compact) < 8 or len(compact) > 120:
            continue
        if looks_like_route_sequence(compact):
            hints.append(compact)
            continue
        if re.search(r"\d+\s*(?:分钟|公里|km|m)", compact):
            hints.append(compact)
            continue
        if "从" in compact and "到" in compact:
            hints.append(compact)
            continue
        if any(keyword in compact for keyword in ROUTE_HINT_KEYWORDS):
            hints.append(compact)

    return dedupe_keep_order(hints, limit=8)


def strip_auth_gate_text(text: str) -> str:
    if any(phrase in text for phrase in AUTH_GATE_PHRASES):
        return ""
    return normalize_text(text, preserve_newlines=True)


def should_skip_card(title: str, text: str, poi_names: list[str], route_hints: list[str]) -> str | None:
    if not text:
        return "auth_gate"
    if not title:
        return "missing_title"
    if ("？" in title or "?" in title or "？" in text or "?" in text) and len(poi_names) < 2 and not route_hints:
        return "question_post"
    if len(text) < 12 and len(poi_names) < 2 and not route_hints:
        return "too_short"
    return None


def build_card(record: dict[str, Any], *, region: str) -> tuple[KnowledgeCard | None, str | None]:
    note_id = normalize_text(record.get("note_id"))
    title = normalize_text(record.get("title"))
    source_url = normalize_text(record.get("url"))

    if not note_id:
        return None, "missing_note_id"
    if not source_url:
        return None, "missing_source_url"

    raw_desc = normalize_text(record.get("desc"), preserve_newlines=True)
    cleaned_desc = strip_auth_gate_text(raw_desc)
    poi_names = extract_poi_names(title, cleaned_desc)
    route_hints = extract_route_hints(cleaned_desc)

    skip_reason = should_skip_card(title, cleaned_desc, poi_names, route_hints)
    if skip_reason:
        return None, skip_reason

    card = KnowledgeCard(
        note_id=note_id,
        title=title,
        text=cleaned_desc,
        source_url=source_url,
        poi_names=poi_names,
        route_hints=route_hints,
        engagement=Engagement(
            likes=to_int(record.get("liked_count")),
            collects=to_int(record.get("collected_count")),
            comments=to_int(record.get("comment_count")),
        ),
        regions=dedupe_keep_order([region]),
        keywords=dedupe_keep_order([normalize_text(record.get("keyword"))]),
    )
    return card, None


def merge_cards(existing: KnowledgeCard, candidate: KnowledgeCard) -> KnowledgeCard:
    text = existing.text if len(existing.text) >= len(candidate.text) else candidate.text
    title = existing.title if len(existing.title) >= len(candidate.title) else candidate.title
    source_url = existing.source_url or candidate.source_url

    return KnowledgeCard(
        note_id=existing.note_id,
        title=title,
        text=text,
        source_url=source_url,
        poi_names=dedupe_keep_order(existing.poi_names + candidate.poi_names, limit=12),
        route_hints=dedupe_keep_order(existing.route_hints + candidate.route_hints, limit=8),
        engagement=Engagement(
            likes=max(existing.engagement.likes, candidate.engagement.likes),
            collects=max(existing.engagement.collects, candidate.engagement.collects),
            comments=max(existing.engagement.comments, candidate.engagement.comments),
        ),
        regions=dedupe_keep_order(existing.regions + candidate.regions),
        keywords=dedupe_keep_order(existing.keywords + candidate.keywords),
    )


def collect_source_paths(pattern: str, limit_files: int | None) -> list[Path]:
    paths = [Path(path) for path in sorted(glob(pattern))]
    paths = [path for path in paths if not path.name.endswith(".rag.json")]
    if limit_files is not None:
        return paths[:limit_files]
    return paths


def sort_cards(cards: list[KnowledgeCard]) -> list[KnowledgeCard]:
    return sorted(
        cards,
        key=lambda card: (
            -card.engagement.collects,
            -card.engagement.likes,
            -card.engagement.comments,
            card.note_id,
        ),
    )


def write_jsonl(path: Path, cards: list[KnowledgeCard]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card.model_dump(), ensure_ascii=False) + "\n")


def write_stats(path: Path, stats: IngestionStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats.model_dump(), handle, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    source_paths = collect_source_paths(args.input_glob, args.limit_files)
    merged_cards: dict[str, KnowledgeCard] = {}
    skip_reasons: Counter[str] = Counter()
    raw_notes_seen = 0
    duplicates_merged = 0

    for path in source_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        region = normalize_text(payload.get("region") or payload.get("keyword") or path.stem)
        for record in iter_note_records(payload):
            raw_notes_seen += 1
            card, skip_reason = build_card(record, region=region)
            if card is None:
                if skip_reason:
                    skip_reasons[skip_reason] += 1
                continue

            existing = merged_cards.get(card.note_id)
            if existing is None:
                merged_cards[card.note_id] = card
                continue

            duplicates_merged += 1
            merged_cards[card.note_id] = merge_cards(existing, card)

    cards = sort_cards(list(merged_cards.values()))
    stats = IngestionStats(
        files_processed=len(source_paths),
        raw_notes_seen=raw_notes_seen,
        unique_cards_written=len(cards),
        duplicates_merged=duplicates_merged,
        skipped_notes=sum(skip_reasons.values()),
        skip_reasons=dict(sorted(skip_reasons.items())),
    )

    write_jsonl(Path(args.output_jsonl), cards)
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
