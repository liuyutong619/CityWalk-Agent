"""Retrieve Xiaohongshu CityWalk chunks from a local or remote Qdrant collection."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from openai import APIConnectionError, NotFoundError, OpenAI

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
DEFAULT_COLLECTION = "xiaohongshu_citywalk_example"
DEFAULT_QDRANT_PATH = "rag/data/qdrant_local_example"
DEFAULT_NOTES_JSONL = "rag/data/example_cards_llm.jsonl"


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        # Prefer explicit values from project .env to avoid stale shell exports.
        os.environ[key.strip()] = value.strip()


def _model_copy(model: Any, **updates: Any):
    if hasattr(model, "model_copy"):
        return model.model_copy(update=updates)
    payload = model.dict()
    payload.update(updates)
    return model.__class__(**payload)


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _model_validate(model_class: Any, payload: dict[str, Any]):
    if hasattr(model_class, "model_validate"):
        return model_class.model_validate(payload)
    return model_class.parse_obj(payload)


def _first_present_env(*keys: str) -> tuple[str | None, str | None]:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value, key
    return None, None


def _resolve_embedding_client_config() -> tuple[dict[str, str], dict[str, str]]:
    api_key, api_key_source = _first_present_env(
        "RAG_EMBEDDING_API_KEY",
        "RAG_EMBEDDING_KEY",
        "OPENAI_API_KEY",
        "OPENAI_APIKEY",
    )
    if not api_key:
        raise RuntimeError(
            "RAG_EMBEDDING_API_KEY or OPENAI_API_KEY is required for embeddings."
        )

    base_url, base_url_source = _first_present_env(
        "RAG_EMBEDDING_BASE_URL",
        "RAG_EMBEDDING_API_BASE",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    )
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return client_kwargs, {
        "api_key_source": api_key_source or "unknown",
        "base_url": base_url or "default OpenAI base_url",
        "base_url_source": base_url_source or "default",
    }


def _resolve_rerank_client_config() -> tuple[dict[str, str], dict[str, str]]:
    api_key, api_key_source = _first_present_env(
        "RAG_RERANK_API_KEY",
        "RAG_RERANK_KEY",
        "LLM_KEY",
    )
    if not api_key:
        raise RuntimeError(
            "RAG_RERANK_API_KEY or LLM_KEY is required for reranking."
        )

    base_url, base_url_source = _first_present_env(
        "RAG_RERANK_BASE_URL",
        "RAG_RERANK_API_BASE",
        "LLM_BASE_URL",
    )
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return client_kwargs, {
        "api_key_source": api_key_source or "unknown",
        "base_url": base_url or "default OpenAI base_url",
        "base_url_source": base_url_source or "default",
    }


def build_openai_client() -> OpenAI:
    load_local_env()
    client_kwargs, _ = _resolve_embedding_client_config()
    return OpenAI(**client_kwargs)


def build_rerank_client() -> OpenAI:
    load_local_env()
    client_kwargs, _ = _resolve_rerank_client_config()
    return OpenAI(**client_kwargs)


def _resolve_note_filter_client_config(model: str | None = None) -> tuple[dict[str, str], dict[str, str]]:
    api_key, api_key_source = _first_present_env(
        "RAG_NOTE_FILTER_API_KEY",
        "RAG_NOTE_FILTER_KEY",
        "LLM_KEY",
        "OPENAI_API_KEY",
        "OPENAI_APIKEY",
    )
    if not api_key:
        raise RuntimeError(
            "RAG_NOTE_FILTER_API_KEY, LLM_KEY, or OPENAI_API_KEY is required for note filtering."
        )

    base_url, base_url_source = _first_present_env(
        "RAG_NOTE_FILTER_BASE_URL",
        "RAG_NOTE_FILTER_API_BASE",
        "LLM_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    )
    if base_url and urlparse(base_url).path.rstrip("/").endswith("/rerank"):
        raise RuntimeError(
            "Note filtering requires a chat-completions base URL, but got a rerank endpoint: "
            f"{base_url}. Set RAG_NOTE_FILTER_BASE_URL or LLM_BASE_URL to your chat provider base URL."
        )
    if model and "/" in model and not base_url:
        raise RuntimeError(
            "Note filter model "
            f"{model!r} likely requires an OpenAI-compatible chat base URL. "
            "Set RAG_NOTE_FILTER_BASE_URL or LLM_BASE_URL before enabling note filtering."
        )

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return client_kwargs, {
        "api_key_source": api_key_source or "unknown",
        "base_url": base_url or "default OpenAI base_url",
        "base_url_source": base_url_source or "default",
    }


def build_note_filter_client(*, model: str | None = None) -> OpenAI:
    load_local_env()
    client_kwargs, _ = _resolve_note_filter_client_config(model=model)
    return OpenAI(**client_kwargs)


try:
    from rag.schema import KnowledgeCard, MatchedChunk, PlannerNoteContext, RetrievalHit, RetrievedNoteEvidence
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from rag.schema import KnowledgeCard, MatchedChunk, PlannerNoteContext, RetrievalHit, RetrievedNoteEvidence


RerankScorer = Callable[[str, list[str]], list[float]]


def parse_args() -> argparse.Namespace:
    load_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="我现在在武汉市的武汉大学的凌波门，想要去东湖走路，给我推荐一条走路不错的路线，一定要走东湖道（在湖中间那个道），就是在绿道上随便走走，相当于经过东湖绿道全景广场、湖心岛动物博物馆这个路，然后到湖北省博物馆结束吧", help="Natural-language query to retrieve against the vector store.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name.")
    parser.add_argument(
        "--qdrant-path",
        default=DEFAULT_QDRANT_PATH,
        help="Local Qdrant storage path. Ignored when --qdrant-url is provided.",
    )
    parser.add_argument("--qdrant-url", default=None, help="Optional remote Qdrant URL.")
    parser.add_argument("--qdrant-api-key", default=None, help="Optional remote Qdrant API key.")
    parser.add_argument(
        "--embedding-model",
        default="openai/text-embedding-3-large",
        help="OpenAI embedding model name for query embedding.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Optional shortened embedding dimension when supported by the model.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="How many vector matches to request before rerank and note aggregation.",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=2,
        help="Optional number of reranked chunk candidates to keep before note aggregation.",
    )
    parser.add_argument(
        "--max-notes",
        type=int,
        default=6,
        help="Maximum number of parent notes to return after rerank and aggregation.",
    )
    parser.add_argument(
        "--notes-jsonl",
        default=DEFAULT_NOTES_JSONL,
        help="Knowledge-card JSONL used to look up the full original note text.",
    )
    parser.add_argument(
        "--rerank-model",
        default=os.environ.get("RAG_RERANK_MODEL"),
        help="Optional OpenAI-compatible chat model used for post-retrieval reranking.",
    )
    parser.add_argument(
        "--note-filter-model",
        default=os.environ.get("RAG_NOTE_FILTER_MODEL"),
        help=(
            "Optional OpenAI-compatible chat model used to filter note candidates for direct prompt relevance "
            "and citywalk planning usefulness."
        ),
    )
    parser.add_argument(
        "--matched-chunks-per-note",
        type=int,
        default=2,
        help="How many supporting chunks to retain under each parent note.",
    )
    parser.add_argument(
        "--full-note-top-k",
        type=int,
        default=2,
        help="How many top notes should include the full original note text.",
    )
    parser.add_argument(
        "--disable-rerank",
        action="store_true",
        help="Skip model reranking and keep dense retrieval ordering only.",
    )
    parser.add_argument(
        "--region",
        action="append",
        default=None,
        help="Optional region filter. Repeat this flag to pass multiple regions.",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=None,
        help="Optional keyword filter. Repeat this flag to pass multiple keywords.",
    )
    parser.add_argument(
        "--poi",
        action="append",
        default=None,
        help="Optional POI filter. Repeat this flag to pass multiple POI names.",
    )
    parser.add_argument(
        "--require-route-hints",
        action="store_true",
        help="Only retrieve chunks whose parent card includes route hints.",
    )
    args = parser.parse_args()
    if args.rerank_top_n is not None and args.rerank_top_n <= 0:
        parser.error("--rerank-top-n must be a positive integer.")
    return args


def require_qdrant():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "qdrant_client is not installed. Install qdrant-client before using the retriever."
        ) from exc
    return QdrantClient, models


def build_client(args: argparse.Namespace):
    QdrantClient, models = require_qdrant()
    if args.qdrant_url:
        client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    else:
        client = QdrantClient(path=args.qdrant_path)
    return client, models


def embed_query(query: str, *, model: str, dimensions: int | None) -> list[float]:
    load_local_env()
    client_kwargs, client_meta = _resolve_embedding_client_config()
    client = OpenAI(**client_kwargs)
    request_kwargs = {"model": model, "input": query, "encoding_format": "float"}
    if dimensions is not None:
        request_kwargs["dimensions"] = dimensions
    active_base_url = client_meta["base_url"]
    active_provider = client_meta["api_key_source"]
    try:
        response = client.embeddings.create(**request_kwargs)
    except APIConnectionError as exc:
        raise RuntimeError(
            "Query embedding failed due to network connectivity. Check outbound access to the embeddings endpoint."
        ) from exc
    except ValueError as exc:
        if "No embedding data received" in str(exc):
            raise RuntimeError(
                "Embedding endpoint returned no data. This usually means provider/base_url is mismatched "
                f"for embeddings. active_provider={active_provider}, active_base_url={active_base_url}"
            ) from exc
        raise

    data = getattr(response, "data", None)
    if not data:
        provider_error = getattr(response, "error", None)
        if isinstance(provider_error, dict):
            message = provider_error.get("message") or "Unknown provider error."
            code = provider_error.get("code")
            raise RuntimeError(
                "Embedding provider returned an error payload instead of embeddings data. "
                f"message={message!r}, code={code!r}, active_provider={active_provider}, active_base_url={active_base_url}"
            )
        raise RuntimeError(
            "Embedding response has empty data. Check provider/base_url compatibility for embeddings. "
            f"active_provider={active_provider}, active_base_url={active_base_url}"
        )
    embedding = getattr(data[0], "embedding", None)
    if not embedding:
        raise RuntimeError(
            "Embedding response data[0] has no embedding vector. Check provider/base_url compatibility for embeddings. "
            f"active_provider={active_provider}, active_base_url={active_base_url}"
        )
    return embedding


def build_filter(args: argparse.Namespace, models):
    must_conditions = []

    if args.region:
        must_conditions.append(
            models.FieldCondition(key="regions", match=models.MatchAny(any=list(dict.fromkeys(args.region))))
        )
    if args.keyword:
        must_conditions.append(
            models.FieldCondition(key="keywords", match=models.MatchAny(any=list(dict.fromkeys(args.keyword))))
        )
    if args.poi:
        must_conditions.append(
            models.FieldCondition(key="poi_names", match=models.MatchAny(any=list(dict.fromkeys(args.poi))))
        )
    if args.require_route_hints:
        must_conditions.append(
            models.FieldCondition(key="has_route_hints", match=models.MatchValue(value=True))
        )

    if not must_conditions:
        return None
    return models.Filter(must=must_conditions)


def collect_hits(raw_hits) -> list[RetrievalHit]:
    results: list[RetrievalHit] = []
    for point in raw_hits:
        payload = dict(point.payload or {})
        note_id = str(payload.get("note_id") or "")
        if not note_id:
            continue
        results.append(
            RetrievalHit(
                score=float(point.score),
                chunk_id=str(payload.get("chunk_id") or ""),
                note_id=note_id,
                title=str(payload.get("title") or ""),
                source_url=str(payload.get("source_url") or ""),
                chunk_text=str(payload.get("chunk_text") or ""),
                poi_names=list(payload.get("poi_names") or []),
                route_hints=list(payload.get("route_hints") or []),
                regions=list(payload.get("regions") or []),
                keywords=list(payload.get("keywords") or []),
                engagement_score=int(payload.get("engagement_score") or 0),
                payload=payload,
            )
        )
    return results


def build_rerank_document(hit: RetrievalHit) -> str:
    sections = [f"标题: {hit.title}"]
    if hit.regions:
        sections.append("区域: " + "，".join(hit.regions))
    if hit.keywords:
        sections.append("关键词: " + "，".join(hit.keywords))
    if hit.poi_names:
        sections.append("POI: " + "，".join(hit.poi_names))
    if hit.route_hints:
        sections.append("路线提示: " + " | ".join(hit.route_hints))
    sections.append(f"正文片段: {hit.chunk_text}")
    return "\n".join(sections)


def _extract_chat_content(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError("Reranker response did not include any choices.")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "") for item in content]
        merged = "".join(part for part in text_parts if part)
        if merged:
            return merged
    raise RuntimeError("Reranker response did not include textual content.")


def _parse_json_payload(content: str) -> dict[str, Any]:
    text = MARKDOWN_FENCE_RE.sub("", (content or "").strip())
    match = JSON_BLOCK_RE.search(text)
    if not match:
        raise RuntimeError(f"Reranker response did not contain JSON: {content}")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise RuntimeError("Reranker JSON payload must be an object.")
    return payload


def _parse_rerank_results(
    results: Any,
    *,
    field_name: str,
    expected_count: int,
    allow_missing: bool = False,
    missing_score: float = -1.0,
) -> list[float]:
    if not isinstance(results, list):
        raise RuntimeError("Reranker payload must contain a results list.")

    scores_by_index: dict[int, float] = {}
    for item in results:
        if not isinstance(item, dict):
            raise RuntimeError("Each reranker result entry must be a JSON object.")
        index = item.get("index")
        score = item.get(field_name)
        if not isinstance(index, int):
            raise RuntimeError(f"Reranker result index must be an integer: {item}")
        if index < 0 or index >= expected_count:
            raise RuntimeError(f"Reranker result index out of range: {item}")
        try:
            numeric_score = float(score)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Reranker result score must be numeric: {item}") from exc
        scores_by_index[index] = numeric_score

    missing = [index for index in range(expected_count) if index not in scores_by_index]
    if missing and not allow_missing:
        raise RuntimeError(f"Reranker response missed candidate indexes: {missing}")
    return [scores_by_index.get(index, missing_score) for index in range(expected_count)]


def _resolve_rerank_top_n(top_n: int | None, total: int) -> int:
    if total <= 0:
        return 0
    if top_n is None:
        return total
    return min(top_n, total)


def _is_http_rerank_endpoint(base_url: str | None) -> bool:
    if not base_url:
        return False
    return urlparse(base_url).path.rstrip("/").endswith("/rerank")


def _http_rerank_scores(
    query: str,
    documents: list[str],
    *,
    model: str,
    api_key: str,
    endpoint: str,
    top_n: int | None,
) -> list[float]:
    requested_top_n = _resolve_rerank_top_n(top_n, len(documents))
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": requested_top_n,
        "return_documents": False,
    }
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(
            f"Rerank endpoint request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            "Rerank request failed due to network connectivity. Check outbound access to the rerank endpoint."
        ) from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Rerank endpoint returned non-JSON content: {body}") from exc
    return _parse_rerank_results(
        payload.get("results"),
        field_name="relevance_score",
        expected_count=len(documents),
        allow_missing=requested_top_n < len(documents),
    )


def _chat_rerank_scores(query: str, documents: list[str], *, model: str, client: OpenAI) -> list[float]:
    candidates_text = "\n\n".join(f"[{index}]\n{document}" for index, document in enumerate(documents))
    system_prompt = (
        "你是一个检索重排器。给定用户意图和候选文档后，"
        "请为每个候选文档输出 0 到 1 之间的相关性分数。"
        "0 表示几乎无关，1 表示高度相关。不要跳过任何候选。"
    )
    user_prompt = f"""用户检索意图：
{query}

候选文档：
{candidates_text}

请只输出 JSON，格式如下：
{{
  "results": [
    {{"index": 0, "score": 0.83}},
    {{"index": 1, "score": 0.27}}
  ]
}}

要求：
1. results 必须覆盖全部候选 index。
2. score 必须是 0 到 1 的浮点数。
3. 不要输出解释，不要输出 Markdown。
"""
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except APIConnectionError as exc:
        raise RuntimeError(
            "Rerank request failed due to network connectivity. Check outbound access to the LLM endpoint."
        ) from exc

    payload = _parse_json_payload(_extract_chat_content(response))
    return _parse_rerank_results(payload.get("results"), field_name="score", expected_count=len(documents))


def llm_rerank_scores(
    query: str,
    documents: list[str],
    *,
    model: str,
    top_n: int | None = None,
) -> list[float]:
    if not model:
        raise RuntimeError("A rerank model is required when reranking is enabled.")
    if not documents:
        return []

    load_local_env()
    client_kwargs, client_meta = _resolve_rerank_client_config()
    if _is_http_rerank_endpoint(client_kwargs.get("base_url")):
        return _http_rerank_scores(
            query,
            documents,
            model=model,
            api_key=client_kwargs["api_key"],
            endpoint=client_kwargs["base_url"],
            top_n=top_n,
        )

    client = OpenAI(**client_kwargs)
    return _chat_rerank_scores(query, documents, model=model, client=client)


def rerank_hits(
    query: str,
    hits: list[RetrievalHit],
    scorer: RerankScorer,
    *,
    top_n: int | None = None,
) -> list[RetrievalHit]:
    if not hits:
        return []
    documents = [build_rerank_document(hit) for hit in hits]
    scores = scorer(query, documents)
    if len(scores) != len(hits):
        raise RuntimeError(
            f"Reranker returned {len(scores)} scores for {len(hits)} candidates."
        )

    reranked_hits = [
        _model_copy(hit, rerank_score=float(score))
        for hit, score in zip(hits, scores)
    ]
    reranked_hits.sort(key=lambda hit: (hit.rerank_score if hit.rerank_score is not None else hit.score, hit.score), reverse=True)
    if top_n is None:
        return reranked_hits
    return reranked_hits[: _resolve_rerank_top_n(top_n, len(reranked_hits))]


def load_note_records_lookup(notes_jsonl: str | Path, note_ids: set[str]) -> dict[str, KnowledgeCard]:
    if not note_ids:
        return {}
    notes_path = Path(notes_jsonl)
    if not notes_path.exists():
        return {}

    lookup: dict[str, KnowledgeCard] = {}
    with notes_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            note_id = str(payload.get("note_id") or "")
            if note_id not in note_ids or note_id in lookup:
                continue
            lookup[note_id] = _model_validate(KnowledgeCard, payload)
            if len(lookup) >= len(note_ids):
                break
    return lookup


def load_note_text_lookup(notes_jsonl: str | Path, note_ids: set[str]) -> dict[str, str]:
    if not note_ids:
        return {}
    notes_path = Path(notes_jsonl)
    if not notes_path.exists():
        return {}

    lookup: dict[str, str] = {}
    with notes_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            note_id = str(payload.get("note_id") or "")
            if note_id not in note_ids or note_id in lookup:
                continue
            lookup[note_id] = str(payload.get("text") or "")
            if len(lookup) >= len(note_ids):
                break
    return lookup


def _merge_unique(values: list[list[str]]) -> list[str]:
    merged: list[str] = []
    for items in values:
        for item in items:
            value = str(item or "").strip()
            if not value or value in merged:
                continue
            merged.append(value)
    return merged


def aggregate_note_hits(
    reranked_hits: list[RetrievalHit],
    *,
    note_records_by_id: dict[str, KnowledgeCard],
    max_notes: int,
    matched_chunks_per_note: int,
) -> list[RetrievedNoteEvidence]:
    grouped: dict[str, list[RetrievalHit]] = defaultdict(list)
    for hit in reranked_hits:
        if hit.note_id:
            grouped[hit.note_id].append(hit)

    note_results: list[RetrievedNoteEvidence] = []
    for note_id, note_hits in grouped.items():
        ordered_hits = sorted(
            note_hits,
            key=lambda hit: (hit.rerank_score if hit.rerank_score is not None else hit.score, hit.score),
            reverse=True,
        )
        primary_hit = ordered_hits[0]
        note_record = note_records_by_id.get(note_id)
        matched_chunks = [
            MatchedChunk(
                chunk_id=hit.chunk_id,
                score=float(hit.rerank_score if hit.rerank_score is not None else hit.score),
                vector_score=float(hit.score),
                rerank_score=float(hit.rerank_score) if hit.rerank_score is not None else None,
                chunk_text=hit.chunk_text,
                poi_names=list(hit.poi_names),
                route_hints=list(hit.route_hints),
                regions=list(hit.regions),
                keywords=list(hit.keywords),
                payload=dict(hit.payload or {}),
            )
            for hit in ordered_hits[:matched_chunks_per_note]
        ]
        note_results.append(
            RetrievedNoteEvidence(
                score=float(primary_hit.rerank_score if primary_hit.rerank_score is not None else primary_hit.score),
                vector_score=float(primary_hit.score),
                rerank_score=float(primary_hit.rerank_score) if primary_hit.rerank_score is not None else None,
                chunk_id=primary_hit.chunk_id,
                note_id=note_id,
                title=(note_record.title if note_record and note_record.title else primary_hit.title),
                source_url=(note_record.source_url if note_record and note_record.source_url else primary_hit.source_url),
                chunk_text=primary_hit.chunk_text,
                poi_names=(list(note_record.poi_names) if note_record and note_record.poi_names else _merge_unique([hit.poi_names for hit in ordered_hits])),
                route_hints=(list(note_record.route_hints) if note_record and note_record.route_hints else _merge_unique([hit.route_hints for hit in ordered_hits])),
                regions=(list(note_record.regions) if note_record and note_record.regions else _merge_unique([hit.regions for hit in ordered_hits])),
                keywords=(list(note_record.keywords) if note_record and note_record.keywords else _merge_unique([hit.keywords for hit in ordered_hits])),
                engagement_score=max((hit.engagement_score for hit in ordered_hits), default=0),
                matched_chunks=matched_chunks,
                full_note_text=(note_record.text if note_record and note_record.text else None),
                payload=dict(primary_hit.payload or {}),
            )
        )

    note_results.sort(key=lambda note: (note.score, note.vector_score), reverse=True)
    return note_results[:max_notes]


def _parse_note_filter_results(results: Any, *, expected_count: int) -> list[bool]:
    if not isinstance(results, list):
        raise RuntimeError("Note-filter payload must contain a results list.")

    keep_by_index: dict[int, bool] = {}
    for item in results:
        if not isinstance(item, dict):
            raise RuntimeError("Each note-filter result entry must be a JSON object.")
        index = item.get("index")
        keep = item.get("keep")
        if not isinstance(index, int):
            raise RuntimeError(f"Note-filter result index must be an integer: {item}")
        if index < 0 or index >= expected_count:
            raise RuntimeError(f"Note-filter result index out of range: {item}")
        if isinstance(keep, bool):
            keep_value = keep
        elif isinstance(keep, str) and keep.lower() in {"true", "false"}:
            keep_value = keep.lower() == "true"
        else:
            raise RuntimeError(f"Note-filter result keep must be boolean: {item}")
        keep_by_index[index] = keep_value

    missing = [index for index in range(expected_count) if index not in keep_by_index]
    if missing:
        raise RuntimeError(f"Note-filter response missed candidate indexes: {missing}")
    return [keep_by_index[index] for index in range(expected_count)]


def filter_notes_for_planning(
    query: str,
    notes: list[RetrievedNoteEvidence],
    *,
    model: str | None,
    client: OpenAI | None = None,
) -> list[RetrievedNoteEvidence]:
    if not notes or not model:
        return notes

    if client is None:
        client = build_note_filter_client(model=model)

    candidates_text = "\n\n".join(
        "\n".join(
            [
                f"[{index}]",
                f"标题: {note.title}",
                f"POI: {'，'.join(note.poi_names) if note.poi_names else '无'}",
                f"路线提示: {' | '.join(note.route_hints) if note.route_hints else '无'}",
                f"区域: {'，'.join(note.regions) if note.regions else '无'}",
                f"关键词: {'，'.join(note.keywords) if note.keywords else '无'}",
                f"全文: {note.full_note_text or note.chunk_text}",
            ]
        )
        for index, note in enumerate(notes)
    )
    system_prompt = (
        "你是一个 citywalk 检索过滤器。你的任务是判断每篇小红书帖子是否应该保留给路线规划阶段。"
        "保留标准只有两个：1. 是否和用户 prompt 直接有关；2. 是否对 citywalk 路线规划有帮助。"
        "只输出 JSON，并在 reason 字段中给出简短的判断原因。"
    )
    user_prompt = f'''用户 prompt：
{query}

候选帖子：
{candidates_text}

请只输出 JSON，格式如下：
{{
  "results": [
    {{"index": 0, "keep": true, "reason": "提到了东湖绿道，符合用户需求"}},
    {{"index": 1, "keep": false, "reason": "这是关于美食探店的，与走路路线无关"}}
  ]
}}

要求：
1. results 必须覆盖全部候选 index。
2. keep 必须是布尔值。
3. 只要帖子和用户 prompt 不直接相关，或者对 citywalk 路线规划帮助不大，就设为 false。
4. 必须在 reason 字段中给出简短的解释。
'''
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except APIConnectionError as exc:
        raise RuntimeError(
            "Note filtering failed due to network connectivity. Check outbound access to the LLM endpoint."
        ) from exc
    except NotFoundError as exc:
        raise RuntimeError(
            f"Note filtering returned 404 for model {model!r}. Check that your chat provider base URL is set via "
            "RAG_NOTE_FILTER_BASE_URL or LLM_BASE_URL, and that the provider actually supports this model."
        ) from exc

    raw_content = _extract_chat_content(response)
    print("\n=== 大模型过滤原因 ===")
    print(raw_content)
    print("======================\n")

    payload = _parse_json_payload(raw_content)
    keep_flags = _parse_note_filter_results(payload.get("results"), expected_count=len(notes))
    return [note for note, keep in zip(notes, keep_flags) if keep]


def project_planner_note_contexts(notes: list[RetrievedNoteEvidence]) -> list[PlannerNoteContext]:
    return [
        PlannerNoteContext(
            title=note.title,
            poi_names=list(note.poi_names),
            route_hints=list(note.route_hints),
            regions=list(note.regions),
            keywords=list(note.keywords),
            full_note_text=note.full_note_text or note.chunk_text or "",
        )
        for note in notes
    ]


def _search_raw_hits(client: Any, *, collection: str, query_vector: list[float], query_filter: Any, top_k: int):
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return list(response.points)
    return client.search(
        collection_name=collection,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )


def retrieve_planner_note_contexts(
    query: str,
    *,
    collection: str = DEFAULT_COLLECTION,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    embedding_model: str = "openai/text-embedding-3-large",
    dimensions: int | None = None,
    top_k: int = 12,
    rerank_top_n: int | None = 2,
    max_notes: int = 6,
    notes_jsonl: str = DEFAULT_NOTES_JSONL,
    rerank_model: str | None = None,
    note_filter_model: str | None = None,
    matched_chunks_per_note: int = 2,
    disable_rerank: bool = False,
    region: list[str] | None = None,
    keyword: list[str] | None = None,
    poi: list[str] | None = None,
    require_route_hints: bool = False,
) -> list[PlannerNoteContext]:
    args = argparse.Namespace(
        query=query,
        collection=collection,
        qdrant_path=qdrant_path,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        embedding_model=embedding_model,
        dimensions=dimensions,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        max_notes=max_notes,
        notes_jsonl=notes_jsonl,
        rerank_model=rerank_model if rerank_model is not None else os.environ.get("RAG_RERANK_MODEL"),
        note_filter_model=(
            note_filter_model if note_filter_model is not None else os.environ.get("RAG_NOTE_FILTER_MODEL")
        ),
        matched_chunks_per_note=matched_chunks_per_note,
        disable_rerank=disable_rerank,
        region=region,
        keyword=keyword,
        poi=poi,
        require_route_hints=require_route_hints,
    )

    client = None
    try:
        client, models = build_client(args)
        query_vector = embed_query(args.query, model=args.embedding_model, dimensions=args.dimensions)
        query_filter = build_filter(args, models)
        raw_hits = _search_raw_hits(
            client,
            collection=args.collection,
            query_vector=query_vector,
            query_filter=query_filter,
            top_k=args.top_k,
        )

        candidate_hits = collect_hits(raw_hits)
        if args.disable_rerank or not args.rerank_model:
            ordered_hits = sorted(candidate_hits, key=lambda hit: hit.score, reverse=True)
        else:
            ordered_hits = rerank_hits(
                query=args.query,
                hits=candidate_hits,
                scorer=lambda current_query, documents: llm_rerank_scores(
                    current_query,
                    documents,
                    model=args.rerank_model,
                    top_n=args.rerank_top_n,
                ),
                top_n=args.rerank_top_n,
            )

        note_ids = {hit.note_id for hit in ordered_hits if hit.note_id}
        note_records = load_note_records_lookup(args.notes_jsonl, note_ids)
        aggregate_limit = len(note_ids) if args.note_filter_model else args.max_notes
        note_results = aggregate_note_hits(
            ordered_hits,
            note_records_by_id=note_records,
            max_notes=max(aggregate_limit, args.max_notes),
            matched_chunks_per_note=args.matched_chunks_per_note,
        )
        filtered_notes = filter_notes_for_planning(
            query=args.query,
            notes=note_results,
            model=args.note_filter_model,
        )
        return project_planner_note_contexts(filtered_notes[: args.max_notes])
    finally:
        if client is not None and hasattr(client, "close"):
            client.close()


def main() -> int:
    args = parse_args()
    planner_payloads = retrieve_planner_note_contexts(
        args.query,
        collection=args.collection,
        qdrant_path=args.qdrant_path,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        embedding_model=args.embedding_model,
        dimensions=args.dimensions,
        top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
        max_notes=args.max_notes,
        notes_jsonl=args.notes_jsonl,
        rerank_model=args.rerank_model,
        note_filter_model=args.note_filter_model,
        matched_chunks_per_note=args.matched_chunks_per_note,
        disable_rerank=args.disable_rerank,
        region=args.region,
        keyword=args.keyword,
        poi=args.poi,
        require_route_hints=args.require_route_hints,
    )
    print(json.dumps([_model_dump(note) for note in planner_payloads], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
