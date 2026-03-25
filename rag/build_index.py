"""Build a local Qdrant index from embedding-ready Xiaohongshu chunks."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Iterable

from openai import APIConnectionError, OpenAI


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


def _first_present_env(*keys: str) -> tuple[str | None, str | None]:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value, key
    return None, None


def build_openai_client() -> OpenAI:
    load_local_env()

    api_key, _ = _first_present_env(
        "RAG_EMBEDDING_API_KEY",
        "RAG_EMBEDDING_KEY",
        "OPENAI_API_KEY",
        "OPENAI_APIKEY",
    )
    if not api_key:
        raise RuntimeError(
            "RAG_EMBEDDING_API_KEY or OPENAI_API_KEY is required for embeddings."
        )

    base_url, _ = _first_present_env(
        "RAG_EMBEDDING_BASE_URL",
        "RAG_EMBEDDING_API_BASE",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    )
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)

try:
    from rag.schema import ChunkRecord
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from rag.schema import ChunkRecord


EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        default="rag/data/xiaohongshu_chunks.jsonl",
        help="Input JSONL file containing embedding-ready chunks.",
    )
    parser.add_argument(
        "--collection",
        default="xiaohongshu_citywalk",
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--qdrant-path",
        default="rag/data/qdrant_local",
        help="Local Qdrant storage path. Ignored when --qdrant-url is provided.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Optional remote Qdrant URL.",
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=None,
        help="Optional remote Qdrant API key.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model name.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Optional shortened embedding dimension when supported by the model.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="How many chunks to send per embeddings request.",
    )
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=128,
        help="How many points to upsert per Qdrant request.",
    )
    parser.add_argument(
        "--manifest-path",
        default="rag/data/qdrant_manifest.json",
        help="Output JSON manifest path.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the collection before indexing.",
    )
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Optional chunk limit for quick verification runs.",
    )
    return parser.parse_args()


def require_qdrant():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "qdrant_client is not installed. Install qdrant-client before building the vector index."
        ) from exc
    return QdrantClient, models


def iter_chunks(path: Path, limit_chunks: int | None) -> Iterable[ChunkRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit_chunks is not None and index >= limit_chunks:
                break
            if not line.strip():
                continue
            yield ChunkRecord.model_validate_json(line)


def batched(items: list[ChunkRecord], batch_size: int) -> Iterable[list[ChunkRecord]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_client(args: argparse.Namespace):
    QdrantClient, models = require_qdrant()
    if args.qdrant_url:
        client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    else:
        client = QdrantClient(path=args.qdrant_path)
    return client, models


def resolve_vector_size(model_name: str, dimensions: int | None) -> int:
    if dimensions is not None:
        return dimensions
    if model_name in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[model_name]
    raise ValueError(
        f"Unknown default dimension for model {model_name!r}. Pass --dimensions explicitly."
    )


def ensure_collection(client, models, *, collection: str, vector_size: int, recreate: bool) -> None:
    if recreate:
        try:
            client.delete_collection(collection_name=collection)
        except Exception:
            pass

    existing_collections = {item.name for item in client.get_collections().collections}
    if collection not in existing_collections:
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    for field_name, field_schema in (
        ("note_id", models.PayloadSchemaType.KEYWORD),
        ("chunk_id", models.PayloadSchemaType.KEYWORD),
        ("chunk_type", models.PayloadSchemaType.KEYWORD),
        ("regions", models.PayloadSchemaType.KEYWORD),
        ("keywords", models.PayloadSchemaType.KEYWORD),
        ("poi_names", models.PayloadSchemaType.KEYWORD),
        ("engagement_score", models.PayloadSchemaType.INTEGER),
    ):
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception:
            continue


def embed_texts(client: OpenAI, texts: list[str], *, model: str, dimensions: int | None) -> list[list[float]]:
    request_kwargs = {"model": model, "input": texts}
    if dimensions is not None:
        request_kwargs["dimensions"] = dimensions
    try:
        response = client.embeddings.create(**request_kwargs)
    except APIConnectionError as exc:
        raise RuntimeError(
            "Embedding request failed due to network connectivity. Check outbound access to the embeddings endpoint."
        ) from exc
    return [item.embedding for item in response.data]


def qdrant_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"citywalk:{chunk_id}"))


def build_payload(chunk: ChunkRecord) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "note_id": chunk.note_id,
        "chunk_type": chunk.chunk_type,
        "chunk_index": chunk.chunk_index,
        "title": chunk.title,
        "source_url": chunk.source_url,
        "chunk_text": chunk.chunk_text,
        "header_text": chunk.header_text,
        "poi_names": chunk.poi_names,
        "route_hints": chunk.route_hints,
        "regions": chunk.regions,
        "keywords": chunk.keywords,
        "engagement_likes": chunk.engagement.likes,
        "engagement_collects": chunk.engagement.collects,
        "engagement_comments": chunk.engagement.comments,
        "engagement_score": chunk.engagement_score,
        "parent_text_length": chunk.parent_text_length,
        "chunk_text_length": chunk.chunk_text_length,
        "has_route_hints": bool(chunk.route_hints),
    }


def main() -> int:
    args = parse_args()
    chunks = list(iter_chunks(Path(args.input_jsonl), args.limit_chunks))
    if not chunks:
        raise ValueError("No chunks found. Run rag/chunking.py first.")

    openai_client = build_openai_client()
    qdrant_client, models = build_client(args)
    try:
        vector_size = resolve_vector_size(args.embedding_model, args.dimensions)
        ensure_collection(
            qdrant_client,
            models,
            collection=args.collection,
            vector_size=vector_size,
            recreate=args.recreate,
        )

        points_written = 0
        for chunk_batch in batched(chunks, args.embed_batch_size):
            vectors = embed_texts(
                openai_client,
                [chunk.embedding_text for chunk in chunk_batch],
                model=args.embedding_model,
                dimensions=args.dimensions,
            )
            point_batch = []
            for chunk, vector in zip(chunk_batch, vectors):
                point_batch.append(
                    models.PointStruct(
                        id=qdrant_point_id(chunk.chunk_id),
                        vector=vector,
                        payload=build_payload(chunk),
                    )
                )

            for start in range(0, len(point_batch), args.upsert_batch_size):
                qdrant_client.upsert(
                    collection_name=args.collection,
                    points=point_batch[start : start + args.upsert_batch_size],
                    wait=True,
                )
                points_written += len(point_batch[start : start + args.upsert_batch_size])

        manifest = {
            "collection": args.collection,
            "input_jsonl": args.input_jsonl,
            "qdrant_path": None if args.qdrant_url else args.qdrant_path,
            "qdrant_url": args.qdrant_url,
            "embedding_model": args.embedding_model,
            "dimensions": vector_size,
            "chunks_indexed": points_written,
            "recreate": args.recreate,
        }
        manifest_path = Path(args.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    finally:
        qdrant_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
