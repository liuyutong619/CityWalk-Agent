"""Structured schemas for cleaned Xiaohongshu RAG knowledge cards."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Engagement(BaseModel):
    """Interaction signals carried over from Xiaohongshu notes."""

    likes: int = Field(default=0, description="点赞数")
    collects: int = Field(default=0, description="收藏数")
    comments: int = Field(default=0, description="评论数")


class KnowledgeCard(BaseModel):
    """Minimal cleaned unit for downstream retrieval and chunking."""

    note_id: str = Field(description="Unique Xiaohongshu note identifier")
    title: str = Field(description="Cleaned note title")
    text: str = Field(description="Cleaned note body text")
    source_url: str = Field(description="Original note URL")
    poi_names: list[str] = Field(default_factory=list, description="POI names heuristically extracted from the note")
    route_hints: list[str] = Field(default_factory=list, description="Route-related hints extracted from the note")
    engagement: Engagement = Field(default_factory=Engagement, description="Raw interaction counters")
    regions: list[str] = Field(default_factory=list, description="District labels inherited from crawl files")
    keywords: list[str] = Field(default_factory=list, description="Search keywords that surfaced this note")


class ChunkRecord(BaseModel):
    """Embedding-ready chunk derived from a single knowledge card."""

    chunk_id: str = Field(description="Stable chunk identifier")
    note_id: str = Field(description="Parent Xiaohongshu note identifier")
    chunk_type: str = Field(default="body", description="Chunk type label")
    chunk_index: int = Field(description="Chunk order within parent note")
    title: str = Field(description="Parent note title")
    source_url: str = Field(description="Original note URL")
    chunk_text: str = Field(description="Chunk body text")
    header_text: str = Field(description="Structured header prepended before embedding")
    embedding_text: str = Field(description="Final text sent to the embedding model")
    poi_names: list[str] = Field(default_factory=list, description="Reliable POI names attached to this chunk")
    route_hints: list[str] = Field(default_factory=list, description="Route hints injected into the chunk header")
    engagement: Engagement = Field(default_factory=Engagement, description="Raw interaction counters")
    engagement_score: int = Field(default=0, description="Heuristic engagement score for tie-breaking")
    regions: list[str] = Field(default_factory=list, description="Region labels inherited from the parent note")
    keywords: list[str] = Field(default_factory=list, description="Keyword labels inherited from the parent note")
    parent_text_length: int = Field(default=0, description="Length of the original note body")
    chunk_text_length: int = Field(default=0, description="Length of the chunk body")


class RetrievalHit(BaseModel):
    """Search result returned by the vector retriever."""

    score: float = Field(description="Vector similarity score")
    chunk_id: str = Field(description="Matched chunk identifier")
    note_id: str = Field(description="Parent note identifier")
    title: str = Field(description="Parent note title")
    source_url: str = Field(description="Original note URL")
    chunk_text: str = Field(description="Matched chunk text")
    poi_names: list[str] = Field(default_factory=list, description="POIs attached to the result")
    route_hints: list[str] = Field(default_factory=list, description="Route hints attached to the result")
    regions: list[str] = Field(default_factory=list, description="Region labels")
    keywords: list[str] = Field(default_factory=list, description="Keyword labels")
    engagement_score: int = Field(default=0, description="Heuristic engagement score")
    rerank_score: float | None = Field(default=None, description="Optional model rerank score")
    payload: dict = Field(default_factory=dict, description="Raw payload returned by the vector store")


class MatchedChunk(BaseModel):
    """Chunk evidence retained under a parent note after reranking."""

    chunk_id: str = Field(description="Matched chunk identifier")
    score: float = Field(description="Effective score used for ranking this chunk")
    vector_score: float = Field(description="Original vector similarity score")
    rerank_score: float | None = Field(default=None, description="Optional model rerank score")
    chunk_text: str = Field(description="Matched chunk text")
    poi_names: list[str] = Field(default_factory=list, description="POIs attached to the chunk")
    route_hints: list[str] = Field(default_factory=list, description="Route hints attached to the chunk")
    regions: list[str] = Field(default_factory=list, description="Region labels attached to the chunk")
    keywords: list[str] = Field(default_factory=list, description="Keywords attached to the chunk")
    payload: dict = Field(default_factory=dict, description="Raw vector-store payload for the chunk")


class RetrievedNoteEvidence(BaseModel):
    """Note-level evidence package returned to downstream agents."""

    score: float = Field(description="Effective note score after rerank and aggregation")
    vector_score: float = Field(description="Original vector score of the primary chunk")
    rerank_score: float | None = Field(default=None, description="Optional model rerank score of the primary chunk")
    chunk_id: str = Field(description="Primary supporting chunk identifier")
    note_id: str = Field(description="Parent note identifier")
    title: str = Field(description="Parent note title")
    source_url: str = Field(description="Original note URL")
    chunk_text: str = Field(description="Primary supporting chunk text")
    poi_names: list[str] = Field(default_factory=list, description="Merged POI names across the note evidence")
    route_hints: list[str] = Field(default_factory=list, description="Merged route hints across the note evidence")
    regions: list[str] = Field(default_factory=list, description="Merged region labels across the note evidence")
    keywords: list[str] = Field(default_factory=list, description="Merged keyword labels across the note evidence")
    engagement_score: int = Field(default=0, description="Heuristic engagement score")
    matched_chunks: list[MatchedChunk] = Field(default_factory=list, description="Top supporting chunks for this note")
    full_note_text: str | None = Field(default=None, description="Optional full original note text for top notes")
    payload: dict = Field(default_factory=dict, description="Raw payload of the primary chunk")


class PlannerNoteContext(BaseModel):
    """Slim note payload forwarded to the route planner."""

    title: str = Field(description="Parent note title")
    poi_names: list[str] = Field(default_factory=list, description="POIs extracted from the full original note")
    route_hints: list[str] = Field(default_factory=list, description="Route hints extracted from the full original note")
    regions: list[str] = Field(default_factory=list, description="Region labels attached to the full original note")
    keywords: list[str] = Field(default_factory=list, description="Keyword labels attached to the full original note")
    full_note_text: str = Field(default="", description="Full original Xiaohongshu note text")


class ChunkingStats(BaseModel):
    """Summary of a chunking run."""

    cards_processed: int = 0
    chunks_written: int = 0
    average_chunk_chars: float = 0.0
    max_chunk_chars: int = 0


class IngestionStats(BaseModel):
    """Summary of a cleaning run."""

    files_processed: int = 0
    raw_notes_seen: int = 0
    unique_cards_written: int = 0
    duplicates_merged: int = 0
    skipped_notes: int = 0
    skip_reasons: dict[str, int] = Field(default_factory=dict)
