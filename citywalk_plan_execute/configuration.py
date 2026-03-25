"""Configuration for CityWalk Plan and Execute agent."""

import os
from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

DEFAULT_CLARIFICATION_MODEL = "openrouter:google/gemini-3-flash-preview"
DEFAULT_CLARIFICATION_MAX_TOKENS = 10000

DEFAULT_INTENT_MODEL = "openrouter:google/gemini-3-flash-preview"
DEFAULT_INTENT_MAX_TOKENS = 10000

DEFAULT_SUPERVISOR_MODEL = "openrouter:google/gemini-3-flash-preview"
DEFAULT_SUPERVISOR_MAX_TOKENS = 10000

DEFAULT_POI_EXPLORER_MODEL = "openrouter:google/gemini-3-flash-preview"
DEFAULT_POI_EXPLORER_MAX_TOKENS = 10000

DEFAULT_ROUTE_PLANNER_MODEL = "openrouter:google/gemini-3-flash-preview"
DEFAULT_ROUTE_PLANNER_MAX_TOKENS = 10000

DEFAULT_ROUTE_POI_ENRICHER_MODEL = "openrouter:google/gemini-3-flash-preview"
DEFAULT_ROUTE_POI_ENRICHER_MAX_TOKENS = 10000

DEFAULT_RAG_COLLECTION = "xiaohongshu_citywalk_example"
DEFAULT_RAG_QDRANT_PATH = "rag/data/qdrant_local_example"
DEFAULT_RAG_EMBEDDING_MODEL = "openai/text-embedding-3-large"
DEFAULT_RAG_TOP_K = 20
DEFAULT_RAG_RERANK_TOP_N = 3
DEFAULT_RAG_MAX_NOTES = 3
DEFAULT_RAG_NOTES_JSONL = "rag/data/example_cards_llm.jsonl"
DEFAULT_RAG_MATCHED_CHUNKS_PER_NOTE = 2

ROLE_MODEL_DEFAULTS: dict[str, dict[str, str | int]] = {
    "clarification": {
        "model": DEFAULT_CLARIFICATION_MODEL,
        "max_tokens": DEFAULT_CLARIFICATION_MAX_TOKENS,
    },
    "intent": {
        "model": DEFAULT_INTENT_MODEL,
        "max_tokens": DEFAULT_INTENT_MAX_TOKENS,
    },
    "supervisor": {
        "model": DEFAULT_SUPERVISOR_MODEL,
        "max_tokens": DEFAULT_SUPERVISOR_MAX_TOKENS,
    },
    "poi_explorer": {
        "model": DEFAULT_POI_EXPLORER_MODEL,
        "max_tokens": DEFAULT_POI_EXPLORER_MAX_TOKENS,
    },
    "route_planner": {
        "model": DEFAULT_ROUTE_PLANNER_MODEL,
        "max_tokens": DEFAULT_ROUTE_PLANNER_MAX_TOKENS,
    },
    "route_poi_enricher": {
        "model": DEFAULT_ROUTE_POI_ENRICHER_MODEL,
        "max_tokens": DEFAULT_ROUTE_POI_ENRICHER_MAX_TOKENS,
    },
}


class Configuration(BaseModel):
    """Configuration for CityWalk agent."""

    # Legacy shared model override
    model: Optional[str] = Field(
        default=None,
        description="Legacy shared model override for all CityWalk roles.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Legacy shared max output tokens override for all CityWalk roles.",
    )

    # Role-specific model settings
    clarification_model: Optional[str] = Field(
        default=None,
        description=f"Model used by the clarification step. Defaults to {ROLE_MODEL_DEFAULTS['clarification']['model']}.",
    )
    clarification_model_max_tokens: Optional[int] = Field(
        default=None,
        description=f"Max output tokens for clarification model. Defaults to {ROLE_MODEL_DEFAULTS['clarification']['max_tokens']}.",
    )
    intent_model: Optional[str] = Field(
        default=None,
        description=f"Model used by the intent parser. Defaults to {ROLE_MODEL_DEFAULTS['intent']['model']}.",
    )
    intent_model_max_tokens: Optional[int] = Field(
        default=None,
        description=f"Max output tokens for intent parser model. Defaults to {ROLE_MODEL_DEFAULTS['intent']['max_tokens']}.",
    )
    supervisor_model: Optional[str] = Field(
        default=None,
        description=f"Model used by the supervisor. Defaults to {ROLE_MODEL_DEFAULTS['supervisor']['model']}.",
    )
    supervisor_model_max_tokens: Optional[int] = Field(
        default=None,
        description=f"Max output tokens for supervisor model. Defaults to {ROLE_MODEL_DEFAULTS['supervisor']['max_tokens']}.",
    )
    poi_explorer_model: Optional[str] = Field(
        default=None,
        description=f"Model used by the POI Explorer sub-agent. Defaults to {ROLE_MODEL_DEFAULTS['poi_explorer']['model']}.",
    )
    poi_explorer_model_max_tokens: Optional[int] = Field(
        default=None,
        description=f"Max output tokens for POI Explorer. Defaults to {ROLE_MODEL_DEFAULTS['poi_explorer']['max_tokens']}.",
    )
    route_planner_model: Optional[str] = Field(
        default=None,
        description=f"Model used by the Route Planner sub-agent. Defaults to {ROLE_MODEL_DEFAULTS['route_planner']['model']}.",
    )
    route_planner_model_max_tokens: Optional[int] = Field(
        default=None,
        description=f"Max output tokens for Route Planner. Defaults to {ROLE_MODEL_DEFAULTS['route_planner']['max_tokens']}.",
    )
    route_poi_enricher_model: Optional[str] = Field(
        default=None,
        description=f"Model used by the Route POI Enricher sub-agent. Defaults to {ROLE_MODEL_DEFAULTS['route_poi_enricher']['model']}.",
    )
    route_poi_enricher_model_max_tokens: Optional[int] = Field(
        default=None,
        description=f"Max output tokens for Route POI Enricher. Defaults to {ROLE_MODEL_DEFAULTS['route_poi_enricher']['max_tokens']}.",
    )

    # Execution limits
    max_replan_count: int = Field(default=3, description="Maximum replanning attempts")
    max_tool_calls: int = Field(default=100, description="Max tool calls per execution")

    # Map API settings
    map_api_key: Optional[str] = Field(default=None, description="Map API key")

    # RAG retrieval settings
    rag_collection: str = Field(default=DEFAULT_RAG_COLLECTION, description="Qdrant collection used for Xiaohongshu note retrieval.")
    rag_qdrant_path: str = Field(default=DEFAULT_RAG_QDRANT_PATH, description="Local Qdrant storage path for RAG retrieval.")
    rag_embedding_model: str = Field(default=DEFAULT_RAG_EMBEDDING_MODEL, description="Embedding model used for RAG query encoding.")
    rag_embedding_dimensions: Optional[int] = Field(default=None, description="Optional shortened embedding dimensions for the RAG embedding model.")
    rag_top_k: int = Field(default=DEFAULT_RAG_TOP_K, description="How many chunk hits to request before note aggregation.")
    rag_rerank_top_n: int = Field(default=DEFAULT_RAG_RERANK_TOP_N, description="How many reranked chunk hits to retain before note aggregation.")
    rag_max_notes: int = Field(default=DEFAULT_RAG_MAX_NOTES, description="Maximum number of note-level Xiaohongshu references to keep for planning.")
    rag_notes_jsonl: str = Field(default=DEFAULT_RAG_NOTES_JSONL, description="Knowledge-card JSONL used to recover full note text.")
    rag_rerank_model: Optional[str] = Field(default=None, description="Optional rerank model for post-retrieval relevance scoring.")
    rag_note_filter_model: Optional[str] = Field(default=None, description="Optional note-filter model used to keep only citywalk-useful notes.")
    rag_matched_chunks_per_note: int = Field(default=DEFAULT_RAG_MATCHED_CHUNKS_PER_NOTE, description="Supporting chunk count retained under each note before projection.")
    rag_disable_rerank: bool = Field(default=False, description="Disable reranking and keep dense retrieval ordering only.")
    rag_require_route_hints: bool = Field(default=False, description="Only retrieve notes whose parent cards include route hints.")

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create Configuration from RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values = {
            field_name: (
                configurable.get(field_name)
                if configurable.get(field_name) is not None
                else os.environ.get(field_name.upper())
            )
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    def model_config_for(self, role: str) -> dict[str, Any]:
        """Resolve model settings for a specific CityWalk role."""
        defaults = ROLE_MODEL_DEFAULTS.get(role)
        if defaults is None:
            raise ValueError(f"Unknown CityWalk model role: {role}")

        role_model = getattr(self, f"{role}_model", None) or self.model or defaults["model"]
        role_max_tokens = getattr(self, f"{role}_model_max_tokens", None)
        if role_max_tokens is None:
            role_max_tokens = self.max_tokens if self.max_tokens is not None else defaults["max_tokens"]

        return {
            "model": role_model,
            "max_tokens": role_max_tokens,
        }

    def rag_retriever_config(self) -> dict[str, Any]:
        """Resolve centralized, non-secret RAG retrieval settings."""
        return {
            "collection": self.rag_collection,
            "qdrant_path": self.rag_qdrant_path,
            "embedding_model": self.rag_embedding_model,
            "dimensions": self.rag_embedding_dimensions,
            "top_k": self.rag_top_k,
            "rerank_top_n": self.rag_rerank_top_n,
            "max_notes": self.rag_max_notes,
            "notes_jsonl": self.rag_notes_jsonl,
            "rerank_model": self.rag_rerank_model,
            "note_filter_model": self.rag_note_filter_model,
            "matched_chunks_per_note": self.rag_matched_chunks_per_note,
            "disable_rerank": self.rag_disable_rerank,
            "require_route_hints": self.rag_require_route_hints,
        }
