"""Main LangGraph implementation for CityWalk Plan and Execute agent.

Architecture: Supervisor (tool-calling) dispatches POI Explorer and Route Planner
sub-agents (ReAct subgraphs).
"""

import json
from typing import Any, Literal, TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, ValidationError

from rag.retriever import retrieve_planner_note_contexts

from citywalk_plan_execute.configuration import Configuration
from citywalk_plan_execute.prompts import (
    CLARIFY_INTENT_PROMPT,
    JSON_FORMATTER_PROMPT,
    PARSE_INTENT_PROMPT,
    POI_EXPLORER_SYSTEM_PROMPT,
    ROUTE_POI_ENRICHER_SYSTEM_PROMPT,
    ROUTE_PLANNER_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
)
from citywalk_plan_execute.state import (
    AgentState,
    AllTasksComplete,
    DispatchPOIExplorer,
    DispatchRoutePlanner,
    IntentClarification,
    POIExplorerState,
    RouteNearbyPOIState,
    RoutePlannerState,
    UserIntent,
)
from citywalk_plan_execute.utils import execute_tool, summarize_tool_result_for_llm
from tools.langchain_tools import (
    ExplorationComplete,
    NearbyPOIEnrichmentComplete,
    PlanningComplete,
    POI_EXPLORER_TOOLS,
    ROUTE_POI_ENRICHER_TOOLS,
    ROUTE_PLANNER_TOOLS,
    SearchRouteNearbyPlaces,
    SubmitNearbyRoutePOIs,
)
from tools.maps_tools import _search_along_polyline
from tools.utils import haversine_distance_meters, merge_polylines, parse_location, parse_polyline


SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Initialize configurable model
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens"),
)


# ============================================================
# JSON model invocation helpers (kept from original)
# ============================================================

def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


import re


def _try_parse_json_object_from_text(text: str) -> dict[str, Any] | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    brace_match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
    if not brace_match:
        return None

    try:
        parsed = json.loads(brace_match.group(1))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _build_json_prompt(prompt: str, schema: type[BaseModel]) -> str:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    return (
        f"{prompt}\n\n"
        "你必须只输出一个 JSON 对象。\n"
        "不要输出 Markdown，不要输出 ```json 代码块，不要输出额外解释。\n"
        "请严格遵循以下 JSON Schema：\n"
        f"{schema_json}"
    )


async def _ainvoke_json_model(
    prompt: str,
    schema: type[SchemaT],
    configurable: Configuration,
    role: str,
) -> SchemaT:
    model = configurable_model.with_config(configurable.model_config_for(role))
    base_prompt = _build_json_prompt(prompt, schema)
    last_error = "未知错误"

    for attempt in range(2):
        current_prompt = base_prompt
        if attempt:
            current_prompt = (
                f"{base_prompt}\n\n"
                "你上一轮输出不合法。\n"
                f"错误摘要：{last_error}\n"
                "现在重新输出一个合法 JSON 对象。"
            )

        response = await model.ainvoke([HumanMessage(content=current_prompt)])
        text = _extract_message_text(response)
        payload = _try_parse_json_object_from_text(text)
        if not isinstance(payload, dict):
            last_error = f"无法从响应中提取 JSON: {text[:240]}"
            continue

        try:
            return schema.model_validate(payload)
        except ValidationError as error:
            last_error = str(error)

    raise ValueError(f"{schema.__name__} 解析失败: {last_error}")


# ============================================================
# Text / normalization helpers (kept from original)
# ============================================================

def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_text_list(value: Any, limit: int = 6) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    elif value is None:
        raw_items = []
    else:
        raw_items = [value]

    normalized: list[str] = []
    for item in raw_items:
        text = _normalize_text(item)
        if text and text not in normalized:
            normalized.append(text)
        if len(normalized) >= limit:
            break
    return normalized


def _model_dump_compat(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    if isinstance(model, dict):
        return dict(model)
    return {}


def _serialize_retrieved_note_contexts(state: AgentState, limit: int = 6) -> list[dict[str, Any]]:
    notes = list(state.get("retrieved_note_contexts") or [])
    serialized: list[dict[str, Any]] = []
    for note in notes[:limit]:
        payload = _model_dump_compat(note)
        if not payload:
            continue
        serialized.append(
            {
                "title": _normalize_text(payload.get("title")),
                "poi_names": _normalize_text_list(payload.get("poi_names"), limit=12),
                "route_hints": _normalize_text_list(payload.get("route_hints"), limit=12),
                "regions": _normalize_text_list(payload.get("regions"), limit=8),
                "keywords": _normalize_text_list(payload.get("keywords"), limit=8),
                "full_note_text": _normalize_text(payload.get("full_note_text")),
            }
        )
    return serialized


def _build_retrieved_notes_reference_rules() -> dict[str, Any]:
    return {
        "xiaohongshu_notes_are_advisory": True,
        "useful_but_not_binding": True,
        "ignore_unreasonable_or_outdated_content": True,
    }


def _summarize_retrieved_note_contexts(note_contexts: list[Any]) -> str:
    if not note_contexts:
        return "未检索到可用小红书帖子，继续按地图与用户约束规划。"

    titles = []
    for note in note_contexts[:3]:
        payload = _model_dump_compat(note)
        title = _normalize_text(payload.get("title"))
        if title:
            titles.append(title)

    title_text = f"，代表帖子：{'；'.join(titles)}" if titles else ""
    return f"已检索到 {len(note_contexts)} 篇相关小红书帖子，可作为路线与点位规划参考{title_text}。"


def _same_coordinates(coord_a: str | None, coord_b: str | None, threshold_meters: float = 30.0) -> bool:
    if not coord_a or not coord_b:
        return False
    try:
        point_a = parse_location(coord_a)
        point_b = parse_location(coord_b)
    except Exception:
        return False
    return haversine_distance_meters(point_a, point_b) <= threshold_meters


def _normalize_intent(intent: UserIntent, user_query: str) -> UserIntent:
    updates: dict[str, Any] = {}
    if intent.start_location and intent.end_location and _normalize_text(intent.start_location) == _normalize_text(intent.end_location):
        updates["return_to_start"] = True
    if not updates:
        return intent
    return intent.model_copy(update=updates)


def _normalize_clarification(clarification: IntentClarification) -> IntentClarification:
    if clarification.is_clear:
        return clarification.model_copy(update={"clarification_question": None})
    if clarification.clarification_question:
        return clarification
    missing = "、".join(clarification.missing_info) if clarification.missing_info else "城市、起点、活动类型"
    fallback_question = f"还差一点关键信息：{missing}。补充一下，我就能开始规划。"
    return clarification.model_copy(update={"clarification_question": fallback_question})


# ============================================================
# POI extraction helpers (kept from original)
# ============================================================

def _normalize_poi_candidate(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "poi_id": _normalize_text(item.get("poi_id")),
        "name": _normalize_text(item.get("name")),
        "address": _normalize_text(item.get("address")),
        "location": _normalize_text(item.get("location")),
        "category": _normalize_text(item.get("category")),
        "distance_meters": item.get("distance_meters"),
        "position_hint": _normalize_text(item.get("position_hint")),
        "source_tool": _normalize_text(item.get("source_tool")),
        "source_keyword": _normalize_text(item.get("source_keyword")),
        "rating": _normalize_text(item.get("rating")),
        "cost": _normalize_text(item.get("cost")),
        "business_hours": _normalize_text(item.get("business_hours")),
        "tag": _normalize_text(item.get("tag")),
        "website": _normalize_text(item.get("website")),
        "tel": _normalize_text(item.get("tel")),
        "matched_keywords": _normalize_text_list(item.get("matched_keywords")),
        "sample_point": _normalize_text(item.get("sample_point")),
        "highlights": _normalize_text_list(item.get("highlights")),
        "description": _normalize_text(item.get("description")),
        "selection_reason": _normalize_text(item.get("selection_reason")),
    }


def _extract_poi_candidates_from_result(action: str, params: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(result, dict) or result.get("status") != "1":
        return []

    candidates: list[dict[str, Any]] = []
    keyword_value = params.get("keyword")
    if isinstance(keyword_value, list):
        source_keyword = "、".join(_normalize_text(item) for item in keyword_value if _normalize_text(item))
    else:
        source_keyword = _normalize_text(keyword_value or params.get("theme"))

    if action in {"search_nearby_places", "search_along_route", "search_along_polyline"}:
        for poi in list(result.get("results") or [])[:]:
            name = _normalize_text(poi.get("name"))
            location = _normalize_text(poi.get("location"))
            if not name or not location:
                continue
            candidates.append(
                {
                    "name": name,
                    "address": _normalize_text(poi.get("address")),
                    "location": location,
                    "category": _normalize_text(poi.get("type")),
                    "distance_meters": poi.get("distance_meters"),
                    "position_hint": _normalize_text(poi.get("position_hint")),
                    "source_tool": action,
                    "source_keyword": source_keyword,
                    "matched_keywords": _normalize_text_list(poi.get("matched_keywords")),
                    "sample_point": _normalize_text(poi.get("sample_point")),
                }
            )
        return candidates

    if action == "search_candidate_corridors":
        for corridor in list(result.get("corridors") or [])[:5]:
            name = _normalize_text(corridor.get("name"))
            location = _normalize_text(corridor.get("center"))
            if not name or not location:
                continue
            theme_tags = list(corridor.get("theme_tags") or [])
            candidates.append(
                {
                    "name": name,
                    "address": _normalize_text(corridor.get("city")),
                    "location": location,
                    "category": "候选漫步廊道",
                    "distance_meters": None,
                    "position_hint": _normalize_text(corridor.get("direction")),
                    "source_tool": action,
                    "source_keyword": "、".join(str(tag).strip() for tag in theme_tags if str(tag).strip()),
                }
            )
        return candidates

    if action == "get_place_details":
        poi = result.get("poi") if isinstance(result.get("poi"), dict) else {}
        name = _normalize_text(poi.get("name"))
        location = _normalize_text(poi.get("location"))
        if not name or not location:
            return []

        candidates.append(
            {
                "poi_id": _normalize_text(poi.get("poi_id")),
                "name": name,
                "address": _normalize_text(poi.get("address")),
                "location": location,
                "category": _normalize_text(poi.get("type")),
                "distance_meters": None,
                "position_hint": "",
                "source_tool": action,
                "source_keyword": _normalize_text(params.get("place_name")),
                "rating": _normalize_text(poi.get("rating")),
                "cost": _normalize_text(poi.get("cost")),
                "business_hours": _normalize_text(poi.get("business_hours")),
                "tag": _normalize_text(poi.get("tag")),
                "website": _normalize_text(poi.get("website")),
                "tel": _normalize_text(poi.get("tel")),
                "highlights": result.get("highlights"),
                "description": _normalize_text(result.get("description")),
            }
        )
        return candidates

    return []


def _merge_poi_candidates(*candidate_groups: list[dict[str, Any]], limit: int = 1000) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []

    for group in candidate_groups:
        for item in group:
            normalized = _normalize_poi_candidate(item)
            name = normalized["name"]
            location = normalized["location"]
            if not name or not location:
                continue
            key = f"{name}|{location}"
            if key not in merged:
                merged[key] = normalized
                ordered_keys.append(key)
                continue

            stored = merged[key]
            for field in [
                "poi_id",
                "address",
                "category",
                "position_hint",
                "source_tool",
                "source_keyword",
                "rating",
                "cost",
                "business_hours",
                "tag",
                "website",
                "tel",
                "sample_point",
                "description",
                "selection_reason",
            ]:
                if not stored.get(field) and normalized.get(field):
                    stored[field] = normalized[field]
            if isinstance(normalized.get("distance_meters"), int):
                if stored.get("distance_meters") is None or (
                    isinstance(stored.get("distance_meters"), int)
                    and normalized["distance_meters"] < stored["distance_meters"]
                ):
                    stored["distance_meters"] = normalized["distance_meters"]
            merged_highlights = _normalize_text_list((stored.get("highlights") or []) + (normalized.get("highlights") or []))
            if merged_highlights:
                stored["highlights"] = merged_highlights
            merged_keywords = _normalize_text_list((stored.get("matched_keywords") or []) + (normalized.get("matched_keywords") or []))
            if merged_keywords:
                stored["matched_keywords"] = merged_keywords

    return [merged[key] for key in ordered_keys[:limit]]


# ============================================================
# Coordinate resolution helpers (kept from original)
# ============================================================

def _resolve_endpoint_coords(
    execution_results: list[dict[str, Any]],
    intent: UserIntent | None = None,
) -> tuple[str | None, str | None]:
    if intent:
        start_location_name = _normalize_text(intent.start_location)
        end_location_name = _normalize_text(intent.end_location)
        start_coords: str | None = None
        end_coords: str | None = None

        for item in execution_results:
            if item.get("action") != "get_coordinates":
                continue
            params = item.get("params", {})
            result = item.get("result", {})
            address = _normalize_text(params.get("address"))
            location = _normalize_text(result.get("location"))
            if not location:
                continue
            if not start_coords and start_location_name and address == start_location_name:
                start_coords = location
            if not end_coords and end_location_name and address == end_location_name:
                end_coords = location

        if start_coords or end_coords:
            return start_coords, end_coords

    coords: list[str] = []
    for item in execution_results:
        if item.get("action") != "get_coordinates":
            continue
        result = item.get("result", {})
        location = _normalize_text(result.get("location"))
        if location:
            coords.append(location)

    if not coords:
        return None, None
    if len(coords) == 1:
        return coords[0], None
    return coords[0], coords[1]


# ============================================================
# Route / polyline helpers (kept from original)
# ============================================================

def _has_successful_route(records: list[dict[str, Any]]) -> bool:
    return any(
        item.get("action") in {"get_walking_route_text", "calculate_walking_route", "get_detailed_walking_route"}
        and isinstance(item.get("result"), dict)
        and item["result"].get("status") == "1"
        for item in records
    )


def _has_detailed_route(records: list[dict[str, Any]]) -> bool:
    return any(
        item.get("action") == "get_detailed_walking_route"
        and isinstance(item.get("result"), dict)
        and item["result"].get("status") == "1"
        for item in records
    )


def _has_submitted_route_plan(records: list[dict[str, Any]]) -> bool:
    return any(item.get("action") == "SubmitRoutePlan" for item in records)


def _latest_submitted_route_plan_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(records):
        if item.get("action") != "SubmitRoutePlan":
            continue
        result = item.get("result", {})
        if isinstance(result, dict):
            return item
    return None


def _latest_successful_multi_waypoint_route_result(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(records):
        if item.get("action") != "plan_multi_waypoint_route":
            continue
        result = item.get("result", {})
        if isinstance(result, dict) and result.get("status") == "1":
            return result
    return None


def _route_plan_stop_locations(route_plan: dict[str, Any] | None) -> list[str]:
    if not isinstance(route_plan, dict):
        return []
    stop_locations: list[str] = []
    for stop in list(route_plan.get("selected_stops") or []):
        if not isinstance(stop, dict):
            continue
        location = _normalize_text(stop.get("location") or stop.get("coordinates"))
        if location:
            stop_locations.append(location)
    return stop_locations


def _resolve_route_plan_stop_coordinates(
    state: AgentState,
    route_plan: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(route_plan, dict):
        return None

    intent = state.get("intent")
    city = _normalize_text(intent.city) if intent else ""
    resolved_stops: list[dict[str, Any]] = []

    for stop in list(route_plan.get("selected_stops") or []):
        if not isinstance(stop, dict):
            continue

        resolved_stop = dict(stop)
        stop_name = _normalize_text(stop.get("name"))
        if stop_name:
            resolved = execute_tool("get_coordinates", {"address": stop_name, "city": city})
            resolved_location = _normalize_text((resolved or {}).get("location"))
            if isinstance(resolved, dict) and resolved.get("status") == "1" and resolved_location:
                resolved_stop["location"] = resolved_location
                if "coordinates" in resolved_stop:
                    resolved_stop["coordinates"] = resolved_location

        resolved_stops.append(resolved_stop)

    resolved_route_plan = dict(route_plan)
    resolved_route_plan["selected_stops"] = resolved_stops
    return resolved_route_plan


def _normalize_waypoint_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_normalize_text(item) for item in value if _normalize_text(item)]


def _same_coordinate_sequence(
    coords_a: list[str],
    coords_b: list[str],
    threshold_meters: float = 30.0,
) -> bool:
    if len(coords_a) != len(coords_b):
        return False
    return all(_same_coordinates(coord_a, coord_b, threshold_meters) for coord_a, coord_b in zip(coords_a, coords_b))


def _matching_multi_waypoint_route_record(
    state: AgentState,
    records: list[dict[str, Any]],
    route_plan: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(route_plan, dict):
        return None

    submitted_polyline = _normalize_text(route_plan.get("route_polyline"))
    submitted_waypoints = _route_plan_stop_locations(route_plan)
    expected_params = _route_materialization_params(state, route_plan) or {}
    expected_origin = _normalize_text(expected_params.get("origin_coords"))
    expected_destination = _normalize_text(expected_params.get("destination_coords"))
    expected_waypoints = _normalize_waypoint_list(expected_params.get("waypoints"))

    for item in reversed(records):
        if item.get("action") != "plan_multi_waypoint_route":
            continue
        result = item.get("result", {})
        if not isinstance(result, dict) or result.get("status") != "1":
            continue

        if submitted_polyline and _normalize_text(result.get("polyline")) == submitted_polyline:
            return item

        params = item.get("params") or {}
        origin_coords = _normalize_text(params.get("origin_coords"))
        destination_coords = _normalize_text(params.get("destination_coords"))
        waypoints = _normalize_waypoint_list(params.get("waypoints"))
        if (
            expected_origin
            and expected_destination
            and _same_coordinates(origin_coords, expected_origin)
            and _same_coordinates(destination_coords, expected_destination)
            and _same_coordinate_sequence(waypoints, expected_waypoints)
        ):
            return item

        if (
            not expected_origin
            and not expected_destination
            and submitted_waypoints
            and _same_coordinate_sequence(waypoints, submitted_waypoints)
        ):
            return item

    return None


def _route_materialization_params(
    state: AgentState,
    route_plan: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(route_plan, dict):
        return None

    intent = state.get("intent")
    start_coords, explicit_end_coords = _resolve_endpoint_coords(state.get("execution_results", []), intent)
    if not start_coords:
        return None

    stop_locations = _route_plan_stop_locations(route_plan)
    return_to_start = bool(intent.return_to_start) if intent else False

    if explicit_end_coords:
        destination_coords = explicit_end_coords
        waypoints = stop_locations
    elif return_to_start:
        destination_coords = start_coords
        waypoints = stop_locations
    elif stop_locations:
        destination_coords = stop_locations[-1]
        waypoints = stop_locations[:-1]
    else:
        return None

    normalized_waypoints = [coords for coords in waypoints if not _same_coordinates(coords, start_coords)]
    normalized_waypoints = [
        coords for coords in normalized_waypoints
        if not _same_coordinates(coords, destination_coords)
    ]

    return {
        "origin_coords": start_coords,
        "destination_coords": destination_coords,
        "waypoints": normalized_waypoints,
    }


def _materialize_submitted_route_plan(
    state: AgentState,
    route_plan: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(route_plan, dict):
        return None, None

    resolved_route_plan = _resolve_route_plan_stop_coordinates(state, route_plan) or dict(route_plan)

    params = _route_materialization_params(state, resolved_route_plan)
    if not params:
        return resolved_route_plan, None

    result = execute_tool("plan_multi_waypoint_route", params)
    materialized_record = {
        "action": "plan_multi_waypoint_route",
        "params": params,
        "result": result,
    }

    enriched_route_plan = dict(resolved_route_plan)
    if isinstance(result, dict) and result.get("status") == "1":
        enriched_route_plan["total_distance_km"] = result.get("total_distance_km")
        enriched_route_plan["total_duration_minutes"] = result.get("total_duration_minutes")
        enriched_route_plan["route_polyline"] = result.get("polyline")
    else:
        enriched_route_plan["total_distance_km"] = None
        enriched_route_plan["total_duration_minutes"] = None
        enriched_route_plan["route_polyline"] = None

    return enriched_route_plan, materialized_record


def _primary_route_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(records):
        if item.get("action") == "get_detailed_walking_route":
            result = item.get("result", {})
            if isinstance(result, dict) and result.get("status") == "1":
                return item
    return None


def _successful_detailed_route_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in records
        if item.get("action") == "get_detailed_walking_route"
        and isinstance(item.get("result"), dict)
        and item["result"].get("status") == "1"
    ]


def _route_progress_label(progress_ratio: float) -> str:
    if progress_ratio <= 0.18:
        return "起步段"
    if progress_ratio <= 0.42:
        return "前段"
    if progress_ratio <= 0.68:
        return "中段"
    if progress_ratio <= 0.9:
        return "后段"
    return "临近终点"


def _annotate_candidates_for_route(
    candidates: list[dict[str, Any]],
    detailed_route_record: dict[str, Any] | None,
    *,
    start_coords: str | None = None,
    end_coords: str | None = None,
    max_candidates: int = 14,
) -> list[dict[str, Any]]:
    if not candidates or not detailed_route_record:
        return []

    result = detailed_route_record.get("result", {})
    polyline = _normalize_text(result.get("full_polyline"))
    if not polyline:
        return []

    polyline_points = parse_polyline(polyline)
    if not polyline_points:
        return []

    route_length = len(polyline_points)
    annotated: list[dict[str, Any]] = []

    for candidate in candidates:
        location = _normalize_text(candidate.get("location"))
        if not location:
            continue
        try:
            poi_point = parse_location(location)
        except Exception:
            continue

        nearest_index = 0
        nearest_distance = float("inf")
        for index, route_point in enumerate(polyline_points):
            distance = haversine_distance_meters(poi_point, route_point)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index

        progress_ratio = nearest_index / max(route_length - 1, 1)
        annotated.append(
            {
                **candidate,
                "_route_index": nearest_index,
                "_route_distance": nearest_distance,
                "_route_progress": round(progress_ratio, 3),
                "_route_position": _route_progress_label(progress_ratio),
            }
        )

    if not annotated:
        return []

    annotated.sort(key=lambda item: (item["_route_distance"], item["_route_index"]))
    return annotated[:max_candidates]


def _poi_matches_existing_location(location: str, items: list[dict[str, Any]]) -> bool:
    if not location:
        return False
    for item in items:
        existing_location = _normalize_text(item.get("location") or item.get("coordinates"))
        if existing_location and _same_coordinates(location, existing_location):
            return True
    return False


def _finalize_nearby_route_pois(
    selected_candidates: list[dict[str, Any]],
    route_polyline: str | None,
    selected_stops: list[dict[str, Any]],
    explored_pois: list[dict[str, Any]],
    *,
    min_route_distance_meters: float = 0.0,
    max_route_distance_meters: float = 500.0,
    limit: int = 6,
) -> list[dict[str, Any]]:
    polyline = _normalize_text(route_polyline)
    if not polyline:
        return []

    if not selected_candidates:
        return []

    annotated = _annotate_candidates_for_route(
        selected_candidates,
        {"result": {"full_polyline": polyline}},
        max_candidates=max(limit * 3, len(selected_candidates) + 2),
    )
    if not annotated:
        return []

    annotated_by_key = {
        f"{_normalize_text(item.get('name'))}|{_normalize_text(item.get('location'))}": item
        for item in annotated
        if _normalize_text(item.get("name")) and _normalize_text(item.get("location"))
    }

    finalized: list[dict[str, Any]] = []
    seen_locations: list[dict[str, Any]] = []
    for candidate in selected_candidates:
        name = _normalize_text(candidate.get("name"))
        location = _normalize_text(candidate.get("location"))
        if not name or not location:
            continue
        annotated_candidate = annotated_by_key.get(f"{name}|{location}")
        if not annotated_candidate:
            continue

        route_distance = annotated_candidate.get("_route_distance")
        if not isinstance(route_distance, (int, float)):
            continue
        if route_distance < min_route_distance_meters or route_distance > max_route_distance_meters:
            continue
        if _poi_matches_existing_location(location, selected_stops):
            continue
        if _poi_matches_existing_location(location, explored_pois):
            continue
        if _poi_matches_existing_location(location, seen_locations):
            continue

        matched_keywords = _normalize_text_list(annotated_candidate.get("matched_keywords"))
        distance_to_route = int(round(route_distance))
        position_hint = _normalize_text(annotated_candidate.get("_route_position") or annotated_candidate.get("position_hint"))
        selection_reason = _normalize_text(candidate.get("selection_reason"))
        if not selection_reason:
            selection_reason = _normalize_text(annotated_candidate.get("selection_reason"))

        finalized.append(
            {
                "name": name,
                "coordinates": location,
                "address": _normalize_text(annotated_candidate.get("address")),
                "category": _normalize_text(annotated_candidate.get("category")),
                "distance_to_route_meters": distance_to_route,
                "position_hint": position_hint,
                "matched_keywords": matched_keywords,
                "selection_reason": selection_reason,
                "source_sample_point": _normalize_text(annotated_candidate.get("sample_point")),
            }
        )
        seen_locations.append({"coordinates": location})
        if len(finalized) >= limit:
            break

    return finalized


def _ordered_route_stops_from_candidates(
    candidates: list[dict[str, Any]],
    detailed_route_record: dict[str, Any] | None,
    *,
    start_coords: str | None = None,
    end_coords: str | None = None,
) -> list[dict[str, Any]]:
    annotated = _annotate_candidates_for_route(
        candidates,
        detailed_route_record,
        start_coords=start_coords,
        end_coords=end_coords,
        max_candidates=max(18, len(candidates) + 4),
    )
    if not annotated:
        return []

    annotated.sort(key=lambda item: item.get("_route_index", 0))
    ordered_stops: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in annotated:
        coordinates = _normalize_text(item.get("location"))
        name = _normalize_text(item.get("name"))
        if not name or not coordinates:
            continue
        dedupe_key = f"{name}|{coordinates}"
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        ordered_stops.append(
            {
                "name": name,
                "address": _normalize_text(item.get("address")),
                "coordinates": coordinates,
                "category": _normalize_text(item.get("category")),
                "position_hint": _normalize_text(item.get("_route_position") or item.get("position_hint")),
                "recommended_stay_minutes": int(item.get("recommended_stay_minutes") or (20 if "咖啡" in _normalize_text(item.get("category")) else 15)),
                "selection_reason": _normalize_text(item.get("selection_reason")),
            }
        )
    return ordered_stops


# ============================================================
# Route chain building helpers (kept from original)
# ============================================================

def _route_cumulative_distances(polyline_points: list[tuple[float, float]]) -> list[float]:
    if not polyline_points:
        return []
    cumulative = [0.0]
    for index in range(1, len(polyline_points)):
        cumulative.append(
            cumulative[-1] + haversine_distance_meters(polyline_points[index - 1], polyline_points[index])
        )
    return cumulative


def _nearest_polyline_index(
    polyline_points: list[tuple[float, float]],
    target_point: tuple[float, float],
    *,
    start_index: int = 0,
) -> tuple[int, float]:
    nearest_index = start_index
    nearest_distance = float("inf")
    for index in range(start_index, len(polyline_points)):
        distance = haversine_distance_meters(target_point, polyline_points[index])
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = index
    return nearest_index, nearest_distance


def _rebalance_segment_minutes(segments: list[dict[str, Any]], target_total: int | None) -> None:
    if not isinstance(target_total, int) or not segments:
        return
    current_total = sum(item.get("walk_minutes", 0) for item in segments if isinstance(item.get("walk_minutes"), int))
    diff = target_total - current_total
    if diff == 0:
        return
    for item in reversed(segments):
        walk_minutes = item.get("walk_minutes")
        if isinstance(walk_minutes, int):
            item["walk_minutes"] = max(0, walk_minutes + diff)
            return


def _estimate_segments_with_route(
    chain_points: list[dict[str, Any]],
    route_record: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not route_record or len(chain_points) < 2:
        return []

    result = route_record.get("result", {})
    polyline = _normalize_text(result.get("full_polyline"))
    polyline_points = parse_polyline(polyline) if polyline else []
    if len(polyline_points) < 2:
        return []

    cumulative_distances = _route_cumulative_distances(polyline_points)
    total_distance = int(result.get("distance_meters") or round(cumulative_distances[-1]) or 0)
    total_minutes = result.get("duration_minutes") if isinstance(result.get("duration_minutes"), int) else None

    markers: list[dict[str, Any]] = []
    search_start = 0
    for point in chain_points:
        coords = _normalize_text(point.get("coordinates"))
        if not coords:
            return []
        try:
            parsed_point = parse_location(coords)
        except Exception:
            return []
        nearest_index, nearest_distance = _nearest_polyline_index(
            polyline_points,
            parsed_point,
            start_index=search_start,
        )
        if nearest_distance > 700:
            return []
        markers.append({"index": nearest_index, "distance_to_route": nearest_distance})
        search_start = nearest_index

    segments: list[dict[str, Any]] = []
    for index in range(len(chain_points) - 1):
        start_marker = markers[index]
        end_marker = markers[index + 1]
        start_index = start_marker["index"]
        end_index = max(start_index, end_marker["index"])
        distance_meters = max(0, int(round(cumulative_distances[end_index] - cumulative_distances[start_index])))
        walk_minutes: int | None = None
        if total_minutes is not None and total_distance > 0:
            walk_minutes = max(1, int(round(total_minutes * distance_meters / total_distance))) if distance_meters > 0 else 0

        segments.append(
            {
                "order": index + 1,
                "from_stop": chain_points[index]["name"],
                "to_stop": chain_points[index + 1]["name"],
                "distance_meters": distance_meters,
                "walk_minutes": walk_minutes,
            }
        )

    _rebalance_segment_minutes(segments, total_minutes)
    return segments


def _matching_detailed_route_record(
    route_records: list[dict[str, Any]],
    origin_coords: str | None,
    destination_coords: str | None,
) -> dict[str, Any] | None:
    if not origin_coords or not destination_coords:
        return None
    for item in reversed(_successful_detailed_route_records(route_records)):
        result = item.get("result", {})
        if _same_coordinates(_normalize_text(result.get("origin")), origin_coords) and _same_coordinates(
            _normalize_text(result.get("destination")),
            destination_coords,
        ):
            return item
    return None


def _pick_outbound_route_record(
    route_records: list[dict[str, Any]],
    start_coords: str | None,
    explicit_end_coords: str | None,
) -> dict[str, Any] | None:
    detailed_records = _successful_detailed_route_records(route_records)
    if not detailed_records:
        return None
    exact_match = _matching_detailed_route_record(route_records, start_coords, explicit_end_coords)
    if exact_match:
        return exact_match
    if start_coords:
        candidates = [
            item
            for item in detailed_records
            if _same_coordinates(_normalize_text((item.get("result") or {}).get("origin")), start_coords)
        ]
        if candidates:
            return max(candidates, key=lambda item: int((item.get("result") or {}).get("distance_meters") or 0))
    return max(detailed_records, key=lambda item: int((item.get("result") or {}).get("distance_meters") or 0))


def _pick_return_route_record(
    route_records: list[dict[str, Any]],
    outbound_route_record: dict[str, Any] | None,
    start_coords: str | None,
    explicit_end_coords: str | None,
) -> dict[str, Any] | None:
    if not start_coords:
        return None
    exact_match = _matching_detailed_route_record(route_records, explicit_end_coords, start_coords)
    if exact_match:
        return exact_match
    outbound_destination = _normalize_text((outbound_route_record or {}).get("result", {}).get("destination"))
    if outbound_destination:
        return _matching_detailed_route_record(route_records, outbound_destination, start_coords)
    return None


def _build_route_chain_record(
    state: AgentState,
    route_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    intent = state.get("intent")
    accepted_pois = list(state.get("explored_pois", []))

    start_coords, explicit_end_coords = _resolve_endpoint_coords(state.get("execution_results", []), intent)
    if not start_coords:
        return None

    outbound_route_record = _pick_outbound_route_record(route_records, start_coords, explicit_end_coords)
    if not outbound_route_record:
        return None
    return_route_record = _pick_return_route_record(route_records, outbound_route_record, start_coords, explicit_end_coords)

    start_name = intent.start_location if intent else "起点"
    explicit_end_name = intent.end_location if intent and intent.end_location else None
    return_to_start = bool(intent.return_to_start) if intent else False

    route_candidates = _annotate_candidates_for_route(
        accepted_pois,
        outbound_route_record,
        start_coords=start_coords,
        end_coords=explicit_end_coords,
        max_candidates=16,
    )
    selected_stops = _ordered_route_stops_from_candidates(
        accepted_pois,
        outbound_route_record,
        start_coords=start_coords,
        end_coords=explicit_end_coords,
    )
    if not selected_stops:
        selected_stops = [
            {
                "name": _normalize_text(item.get("name")),
                "address": _normalize_text(item.get("address")),
                "coordinates": _normalize_text(item.get("location")),
                "category": _normalize_text(item.get("category")),
                "position_hint": _normalize_text(item.get("position_hint")),
                "recommended_stay_minutes": int(item.get("recommended_stay_minutes") or 15),
                "selection_reason": _normalize_text(item.get("selection_reason")),
            }
            for item in accepted_pois
            if _normalize_text(item.get("name")) and _normalize_text(item.get("location"))
        ]

    display_stops: list[dict[str, Any]] = []
    for stop in selected_stops:
        coordinates = _normalize_text(stop.get("coordinates") or stop.get("location"))
        if not coordinates:
            continue
        display_stops.append(
            {
                "name": _normalize_text(stop.get("name")) or "停靠点",
                "address": _normalize_text(stop.get("address")),
                "coordinates": coordinates,
                "category": _normalize_text(stop.get("category")),
                "position_hint": _normalize_text(stop.get("position_hint")),
                "recommended_stay_minutes": int(stop.get("recommended_stay_minutes") or 15),
                "selection_reason": _normalize_text(stop.get("selection_reason")),
            }
        )

    outbound_result = outbound_route_record.get("result", {})
    outbound_destination = _normalize_text(outbound_result.get("destination")) or explicit_end_coords
    outbound_end_name = explicit_end_name or _normalize_text((outbound_route_record.get("params") or {}).get("to_name")) or "终点"

    if explicit_end_coords and explicit_end_name:
        display_stops.append(
            {
                "name": explicit_end_name,
                "address": "",
                "coordinates": explicit_end_coords,
                "category": "终点",
                "position_hint": "终点",
                "recommended_stay_minutes": 20,
            }
        )
    elif return_to_start and outbound_destination and not any(
        _same_coordinates(_normalize_text(item.get("coordinates")), outbound_destination) for item in display_stops
    ):
        display_stops.append(
            {
                "name": outbound_end_name,
                "address": "",
                "coordinates": outbound_destination,
                "category": "折返点",
                "position_hint": "折返前",
                "recommended_stay_minutes": 15,
            }
        )

    outbound_chain_points = [{"name": start_name, "coordinates": start_coords}] + [
        {"name": stop["name"], "coordinates": stop["coordinates"]} for stop in display_stops
    ]
    segments = _estimate_segments_with_route(outbound_chain_points, outbound_route_record)

    if return_to_start:
        last_outbound_point = outbound_chain_points[-1] if outbound_chain_points else {"name": start_name, "coordinates": start_coords}
        return_segments = _estimate_segments_with_route(
            [
                last_outbound_point,
                {"name": f"返回{start_name}", "coordinates": start_coords},
            ],
            return_route_record,
        )
        if return_segments:
            for item in return_segments:
                item["order"] = len(segments) + item["order"]
            segments.extend(return_segments)
            display_stops.append(
                {
                    "name": f"返回{start_name}",
                    "address": "",
                    "coordinates": start_coords,
                    "category": "返回起点",
                    "position_hint": "返回",
                    "recommended_stay_minutes": 0,
                    "is_return_to_start": True,
                }
            )

    for index, stop in enumerate(display_stops, start=1):
        stop["order"] = index
        segment = segments[index - 1] if index - 1 < len(segments) else {}
        stop["walk_from_previous_minutes"] = segment.get("walk_minutes")

    total_stay_minutes = sum(int(stop.get("recommended_stay_minutes") or 0) for stop in display_stops)
    total_walking_minutes = sum(item.get("walk_minutes", 0) for item in segments if isinstance(item.get("walk_minutes"), int))
    total_distance_meters = sum(item.get("distance_meters", 0) for item in segments if isinstance(item.get("distance_meters"), int))
    final_polyline = merge_polylines(
        [
            _normalize_text(outbound_result.get("full_polyline")),
            _normalize_text((return_route_record or {}).get("result", {}).get("full_polyline")),
        ]
    )
    middle_stop_names = [stop["name"] for stop in display_stops if not stop.get("is_return_to_start") and stop["name"] != explicit_end_name]

    if explicit_end_name and return_to_start:
        route_title = f"{start_name}到{explicit_end_name}往返漫步"
    elif explicit_end_name:
        route_title = f"{start_name}到{explicit_end_name}漫步"
    elif return_to_start:
        route_title = f"{start_name}环线漫步"
    else:
        route_title = f"{start_name}出发漫步"

    if explicit_end_name and return_to_start:
        route_summary = (
            f"从{start_name}步行前往{explicit_end_name}，沿线串联"
            f"{'、'.join(middle_stop_names) if middle_stop_names else '沿途步道'}，最后回到起点。"
        )
    elif explicit_end_name:
        route_summary = (
            f"从{start_name}步行前往{explicit_end_name}，沿线串联"
            f"{'、'.join(middle_stop_names) if middle_stop_names else '沿途步道'}，整体偏向不绕路的轻松漫步。"
        )
    elif return_to_start:
        route_summary = (
            f"从{start_name}出发，沿线串联"
            f"{'、'.join(middle_stop_names) if middle_stop_names else '周边步道'}，最后回到起点。"
        )
    else:
        route_summary = (
            f"从{start_name}出发，沿线串联"
            f"{'、'.join(stop['name'] for stop in display_stops) if display_stops else '周边步道'}，整体偏向轻松散步。"
        )

    return {
        "action": "route_stop_chain",
        "params": {
            "start_name": start_name,
            "end_name": explicit_end_name,
            "return_to_start": return_to_start,
        },
        "result": {
            "status": "1",
            "route_title": route_title,
            "route_summary": route_summary,
            "stops": display_stops,
            "segments": segments,
            "total_walking_minutes": total_walking_minutes or outbound_result.get("duration_minutes"),
            "total_duration_minutes": (total_walking_minutes or outbound_result.get("duration_minutes", 0)) + total_stay_minutes,
            "full_polyline": final_polyline,
            "distance_meters": total_distance_meters or outbound_result.get("distance_meters"),
            "return_to_start": return_to_start,
            "candidate_count": len(route_candidates),
        },
        "reason": "基于主路线和沿线 POI 顺序自动生成最终串联结果",
    }


def _resolve_final_route_output_data(state: AgentState) -> dict[str, Any]:
    accepted_routes = list(state.get("planned_routes", []))
    latest_multi_waypoint_result = _latest_successful_multi_waypoint_route_result(accepted_routes)

    latest_route_plan_record = _latest_submitted_route_plan_record(accepted_routes)
    route_plan = (
        latest_route_plan_record.get("result")
        if latest_route_plan_record and isinstance(latest_route_plan_record.get("result"), dict)
        else state.get("route_plan")
    )

    if route_plan:
        selected_stops = list(route_plan.get("selected_stops", []))
        matching_multi_waypoint_record = _matching_multi_waypoint_route_record(state, accepted_routes, route_plan)
        matching_multi_waypoint_result = (
            matching_multi_waypoint_record.get("result", {})
            if matching_multi_waypoint_record
            else {}
        )

        total_distance_km = route_plan.get("total_distance_km")
        total_duration_minutes = route_plan.get("total_duration_minutes")
        route_polyline = route_plan.get("route_polyline")
        if matching_multi_waypoint_result:
            if total_distance_km is None:
                total_distance_km = matching_multi_waypoint_result.get("total_distance_km")
            if total_duration_minutes is None:
                total_duration_minutes = matching_multi_waypoint_result.get("total_duration_minutes")
            if not route_polyline:
                route_polyline = matching_multi_waypoint_result.get("polyline")

        if total_distance_km is None or total_duration_minutes is None:
            total_duration = 0
            for record in reversed(accepted_routes):
                if record.get("action") != "get_detailed_walking_route":
                    continue
                result = record.get("result", {})
                if result.get("status") != "1":
                    continue
                duration_minutes = result.get("duration_minutes")
                if isinstance(duration_minutes, int):
                    total_duration += duration_minutes

            if total_duration_minutes is None and total_duration > 0:
                total_duration_minutes = total_duration

        stops = [
            {
                "order": idx + 1,
                "name": stop.get("name"),
                "coordinates": stop.get("location"),
                "walk_from_previous_minutes": None,
                "recommended_stay_minutes": 15,
            }
            for idx, stop in enumerate(selected_stops)
        ]
        return {
            "route_plan": route_plan,
            "selected_stops": selected_stops,
            "stops": stops,
            "route_summary": route_plan.get("reasoning", ""),
            "total_duration_minutes": total_duration_minutes,
            "route_polyline": route_polyline,
        }

    route_chain_record = _build_route_chain_record(state, accepted_routes)
    chain_result = route_chain_record.get("result", {}) if route_chain_record else {}
    stops = list(chain_result.get("stops", []))
    return {
        "route_plan": None,
        "selected_stops": stops,
        "stops": stops,
        "route_summary": chain_result.get("route_summary", ""),
        "total_duration_minutes": None,
        "route_polyline": latest_multi_waypoint_result.get("polyline") if latest_multi_waypoint_result else None,
    }


def _build_route_poi_enricher_context(state: AgentState, route_data: dict[str, Any]) -> str:
    intent = state.get("intent")
    selected_stops = list(route_data.get("selected_stops") or [])
    explored_pois = list(state.get("explored_pois", []))
    route_polyline = _normalize_text(route_data.get("route_polyline"))
    retrieved_note_contexts = _serialize_retrieved_note_contexts(state)

    return json.dumps(
        {
            "task": "为最终路线补充主路线周围的可选探索点，不改变主路线本身。",
            "intent": intent.model_dump() if intent else None,
            "selected_stops": [
                {
                    "name": _normalize_text(stop.get("name")),
                    "coordinates": _normalize_text(stop.get("location") or stop.get("coordinates")),
                    "selection_reason": _normalize_text(stop.get("selection_reason") or stop.get("reason")),
                }
                for stop in selected_stops
                if _normalize_text(stop.get("name"))
            ],
            "explored_pois": [
                {
                    "name": _normalize_text(poi.get("name")),
                    "category": _normalize_text(poi.get("category")),
                    "source_keyword": _normalize_text(poi.get("source_keyword")),
                    "selection_reason": _normalize_text(poi.get("selection_reason")),
                }
                for poi in explored_pois[:10]
                if _normalize_text(poi.get("name"))
            ],
            "retrieved_note_contexts": retrieved_note_contexts,
            "reference_rules": _build_retrieved_notes_reference_rules(),
            "route_polyline_available": bool(route_polyline),
            "route_point_count": len(parse_polyline(route_polyline)) if route_polyline else 0,
            "rules": {
                "main_route_must_stay_unchanged": True,
                "candidate_distance_to_route_meters": [10, 250],
                "max_optional_pois": 6,
                "do_not_repeat_selected_stops": True,
            },
        },
        ensure_ascii=False,
    )


# ============================================================
# Context builders (replace _render_... functions)
# ============================================================

def _build_poi_context(state: AgentState, task_description: str) -> str:
    """Build JSON context for POI Explorer sub-agent."""
    intent = state.get("intent")
    start_coords, end_coords = _resolve_endpoint_coords(state.get("execution_results", []), intent)
    retrieved_note_contexts = _serialize_retrieved_note_contexts(state)
    return json.dumps({
        "task": task_description,
        "intent": intent.model_dump() if intent else None,
        "known_coordinates": {"start": start_coords, "end": end_coords},
        "existing_pois": [
            {
                "name": p.get("name"),
                "location": p.get("location"),
                "category": p.get("category"),
                "rating": p.get("rating"),
                "highlights": p.get("highlights"),
                "description": p.get("description"),
                "selection_reason": p.get("selection_reason"),
            }
            for p in state.get("explored_pois", [])[:8]
        ],
        "retrieved_note_contexts": retrieved_note_contexts,
        "reference_rules": _build_retrieved_notes_reference_rules(),
    }, ensure_ascii=False, indent=2)


def _build_route_context(state: AgentState, task_description: str) -> str:
    """Build JSON context for Route Planner sub-agent."""
    intent = state.get("intent")
    start_coords, end_coords = _resolve_endpoint_coords(state.get("execution_results", []), intent)
    explored_pois = state.get("explored_pois", [])
    retrieved_note_contexts = _serialize_retrieved_note_contexts(state)
    return json.dumps({
        "task": task_description,
        "intent": intent.model_dump() if intent else None,
        "known_coordinates": {"start": start_coords, "end": end_coords},
        "poi_list": [
            {
                "name": p.get("name"),
                "location": p.get("location"),
                "category": p.get("category"),
                "rating": p.get("rating"),
                "highlights": p.get("highlights"),
                "description": p.get("description"),
                "selection_reason": p.get("selection_reason"),
            }
            for p in explored_pois[:]
        ],
        "existing_routes": [
            {
                "action": r.get("action"),
                "origin": (r.get("result") or {}).get("origin"),
                "destination": (r.get("result") or {}).get("destination"),
                "distance_meters": (r.get("result") or {}).get("distance_meters"),
                "duration_minutes": (r.get("result") or {}).get("duration_minutes"),
            }
            for r in state.get("planned_routes", [])[:5]
        ],
        "retrieved_note_contexts": retrieved_note_contexts,
        "reference_rules": _build_retrieved_notes_reference_rules(),
    }, ensure_ascii=False, indent=2)


def _build_supervisor_status(state: AgentState) -> str:
    """Build status summary for supervisor messages."""
    intent = state.get("intent")
    retrieved_info = list(state.get("retrieved_info", []))
    retrieved_note_contexts = _serialize_retrieved_note_contexts(state, limit=3)
    explored_pois = state.get("explored_pois", [])
    planned_routes = state.get("planned_routes", [])

    lines = []
    lines.append(f"用户意图: {intent.model_dump_json(ensure_ascii=False) if intent else '未解析'}")
    if retrieved_info:
        lines.append(f"小红书检索: {retrieved_info[-1]}")
    if retrieved_note_contexts:
        lines.append(f"相关小红书帖子: {len(state.get('retrieved_note_contexts', []))} 篇")
        for note in retrieved_note_contexts:
            hints: list[str] = []
            if note.get("poi_names"):
                hints.append(f"POI: {'、'.join(note['poi_names'][:4])}")
            if note.get("route_hints"):
                hints.append(f"路线提示: {' | '.join(note['route_hints'][:2])}")
            suffix = f"（{'；'.join(hints)}）" if hints else ""
            lines.append(f"  - {note.get('title') or '?'}{suffix}")
    else:
        lines.append("未检索到可用小红书帖子")
    lines.append(f"已探索 POI: {len(explored_pois)} 个")
    if explored_pois:
        names = [p.get("name", "?") for p in explored_pois[:5]]
        lines.append(f"  代表: {', '.join(names)}")
    lines.append(f"已规划路线: {len(planned_routes)} 条")

    route_plan = state.get("route_plan")
    if not isinstance(route_plan, dict):
        latest_route_plan_record = _latest_submitted_route_plan_record(planned_routes)
        route_plan = latest_route_plan_record.get("result") if latest_route_plan_record else None

    if isinstance(route_plan, dict):
        stop_names = [
            _normalize_text(stop.get("name"))
            for stop in list(route_plan.get("selected_stops") or [])
            if isinstance(stop, dict) and _normalize_text(stop.get("name"))
        ]
        route_text = " -> ".join(stop_names[:6]) if stop_names else "0 个停靠点"
        metrics = []
        total_distance_km = route_plan.get("total_distance_km")
        total_duration_minutes = route_plan.get("total_duration_minutes")
        if isinstance(total_distance_km, (int, float)):
            metrics.append(f"约 {total_distance_km} 公里")
        if isinstance(total_duration_minutes, (int, float)):
            metrics.append(f"约 {int(total_duration_minutes)} 分钟")
        metric_text = f"（{'，'.join(metrics)}）" if metrics else ""
        lines.append(f"  最新 SubmitRoutePlan: {route_text}{metric_text}")
    else:
        for r in planned_routes[-3:]:
            res = r.get("result", {})
            lines.append(f"  - {r.get('action')}: {res.get('distance_meters')}米, {res.get('duration_minutes')}分钟")
    return "\n".join(lines)


def _summarize_poi_result(sub_result: dict) -> str:
    """Summarize POI explorer sub-graph result for supervisor.

    包含每个精选 POI 的名称和选择理由，让 Supervisor 能评判质量。
    """
    pois = sub_result.get("explored_pois", [])
    candidate_pool = sub_result.get("_candidate_pool", [])
    if not pois:
        return "POI Explorer 未找到有效兴趣点。"
    lines = [f"POI Explorer 从 {len(candidate_pool)} 个候选中精选了 {len(pois)} 个:"]
    for p in pois[:8]:
        name = p.get("name", "?")
        reason = _normalize_text(p.get("selection_reason"))
        if reason:
            lines.append(f"  - {name}: {reason}")
        else:
            category = _normalize_text(p.get("category", "")).split(";")[0]
            lines.append(f"  - {name} ({category})" if category else f"  - {name}")
    return "\n".join(lines)


def _summarize_route_result(sub_result: dict) -> str:
    """Summarize Route Planner sub-graph result for supervisor."""
    records = sub_result.get("route_records", [])
    if not records:
        return "Route Planner 未生成有效路线。"

    latest_route_plan_record = _latest_submitted_route_plan_record(records)
    if latest_route_plan_record:
        route_plan = latest_route_plan_record.get("result", {})
        stop_names = [
            _normalize_text(stop.get("name"))
            for stop in list(route_plan.get("selected_stops") or [])
            if isinstance(stop, dict) and _normalize_text(stop.get("name"))
        ]
        route_text = " -> ".join(stop_names[:6]) if stop_names else "0 个停靠点"
        metrics = []
        total_distance_km = route_plan.get("total_distance_km")
        total_duration_minutes = route_plan.get("total_duration_minutes")
        if isinstance(total_distance_km, (int, float)):
            metrics.append(f"约 {total_distance_km} 公里")
        if isinstance(total_duration_minutes, (int, float)):
            metrics.append(f"约 {int(total_duration_minutes)} 分钟")
        metric_text = f"（{'，'.join(metrics)}）" if metrics else ""
        return f"Route Planner 最新 SubmitRoutePlan: {route_text}{metric_text}。"

    summaries = []
    for r in records[-3:]:
        res = r.get("result", {})
        if res.get("status") == "1":
            summaries.append(f"{res.get('distance_meters')}米/{res.get('duration_minutes')}分钟")
    return f"Route Planner 规划了 {len(records)} 条路线: {'; '.join(summaries) if summaries else '结果待确认'}。"


def _summarize_route_poi_enricher_result(sub_result: dict) -> str:
    pois = list(sub_result.get("nearby_route_pois", []))
    if not pois:
        return "Route POI Enricher 未补充额外可选点位。"

    lines = [f"Route POI Enricher 补充了 {len(pois)} 个主路线周边可选点位:"]
    for poi in pois[:6]:
        name = _normalize_text(poi.get("name")) or "?"
        distance = poi.get("distance_to_route_meters")
        position_hint = _normalize_text(poi.get("position_hint"))
        matched_keywords = "、".join(_normalize_text_list(poi.get("matched_keywords")))
        extras: list[str] = []
        if isinstance(distance, int):
            extras.append(f"距主路线约{distance}米")
        if position_hint:
            extras.append(position_hint)
        if matched_keywords:
            extras.append(f"匹配 {matched_keywords}")
        detail = f"（{'，'.join(extras)}）" if extras else ""
        lines.append(f"  - {name}{detail}")
    return "\n".join(lines)


def _format_poi_search_result(action: str, result: dict[str, Any]) -> str:
    """为 POI Explorer 的 ToolMessage 生成带名称和分类的搜索结果列表。

    比 summarize_tool_result_for_llm 更详细，让 LLM 能逐条判断哪些有趣。
    """
    if not isinstance(result, dict) or result.get("status") != "1":
        error = (result or {}).get("error", "未知错误") if isinstance(result, dict) else str(result)
        return f"搜索失败: {error}"

    if action == "get_place_details":
        poi = result.get("poi") if isinstance(result.get("poi"), dict) else {}
        name = _normalize_text(poi.get("name")) or "?"
        category = (_normalize_text(poi.get("type")) or "").split(";")[0]
        address = _normalize_text(poi.get("address"))
        rating = _normalize_text(poi.get("rating"))
        business_hours = _normalize_text(poi.get("business_hours"))
        highlights = list(result.get("highlights") or [])
        description = _normalize_text(result.get("description"))

        lines = [f"点位详情: {name} ({category or '未知类型'})"]
        if address:
            lines.append(f"  - 地址: {address}")
        if rating:
            lines.append(f"  - 评分: {rating}")
        if business_hours:
            lines.append(f"  - 营业时间: {business_hours}")
        if highlights:
            lines.append(f"  - 亮点: {'、'.join(str(h) for h in highlights[:4])}")
        if description:
            lines.append(f"  - 简介: {description}")
        matches = list(result.get("candidate_matches") or [])
        if len(matches) > 1:
            lines.append(
                "  - 同名候选: "
                + "；".join(
                    f"{_normalize_text(m.get('name'))}({_normalize_text(m.get('address'))})"
                    for m in matches[:3]
                    if _normalize_text(m.get("name"))
                )
            )
        return "\n".join(lines)

    # search_nearby_places / search_along_route / search_along_polyline
    pois = list(result.get("results") or [])
    if pois:
        lines = [f"找到 {len(pois)} 个地点:"]
        for poi in pois[:10]:
            name = _normalize_text(poi.get("name")) or "?"
            category = (_normalize_text(poi.get("type")) or "").split(";")[0]
            poi_id = _normalize_text(poi.get("poi_id"))
            dist = poi.get("distance_meters")
            dist_str = f", 距{dist}米" if isinstance(dist, int) else ""
            poi_id_text = f", poi_id={poi_id}" if poi_id else ""
            lines.append(f"  - {name} ({category}{dist_str}{poi_id_text})")
        if len(pois) > 10:
            lines.append(f"  ... 等共 {len(pois)} 个")
        return "\n".join(lines)

    # search_candidate_corridors
    corridors = list(result.get("corridors") or [])
    if corridors:
        lines = [f"找到 {len(corridors)} 条候选廊道:"]
        for c in corridors[:5]:
            name = _normalize_text(c.get("name")) or "?"
            samples = ", ".join(str(s) for s in (c.get("sample_pois") or [])[:3])
            lines.append(f"  - {name} (代表点位: {samples})")
        return "\n".join(lines)

    # get_coordinates
    if result.get("location"):
        return f"定位成功: {result.get('formatted_address', '?')}，坐标 {result['location']}"

    return summarize_tool_result_for_llm(action, result)


# ============================================================
# POI Explorer ReAct Subgraph
# ============================================================

async def _poi_explorer_llm(state: POIExplorerState, config: RunnableConfig):
    """LLM node: decide which tools to call."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        configurable_model
        .bind_tools(POI_EXPLORER_TOOLS)
        .with_config(configurable.model_config_for("poi_explorer"))
    )
    messages = state.get("messages", [])
    response = await model.ainvoke(messages)
    return Command(
        goto="poi_explorer_tools",
        update={
            "messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0),
        },
    )


async def _poi_explorer_tools(state: POIExplorerState, config: RunnableConfig):
    """Execute tool calls, return ToolMessages.

    数据流：
    - 搜索工具 → 结果暂存到 _candidate_pool
    - SelectPOIs → LLM 从候选池中精选，写入 explored_pois
    - ExplorationComplete → 纯结束信号，goto END
    """
    configurable = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])
    last_msg = messages[-1]

    if not last_msg.tool_calls:
        return Command(goto=END)

    current_tool_calls = state.get("tool_call_iterations", 0)
    max_tool_calls = configurable.max_tool_calls
    at_tool_limit = current_tool_calls >= max_tool_calls

    tool_messages = []
    candidate_pool: list[dict] = list(state.get("_candidate_pool", []))
    explored_pois: list[dict] = list(state.get("explored_pois", []))
    should_end = False
    executed_non_control_calls = 0

    for tc in last_msg.tool_calls:
        if tc["name"] == "ExplorationComplete":
            # 纯结束信号
            print(f"  🏁 [POI Explorer] ExplorationComplete: {tc['args'].get('summary', '')[:]}", flush=True)
            tool_messages.append(ToolMessage(
                content=tc["args"].get("summary", "探索完成"),
                tool_call_id=tc["id"],
                name=tc["name"],
            ))
            should_end = True
            continue

        if tc["name"] == "SelectPOIs":
            # 筛选：从候选池中匹配 LLM 选中的名称
            selected_names = {
                _normalize_text(p.get("name"))
                for p in tc["args"].get("selected_pois", [])
                if _normalize_text(p.get("name"))
            }
            reasons_map = {
                _normalize_text(p.get("name")): _normalize_text(p.get("reason"))
                for p in tc["args"].get("selected_pois", [])
                if _normalize_text(p.get("name"))
            }
            newly_selected = [
                {**c, "selection_reason": reasons_map.get(_normalize_text(c.get("name")), "")}
                for c in candidate_pool
                if _normalize_text(c.get("name")) in selected_names
            ]
            explored_pois = _merge_poi_candidates(explored_pois, newly_selected)

            selected_summary = "、".join(selected_names) if selected_names else "无"
            print(f"  ✂️  [POI Explorer] SelectPOIs: 从 {len(candidate_pool)} 个候选中精选 {len(newly_selected)} 个（仅为节点排列，并非是最终路线）：", flush=True)
            for p in tc["args"].get("selected_pois", [])[:6]:
                print(f"      ├─ {p.get('name', '?')}: {p.get('reason', '')[:60]}", flush=True)
            tool_messages.append(ToolMessage(
                content=f"已精选 {len(newly_selected)} 个点位: {selected_summary}",
                tool_call_id=tc["id"],
                name=tc["name"],
            ))
            continue

        # 普通搜索工具
        if at_tool_limit or (current_tool_calls + executed_non_control_calls) >= max_tool_calls:
            tool_messages.append(ToolMessage(
                content=f"已达到工具调用上限({max_tool_calls})，请调用 SelectPOIs 与 ExplorationComplete 收尾。",
                tool_call_id=tc["id"],
                name=tc["name"],
            ))
            continue

        params_brief = json.dumps(tc["args"], ensure_ascii=False)
        if len(params_brief) > 120:
            params_brief = params_brief[:117] + "..."
        print(f"  🔍 [POI Explorer] {tc['name']}({params_brief})", flush=True)

        result = execute_tool(tc["name"], tc["args"])
        executed_non_control_calls += 1

        # 输出结果摘要
        result_summary = summarize_tool_result_for_llm(tc["name"], result)
        print(f"      └─ {result_summary[:500]}", flush=True)

        extracted = _extract_poi_candidates_from_result(tc["name"], tc["args"], result)
        if extracted:
            candidate_pool = _merge_poi_candidates(candidate_pool, extracted)
            print(f"      └─ 候选池: +{len(extracted)} → 共 {len(candidate_pool)} 个", flush=True)

        tool_messages.append(ToolMessage(
            content=_format_poi_search_result(tc["name"], result),
            tool_call_id=tc["id"],
            name=tc["name"],
        ))

    update: dict[str, Any] = {
        "messages": tool_messages,
        "_candidate_pool": candidate_pool,
        "explored_pois": explored_pois,
        "tool_call_iterations": current_tool_calls + executed_non_control_calls,
    }
    if should_end:
        if not explored_pois and candidate_pool:
            # 兜底：如果 LLM 忘了调 SelectPOIs 就结束了，保留全部候选
            update["explored_pois"] = candidate_pool
        return Command(goto=END, update=update)

    if at_tool_limit:
        if not explored_pois and candidate_pool:
            # 已达上限且未完成筛选时，兜底保留候选
            update["explored_pois"] = candidate_pool
        return Command(goto=END, update=update)

    return Command(goto="poi_explorer", update=update)


# Build POI Explorer subgraph
_poi_builder = StateGraph(POIExplorerState, config_schema=Configuration)
_poi_builder.add_node("poi_explorer", _poi_explorer_llm)
_poi_builder.add_node("poi_explorer_tools", _poi_explorer_tools)
_poi_builder.add_edge(START, "poi_explorer")
poi_explorer_subgraph = _poi_builder.compile()


# ============================================================
# Route Planner ReAct Subgraph
# ============================================================

async def _route_planner_llm(state: RoutePlannerState, config: RunnableConfig):
    """LLM node: decide which route tools to call."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        configurable_model
        .bind_tools(ROUTE_PLANNER_TOOLS)
        .with_config(configurable.model_config_for("route_planner"))
    )
    messages = state.get("messages", [])
    try:
        response = await model.ainvoke(messages)
    except Exception as exc:
        print(f"❌ [Route Planner] LLM 调用失败: {type(exc).__name__}: {exc}", flush=True)
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            print(f"   status_code: {status_code}", flush=True)
        body = getattr(exc, "body", None)
        if body:
            print(f"   raw_body: {body}", flush=True)
        raw_response = getattr(exc, "raw_response", None)
        if raw_response is not None and not body:
            raw_text = getattr(raw_response, "text", None)
            if raw_text:
                print(f"   raw_body: {raw_text}", flush=True)
        raise
    return Command(
        goto="route_planner_tools",
        update={
            "messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0),
        },
    )


async def _route_planner_tools(state: RoutePlannerState, config: RunnableConfig):
    """Execute route tool calls, return ToolMessages."""
    configurable = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])
    last_msg = messages[-1]
    current_tool_calls = state.get("tool_call_iterations", 0)
    max_tool_calls = configurable.max_tool_calls
    existing_records: list[dict] = list(state.get("route_records", []))
    has_submitted_plan = _has_submitted_route_plan(existing_records)

    if not last_msg.tool_calls:
        if has_submitted_plan:
            return Command(goto=END)

        reminder = (
            f"你还没有调用 SubmitRoutePlan，当前不能结束。"
            f"{' 已达到工具调用上限，不要再调用新的路线工具；' if current_tool_calls >= max_tool_calls else ''}"
            "请基于现有结果调用 SubmitRoutePlan，填写 selected_stops、reasoning，然后再调用 PlanningComplete。"
        )
        return Command(
            goto="route_planner",
            update={
                "messages": [HumanMessage(content=reminder)],
                "route_records": existing_records,
                "tool_call_iterations": current_tool_calls,
            },
        )

    planning_complete = any(tc["name"] == "PlanningComplete" for tc in last_msg.tool_calls)
    at_tool_limit = current_tool_calls >= max_tool_calls

    tool_messages = []
    new_records: list[dict] = existing_records
    executed_non_control_calls = 0

    for tc in last_msg.tool_calls:
        if tc["name"] == "SubmitRoutePlan":
            print(f"  📍 [Route Planner] SubmitRoutePlan: {len(tc['args'].get('selected_stops', []))} stops", flush=True)
            new_records.append({
                "action": "SubmitRoutePlan",
                "params": tc["args"],
                "result": tc["args"],
            })
            tool_messages.append(ToolMessage(
                content=f"已记录路线方案：{len(tc['args'].get('selected_stops', []))} 个停靠点",
                tool_call_id=tc["id"],
                name=tc["name"],
            ))
            continue

        if tc["name"] == "PlanningComplete":
            print(f"  🏁 [Route Planner] PlanningComplete: {tc['args'].get('summary', '')[:500]}", flush=True)
            tool_messages.append(ToolMessage(
                content=tc["args"].get("summary", "规划完成"),
                tool_call_id=tc["id"],
                name=tc["name"],
            ))
            continue

        if at_tool_limit or (current_tool_calls + executed_non_control_calls) >= max_tool_calls:
            tool_messages.append(ToolMessage(
                content=(
                    f"已达到工具调用上限({max_tool_calls})，不要再调用新的路线工具；"
                    "请基于已有结果调用 SubmitRoutePlan 提交方案，再调用 PlanningComplete 收尾。"
                ),
                tool_call_id=tc["id"],
                name=tc["name"],
            ))
            continue

        params_brief = json.dumps(tc["args"], ensure_ascii=False)
        if len(params_brief) > 120:
            params_brief = params_brief[:117] + "..."
        print(f"  🗺️  [Route Planner] {tc['name']}({params_brief})", flush=True)

        result = execute_tool(tc["name"], tc["args"])
        executed_non_control_calls += 1

        result_summary = summarize_tool_result_for_llm(tc["name"], result)
        print(f"      └─ {result_summary[:500]}", flush=True)

        # Record full result in structured state
        new_records.append({
            "action": tc["name"],
            "params": tc["args"],
            "result": result,
        })

        # Feed LLM a compact summary (full data lives in route_records)
        tool_messages.append(ToolMessage(
            content=summarize_tool_result_for_llm(tc["name"], result),
            tool_call_id=tc["id"],
            name=tc["name"],
        ))

    update: dict[str, Any] = {
        "messages": tool_messages,
        "route_records": new_records,
        "tool_call_iterations": current_tool_calls + executed_non_control_calls,
    }
    has_submitted_plan = _has_submitted_route_plan(new_records)
    if (planning_complete or at_tool_limit) and not has_submitted_plan:
        update["messages"] = tool_messages + [
            HumanMessage(
                content=(
                    f"当前还缺少 SubmitRoutePlan，不能结束。"
                    f"{' 已达到工具调用上限，不要再调用新的路线工具；' if at_tool_limit else ''}"
                    "请基于现有结果调用 SubmitRoutePlan，填写 selected_stops、reasoning，然后再调用 PlanningComplete。"
                )
            )
        ]
        return Command(goto="route_planner", update=update)
    if planning_complete or at_tool_limit:
        return Command(goto=END, update=update)
    return Command(goto="route_planner", update=update)


# Build Route Planner subgraph
_route_builder = StateGraph(RoutePlannerState, config_schema=Configuration)
_route_builder.add_node("route_planner", _route_planner_llm)
_route_builder.add_node("route_planner_tools", _route_planner_tools)
_route_builder.add_edge(START, "route_planner")
route_planner_subgraph = _route_builder.compile()


# ============================================================
# Route POI Enricher ReAct Subgraph
# ============================================================

async def _route_poi_enricher_llm(state: RouteNearbyPOIState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    model = (
        configurable_model
        .bind_tools(ROUTE_POI_ENRICHER_TOOLS)
        .with_config(configurable.model_config_for("route_poi_enricher"))
    )
    messages = state.get("messages", [])
    response = await model.ainvoke(messages)
    return Command(
        goto="route_poi_enricher_tools",
        update={
            "messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0),
        },
    )


async def _route_poi_enricher_tools(state: RouteNearbyPOIState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])
    last_msg = messages[-1]

    if not last_msg.tool_calls:
        return Command(goto=END)

    current_tool_calls = state.get("tool_call_iterations", 0)
    max_tool_calls = configurable.max_tool_calls
    at_tool_limit = current_tool_calls >= max_tool_calls

    tool_messages = []
    candidate_pool: list[dict[str, Any]] = list(state.get("_candidate_pool", []))
    nearby_route_pois: list[dict[str, Any]] = list(state.get("nearby_route_pois", []))
    route_polyline = _normalize_text(state.get("route_polyline"))
    selected_stops = list(state.get("selected_stops", []))
    explored_pois = list(state.get("explored_pois", []))
    should_end = False
    executed_non_control_calls = 0

    for tc in last_msg.tool_calls:
        if tc["name"] == "NearbyPOIEnrichmentComplete":
            print(f"  🏁 [Route POI Enricher] NearbyPOIEnrichmentComplete: {tc['args'].get('summary', '')[:500]}", flush=True)
            tool_messages.append(ToolMessage(
                content=tc["args"].get("summary", "路线周边点位补充完成"),
                tool_call_id=tc["id"],
                name=tc["name"],
            ))
            should_end = True
            continue

        if tc["name"] == "SubmitNearbyRoutePOIs":
            submitted_pois = list(tc["args"].get("selected_pois", []))
            candidate_map = {
                _normalize_text(item.get("name")): item
                for item in candidate_pool
                if _normalize_text(item.get("name"))
            }
            selected_candidates: list[dict[str, Any]] = []
            for item in submitted_pois:
                name = _normalize_text(item.get("name"))
                candidate = candidate_map.get(name)
                if not candidate:
                    continue
                selected_candidates.append(
                    {
                        **candidate,
                        "selection_reason": _normalize_text(item.get("reason")),
                    }
                )

            nearby_route_pois = _finalize_nearby_route_pois(
                selected_candidates,
                route_polyline,
                selected_stops,
                explored_pois,
            )

            print(f"  📌 [Route POI Enricher] SubmitNearbyRoutePOIs: 提交 {len(nearby_route_pois)} 个可选点位", flush=True)
            for poi in nearby_route_pois[:6]:
                print(f"      ├─ {poi.get('name', '?')}: {poi.get('selection_reason', '')[:80]}", flush=True)
            tool_messages.append(ToolMessage(
                content=f"已记录 {len(nearby_route_pois)} 个主路线周边可选点位。",
                tool_call_id=tc["id"],
                name=tc["name"],
            ))
            continue

        if at_tool_limit or (current_tool_calls + executed_non_control_calls) >= max_tool_calls:
            tool_messages.append(ToolMessage(
                content=f"已达到工具调用上限({max_tool_calls})，请调用 SubmitNearbyRoutePOIs 与 NearbyPOIEnrichmentComplete 收尾。",
                tool_call_id=tc["id"],
                name=tc["name"],
            ))
            continue

        params_brief = json.dumps(tc["args"], ensure_ascii=False)
        if len(params_brief) > 120:
            params_brief = params_brief[:117] + "..."
        print(f"  🧭 [Route POI Enricher] {tc['name']}({params_brief})", flush=True)

        if tc["name"] == "SearchRouteNearbyPlaces":
            if not route_polyline:
                result = {"status": "0", "error": "当前没有可用的最终路线 polyline"}
            else:
                result = _search_along_polyline(
                    route_polyline,
                    tc["args"].get("keyword"),
                    int(tc["args"].get("radius", 250) or 250),
                )
            action_name = "search_along_polyline"
        else:
            result = execute_tool(tc["name"], tc["args"])
            action_name = tc["name"]

        executed_non_control_calls += 1
        result_summary = summarize_tool_result_for_llm(action_name, result)
        print(f"      └─ {result_summary[:500]}", flush=True)

        extracted = _extract_poi_candidates_from_result(action_name, tc["args"], result)
        if extracted:
            candidate_pool = _merge_poi_candidates(candidate_pool, extracted)
            print(f"      └─ 候选池: +{len(extracted)} → 共 {len(candidate_pool)} 个", flush=True)

        tool_messages.append(ToolMessage(
            content=_format_poi_search_result(action_name, result),
            tool_call_id=tc["id"],
            name=tc["name"],
        ))

    update: dict[str, Any] = {
        "messages": tool_messages,
        "_candidate_pool": candidate_pool,
        "nearby_route_pois": nearby_route_pois,
        "tool_call_iterations": current_tool_calls + executed_non_control_calls,
    }
    if should_end or at_tool_limit:
        return Command(goto=END, update=update)

    return Command(goto="route_poi_enricher", update=update)


_route_poi_enricher_builder = StateGraph(RouteNearbyPOIState, config_schema=Configuration)
_route_poi_enricher_builder.add_node("route_poi_enricher", _route_poi_enricher_llm)
_route_poi_enricher_builder.add_node("route_poi_enricher_tools", _route_poi_enricher_tools)
_route_poi_enricher_builder.add_edge(START, "route_poi_enricher")
route_poi_enricher_subgraph = _route_poi_enricher_builder.compile()


# ============================================================
# Main graph nodes: clarify_intent, parse_intent (kept)
# ============================================================

async def clarify_intent(state: AgentState, config: RunnableConfig) -> Command[Literal["parse_intent", "__end__"]]:
    """Check if user intent is clear, ask for clarification if needed."""
    print("⏳ [Node] clarify_intent: 检查用户意图是否清晰...", flush=True)
    configurable = Configuration.from_runnable_config(config)

    history = state.get("conversation_history", [])
    history_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in history) if history else "无"
    prompt = CLARIFY_INTENT_PROMPT.format(
        conversation_history=history_text,
        user_query=state["user_query"],
    )
    clarification = _normalize_clarification(
        await _ainvoke_json_model(prompt, IntentClarification, configurable, role="clarification")
    )

    if not clarification.is_clear:
        question = clarification.clarification_question
        return Command(
            goto=END,
            update={
                "clarification": clarification,
                "conversation_history": [{"role": "assistant", "content": question}],
                "final_output": {
                    "need_clarification": True,
                    "question": question,
                    "missing_info": clarification.missing_info,
                },
            },
        )

    return Command(
        goto="parse_intent",
        update={
            "clarification": clarification,
            "conversation_history": [{"role": "user", "content": state["user_query"]}],
        },
    )


async def parse_intent(state: AgentState, config: RunnableConfig) -> Command[Literal["info_retriever"]]:
    """Parse user query into structured intent."""
    print("⏳ [Node] parse_intent: 解析用户意图...", flush=True)
    configurable = Configuration.from_runnable_config(config)

    history = state.get("conversation_history", [])
    history_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in history) if history else "无"
    prompt = PARSE_INTENT_PROMPT.format(
        conversation_history=history_text,
        user_query=state["user_query"],
    )
    intent = _normalize_intent(
        await _ainvoke_json_model(prompt, UserIntent, configurable, role="intent"),
        state["user_query"],
    )

    print("⏳ [Node] parse_intent: 自动获取起点和终点坐标...", flush=True)
    initial_results: list[dict] = []
    step_id = 1

    if intent.start_location:
        res = execute_tool("get_coordinates", {"address": intent.start_location, "city": intent.city})
        initial_results.append(
            {
                "step_id": step_id,
                "action": "get_coordinates",
                "params": {"address": intent.start_location, "city": intent.city},
                "result": res,
                "reason": "为后续规划获取起点坐标",
            }
        )
        step_id += 1

    if intent.end_location:
        res = execute_tool("get_coordinates", {"address": intent.end_location, "city": intent.city})
        initial_results.append(
            {
                "step_id": step_id,
                "action": "get_coordinates",
                "params": {"address": intent.end_location, "city": intent.city},
                "result": res,
                "reason": "为后续规划获取终点坐标",
            }
        )

    return Command(
        goto="info_retriever",
        update={
            "intent": intent,
            "execution_results": initial_results,
            "retrieved_info": [],
            "retrieved_note_contexts": [],
            "supervisor_iterations": 0,
            "replan_count": 0,
        },
    )


# ============================================================
# Supervisor node pair (supervisor + supervisor_tools)
# ============================================================

DISPATCH_TOOLS = [DispatchPOIExplorer, DispatchRoutePlanner, AllTasksComplete]


async def supervisor(state: AgentState, config: RunnableConfig):
    """Supervisor LLM: decide which sub-agent to dispatch."""
    print("⏳ [Node] supervisor: 审视全局并分配任务...", flush=True)
    configurable = Configuration.from_runnable_config(config)

    model = (
        configurable_model
        .bind_tools(DISPATCH_TOOLS)
        .with_config(configurable.model_config_for("supervisor"))
    )

    messages = state.get("supervisor_messages", [])
    response = await model.ainvoke(messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "supervisor_iterations": state.get("supervisor_iterations", 0) + 1,
        },
    )


async def supervisor_tools(state: AgentState, config: RunnableConfig):
    """Execute supervisor tool calls — dispatch sub-agents or finish."""
    configurable = Configuration.from_runnable_config(config)
    messages = state.get("supervisor_messages", [])
    last_msg = messages[-1]

    max_supervisor_loops = max(6, configurable.max_replan_count * 3)
    exceeded = state.get("supervisor_iterations", 0) >= max_supervisor_loops

    # Exit conditions
    no_tool_calls = not last_msg.tool_calls
    all_complete = any(tc["name"] == "AllTasksComplete" for tc in (last_msg.tool_calls or []))

    if exceeded or no_tool_calls or all_complete:
        print("  └─ Supervisor: 所有任务完成，进入路线周边点位补充", flush=True)
        return Command(goto="route_poi_enricher")

    tool_messages = []
    update: dict[str, Any] = {}

    for tc in last_msg.tool_calls:
        if tc["name"] == "DispatchPOIExplorer":
            print(f"  └─ Supervisor 派遣 POI Explorer: {tc['args'].get('task_description', '')[:]}", flush=True)
            task_desc = tc["args"].get("task_description", "探索兴趣点")
            result = await poi_explorer_subgraph.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=POI_EXPLORER_SYSTEM_PROMPT),
                        HumanMessage(content=_build_poi_context(state, task_desc)),
                    ],
                    "tool_call_iterations": 0,
                    "_candidate_pool": [],
                    "explored_pois": [],
                },
                config,
            )
            new_pois = result.get("explored_pois", [])
            update["explored_pois"] = _merge_poi_candidates(
                list(state.get("explored_pois", [])), new_pois
            )
            tool_messages.append(ToolMessage(
                content=_summarize_poi_result(result),
                tool_call_id=tc["id"],
                name=tc["name"],
            ))

        elif tc["name"] == "DispatchRoutePlanner":
            print(f"  └─ Supervisor 派遣 Route Planner: {tc['args'].get('task_description', '')[:]}", flush=True)
            task_desc = tc["args"].get("task_description", "规划路线")
            result = await route_planner_subgraph.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=ROUTE_PLANNER_SYSTEM_PROMPT),
                        HumanMessage(content=_build_route_context(state, task_desc)),
                    ],
                    "tool_call_iterations": 0,
                    "route_records": [],
                },
                config,
            )
            new_records = list(result.get("route_records", []))

            # Keep the most recent submission when Route Planner revises within one run.
            route_plan = _latest_submitted_route_plan_record(new_records)
            if route_plan:
                materialized_route_plan, materialized_route_record = _materialize_submitted_route_plan(
                    state,
                    route_plan.get("result"),
                )
                if materialized_route_plan:
                    route_plan["result"] = materialized_route_plan
                    update["route_plan"] = materialized_route_plan
                if materialized_route_record:
                    new_records.append(materialized_route_record)

            update["planned_routes"] = list(state.get("planned_routes", [])) + new_records

            tool_messages.append(ToolMessage(
                content=_summarize_route_result(result),
                tool_call_id=tc["id"],
                name=tc["name"],
            ))

        elif tc["name"] == "AllTasksComplete":
            tool_messages.append(ToolMessage(
                content=tc["args"].get("summary", "完成"),
                tool_call_id=tc["id"],
                name=tc["name"],
            ))

    # Append current status for next supervisor iteration
    status_msg = HumanMessage(content=f"当前全局状态更新：\n{_build_supervisor_status({**state, **update})}")
    update["supervisor_messages"] = tool_messages + [status_msg]

    return Command(goto="supervisor", update=update)


async def route_poi_enricher_node(state: AgentState, config: RunnableConfig) -> Command[Literal["json_formatter"]]:
    print("⏳ [Node] route_poi_enricher: 补充主路线周边可选点位...", flush=True)
    route_data = _resolve_final_route_output_data(state)
    route_polyline = _normalize_text(route_data.get("route_polyline"))

    if not route_polyline:
        print("  └─ Route POI Enricher: 没有可用的最终路线 polyline，跳过", flush=True)
        return Command(goto="json_formatter", update={"nearby_route_pois": []})

    sub_result = await route_poi_enricher_subgraph.ainvoke(
        {
            "messages": [
                SystemMessage(content=ROUTE_POI_ENRICHER_SYSTEM_PROMPT),
                HumanMessage(content=_build_route_poi_enricher_context(state, route_data)),
            ],
            "tool_call_iterations": 0,
            "route_polyline": route_polyline,
            "selected_stops": list(route_data.get("selected_stops") or []),
            "explored_pois": list(state.get("explored_pois", [])),
            "_candidate_pool": [],
            "nearby_route_pois": [],
        },
        config,
    )
    nearby_route_pois = list(sub_result.get("nearby_route_pois", []))
    print(f"  └─ {_summarize_route_poi_enricher_result(sub_result)}", flush=True)
    return Command(
        goto="json_formatter",
        update={"nearby_route_pois": nearby_route_pois},
    )


# ============================================================
# Placeholder / unchanged nodes
# ============================================================

async def info_retriever_node(state: AgentState, config: RunnableConfig) -> Command[Literal["supervisor"]]:
    print("⏳ [Node] info_retriever: 检索相关小红书帖子...", flush=True)
    configurable = Configuration.from_runnable_config(config)
    retrieved_info = list(state.get("retrieved_info", []))
    note_contexts = []

    try:
        note_contexts = retrieve_planner_note_contexts(
            state.get("user_query", ""),
            **configurable.rag_retriever_config(),
        )
        retrieval_summary = _summarize_retrieved_note_contexts(note_contexts)
    except Exception as exc:
        print(f"  └─ RAG 检索失败，已降级继续规划: {type(exc).__name__}: {exc}", flush=True)
        note_contexts = []
        retrieval_summary = f"小红书检索失败，已跳过并继续规划：{type(exc).__name__}: {exc}"

    status_text = (
        "用户意图已解析完成，且已完成小红书帖子检索准备。\n"
        f"{_build_supervisor_status({**state, 'retrieved_info': retrieved_info + [retrieval_summary], 'retrieved_note_contexts': note_contexts})}"
    )

    return Command(
        goto="supervisor",
        update={
            "retrieved_info": retrieved_info + [retrieval_summary],
            "retrieved_note_contexts": note_contexts,
            "supervisor_messages": [
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                HumanMessage(content=status_text),
            ],
            "supervisor_iterations": 0,
        },
    )


async def json_formatter_node(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    print("⏳ [Node] json_formatter: 输出最终 JSON...", flush=True)
    intent = state.get("intent")
    accepted_pois = list(state.get("explored_pois", []))
    accepted_routes = list(state.get("planned_routes", []))
    route_data = _resolve_final_route_output_data(state)
    stops = list(route_data.get("stops") or [])
    route_summary = _normalize_text(route_data.get("route_summary"))
    total_duration_minutes = route_data.get("total_duration_minutes")
    route_polyline = _normalize_text(route_data.get("route_polyline"))
    nearby_route_pois = list(state.get("nearby_route_pois", []))

    formatter_note = JSON_FORMATTER_PROMPT.format(
        retrieved_info=state.get("retrieved_info", []),
        explored_pois=accepted_pois,
        planned_routes=accepted_routes,
    )

    output = {
        "task_type": "citywalk_plan",
        "stops": stops,
        "route_title": intent.activity_type + "路线" if intent else "路线",
        "route_summary": route_summary or formatter_note,
        "total_duration_minutes": None,
        "total_walking_minutes": total_duration_minutes,
        "route_polyline": route_polyline,
        "nearby_route_pois": nearby_route_pois,
    }
    return Command(goto=END, update={"final_output": output})


# ============================================================
# Build the final graph
# ============================================================

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("clarify_intent", clarify_intent)
    graph.add_node("parse_intent", parse_intent)
    graph.add_node("supervisor", supervisor)
    graph.add_node("supervisor_tools", supervisor_tools)
    graph.add_node("route_poi_enricher", route_poi_enricher_node)
    graph.add_node("info_retriever", info_retriever_node)
    graph.add_node("json_formatter", json_formatter_node)

    graph.add_edge(START, "clarify_intent")
    # Remaining routing is dynamic via Command(goto=...)

    return graph.compile()
