"""Utility functions for CityWalk agent."""

import sys
import os
from typing import Any
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.maps_tools import (
    get_coordinates,
    search_nearby_places,
    get_place_details,
    calculate_walking_route,
    get_detailed_walking_route,
    get_walking_route_text,
    search_along_route,
    search_candidate_corridors,
    plan_multi_waypoint_route,
    calculate_distance_matrix,
    evaluate_detour_impact,
)


TOOL_REGISTRY = {
    "get_coordinates": lambda params: get_coordinates(
        params.get("address"), params.get("city")
    ),
    "search_nearby_places": lambda params: search_nearby_places(
        params.get("keyword"), params.get("location_coords"), params.get("radius", 1000)
    ),
    "get_place_details": lambda params: get_place_details(
        params.get("place_name"),
        params.get("city"),
        params.get("location_coords"),
        params.get("radius", 1200),
    ),
    "calculate_walking_route": lambda params: calculate_walking_route(
        params.get("origin_coords"), params.get("destination_coords")
    ),
    "get_detailed_walking_route": lambda params: get_detailed_walking_route(
        params.get("origin_coords"), params.get("destination_coords")
    ),
    "get_walking_route_text": lambda params: get_walking_route_text(
        params.get("origin_coords"), params.get("destination_coords"), params.get("detail_level", "low"), params.get("travel_mode", "步行")
    ),
    "search_along_route": lambda params: search_along_route(
        params.get("origin_coords"), params.get("destination_coords"), params.get("keyword"), params.get("radius", 200)
    ),
    "search_candidate_corridors": lambda params: search_candidate_corridors(
        params.get("start_location_coords"),
        params.get("city"),
        params.get("theme"),
        params.get("max_radius", 3000),
    ),
    "plan_multi_waypoint_route": lambda params: plan_multi_waypoint_route(
        params.get("origin_coords"),
        params.get("destination_coords"),
        params.get("waypoints", []),
    ),
    "calculate_distance_matrix": lambda params: calculate_distance_matrix(
        params.get("origins", []),
        params.get("destinations", []),
    ),
    "evaluate_detour_impact": lambda params: evaluate_detour_impact(
        params.get("prev_stop_coords"),
        params.get("next_stop_coords"),
        params.get("candidate_poi_coords"),
    ),
}


def execute_tool(action: str, params: dict) -> dict:
    """Execute a tool by name."""
    tool = TOOL_REGISTRY.get(action)
    if not tool:
        return {"error": f"Unknown tool: {action}"}
    try:
        # Normalize and validate coordinates format
        for key in ["location_coords", "origin_coords", "destination_coords", "start_location_coords"]:
            if key in params and params[key]:
                coords = params[key]
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    coords = f"{coords[0]},{coords[1]}"
                    params[key] = coords
                elif isinstance(coords, str):
                    coords = coords.strip()
                    if coords.startswith("[") and coords.endswith("]"):
                        coords = coords[1:-1].replace(" ", "")
                        params[key] = coords

                if isinstance(coords, str) and "," not in coords:
                    return {"error": f"Invalid coordinate format for {key}: {coords}. Expected 'lng,lat'"}

        return tool(params)
    except Exception as e:
        return {"error": str(e)}


def _compact_text(value: Any, limit: int = 160) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _top_names(items: list[dict], limit: int = 4) -> str:
    names = []
    for item in items:
        name = str(item.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return "、".join(names)


def summarize_tool_result_for_llm(action: str, result: dict) -> str:
    """Summarize raw tool output into compact text for downstream LLM prompts."""
    if not isinstance(result, dict):
        return _compact_text(result)

    if result.get("status") == "0" or result.get("error"):
        return f"失败: {result.get('error') or '未知错误'}"

    if action == "get_coordinates":
        return (
            f"定位成功，坐标 {result.get('location')}，"
            f"标准地址 {result.get('formatted_address') or '未知'}。"
        )

    if action in {"calculate_walking_route", "get_walking_route_text", "get_detailed_walking_route"}:
        summary = result.get("route_summary")
        if summary:
            return _compact_text(summary, limit=220)
        return (
            f"路线约 {result.get('duration_minutes')} 分钟，"
            f"{result.get('distance_meters')} 米，共 {result.get('step_count') or '若干'} 段。"
        )

    if action in {"search_nearby_places", "search_along_route", "search_along_polyline"}:
        summary = result.get("along_route_summary")
        if summary:
            return _compact_text(summary, limit=220)
        names = _top_names(list(result.get("results") or []) or list(result.get("top_pois") or []))
        return (
            f"找到 {result.get('result_count') or len(result.get('results') or [])} 个相关地点"
            f"{f'，包括 {names}' if names else ''}。"
        )

    if action == "get_place_details":
        poi = result.get("poi") if isinstance(result.get("poi"), dict) else {}
        name = _compact_text(poi.get("name") or "该点位", limit=40)
        description = _compact_text(result.get("description") or "", limit=140)
        highlights = "、".join(str(x) for x in (result.get("highlights") or [])[:3])
        extras = []
        if highlights:
            extras.append(f"关键信息: {highlights}")
        if description:
            extras.append(description)
        return f"{name} 详情已获取。{' '.join(extras)}".strip()

    if action == "search_candidate_corridors":
        corridors = list(result.get("corridors") or [])
        corridor_names = "、".join(
            str(item.get("name") or "").strip()
            for item in corridors[:3]
            if str(item.get("name") or "").strip()
        )
        return (
            f"找到 {result.get('corridor_count') or len(corridors)} 条候选廊道"
            f"{f'：{corridor_names}' if corridor_names else ''}。"
        )

    if action == "route_stop_chain":
        stops = list(result.get("stops") or [])
        stop_names = "、".join(
            str(item.get("name") or "").strip()
            for item in stops[:4]
            if str(item.get("name") or "").strip()
        )
        return (
            f"串联出 {len(stops)} 个停靠点"
            f"{f'：{stop_names}' if stop_names else ''}，"
            f"总步行约 {result.get('total_walking_minutes')} 分钟。"
        )

    return _compact_text(result, limit=220)
