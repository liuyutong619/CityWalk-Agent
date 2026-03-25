import ast
import json
import math
import os
import re
import time
from collections import defaultdict
from typing import Any

import requests
from dotenv import load_dotenv

try:
    from tools.utils import (
        bearing_to_direction,
        format_location,
        merge_polylines,
        normalize_theme_keywords,
        parse_location,
        sample_polyline_points,
    )
except ModuleNotFoundError:
    from utils import (
        bearing_to_direction,
        format_location,
        merge_polylines,
        normalize_theme_keywords,
        parse_location,
        sample_polyline_points,
    )

load_dotenv()
AMAP_KEY = os.getenv("AMAP_KEY")

if not AMAP_KEY:
    raise RuntimeError("缺少 AMAP_KEY，请在项目根目录的 .env 文件中配置")

AMAP_BASE_URL = "https://restapi.amap.com"
DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_SEARCH_LIMIT = 8


def _amap_get(path: str, params: dict) -> dict:
    url = f"{AMAP_BASE_URL}{path}"
    merged_params = {"key": AMAP_KEY, **params}
    response = requests.get(url, params=merged_params, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "1":
        info = payload.get("info") or "高德接口调用失败"
        infocode = payload.get("infocode")
        raise RuntimeError(f"{info} (infocode={infocode})")
    time.sleep(0.35)
    return payload


def _unique_non_empty(values: list[Any]) -> list[str]:
    deduplicated = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in deduplicated:
            deduplicated.append(text)
    return deduplicated


def _shrink_preserving_order(items: list[str], max_items: int) -> list[str]:
    if len(items) <= max_items:
        return items
    if max_items <= 1:
        return items[:1]

    selected = [items[0]]
    middle_slots = max_items - 2
    for slot in range(middle_slots):
        mapped_index = round((slot + 1) * (len(items) - 1) / (middle_slots + 1))
        candidate = items[mapped_index]
        if candidate not in selected:
            selected.append(candidate)

    if items[-1] not in selected:
        selected.append(items[-1])
    return selected[:max_items]


def _normalize_keywords(keyword: str | list[str]) -> list[str]:
    """Normalize keyword input into a deduplicated list."""
    if isinstance(keyword, list):
        raw_items = keyword
    else:
        text = str(keyword or "").strip()
        if not text:
            return []

        # Tolerate stringified arrays from tool-calling outputs, e.g. '["咖啡馆","公园"]'
        if text.startswith("[") and text.endswith("]"):
            parsed_items: list[Any] | None = None
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    parsed_items = parsed
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, list):
                        parsed_items = parsed
                except Exception:
                    parsed_items = None
            if parsed_items is not None:
                raw_items = parsed_items
            else:
                for sep in ["，", ",", "；", ";", "|", "\n"]:
                    text = text.replace(sep, " ")
                raw_items = text.split(" ")
        else:
            for sep in ["，", ",", "；", ";", "|", "\n"]:
                text = text.replace(sep, " ")
            raw_items = text.split(" ")

    normalized = []
    for item in raw_items:
        text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized[:6]


def _strip_parenthetical_suffix(text: str) -> str:
    return re.sub(r"\s*[（(][^()（）]{0,24}[)）]\s*$", "", str(text or "").strip()).strip()


def _build_place_query_variants(text: str) -> list[str]:
    variants: list[str] = []
    for candidate in [str(text or "").strip(), _strip_parenthetical_suffix(text)]:
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def _format_step_brief(step: dict[str, Any]) -> str | None:
    instruction = str(step.get("instruction") or "").strip()
    road = str(step.get("road") or "").strip()
    distance_meters = step.get("distance_meters")
    duration_minutes = step.get("duration_minutes")

    if instruction:
        base_text = instruction
    elif road:
        base_text = f"沿{road}步行"
    else:
        return None

    metrics = []
    if isinstance(distance_meters, int) and distance_meters > 0:
        metrics.append(f"{distance_meters}米")
    if isinstance(duration_minutes, int) and duration_minutes > 0:
        metrics.append(f"约{duration_minutes}分钟")
    if metrics:
        return f"{base_text}（{'，'.join(metrics)}）"
    return base_text


def _extract_key_steps(steps: list[dict[str, Any]], max_items: int = 5) -> list[str]:
    step_briefs = _unique_non_empty([_format_step_brief(step) for step in steps])
    return _shrink_preserving_order(step_briefs, max_items)


def _build_route_summary_text(detailed_route: dict[str, Any], travel_mode: str = "步行") -> tuple[str, list[str], list[str]]:
    steps = list(detailed_route.get("steps") or [])
    distance_meters = detailed_route.get("distance_meters")
    duration_minutes = detailed_route.get("duration_minutes")
    step_count = detailed_route.get("step_count") or len(steps)

    key_steps = _extract_key_steps(steps)
    major_roads = _shrink_preserving_order(
        _unique_non_empty([step.get("road") for step in steps]),
        4,
    )

    summary_parts = [
        f"这段{travel_mode}全程约{duration_minutes}分钟、{distance_meters}米，共{step_count}段。"
    ]
    if major_roads:
        summary_parts.append(f"主要经过 {'、'.join(major_roads)}。")
    if key_steps:
        summary_parts.append(f"关键走法：{'；'.join(key_steps[:3])}。")

    return "".join(summary_parts), key_steps, major_roads


def _position_hint(sample_index: int, sample_point_count: int) -> str:
    if sample_point_count <= 1:
        return "沿线中段"
    ratio = sample_index / sample_point_count
    if ratio <= 0.2:
        return "起点附近"
    if ratio <= 0.45:
        return "前段"
    if ratio <= 0.7:
        return "中段"
    if ratio <= 0.9:
        return "后段"
    return "终点附近"


def _build_along_route_summary_text(
    keyword: str,
    radius: int,
    sampled_point_count: int,
    results: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], str]:
    top_pois = []
    position_counts = defaultdict(int)

    for poi in results:
        sample_index = int(poi.get("first_seen_sample_index") or 0)
        position_hint = _position_hint(sample_index, sampled_point_count)
        position_counts[position_hint] += 1
        top_pois.append(
            {
                "name": poi.get("name"),
                "address": poi.get("address"),
                "distance_meters": poi.get("distance_meters"),
                "type": poi.get("type"),
                "district": poi.get("adname"),
                "position_hint": position_hint,
            }
        )

    top_pois = top_pois[:5]
    hotspot_parts = [
        f"{position}约{count}处"
        for position, count in sorted(position_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    ]
    hotspot_text = "、".join(hotspot_parts) if hotspot_parts else "分布较为分散"
    top_names = _unique_non_empty([poi.get("name") for poi in top_pois])[:4]

    summary = f"沿这条路线在约{radius}米范围内，共发现 {len(results)} 个与“{keyword}”相关的点位。"
    distribution = f"它们主要集中在{hotspot_text}。"
    if top_names:
        summary += f" 优先可关注：{'、'.join(top_names)}。"
    return summary, top_pois, distribution


def get_coordinates(address: str, city: str) -> dict:
    """
    坐标解析工具：通过高德 POI 搜索将人类可读的位置名称解析为经纬度坐标。
    适合站点、景点、广场、展馆等 POI 名称；会优先尝试原始名称，再尝试去掉括号修饰后的名称。
    """
    try:
        address = str(address or "").strip()
        if not address:
            return {"status": "0", "error": "address 不能为空"}

        for query in _build_place_query_variants(address):
            payload = _amap_get(
                "/v3/place/text",
                {
                    "keywords": query,
                    "city": city or "",
                    "citylimit": "true" if city else "false",
                    "offset": 5,
                    "page": 1,
                    "extensions": "all",
                },
            )
            pois = list(payload.get("pois") or [])
            if not pois:
                continue

            exact = [
                poi for poi in pois
                if _strip_parenthetical_suffix(poi.get("name")) == _strip_parenthetical_suffix(query)
            ]
            poi = exact[0] if exact else pois[0]

            pname = str(poi.get("pname") or "").strip()
            cityname = str(poi.get("cityname") or "").strip()
            district = str(poi.get("adname") or "").strip()
            poi_address = str(poi.get("address") or "").strip()
            poi_name = str(poi.get("name") or address).strip()
            formatted_address = "".join(part for part in [pname, cityname, district, poi_address or poi_name] if part)

            return {
                "status": "1",
                "location": poi.get("location"),
                "formatted_address": formatted_address or poi_name,
                "province": pname or None,
                "city": cityname or None,
                "district": district or None,
                "adcode": poi.get("adcode"),
                "poi_id": poi.get("id"),
                "poi_name": poi_name,
                "candidate_matches": [
                    {
                        "name": item.get("name"),
                        "location": item.get("location"),
                        "address": item.get("address"),
                        "type": item.get("type"),
                    }
                    for item in pois[:3]
                ],
            }

        return {"status": "0", "error": f"未找到与“{address}”相关的点位坐标"}
    except Exception as error:
        return {"status": "0", "error": str(error)}


def search_nearby_places(keyword: str | list[str], location_coords: str, radius: int = 1000) -> dict:
    """
    周边搜索工具：根据中心点经纬度，搜索附近的兴趣点（如咖啡馆、餐厅、景点）。
    """
    try:
        keywords = _normalize_keywords(keyword)
        if not keywords:
            return {"status": "0", "error": "keyword 不能为空"}

        deduplicated_pois: dict[str, dict[str, Any]] = {}
        for single_keyword in keywords:
            payload = _amap_get(
                "/v3/place/around",
                {
                    "keywords": single_keyword,
                    "location": location_coords,
                    "radius": radius,
                    "offset": DEFAULT_SEARCH_LIMIT,
                    "page": 1,
                    "extensions": "base",
                    "sortrule": "distance",
                },
            )
            pois = payload.get("pois") or []
            for poi in pois:
                dedupe_key = poi.get("id") or f"{poi.get('name')}|{poi.get('location')}"
                stored = deduplicated_pois.get(dedupe_key)
                if stored is None:
                    distance_text = poi.get("distance")
                    distance_meters = int(distance_text) if str(distance_text).isdigit() else None
                    deduplicated_pois[dedupe_key] = {
                        "poi_id": poi.get("id"),
                        "name": poi.get("name"),
                        "address": poi.get("address"),
                        "location": poi.get("location"),
                        "distance_meters": distance_meters,
                        "type": poi.get("type"),
                        "typecode": poi.get("typecode"),
                        "adname": poi.get("adname"),
                        "matched_keywords": [single_keyword],
                    }
                    continue

                matched_keywords = list(stored.get("matched_keywords") or [])
                if single_keyword not in matched_keywords:
                    matched_keywords.append(single_keyword)
                    stored["matched_keywords"] = matched_keywords

                new_distance_text = poi.get("distance")
                new_distance = int(new_distance_text) if str(new_distance_text).isdigit() else None
                old_distance = stored.get("distance_meters")
                if old_distance is None and new_distance is not None:
                    stored["distance_meters"] = new_distance
                elif isinstance(old_distance, int) and isinstance(new_distance, int) and new_distance < old_distance:
                    stored["distance_meters"] = new_distance

        if not deduplicated_pois:
            return {"status": "0", "error": "附近没有找到相关地点"}

        results = sorted(
            deduplicated_pois.values(),
            key=lambda item: item.get("distance_meters") if isinstance(item.get("distance_meters"), int) else 999999,
        )
        return {
            "status": "1",
            "keyword": "、".join(keywords),
            "keywords": keywords,
            "result_count": len(results),
            "results": results,
        }
    except Exception as error:
        return {"status": "0", "error": str(error)}


def get_place_details(
    place_name: str,
    city: str | None = None,
    location_coords: str | None = None,
    radius: int = 1200,
) -> dict:
    """
    点位详情工具：根据地名查询单个点位的说明性信息（评分、营业时间、标签等）。
    可用于判断该点位是否值得纳入候选 POI。
    """
    try:
        place_name = str(place_name or "").strip()
        if not place_name:
            return {"status": "0", "error": "place_name 不能为空"}

        pois: list[dict[str, Any]] = []
        if location_coords:
            payload = _amap_get(
                "/v3/place/around",
                {
                    "keywords": place_name,
                    "location": location_coords,
                    "radius": radius,
                    "offset": 5,
                    "page": 1,
                    "extensions": "all",
                    "sortrule": "distance",
                },
            )
            pois = list(payload.get("pois") or [])
        else:
            payload = _amap_get(
                "/v3/place/text",
                {
                    "keywords": place_name,
                    "city": city or "",
                    "citylimit": "true" if city else "false",
                    "offset": 5,
                    "page": 1,
                    "extensions": "all",
                },
            )
            pois = list(payload.get("pois") or [])

        if not pois:
            return {"status": "0", "error": f"未找到与“{place_name}”相关的点位"}

        exact = [poi for poi in pois if str(poi.get("name") or "").strip() == place_name]
        poi = exact[0] if exact else pois[0]
        biz_ext = poi.get("biz_ext") or {}
        photos = list(poi.get("photos") or [])

        rating = str(biz_ext.get("rating") or "").strip()
        cost = str(biz_ext.get("cost") or "").strip()
        business_hours = str(
            poi.get("business_hours")
            or poi.get("opentime")
            or poi.get("open_time")
            or ""
        ).strip()
        tag = str(poi.get("tag") or "").strip()
        website = str(poi.get("website") or "").strip()
        tel = str(poi.get("tel") or "").strip()

        highlights = _unique_non_empty(
            [
                f"评分 {rating}" if rating else "",
                f"人均 {cost} 元" if cost else "",
                f"营业时间 {business_hours}" if business_hours else "",
                tag,
                poi.get("type"),
                poi.get("business_area"),
            ]
        )

        description_parts = [
            f"{poi.get('name') or '该点位'} 属于 {str(poi.get('type') or '未知类型').split(';')[0]}。"
        ]
        if rating:
            description_parts.append(f"评分约 {rating}。")
        if cost:
            description_parts.append(f"参考消费约 {cost} 元。")
        if business_hours:
            description_parts.append(f"营业时间：{business_hours}。")
        if tag:
            description_parts.append(f"标签：{tag}。")

        return {
            "status": "1",
            "poi": {
                "poi_id": poi.get("id"),
                "name": poi.get("name"),
                "address": poi.get("address"),
                "location": poi.get("location"),
                "type": poi.get("type"),
                "typecode": poi.get("typecode"),
                "adname": poi.get("adname"),
                "cityname": poi.get("cityname"),
                "pname": poi.get("pname"),
                "tel": tel or None,
                "website": website or None,
                "business_hours": business_hours or None,
                "rating": rating or None,
                "cost": cost or None,
                "tag": tag or None,
                "photo_count": len(photos),
                "photo_urls": [p.get("url") for p in photos[:3] if p.get("url")],
            },
            "highlights": highlights,
            "description": "".join(description_parts),
            "candidate_matches": [
                {
                    "name": p.get("name"),
                    "location": p.get("location"),
                    "address": p.get("address"),
                    "type": p.get("type"),
                }
                for p in pois[:3]
            ],
        }
    except Exception as error:
        return {"status": "0", "error": str(error)}


def get_detailed_walking_route(origin_coords: str, destination_coords: str) -> dict:
    """
    详细步行路线工具：返回总距离、总时长、完整 polyline 以及逐段 steps。
    """
    try:
        payload = _amap_get(
            "/v3/direction/walking",
            {
                "origin": origin_coords,
                "destination": destination_coords,
            },
        )
        route = payload.get("route") or {}
        paths = route.get("paths") or []
        if not paths:
            return {"status": "0", "error": "无法规划步行路线"}

        path = paths[0]
        raw_steps = path.get("steps") or []
        parsed_steps = []
        step_polylines = []
        for index, step in enumerate(raw_steps, start=1):
            duration_seconds = int(step.get("duration", 0) or 0)
            polyline = step.get("polyline") or ""
            step_polylines.append(polyline)
            parsed_steps.append(
                {
                    "order": index,
                    "instruction": step.get("instruction"),
                    "road": step.get("road"),
                    "orientation": step.get("orientation"),
                    "action": step.get("action"),
                    "assistant_action": step.get("assistant_action"),
                    "walk_type": step.get("walk_type"),
                    "distance_meters": int(step.get("distance", 0) or 0),
                    "duration_minutes": math.ceil(duration_seconds / 60) if duration_seconds else 0,
                    "polyline": polyline,
                }
            )

        total_duration_seconds = int(path.get("duration", 0) or 0)
        full_polyline = merge_polylines(step_polylines)
        return {
            "status": "1",
            "origin": origin_coords,
            "destination": destination_coords,
            "distance_meters": int(path.get("distance", 0) or 0),
            "duration_minutes": math.ceil(total_duration_seconds / 60) if total_duration_seconds else 0,
            "step_count": len(parsed_steps),
            "full_polyline": full_polyline,
            "steps": parsed_steps,
        }
    except Exception as error:
        return {"status": "0", "error": str(error)}


def get_walking_route_text(origin_coords: str, destination_coords: str, detail_level: str = "low", travel_mode: str = "步行") -> dict:
    """
    面向 LLM 的步行路线摘要工具：返回可读的路线说明，不暴露长 polyline。
    detail_level参数控制返回粒度："low" 仅返回摘要，"high" 补充完整的路段步骤。
    travel_mode参数支持配置为 "步行" 或 "骑行" 等，以调整生成的文字。
    """
    detailed_route = get_detailed_walking_route(origin_coords, destination_coords)
    if detailed_route.get("status") != "1":
        return {"status": "0", "error": detailed_route.get("error", "未知的路线规划错误")}

    route_summary, key_steps, major_roads = _build_route_summary_text(detailed_route, travel_mode=travel_mode)
    
    result = {
        "status": "1",
        "origin": origin_coords,
        "destination": destination_coords,
        "distance_meters": detailed_route.get("distance_meters"),
        "duration_minutes": detailed_route.get("duration_minutes"),
        "step_count": detailed_route.get("step_count"),
        "route_summary": route_summary,
    }

    if detail_level == "high":
        result["key_steps"] = key_steps
        
    return result


def calculate_walking_route(origin_coords: str, destination_coords: str) -> dict:
    """
    步行路线规划工具：计算两个经纬度坐标之间的步行距离和预计耗时。
    """
    detailed_route = get_detailed_walking_route(origin_coords, destination_coords)
    if detailed_route.get("status") != "1":
        return detailed_route

    return {
        "status": "1",
        "distance_meters": detailed_route["distance_meters"],
        "duration_minutes": detailed_route["duration_minutes"],
    }


def _search_along_polyline(polyline: str, keyword: str | list[str], radius: int = 200) -> dict:
    """
    内部方法：对一条步行路线采样，并在采样点周边搜索兴趣点。
    """
    try:
        keywords = _normalize_keywords(keyword)
        if not keywords:
            return {"status": "0", "error": "keyword 不能为空"}

        sampled_points = sample_polyline_points(polyline)
        if not sampled_points:
            return {"status": "0", "error": "无法从 polyline 中解析出有效坐标点"}

        deduplicated_pois = {}
        for sample_index, point in enumerate(sampled_points, start=1):
            point_text = format_location(point)
            for single_keyword in keywords:
                nearby_result = search_nearby_places(single_keyword, point_text, radius)
                if nearby_result.get("status") != "1":
                    continue

                for poi in nearby_result.get("results", []):
                    dedupe_key = poi.get("poi_id") or f"{poi.get('name')}|{poi.get('location')}"
                    if dedupe_key not in deduplicated_pois:
                        deduplicated_pois[dedupe_key] = {
                            **poi,
                            "first_seen_sample_index": sample_index,
                            "sample_point": point_text,
                            "matched_keywords": [single_keyword],
                        }
                        continue
                    stored = deduplicated_pois[dedupe_key]
                    matched_keywords = list(stored.get("matched_keywords") or [])
                    if single_keyword not in matched_keywords:
                        matched_keywords.append(single_keyword)
                        stored["matched_keywords"] = matched_keywords

        if not deduplicated_pois:
            return {"status": "0", "error": "路线沿途没有找到相关地点"}

        sorted_results = sorted(
            deduplicated_pois.values(),
            key=lambda item: (
                item.get("first_seen_sample_index", 999),
                item.get("distance_meters") if item.get("distance_meters") is not None else 999999,
            ),
        )

        return {
            "status": "1",
            "keyword": "、".join(keywords),
            "keywords": keywords,
            "radius": radius,
            "sample_point_count": len(sampled_points),
            "sample_points": [format_location(point) for point in sampled_points],
            "result_count": len(sorted_results),
            "results": sorted_results,
        }
    except Exception as error:
        return {"status": "0", "error": str(error)}


def search_along_route(
    origin_coords: str,
    destination_coords: str,
    keyword: str | list[str],
    radius: int = 200,
) -> dict:
    """
    沿路线搜索工具：输入两点的经纬度，自动规划路线并沿途搜索兴趣点。
    """
    try:
        route_detail = get_detailed_walking_route(origin_coords, destination_coords)
        if route_detail.get("status") != "1":
            return route_detail
            
        polyline = route_detail.get("full_polyline")
        if not polyline:
            return {"status": "0", "error": "未能生成有效的路线"}
            
        return _search_along_polyline(polyline, keyword, radius)
    except Exception as error:
        return {"status": "0", "error": str(error)}


def search_candidate_corridors(
    start_location_coords: str,
    city: str,
    theme: str | list[str],
    max_radius: int = 3000,
) -> dict:
    """
    候选路线廊道搜索工具：围绕起点搜索不同方向上更有潜力漫步的片区。
    """
    try:
        try:
            origin_point = parse_location(start_location_coords)
        except Exception as error:
            return {
                "status": "0",
                "error": (
                    f"start_location_coords 无法解析: {start_location_coords}。"
                    "期望格式为 '经度,纬度'。"
                ),
                "error_code": "invalid_start_location_coords",
                "diagnosis": "问题更可能来自 start_location_coords，而不是 theme。",
                "start_location_coords": start_location_coords,
                "theme": theme,
                "theme_keywords": [],
                "search_attempts": [],
                "details": str(error),
            }
        theme_keywords = normalize_theme_keywords(theme)
        grouped_pois = {}
        search_attempts = []

        for keyword in theme_keywords:
            nearby_result = search_nearby_places(keyword, start_location_coords, max_radius)
            attempt = {
                "keyword": keyword,
                "status": nearby_result.get("status"),
                "result_count": int(nearby_result.get("result_count") or len(nearby_result.get("results") or [])),
            }
            if nearby_result.get("status") != "1":
                attempt["error"] = nearby_result.get("error") or "未知错误"
                search_attempts.append(attempt)
                continue
            search_attempts.append(attempt)

            for poi in nearby_result.get("results", []):
                dedupe_key = poi.get("poi_id") or f"{poi.get('name')}|{poi.get('location')}"
                stored_poi = grouped_pois.setdefault(
                    dedupe_key,
                    {
                        **poi,
                        "matched_keywords": [],
                    },
                )
                if keyword not in stored_poi["matched_keywords"]:
                    stored_poi["matched_keywords"].append(keyword)

        if not grouped_pois:
            no_match_attempts = [
                item
                for item in search_attempts
                if item.get("status") != "1" and item.get("error") == "附近没有找到相关地点"
            ]
            hard_fail_attempts = [
                item
                for item in search_attempts
                if item.get("status") != "1" and item.get("error") != "附近没有找到相关地点"
            ]

            if no_match_attempts and not hard_fail_attempts:
                attempted_keywords = [item["keyword"] for item in no_match_attempts]
                return {
                    "status": "0",
                    "error": (
                        "未找到可用的候选路线廊道：start_location_coords 已成功解析，"
                        f"但 theme 关键词 {attempted_keywords} 在该坐标周边 {max_radius} 米内均未搜到结果。"
                        "更可能是 theme 与周边 POI 不匹配，而不是 start_location_coords 格式错误。"
                    ),
                    "error_code": "theme_no_matches",
                    "diagnosis": "问题更可能来自 theme，而不是 start_location_coords。",
                    "start_location_coords": start_location_coords,
                    "theme": theme,
                    "theme_keywords": theme_keywords,
                    "search_attempts": search_attempts,
                }

            if hard_fail_attempts and not no_match_attempts:
                failure_summary = "；".join(
                    f"{item['keyword']}: {item.get('error', '未知错误')}"
                    for item in hard_fail_attempts[:4]
                )
                return {
                    "status": "0",
                    "error": (
                        "未找到可用的候选路线廊道：start_location_coords 已成功解析，但周边检索阶段失败。"
                        f"失败详情：{failure_summary}"
                    ),
                    "error_code": "nearby_search_failed",
                    "diagnosis": "坐标格式本身看起来有效，但无法完成周边搜索。",
                    "start_location_coords": start_location_coords,
                    "theme": theme,
                    "theme_keywords": theme_keywords,
                    "search_attempts": search_attempts,
                }

            failure_summary = "；".join(
                f"{item['keyword']}: {item.get('error', '无结果')}"
                for item in search_attempts[:4]
            )
            return {
                "status": "0",
                "error": (
                    "未找到可用的候选路线廊道：start_location_coords 已成功解析，但未收集到可用于聚类的 POI。"
                    f"检索摘要：{failure_summary}"
                ),
                "error_code": "empty_corridor_candidates",
                "diagnosis": "无法仅凭当前结果判断是 theme 还是位置分布导致。",
                "start_location_coords": start_location_coords,
                "theme": theme,
                "theme_keywords": theme_keywords,
                "search_attempts": search_attempts,
            }

        corridor_groups = defaultdict(list)
        for poi in grouped_pois.values():
            poi_location = poi.get("location")
            if not poi_location:
                continue
            try:
                target_point = parse_location(poi_location)
            except Exception:
                continue
            direction = bearing_to_direction(origin_point, target_point)
            corridor_groups[direction].append(poi)

        if not corridor_groups:
            return {
                "status": "0",
                "error": (
                    "未找到可用的候选路线廊道：已搜到 POI，但这些结果缺少可用坐标，无法按方向聚类。"
                ),
                "error_code": "corridor_grouping_failed",
                "diagnosis": "问题不在 theme，也不在 start_location_coords 格式，而在返回 POI 数据缺少 location。",
                "start_location_coords": start_location_coords,
                "theme": theme,
                "theme_keywords": theme_keywords,
                "search_attempts": search_attempts,
                "matched_poi_count": len(grouped_pois),
            }

        corridors = []
        theme_label = "/".join(theme_keywords[:2])
        for direction, pois in corridor_groups.items():
            pois.sort(
                key=lambda item: item.get("distance_meters") if item.get("distance_meters") is not None else 999999
            )
            representative_poi = pois[0]
            matched_keywords = []
            for poi in pois:
                for keyword in poi.get("matched_keywords", []):
                    if keyword not in matched_keywords:
                        matched_keywords.append(keyword)

            type_tags = []
            for poi in pois:
                poi_type = poi.get("type")
                if poi_type and poi_type not in type_tags:
                    type_tags.append(poi_type)

            score = round(
                len(pois) * 2.0
                + len(matched_keywords) * 1.5
                - (representative_poi.get("distance_meters") or 0) / max(max_radius, 1),
                2,
            )
            corridors.append(
                {
                    "name": f"{direction}向{theme_label}漫步廊道",
                    "city": city,
                    "direction": direction,
                    "center": representative_poi.get("location"),
                    "poi_count": len(pois),
                    "theme_tags": matched_keywords[:5],
                    "sample_pois": [poi.get("name") for poi in pois[:3]],
                    "sample_types": type_tags[:3],
                    "score": score,
                }
            )

        corridors.sort(key=lambda item: (-item["score"], -item["poi_count"], item["direction"]))
        return {
            "status": "1",
            "city": city,
            "start_location_coords": start_location_coords,
            "theme_keywords": theme_keywords,
            "corridor_count": min(len(corridors), 5),
            "corridors": corridors[:5],
        }
    except Exception as error:
        return {"status": "0", "error": str(error)}


def run_demo() -> None:
    print("=== 新工具测试：绕路检测 ===\n")

    # 测试数据：凌波门->万林博物馆->东湖听涛（绕路路线）
    A = "114.370934,30.543459"  # 凌波门东湖观景点
    B = "114.363056,30.536666"  # 武汉大学万林艺术博物馆
    C = "114.375284,30.568065"  # 东湖听涛景区

    print("测试路线：A(凌波门) -> B(万林博物馆) -> C(东湖听涛)")
    print(f"A坐标: {A}")
    print(f"B坐标: {B}")
    print(f"C坐标: {C}\n")

    # 测试1: 距离矩阵
    print("--- 测试1: calculate_distance_matrix ---")
    matrix_result = calculate_distance_matrix([A, B, C], [A, B, C])
    if matrix_result.get("status") == "1":
        print(f"矩阵计算成功，节点数: {len(matrix_result['nodes'])}")
        print("节点信息:")
        for node in matrix_result["nodes"]:
            print(f"  [{node['index']}] {node['coords']}")
        print("\n距离矩阵 (km):")
        for i, row in enumerate(matrix_result["distances_km"]):
            print(f"  [{i}] -> {row}")
        print("\n时长矩阵 (分钟):")
        for i, row in enumerate(matrix_result["durations_minutes"]):
            print(f"  [{i}] -> {row}")
    print()

    print("等待2秒避免QPS超限...")
    time.sleep(2)

    # 测试2: 评估B是否在A->C之间绕路
    print("--- 测试2: evaluate_detour_impact (A->B->C中的B) ---")
    detour_result = evaluate_detour_impact(A, C, B)
    if detour_result.get("status") == "1":
        print(f"  方向夹角: {detour_result['direction_angle']}度")
        print(f"  是否折返: {detour_result['is_backtracking']}")
        print(f"  直达距离: {detour_result['direct_distance']}m")
        print(f"  绕路距离: {detour_result['detour_distance']}m")
        print(f"  增加距离: {detour_result['extra_distance']}m")
        print(f"  绕路比例: {detour_result['detour_ratio']}")
        print(f"  判定结果: {detour_result['verdict']}")
    else:
        print(f"  错误: {detour_result.get('error')}")
    print()

    print("等待2秒避免QPS超限...")
    time.sleep(2)

    # 测试3: 多点路线规划
    print("--- 测试3: plan_multi_waypoint_route (A->B->C) ---")
    route_result = plan_multi_waypoint_route(A, C, [B])
    if route_result.get("status") == "1":
        print(f"  总距离: {route_result['total_distance_km']}km")
        print(f"  总时长: {route_result['total_duration_minutes']}分钟")
        print(f"  分段数: {route_result['segment_count']}")
    else:
        print(f"  错误: {route_result.get('error')}")
    print()

    print("等待2秒避免QPS超限...")
    time.sleep(2)

    # 额外调试：测试单段路线
    print("--- 调试: 测试单段路线 (A->B) ---")
    debug_route = calculate_walking_route(A, B)
    print(f"  状态: {debug_route.get('status')}")
    if debug_route.get("status") == "1":
        print(f"  距离: {debug_route.get('distance_meters')}m")
        print(f"  时长: {debug_route.get('duration_minutes')}分钟")
    else:
        print(f"  错误: {debug_route.get('error')}")
    print()


def plan_multi_waypoint_route(origin_coords: str, destination_coords: str, waypoints: list[str]) -> dict:
    """
    多点路线规划：规划经过多个途经点的完整步行路线。
    通过分段调用步行路线API并拼接结果。
    """
    try:
        all_points = [origin_coords] + waypoints + [destination_coords]
        total_distance = 0
        total_duration = 0
        all_steps = []
        polylines = []

        for i in range(len(all_points) - 1):
            segment = get_detailed_walking_route(all_points[i], all_points[i + 1])
            if segment.get("status") != "1":
                return {"status": "0", "error": f"无法规划第{i+1}段路线"}

            total_distance += int(segment.get("distance_meters", 0))
            total_duration += int(segment.get("duration_minutes", 0))
            all_steps.extend(segment.get("steps", []))
            polylines.append(segment.get("full_polyline", ""))

        return {
            "status": "1",
            "total_distance": total_distance,
            "total_duration": total_duration,
            "total_distance_km": round(total_distance / 1000, 2),
            "total_duration_minutes": total_duration,
            "polyline": ";".join(p for p in polylines if p),
            "steps": all_steps,
            "segment_count": len(all_points) - 1,
        }
    except Exception as error:
        return {"status": "0", "error": str(error)}


def calculate_distance_matrix(origins: list[str], destinations: list[str]) -> dict:
    """
    距离矩阵：批量计算多个起点到多个终点的步行距离和时长。
    返回矩阵格式，便于LLM快速查找和对比。
    """
    try:
        # 构建完整节点列表（去重）
        all_coords = []
        coord_to_index = {}
        for coord in origins + destinations:
            if coord not in coord_to_index:
                coord_to_index[coord] = len(all_coords)
                all_coords.append(coord)

        n = len(all_coords)
        distances_km = [[0.0] * n for _ in range(n)]
        durations_minutes = [[0] * n for _ in range(n)]

        # 对每个节点作为终点调用API
        for dest_idx, dest in enumerate(all_coords):
            origins_str = "|".join(all_coords)
            payload = _amap_get(
                "/v3/distance",
                {
                    "origins": origins_str,
                    "destination": dest,
                    "type": "1",
                },
            )

            results = payload.get("results") or []
            for origin_idx, result in enumerate(results):
                dist = int(result.get("distance", 0))
                dur = int(result.get("duration", 0))
                distances_km[origin_idx][dest_idx] = round(dist / 1000, 2)
                durations_minutes[origin_idx][dest_idx] = math.ceil(dur / 60) if dur > 0 else 0

        return {
            "status": "1",
            "nodes": [{"index": i, "coords": coord} for i, coord in enumerate(all_coords)],
            "distances_km": distances_km,
            "durations_minutes": durations_minutes,
        }
    except Exception as error:
        return {"status": "0", "error": str(error)}


def evaluate_detour_impact(prev_stop_coords: str, next_stop_coords: str, candidate_poi_coords: str) -> dict:
    """
    评估在两个停靠点之间插入POI的绕路影响。
    使用方向夹角判断是否折返，同时计算距离增加。
    """
    try:
        # 解析坐标
        prev_lng, prev_lat = map(float, prev_stop_coords.split(','))
        next_lng, next_lat = map(float, next_stop_coords.split(','))
        poi_lng, poi_lat = map(float, candidate_poi_coords.split(','))

        # 计算方向向量
        vec1 = (poi_lng - prev_lng, poi_lat - prev_lat)  # prev->POI
        vec2 = (next_lng - poi_lng, next_lat - poi_lat)  # POI->next

        # 计算夹角
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        mag1 = (vec1[0]**2 + vec1[1]**2) ** 0.5
        mag2 = (vec2[0]**2 + vec2[1]**2) ** 0.5

        if mag1 > 0 and mag2 > 0:
            cos_angle = max(-1, min(1, dot_product / (mag1 * mag2)))
            angle_deg = math.acos(cos_angle) * 180 / math.pi
        else:
            angle_deg = 0

        # 计算距离
        direct = calculate_walking_route(prev_stop_coords, next_stop_coords)
        leg1 = calculate_walking_route(prev_stop_coords, candidate_poi_coords)
        leg2 = calculate_walking_route(candidate_poi_coords, next_stop_coords)

        direct_distance = int(direct.get("distance_meters", 0)) if direct.get("status") == "1" else 0
        detour_distance = 0
        if leg1.get("status") == "1" and leg2.get("status") == "1":
            detour_distance = int(leg1.get("distance_meters", 0)) + int(leg2.get("distance_meters", 0))

        extra_distance = detour_distance - direct_distance
        detour_ratio = round(extra_distance / direct_distance, 2) if direct_distance > 0 else 0

        # 判断是否折返或明显绕路
        is_backtracking = angle_deg > 120
        is_significant_detour = extra_distance > 500 or detour_ratio > 0.3

        return {
            "status": "1",
            "direction_angle": round(angle_deg, 1),
            "is_backtracking": is_backtracking,
            "direct_distance": direct_distance,
            "detour_distance": detour_distance,
            "extra_distance": extra_distance,
            "extra_minutes": extra_distance // 80,
            "detour_ratio": detour_ratio,
            "is_significant_detour": is_significant_detour,
            "verdict": "折返" if is_backtracking else ("明显绕路" if is_significant_detour else "可接受"),
        }
    except Exception as error:
        return {"status": "0", "error": str(error)}


__all__ = [
    "calculate_walking_route",
    "get_coordinates",
    "get_place_details",
    "get_detailed_walking_route",
    "get_walking_route_text",
    "run_demo",
    "search_along_route",
    "search_candidate_corridors",
    "search_nearby_places",
    "plan_multi_waypoint_route",
    "calculate_distance_matrix",
    "evaluate_detour_impact",
]


if __name__ == "__main__":
    run_demo()
