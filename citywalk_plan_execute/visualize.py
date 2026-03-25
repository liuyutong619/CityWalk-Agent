"""Format agent output for map visualization."""


def format_for_visualization(state: dict) -> dict:
    """Format agent state to match visualize_amap.py expected format."""
    final_output = state.get("final_output", {})
    intent = state.get("intent")
    execution_results = state.get("execution_results", [])
    nearby_route_pois = []

    # Extract coordinates from execution results
    start_point = {}
    stops = []

    for result in execution_results:
        action = result.get("action")
        res = result.get("result", {})

        # Get start point coordinates
        if action == "get_coordinates" and not start_point:
            if res.get("status") == "1":
                start_point = {
                    "input_name": intent.start_location if intent else "",
                    "resolved_name": res.get("formatted_address", ""),
                    "coordinates": res.get("location", ""),
                }

    final_stops = list(final_output.get("stops") or []) if isinstance(final_output, dict) else []
    if isinstance(final_output, dict):
        nearby_route_pois = [
            {
                "name": item.get("name") or "附近可选点",
                "coordinates": item.get("coordinates"),
                "address": item.get("address"),
                "category": item.get("category"),
                "distance_to_route_meters": item.get("distance_to_route_meters"),
                "position_hint": item.get("position_hint"),
                "matched_keywords": item.get("matched_keywords") or [],
                "selection_reason": item.get("selection_reason"),
                "source_sample_point": item.get("source_sample_point"),
            }
            for item in list(final_output.get("nearby_route_pois") or [])
            if item.get("coordinates")
        ]
    if final_stops:
        stops = [
            {
                "order": item.get("order", index + 1),
                "name": item.get("name") or f"站点{index + 1}",
                "coordinates": item.get("coordinates"),
                "walk_from_previous_minutes": item.get("walk_from_previous_minutes"),
                "recommended_stay_minutes": item.get("recommended_stay_minutes", 5),
            }
            for index, item in enumerate(final_stops)
            if item.get("coordinates")
        ]
    else:
        for result in execution_results:
            action = result.get("action")
            res = result.get("result", {})

            if action == "get_detailed_walking_route" and res.get("status") == "1":
                steps = res.get("steps", [])
                for idx, step in enumerate(steps[:5], 1):
                    polyline = step.get("polyline", "")
                    if polyline:
                        coords = polyline.split(";")[0] if ";" in polyline else polyline
                        stops.append({
                            "order": idx,
                            "name": step.get("road") or f"站点{idx}",
                            "coordinates": coords,
                            "walk_from_previous_minutes": step.get("duration_minutes", 0),
                            "recommended_stay_minutes": 5,
                        })

    if not start_point:
        if stops:
            start_point = {
                "input_name": intent.start_location if intent else "",
                "resolved_name": stops[0].get("name", ""),
                "coordinates": stops[0].get("coordinates"),
            }
        elif isinstance(final_output, dict):
            route_polyline = final_output.get("route_polyline")
            if isinstance(route_polyline, str) and route_polyline.strip():
                first_point = next((point for point in route_polyline.split(";") if "," in point), "")
                if first_point:
                    start_point = {
                        "input_name": intent.start_location if intent else "",
                        "resolved_name": "",
                        "coordinates": first_point,
                    }

    return {
        "user_query": state.get("user_query"),
        "intent": intent.dict() if intent else {},
        "start_point": start_point,
        "final_output": {
            "task_type": final_output.get("task_type", "citywalk_plan") if isinstance(final_output, dict) else "citywalk_plan",
            "stops": stops,
            "route_title": (
                final_output.get("route_title")
                if isinstance(final_output, dict) and final_output.get("route_title")
                else (f"{intent.activity_type}路线" if intent else "路线")
            ),
            "route_summary": final_output.get("route_summary") if isinstance(final_output, dict) else None,
            "total_duration_minutes": final_output.get("total_duration_minutes") if isinstance(final_output, dict) else None,
            "total_walking_minutes": final_output.get("total_walking_minutes") if isinstance(final_output, dict) else None,
            "route_polyline": final_output.get("route_polyline") if isinstance(final_output, dict) else None,
            "nearby_route_pois": nearby_route_pois,
        },
    }
