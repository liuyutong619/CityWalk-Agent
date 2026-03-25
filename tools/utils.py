import math
import re

DEFAULT_ALONG_ROUTE_SAMPLE_DISTANCE = 250
DEFAULT_ALONG_ROUTE_MAX_SAMPLES = 5
THEME_KEYWORDS = {
    "烟火气": ["小吃", "面馆", "咖啡馆", "便利店", "甜品店", "步行街"],
    "生活感": ["便利店", "社区商业", "早餐店", "咖啡馆", "甜品店"],
    "文艺": ["书店", "咖啡馆", "美术馆", "展览馆", "创意园"],
    "老街": ["老街", "历史建筑", "名人故居", "古建筑", "步行街"],
    "水岸": ["江边", "河边", "湖景", "公园", "绿道"],
    "公园": ["公园", "绿地", "湖景", "绿道"],
}


def parse_location(location_text: str) -> tuple[float, float]:
    longitude_text, latitude_text = location_text.split(",")
    return float(longitude_text), float(latitude_text)


def format_location(point: tuple[float, float]) -> str:
    longitude, latitude = point
    return f"{longitude:.6f},{latitude:.6f}"


def parse_polyline(polyline: str) -> list[tuple[float, float]]:
    points = []
    for point_text in polyline.split(";"):
        point_text = point_text.strip()
        if not point_text:
            continue
        points.append(parse_location(point_text))
    return points


def deduplicate_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduplicated_points = []
    for point in points:
        if not deduplicated_points or point != deduplicated_points[-1]:
            deduplicated_points.append(point)
    return deduplicated_points


def merge_polylines(step_polylines: list[str]) -> str:
    merged_points = []
    for polyline in step_polylines:
        merged_points.extend(parse_polyline(polyline))
    merged_points = deduplicate_points(merged_points)
    return ";".join(format_location(point) for point in merged_points)


def haversine_distance_meters(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    lon1, lat1 = point_a
    lon2, lat2 = point_b
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    hav = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    earth_radius_meters = 6371000
    return 2 * earth_radius_meters * math.asin(math.sqrt(hav))


def sample_polyline_points(
    polyline: str,
    sample_distance_meters: int = DEFAULT_ALONG_ROUTE_SAMPLE_DISTANCE,
    max_samples: int = DEFAULT_ALONG_ROUTE_MAX_SAMPLES,
) -> list[tuple[float, float]]:
    points = parse_polyline(polyline)
    if not points:
        return []
    if len(points) <= max_samples:
        return points

    sampled_points = [points[0]]
    distance_since_last_sample = 0.0

    for index in range(1, len(points) - 1):
        previous_point = points[index - 1]
        current_point = points[index]
        distance_since_last_sample += haversine_distance_meters(previous_point, current_point)
        if distance_since_last_sample >= sample_distance_meters:
            sampled_points.append(current_point)
            distance_since_last_sample = 0.0
            if len(sampled_points) >= max_samples - 1:
                break

    if sampled_points[-1] != points[-1]:
        sampled_points.append(points[-1])

    if len(sampled_points) <= max_samples:
        return sampled_points

    reduced_points = []
    for index in range(max_samples):
        mapped_index = round(index * (len(sampled_points) - 1) / (max_samples - 1))
        reduced_points.append(sampled_points[mapped_index])
    return deduplicate_points(reduced_points)


def bearing_to_direction(origin: tuple[float, float], target: tuple[float, float]) -> str:
    lon1, lat1 = map(math.radians, origin)
    lon2, lat2 = map(math.radians, target)
    delta_lon = lon2 - lon1
    x_value = math.sin(delta_lon) * math.cos(lat2)
    y_value = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    bearing = (math.degrees(math.atan2(x_value, y_value)) + 360) % 360
    directions = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    return directions[round(bearing / 45) % 8]


def normalize_theme_keywords(theme: str | list[str]) -> list[str]:
    if isinstance(theme, str):
        raw_items = [item.strip() for item in re.split(r"[、，,/]", theme) if item.strip()]
    else:
        raw_items = [str(item).strip() for item in theme if str(item).strip()]

    if not raw_items:
        raw_items = ["烟火气"]

    expanded_keywords = []
    for item in raw_items:
        expanded_keywords.append(item)
        expanded_keywords.extend(THEME_KEYWORDS.get(item, []))

    deduplicated_keywords = []
    for keyword in expanded_keywords:
        if keyword not in deduplicated_keywords:
            deduplicated_keywords.append(keyword)
    return deduplicated_keywords[:6]


__all__ = [
    "DEFAULT_ALONG_ROUTE_MAX_SAMPLES",
    "DEFAULT_ALONG_ROUTE_SAMPLE_DISTANCE",
    "THEME_KEYWORDS",
    "bearing_to_direction",
    "deduplicate_points",
    "format_location",
    "haversine_distance_meters",
    "merge_polylines",
    "normalize_theme_keywords",
    "parse_location",
    "parse_polyline",
    "sample_polyline_points",
]
