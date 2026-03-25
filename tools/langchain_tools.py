"""LangChain @tool wrappers for maps_tools functions.

Wraps the raw maps_tools functions as LangChain Tool objects for use with
bind_tools() in ReAct-style agent loops. Original functions are not modified.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.maps_tools import (
    get_coordinates as _get_coordinates,
    search_nearby_places as _search_nearby_places,
    get_place_details as _get_place_details,
    get_detailed_walking_route as _get_detailed_walking_route,
    get_walking_route_text as _get_walking_route_text,
    calculate_walking_route as _calculate_walking_route,
    search_along_route as _search_along_route,
    search_candidate_corridors as _search_candidate_corridors,
    plan_multi_waypoint_route as _plan_multi_waypoint_route,
    calculate_distance_matrix as _calculate_distance_matrix,
    evaluate_detour_impact as _evaluate_detour_impact,
)


# ---- LangChain @tool wrappers ----

@tool
def get_coordinates(address: str, city: str) -> dict:
    """
    位置解析：通过 POI 搜索将位置名称转换为经纬度坐标。必须先调用此工具获取坐标。
    返回 location 格式固定为 "经度,纬度"（例如 "114.357528,30.586428"）。
    """
    return _get_coordinates(address, city)


@tool
def search_nearby_places(keyword: str | list[str], location_coords: str, radius: int = 1000) -> dict:
    """
    周边搜索：根据中心点坐标搜索附近兴趣点（如咖啡馆、餐厅、景点）。
    keyword：附近兴趣点的主题，比如公园、餐厅、游乐园等等，若输入过于具体，可能输出结果为空。
             多关键词请直接传数组，如 ["公园","景点"]；不要传字符串化数组如 '["公园","景点"]'。
    location_coords：中心点坐标，必须是 "经度,纬度" 字符串；不要传 [经度, 纬度] 数组。
    radius：以中心点为圆心的搜索半径，单位是米
    """
    return _search_nearby_places(keyword, location_coords, radius)


@tool
def get_place_details(
    place_name: str,
    city: str | None = None,
    location_coords: str | None = None,
    radius: int = 1200,
) -> dict:
    """
    点位详情：根据地名获取说明性信息；可传 city 或 location_coords 提高命中。
    place_name：必传。要查询的地名
    location_coords：可选是否传入，若传入必须是 "经度,纬度" 字符串；不要传 [经度, 纬度] 数组。
    """
    return _get_place_details(place_name, city, location_coords, radius)


@tool
def get_detailed_walking_route(origin_coords: str, destination_coords: str) -> dict:
    """
    详细步行路线：返回总距离、总时长、完整 polyline 以及逐段 steps。这里返回的是完整的路线规划，包括转向点的经纬度等详细信息，适合在需要给出详细路线时调用。
    origin_coords / destination_coords：都必须是 "经度,纬度" 字符串；不要传 [经度, 纬度] 数组。
    """
    return _get_detailed_walking_route(origin_coords, destination_coords)


@tool
def get_walking_route_text(
    origin_coords: str,
    destination_coords: str,
    detail_level: str = "low",
    travel_mode: str = "步行",
) -> dict:
    """步行路线摘要：返回可读的路线说明。
    origin_coords / destination_coords：都必须是 "经度,纬度" 字符串；不要传 [经度, 纬度] 数组。
    detail_level: 可选 "low" 或 "high"。
    travel_mode: 交通方式，可选 "步行"、"骑行"、"公交"、"驾车" 等。
    """
    return _get_walking_route_text(origin_coords, destination_coords, detail_level, travel_mode)


@tool
def calculate_walking_route(origin_coords: str, destination_coords: str) -> dict:
    """
    步行距离计算：计算两点之间的步行距离和预计耗时。只会返回距离和耗时，不会返回路线。
    origin_coords / destination_coords：都必须是 "经度,纬度" 字符串；不要传 [经度, 纬度] 数组。
    """
    return _calculate_walking_route(origin_coords, destination_coords)


@tool
def search_along_route(
    origin_coords: str,
    destination_coords: str,
    keyword: str | list[str],
    radius: int = 200,
) -> dict:
    """
    沿路线搜索：输入起终点坐标，自动规划路线并沿途搜索兴趣点。
    origin_coords：起点坐标，必须是 "经度,纬度" 字符串；不要传 [经度, 纬度] 数组。
    destination_coords：终点坐标，必须是 "经度,纬度" 字符串；不要传 [经度, 纬度] 数组。
    keyword：附近兴趣点的主题，比如公园、餐厅等等，若输入过于具体，可能输出结果为空。
             多关键词请直接传数组，如 ["公园","景点"]；不要传字符串化数组如 '["公园","景点"]'。
    radius：沿路线两侧的搜索半径，单位米，如果结果不理想，可能是半径太小，请调大半径比如500->3000！
    """
    return _search_along_route(origin_coords, destination_coords, keyword, radius)


@tool
def search_candidate_corridors(
    start_location_coords: str,
    city: str,
    theme: str,
    max_radius: int = 3000,
) -> dict:
    """
    候选廊道搜索：围绕起点搜索不同方向上有潜力漫步的片区。会返回不同方向的大致信息，按照东南西北为类别，返回各方向可能包含什么东西，适合没有明确线路时调用。
    start_location_coords：必须是 "经度,纬度" 字符串；不要传 [经度, 纬度] 数组。
    """
    return _search_candidate_corridors(start_location_coords, city, theme, max_radius)


@tool
def plan_multi_waypoint_route(
    origin_coords: str,
    destination_coords: str,
    waypoints: list[str],
) -> dict:
    """
    多点路线规划：按给定顺序规划经过多个POI的完整步行路线。

    使用场景：
    - 已确定POI访问顺序后，获取完整路线的总距离、总时长和详细路径
    - 验证规划方案的可行性（总距离是否合理）
    - 获取最终输出所需的polyline数据

    参数：
    - origin_coords：起点坐标，格式 "经度,纬度"（如 "114.370,30.543"）
    - destination_coords：终点坐标，格式 "经度,纬度"
    - waypoints：途经点坐标列表，按访问顺序排列，格式 ["经度1,纬度1", "经度2,纬度2", ...]

    返回：
    - total_distance_km：总距离（公里）
    - total_duration_minutes：总时长（分钟）
    - polyline：完整路径的坐标串
    - segment_count：分段数量

    注意：此工具不负责优化顺序，只是按给定顺序规划路线。
    """
    return _plan_multi_waypoint_route(origin_coords, destination_coords, waypoints)


@tool
def calculate_distance_matrix(origins: list[str], destinations: list[str]) -> dict:
    """
    距离矩阵：一次性获取多个POI之间的所有距离和时长关系。

    使用场景：
    - 在筛选和排序POI前，先获取完整的距离矩阵
    - 快速对比不同POI顺序的总距离（如 A->B->C vs A->C->B）
    - 判断某个POI是否偏离主路线（通过对比不同路径的距离）
    - 一次API调用获取所有数据，避免重复调用

    参数：
    - origins：起点坐标列表，格式 ["经度1,纬度1", "经度2,纬度2", ...]
    - destinations：终点坐标列表，格式同上（通常与origins相同，构建完整矩阵）

    返回：
    - nodes：节点列表 [{"index": 0, "coords": "114.37,30.54"}, ...]
    - distances_km：距离矩阵（二维数组），distances_km[i][j] 表示节点i到节点j的距离（公里）
    - durations_minutes：时长矩阵（二维数组），durations_minutes[i][j] 表示节点i到节点j的时长（分钟）

    示例：如果有3个POI [A, B, C]，返回的矩阵可以快速查询任意两点间距离。
    """
    return _calculate_distance_matrix(origins, destinations)


@tool
def evaluate_detour_impact(
    prev_stop_coords: str,
    next_stop_coords: str,
    candidate_poi_coords: str,
) -> dict:
    """
    绕路评估：判断在两个已确定的停靠点之间插入候选POI是否会导致折返或明显绕路。

    使用场景：
    - 已有部分POI顺序（如 A->B），考虑是否在中间插入新POI（A->C->B）
    - 判断插入后是否会"往回走"（折返）或大幅增加距离
    - 辅助决策：保留还是删除某个POI

    判断标准：
    - 方向夹角 > 120度：折返（从A到C，再从C到B时方向相反）

    参数：
    - prev_stop_coords：前一个停靠点坐标，格式 "经度,纬度"
    - next_stop_coords：后一个停靠点坐标，格式 "经度,纬度"
    - candidate_poi_coords：候选POI坐标，格式 "经度,纬度"

    返回：
    - direction_angle：方向夹角（度），>120度表示折返
    - is_backtracking：是否折返（布尔值）
    - direct_distance：前后停靠点直达距离（米）
    - detour_distance：插入POI后的总距离（米）
    - extra_distance：增加的距离（米）
    - detour_ratio：绕路比例（增加距离/直达距离）
    - verdict：综合判定结果（"折返" / "明显绕路" / "可接受"）

    示例：路线 A->B->C，评估B是否绕路，调用 evaluate_detour_impact(A, C, B)
    """
    return _evaluate_detour_impact(prev_stop_coords, next_stop_coords, candidate_poi_coords)


# ---- Completion / selection signal models (used as bind_tools targets) ----

class SelectedPOI(BaseModel):
    """被精选的兴趣点"""
    name: str = Field(description="兴趣点名称，必须与候选列表中的名称完全一致")
    reason: str = Field(description="选择理由：为什么这个点位适合本次 CityWalk")


class SelectPOIs(BaseModel):
    """从当前候选池中精选有趣的点位。审视所有搜索到的候选，只挑真正值得停留的。
    可以在搜索过程中随时调用，也可以在搜索结束前集中调用一次。"""
    selected_pois: list[SelectedPOI] = Field(
        description="精选的兴趣点列表。只保留有趣的、符合用户意图的，去掉无聊的、不相关的、重复的。质量比数量重要。"
    )


class ExplorationComplete(BaseModel):
    """搜索与筛选都已完成，调用此工具结束 POI 探索阶段。
    注意：结束前务必先调用 SelectPOIs 完成筛选，否则没有任何点位会被保留。"""
    summary: str = Field(description="探索过程与结果的简要总结")


class SubmitRoutePlan(BaseModel):
    """提交路线编排方案，由程序补全完整路线详情。"""
    selected_stops: list[dict] = Field(description="编排后的停靠点列表，格式：[{'name': '景点名', 'location': '经度,纬度', 'reason': '选择理由'}]")
    reasoning: str = Field(description="编排思路：为什么选这些点、为什么这个顺序、删除了哪些绕路点")


class PlanningComplete(BaseModel):
    """路线规划完成，调用此工具结束规划并返回结果摘要。"""
    summary: str = Field(description="规划结果的简要总结")


class SearchRouteNearbyPlaces(BaseModel):
    """沿最终主路线搜索周边可选点位。系统会自动使用已经确定的 route_polyline。"""
    keyword: str | list[str] = Field(
        description="沿最终路线搜索的具体检索词。优先使用可被地图理解的具体品类或地点类型；多关键词可直接传数组。"
    )
    radius: int = Field(default=250, description="沿主路线两侧搜索半径，单位米。")


class SelectedNearbyPOI(BaseModel):
    """被选中的主路线附近可选点位"""
    name: str = Field(description="点位名称，必须与候选列表中的名称完全一致")
    reason: str = Field(description="为什么它值得作为主路线旁边的可选探索点")


class SubmitNearbyRoutePOIs(BaseModel):
    """提交最终保留的主路线附近可选点位。"""
    selected_pois: list[SelectedNearbyPOI] = Field(
        description="最终保留的可选探索点位列表，最多 6 个；如果没有合适点位，可提交空列表。"
    )


class NearbyPOIEnrichmentComplete(BaseModel):
    """路线周边可选点补充完成。"""
    summary: str = Field(description="补充结果的简要总结")


# Convenience collections for sub-agent tool binding
POI_EXPLORER_TOOLS = [
    search_along_route,
    search_nearby_places,
    get_place_details,
    search_candidate_corridors,
    get_coordinates,
    SelectPOIs,
    ExplorationComplete,
]

ROUTE_PLANNER_TOOLS = [
    calculate_walking_route,
    calculate_distance_matrix,
    evaluate_detour_impact,
    SubmitRoutePlan,
    PlanningComplete,
]

ROUTE_POI_ENRICHER_TOOLS = [
    SearchRouteNearbyPlaces,
    get_place_details,
    SubmitNearbyRoutePOIs,
    NearbyPOIEnrichmentComplete,
]
