#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

DEFAULT_INPUT_JSON = Path(__file__).with_name("result.json")
DEFAULT_OUTPUT_HTML = Path(__file__).with_name("citywalk_amap.html")


def load_amap_js_config():
    """从.env文件加载高德前端 JS API 所需的 KEY 和安全密钥"""
    env_file = Path(__file__).parent / '.env'
    config = {
        'AMAP_JS_KEY': os.getenv('AMAP_JS_KEY', 'YOUR_AMAP_JS_KEY'),
        'AMAP_JS_SECURITY_CODE': os.getenv('AMAP_JS_SECURITY_CODE', 'YOUR_SECURITY_CODE_HERE'),
    }
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                # 忽略注释和空行
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, val = line.split('=', 1)
                    key = key.strip()
                    val = val.strip()
                    if key == 'AMAP_JS_KEY':
                        config['AMAP_JS_KEY'] = val
                    elif key == 'AMAP_JS_SECURITY_CODE':
                        config['AMAP_JS_SECURITY_CODE'] = val
    return config['AMAP_JS_KEY'], config['AMAP_JS_SECURITY_CODE']


def parse_args():
    parser = argparse.ArgumentParser(description="根据 CityWalk JSON 结果生成高德地图 HTML")
    parser.add_argument(
        "input_json",
        nargs="?",
        default=str(DEFAULT_INPUT_JSON),
        help="CityWalk 结果 JSON 路径，默认读取 result.json",
    )
    parser.add_argument(
        "--output-html",
        default=str(DEFAULT_OUTPUT_HTML),
        help="输出 HTML 路径，默认写入 citywalk_amap.html",
    )
    return parser.parse_args()


def load_citywalk_data(input_path: Path):
    text = input_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        for line in reversed(text.splitlines()):
            candidate = line.strip()
            if not candidate.startswith('{'):
                continue
            try:
                data = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        else:
            raise ValueError(
                f"无法从 {input_path} 解析出有效 JSON。请优先使用 run_langgraph_citywalk_demo.py 新生成的纯 JSON 文件。"
            ) from None

    if not isinstance(data, dict):
        raise ValueError(f"{input_path} 的顶层内容不是 JSON 对象。")
    return data


def parse_coordinates(raw_coordinates, field_name):
    if not isinstance(raw_coordinates, str) or "," not in raw_coordinates:
        raise ValueError(f"{field_name} 缺少合法 coordinates，当前值: {raw_coordinates!r}")
    lng_text, lat_text = raw_coordinates.split(",", 1)
    try:
        return float(lng_text), float(lat_text)
    except ValueError as exc:
        raise ValueError(f"{field_name} 的 coordinates 不是合法经纬度: {raw_coordinates!r}") from exc


def format_minutes_label(minutes):
    if minutes is None:
        return "未知"
    if isinstance(minutes, float) and minutes.is_integer():
        minutes = int(minutes)
    return f"{minutes}分钟"


def generate_amap_html(data, amap_js_key, amap_security_code):
    """生成高德地图HTML"""
    final = data.get("final_output", {})
    start_point = data.get("start_point", {})
    stops = final.get("stops", [])
    nearby_route_pois = final.get("nearby_route_pois", [])
    route_polyline = final.get("route_polyline")

    if not isinstance(final, dict) or not isinstance(start_point, dict):
        raise ValueError("输入 JSON 缺少 final_output 或 start_point，无法绘制地图。")
    if not isinstance(route_polyline, str) or not route_polyline.strip():
        raise ValueError("输入 JSON 缺少 final_output.route_polyline，无法直接绘制 Agent 规划路线。")

    # 构建标记点数据
    route_path = []
    for point in route_polyline.split(";"):
        if "," not in point:
            continue
        lng, lat = point.split(",", 1)
        try:
            route_path.append([float(lng), float(lat)])
        except ValueError:
            continue
    if len(route_path) < 2:
        raise ValueError("final_output.route_polyline 解析后的坐标点不足，无法绘制路线。")

    start_coordinates = start_point.get("coordinates")
    if isinstance(start_coordinates, str) and "," in start_coordinates:
        start_lng, start_lat = parse_coordinates(start_coordinates, "start_point")
    elif stops:
        start_lng, start_lat = parse_coordinates(stops[0].get("coordinates"), "stop[0]")
    else:
        start_lng, start_lat = route_path[0]
    start_name = (
        start_point.get("resolved_name")
        or start_point.get("input_name")
        or (stops[0].get("name") if stops else "")
        or "起点"
    )

    markers = [{
        "lng": start_lng,
        "lat": start_lat,
        "title": "起点",
        "content": start_name,
        "type": "start",
        "order": 0,
        "name": start_name,
    }]

    for stop in stops:
        coords = stop.get("coordinates")
        if not coords:
            continue
        lng, lat = parse_coordinates(coords, f"stop[{stop.get('order', '?')}]")
        markers.append({
            "lng": lng,
            "lat": lat,
            "title": f"{stop['order']}. {stop['name']}",
            "content": (
                f"{stop['name']}<br>"
                f"步行: {format_minutes_label(stop.get('walk_from_previous_minutes'))}<br>"
                f"停留: {format_minutes_label(stop.get('recommended_stay_minutes'))}"
            ),
            "type": "stop",
            "order": stop['order'],
            "name": stop['name'],
        })

    for poi in nearby_route_pois:
        coords = poi.get("coordinates")
        if not coords:
            continue
        try:
            lng, lat = parse_coordinates(coords, poi.get("name") or "nearby_route_poi")
        except ValueError:
            continue
        
        category = poi.get("category", "")
        # 根据分类匹配一些有趣的 Emoji 💡
        emoji = "✨"
        if any(kw in category for kw in ["餐饮", "美食", "餐厅", "吃"]):
            emoji = "🍴"
        elif "咖啡" in category:
            emoji = "☕"
        elif any(kw in category for kw in ["奶茶", "排档", "茶"]):
            emoji = "🧋"
        elif any(kw in category for kw in ["甜品", "烘焙", "面包"]):
            emoji = "🥐"
        elif any(kw in category for kw in ["酒吧", "居酒屋", "酒"]):
            emoji = "🍻"
        elif any(kw in category for kw in ["公园", "森林", "绿地", "植物园"]):
            emoji = "🌳"
        elif any(kw in category for kw in ["山", "步道", "观景"]):
            emoji = "⛰️"
        elif any(kw in category for kw in ["湖", "水", "江", "河", "海"]):
            emoji = "🌊"
        elif any(kw in category for kw in ["博物馆", "美术馆", "展览", "画廊"]):
            emoji = "🏛️"
        elif any(kw in category for kw in ["书店", "阅读", "书"]):
            emoji = "📖"
        elif any(kw in category for kw in ["寺", "庙", "道观", "教堂", "遗址"]):
            emoji = "⛩️"
        elif any(kw in category for kw in ["商场", "购物", "市集"]):
            emoji = "🛍️"
        elif any(kw in category for kw in ["游乐场", "乐园", "玩乐"]):
            emoji = "🎡"
        elif any(kw in category for kw in ["风景", "景点", "地标", "打卡"]):
            emoji = "📸"

        matched_keywords = poi.get("matched_keywords") or []
        matched_keywords_text = "、".join(str(item) for item in matched_keywords if str(item).strip())
        detail_lines = []
        if poi.get("category"):
            detail_lines.append(f"类型: {poi['category']}")
        if poi.get("distance_to_route_meters") is not None:
            detail_lines.append(f"距主路线: {poi['distance_to_route_meters']}米")
        if poi.get("position_hint"):
            detail_lines.append(f"位置: {poi['position_hint']}")
        if matched_keywords_text:
            detail_lines.append(f"匹配: {matched_keywords_text}")
        if poi.get("selection_reason"):
            detail_lines.append(str(poi["selection_reason"]))

        markers.append({
            "lng": lng,
            "lat": lat,
            "title": poi.get("name") or "附近可选点",
            "content": "<br>".join(detail_lines),
            "type": "nearby",
            "order": None,
            "name": f"{emoji} {poi.get('name') or '附近可选点'}",
        })

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{final.get('route_title', 'CityWalk路线')}</title>
    <script>
        window._AMapSecurityConfig = {{
            securityJsCode: "{amap_security_code}"
        }}
    </script>
    <script src="https://webapi.amap.com/maps?v=2.0&key={amap_js_key}"></script>
    <style>
        #container {{width: 100%; height: 100vh; margin: 0; padding: 0;}}
        .info {{position: absolute; top: 10px; left: 10px; background: white;
                padding: 15px; border-radius: 5px; box-shadow: 0 2px 6px rgba(0,0,0,0.3); z-index: 999;}}
        
        /* 自定义标记点的样式 */
        .custom-marker {{
            background-color: white;
            border: 2px solid #3366FF;
            border-radius: 20px;
            padding: 4px 10px;
            color: #333;
            font-size: 13px;
            font-weight: bold;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            white-space: nowrap;
            /* 居中偏移 */
            transform: translate(-50%, -50%);
        }}
        .custom-marker.start {{
            border-color: #FF5722;
            background-color: #FFF3E0;
            color: #E64A19;
        }}
        .custom-marker.nearby {{
            border-color: #D97706;
            background-color: #FFF7ED;
            color: #B45309;
        }}
        .custom-marker-arrow {{
            position: absolute;
            bottom: -6px;
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 0;
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
            border-top: 6px solid #3366FF;
        }}
        .custom-marker.start .custom-marker-arrow {{
            border-top-color: #FF5722;
        }}
        .custom-marker.nearby .custom-marker-arrow {{
            border-top-color: #D97706;
        }}
    </style>
</head>
<body>
    <div class="info">
        <h3>{final.get('route_title', 'CityWalk路线')}</h3>
        <p>总时长: {final.get('total_duration_minutes', 0)}分钟 | 步行: {final.get('total_walking_minutes', 0)}分钟</p>
    </div>
    <div id="container"></div>
    <script>
        console.log('开始初始化地图...');
        var map = new AMap.Map('container', {{
            zoom: 15,
            center: [{start_lng}, {start_lat}]
        }});
        console.log('地图初始化完成');

        var markers = {json.dumps(markers, ensure_ascii=False)};
        var routePath = {json.dumps(route_path)};
        var coordOffsets = {{}}; // 用于记录每个坐标出现的次数以避免重叠

        markers.forEach(function(m) {{
            // 生成自定义 HTML 内容作为 Marker
            var isStart = m.type === 'start';
            var isNearby = m.type === 'nearby';
            var indexText = m.order;
            var nameText = m.name;
            var className = 'custom-marker';
            if (isStart) {{
                className += ' start';
            }} else if (isNearby) {{
                className += ' nearby';
            }}
            var labelText = isStart || isNearby ? nameText : indexText + '. ' + nameText;

            var markerContent = document.createElement('div');
            markerContent.className = className;
            markerContent.innerHTML = '<span>' + labelText + '</span><div class="custom-marker-arrow"></div>';

            var coordKey = m.lng.toFixed(5) + ',' + m.lat.toFixed(5);
            var overlapCount = coordOffsets[coordKey] || 0;
            coordOffsets[coordKey] = overlapCount + 1;

            var offsetY = -10;
            if (overlapCount > 0) {{
                offsetY = -10 - (overlapCount * 32); // 发生重叠时向上偏移排列，避免字看不清
            }}

            var marker = new AMap.Marker({{
                position: [m.lng, m.lat],
                title: m.title,
                content: markerContent, // 使用自定义的 DOM 节点
                offset: new AMap.Pixel(0, offsetY), // 动态调整Y轴偏移，防止遮挡
                zIndex: 100 + overlapCount // 让后添加的标记在更上层显示
            }});

            var infoWindow = new AMap.InfoWindow({{
                content: '<div style="padding:10px;"><b>' + m.title + '</b><br>' + m.content + '</div>'
            }});

            marker.on('click', function() {{
                infoWindow.open(map, marker.getPosition());
            }});

            map.add(marker);
        }});

        var routePolyline = new AMap.Polyline({{
            path: routePath,
            strokeColor: '#FF3333',
            strokeWeight: 8,
            strokeOpacity: 0.8,
            showDir: true
        }});
        map.add(routePolyline);
        map.setFitView();

    </script>
</body>
</html>"""
    return html

if __name__ == "__main__":
    args = parse_args()
    amap_js_key, amap_security_code = load_amap_js_config()
    input_path = Path(args.input_json).expanduser()
    output_path = Path(args.output_html).expanduser()
    data = load_citywalk_data(input_path)

    html = generate_amap_html(data, amap_js_key, amap_security_code)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"高德地图已保存到: {output_path}")
