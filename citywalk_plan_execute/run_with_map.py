"""Run CityWalk Plan and Execute agent with visualization."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from citywalk_plan_execute import build_graph
from citywalk_plan_execute.visualize import format_for_visualization
from visualize_amap import generate_amap_html, load_amap_js_config

# 加载 .env 文件
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"


DEFAULT_QUERY = "我现在在武汉市的武汉大学的凌波门，想要去东湖走路，给我推荐一条走路不错的路线，一定要走东湖道（在湖中间那个道），就是在绿道上随便走走，相当于经过东湖绿道全景广场、湖心岛动物博物馆这个路，然后到湖北省博物馆结束吧"
#"我是一个成都的熊熊，给我推荐一个武汉的citywalk路线,我很随性，随便规划，我的起点是武汉站，到时候还要回来，我想和我的熊熊朋友手牵手看樱花，半天时间"
#"我现在在武汉市的武汉大学的凌波门，想要去东湖走路，给我推荐一条走路不错的路线，一定要走东湖道（在湖中间那个道），就是在绿道上随便走走，相当于经过东湖绿道全景广场、湖心岛动物博物馆这个路，然后到湖北省博物馆结束吧"
#"我是一个成都的熊熊，给我推荐一个武汉的citywalk路线,我很随性，随便规划，我的起点是武汉站，到时候还要回来，我想和我的熊熊朋友手牵手看樱花，半天时间"
#"我现在在武汉市的武汉大学的凌波门，想要去东湖走路，给我推荐一条走路不错的路线，一定要走东湖道（在湖中间那个道），就是在绿道上随便走走，相当于经过东湖绿道全景广场、湖心岛动物博物馆这个路，然后到湖北省博物馆结束吧"
#"我现在在武汉市的武汉大学的凌波门，想要去东湖走路，给我推荐一条走路不错的路线，一定要走东湖绿道，就是在绿道上随便走走，然后到湖北省博物馆结束吧"
#"我现在在汉口江滩的三阳广场，想沿着江边纯步行，去看看知音号，然后去江汉关博物馆，最后走到江汉路步行街吃点东西，全程只走路，帮我规划一条最顺畅不绕路的步行路线，大概需要走多久？"
#"我住在武昌的昙华林，今天大概只有3个小时散步。我想徒步去粮道街吃个过早，然后再走到黄鹤楼的外面拍张照，最后走回昙华林。全程必须纯步行，绝对不能坐任何车，路线尽量安排有武汉老街巷感觉的。"
#"我在东湖绿道的楚风园入口，想在东湖边纯走路散心。给我规划一条大概5公里的纯步行路线，要求一直在水边走，最后走到磨山景区的某个出口结束，绝不允许出现需要坐船或观光车的节点。"
#"我带了一只小狗在解放公园南门，想在附近完全步行遛狗。规划一个1小时左右的纯步行环线，途径有树荫的安静小路，并且最后必须步行回到解放公园南门。注意不经过任何需要坐车的路段，步行距离控制在3-4公里。"
#"我想挑战一次极限纯徒步：起点是武汉大学凌波门，我想纯走路去湖北省博物馆看看，然后再纯走路去楚河汉街。我发誓绝不借助任何交通工具，请帮我规划这条完全靠双脚的路线，并计算步数和耗时。"


def _resolve_queries() -> list[str]:
    cli_queries = [item.strip() for item in sys.argv[1:] if item.strip()]
    if cli_queries:
        return cli_queries

    env_query = os.environ.get("CITYWALK_QUERY", "").strip()
    if env_query:
        return [env_query]

    return [DEFAULT_QUERY]


def _build_initial_state(query: str) -> dict:
    return {
        "user_query": query,
        "conversation_history": [],
        "clarification": None,
        "intent": None,
        "retrieved_info": [],
        "explored_pois": [],
        "planned_routes": [],
        "route_plan": None,
        "nearby_route_pois": [],
        "execution_results": [],
        "evaluation": None,
        "supervisor_messages": [],
        "supervisor_iterations": 0,
        "replan_count": 0,
        "final_output": None,
    }


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _to_json_safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_to_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


async def main():
    queries = _resolve_queries()

    graph = build_graph()

    config = {
        "configurable": {
            "max_replan_count": 3,
        }
    }

    for index, query in enumerate(queries, start=1):
        print(f"用户输入[{index}/{len(queries)}]: {query}\n")
        result = await graph.ainvoke(_build_initial_state(query), config)

        if result.get("final_output", {}).get("need_clarification"):
            print("需要澄清:")
            print(result["final_output"]["question"])
            continue

        viz_data = format_for_visualization(result)
        full_state = _to_json_safe(result)
        (Path(__file__).parent.parent / f"full_state_{index}.json").write_text(
            json.dumps(full_state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        suffix = "" if len(queries) == 1 else f"_{index}"
        output_json = Path(__file__).parent.parent / f"plan_execute_result{suffix}.json"
        output_json.write_text(json.dumps(viz_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已保存: {output_json}")

        if viz_data.get("final_output", {}).get("route_polyline"):
            amap_js_key, amap_security_code = load_amap_js_config()
            html = generate_amap_html(viz_data, amap_js_key, amap_security_code)

            output_html = Path(__file__).parent.parent / f"plan_execute_map{suffix}.html"
            output_html.write_text(html, encoding="utf-8")
            print(f"地图已生成: {output_html}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
