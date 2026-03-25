import asyncio
import html
import os
import json
import textwrap
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Import agent things
from citywalk_plan_execute import build_graph
from citywalk_plan_execute.visualize import format_for_visualization
from citywalk_plan_execute.configuration import Configuration
from visualize_amap import generate_amap_html, load_amap_js_config

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"

st.set_page_config(page_title="CityWalk Planner", layout="wide")


def _build_initial_state(query: str, history: list = None) -> dict:
    if history is None:
        history = []
    return {
        "user_query": query,
        "conversation_history": history,
        "clarification": None,
        "intent": None,
        "retrieved_info": [],
        "explored_pois": [],
        "planned_routes": [],
        "route_plan": None,
        "execution_results": [],
        "evaluation": None,
        "supervisor_messages": [],
        "supervisor_iterations": 0,
        "replan_count": 0,
        "final_output": None,
    }


NODE_DESCRIPTIONS = {
    "clarify_intent": ("🧐 分析需求", "正在理解你的意图并提取关键信息..."),
    "parse_intent": ("🗺️ 解析意图", "正在定位您的位置信息与坐标..."),
    "info_retriever": ("📚 检索经验", "正在从小红书等社区检索真实的 CityWalk 经验与避坑指南..."),
    "supervisor": ("👨‍💼 规划调度 (Supervisor)", "正在统筹规划全局，审视目标与分配子任务..."),
    "supervisor_tools": ("⚙️ 执行调度 (Supervisor)", "正在调度探索节点和规划节点，监控执行结果..."),
    "poi_explorer": ("🕵️‍♂️ 探索任务 (POI Explorer)", "Sub-Agent 正在多方位搜索您周边的路线、景点与宝藏小店..."),
    "poi_explorer_tools": ("🔍 仔细摸排 (POI Explorer)", "Sub-Agent 正在对找到的地点进行严格筛选和精选..."),
    "route_planner": ("🧭 路线规划 (Route Planner)", "Sub-Agent 正在为您编排最顺路的行走动线..."),
    "route_planner_tools": ("📏 测量计算 (Route Planner)", "Sub-Agent 正在计算具体的步行路线、时间和连通性..."),
    "route_poi_enricher": ("✨ 丰富路线 (POI Enricher)", "正在为主路线周边补充可选的探索点位..."),
    "route_poi_enricher_tools": ("📍 筛选周边 (POI Enricher)", "正在筛选主路线周边最值得打卡的宝藏地点..."),
    "json_formatter": ("📝 结果整理", "路线生成完毕，正在为您格式化并生成可视化地图数据...")
}


def _truncate_text(text: str, limit: int = 220) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _extract_text_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text).strip())
        return "\n".join(parts)
    return str(content).strip()


def _tool_call_label(name: str) -> str:
    labels = {
        "DispatchPOIExplorer": "正在派遣 POI Explorer",
        "DispatchRoutePlanner": "正在派遣 Route Planner",
        "AllTasksComplete": "正在确认全部任务完成",
        "SelectPOIs": "正在精选候选 POI",
        "ExplorationComplete": "正在收尾探索结果",
        "SubmitRoutePlan": "正在提交路线方案",
        "PlanningComplete": "正在收尾路线规划",
    }
    return labels.get(name, f"正在调用工具 {name}")


def _tool_call_detail(name: str, args: dict) -> str:
    args = args or {}

    if name in {"DispatchPOIExplorer", "DispatchRoutePlanner"}:
        return str(args.get("task_description") or "").strip()

    if name in {"AllTasksComplete", "ExplorationComplete", "PlanningComplete"}:
        return str(args.get("summary") or "").strip()

    if name == "SelectPOIs":
        selected = args.get("selected_pois") or []
        names = [str(item.get("name")).strip() for item in selected if isinstance(item, dict) and item.get("name")]
        if names:
            return "、".join(names[:4])
        if selected:
            return f"{len(selected)} 个候选点位"
        return ""

    if name == "SubmitRoutePlan":
        stops = args.get("selected_stops") or []
        names = [str(item.get("name")).strip() for item in stops if isinstance(item, dict) and item.get("name")]
        if names:
            return " -> ".join(names[:4])
        if stops:
            return f"{len(stops)} 个停靠点"
        return ""

    for key in ("keyword", "keywords", "query", "origin", "destination", "location", "center", "task_description"):
        value = args.get(key)
        if not value:
            continue
        if isinstance(value, list):
            joined = "、".join(str(item).strip() for item in value[:4] if str(item).strip())
            if joined:
                return joined
        elif not isinstance(value, dict):
            return str(value).strip()

    return ""


def _summarize_tool_calls(tool_calls: list) -> str:
    if not tool_calls:
        return ""

    summaries = []
    for tool_call in tool_calls[:2]:
        if not isinstance(tool_call, dict):
            continue
        name = tool_call.get("name") or "未知工具"
        detail = _tool_call_detail(name, tool_call.get("args") or {})
        summary = _tool_call_label(name)
        if detail:
            summary = f"{summary}：{detail}"
        summaries.append(_truncate_text(summary, 140))

    if not summaries:
        return ""
    if len(tool_calls) > len(summaries):
        summaries.append(f"另有 {len(tool_calls) - len(summaries)} 个动作待执行")
    return "\n".join(summaries)


def _get_latest_message(update: dict):
    for key in ("supervisor_messages", "messages"):
        messages = update.get(key)
        if not isinstance(messages, list):
            continue
        for message in reversed(messages):
            if message is not None:
                return message
    return None


def _compact_supervisor_status(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return ""

    preferred_prefixes = (
        "已探索 POI",
        "代表:",
        "已规划路线",
        "最新 SubmitRoutePlan",
        "- ",
    )
    preferred_lines = [line for line in lines if line.startswith(preferred_prefixes)]
    chosen_lines = preferred_lines[:4] if preferred_lines else lines[:4]
    return _truncate_text("\n".join(chosen_lines), 240)


def _build_dynamic_status(node_name: str, updates: dict) -> tuple[str, str]:
    title, fallback_desc = NODE_DESCRIPTIONS.get(
        node_name,
        (f"🔄 内部流转：{node_name}", "子任务正在计算流转中..."),
    )

    message = _get_latest_message(updates)
    if message is None:
        return title, fallback_desc

    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        summary = _summarize_tool_calls(tool_calls)
        if summary:
            return title, summary

    content = _extract_text_content(getattr(message, "content", None))
    if node_name == "supervisor_tools":
        content = _compact_supervisor_status(content)

    if node_name == "poi_explorer_tools":
        stats = []
        candidate_pool = updates.get("_candidate_pool") or []
        explored_pois = updates.get("explored_pois") or []
        if candidate_pool:
            stats.append(f"候选池 {len(candidate_pool)} 个")
        if explored_pois:
            stats.append(f"已精选 {len(explored_pois)} 个")
        if stats:
            stat_line = "，".join(stats)
            content = f"{content}\n{stat_line}".strip() if content else stat_line

    if node_name == "route_planner_tools":
        route_records = updates.get("route_records") or []
        submitted = sum(1 for record in route_records if isinstance(record, dict) and record.get("action") == "SubmitRoutePlan")
        if submitted:
            submitted_line = f"已提交 {submitted} 版路线方案"
            content = f"{content}\n{submitted_line}".strip() if content else submitted_line

    if content:
        return title, _truncate_text(content, 260)
    return title, fallback_desc


def get_breathing_html(title, desc):
    safe_title = html.escape(title)
    safe_desc = html.escape(desc)
    return textwrap.dedent(
        f"""
        <style>
        .breathe-container {{
            display: flex;
            align-items: center;
            background-color: #f4f9ff;
            padding: 16px 20px;
            border-radius: 12px;
            border-left: 6px solid #2196F3;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }}
        .breathe-dot {{
            width: 18px;
            height: 18px;
            background-color: #2196F3;
            border-radius: 50%;
            margin-right: 18px;
            flex-shrink: 0;
            animation: breathing 1.8s infinite ease-in-out;
        }}
        @keyframes breathing {{
            0% {{ box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.4); transform: scale(0.9); opacity: 0.8; }}
            50% {{ box-shadow: 0 0 0 12px rgba(33, 150, 243, 0%); transform: scale(1.1); opacity: 1; }}
            100% {{ box-shadow: 0 0 0 0 rgba(33, 150, 243, 0); transform: scale(0.9); opacity: 0.8; }}
        }}
        .step-text-container {{
            display: flex;
            flex-direction: column;
        }}
        .step-title {{
            font-size: 16px;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 4px;
            font-family: sans-serif;
        }}
        .step-desc {{
            font-size: 14px;
            color: #555;
            font-family: sans-serif;
            white-space: pre-line;
            line-height: 1.5;
        }}
        </style>
        <div class="breathe-container">
            <div class="breathe-dot"></div>
            <div class="step-text-container">
                <div class="step-title">{safe_title}</div>
                <div class="step-desc">{safe_desc}</div>
            </div>
        </div>
        """
    ).strip()


def _estimate_status_card_height(desc: str) -> int:
    lines = [line for line in (desc or "").splitlines() if line.strip()]
    if not lines:
        return 110

    visual_lines = 0
    for line in lines:
        visual_lines += max(1, (len(line) + 27) // 28)

    return max(110, min(220, 72 + visual_lines * 24))


def render_status_card(placeholder, title: str, desc: str):
    with placeholder.container():
        components.html(
            get_breathing_html(title, desc),
            height=_estimate_status_card_height(desc),
            scrolling=False,
        )

async def run_planner(query: str, history: list, status_placeholder, log_expander):
    graph = build_graph()
    config = {
        "configurable": {
            "max_replan_count": 3,
        }
    }
    
    merged_state = _build_initial_state(query, history)
    
    # Render first state
    render_status_card(status_placeholder, "🚀 启动规划", "正在初始化 CityWalk Agent...")
    
    # Use subgraphs=True to yield events from sub-agents (poi_explorer, route_planner)
    async for chunk in graph.astream(merged_state, config, stream_mode="updates", subgraphs=True):
        
        # When subgraphs=True, chunk is a tuple: (namespace, state_update)
        if isinstance(chunk, tuple):
            namespace, state_update = chunk
        else:
            state_update = chunk
        
        # We only care about iterating dict updates
        if isinstance(state_update, dict):
            for node_name, updates in state_update.items():
                if isinstance(updates, dict):
                    # Only apply updates to main state if they are relevant to main graph
                    # To avoid sub-graph keys polluting main state or throwing errors,
                    # we do a loose merge. For display we mostly just care about node execution.
                    for k, v in updates.items():
                        if k in merged_state:
                            if isinstance(v, list) and isinstance(merged_state.get(k), list):
                                merged_state[k] = merged_state[k] + v
                            else:
                                merged_state[k] = v
                    
                    title, desc = _build_dynamic_status(node_name, updates)
                    render_status_card(status_placeholder, title, desc)
                    log_expander.write(f"✅ 完成执行: **{node_name}**")
            
    status_placeholder.empty() # Clear breathing text when done
    return merged_state


def main():
    st.title("🚶 CityWalk Agent")    
    # 动态加载 Configuration 读取默认模型名称展示给用户
    default_config = Configuration()
    model_name = default_config.model_config_for("supervisor")["model"]
    # 如果带有 'openrouter:' 前缀等，稍微清洗一下作为展示
    display_model_name = model_name.split(":")[-1] if ":" in model_name else model_name
    
    st.markdown(f"**Powered by `{display_model_name}`**")
    st.markdown("输入你的 CityWalk 需求，AI 将为你规划沿途路线并生成高德地图展示。")

    if "history" not in st.session_state:
        st.session_state.history = []

    # UI for query
    query = st.text_area("告诉 AI 你的想法 (例如：我带了一只修狗，现在在汉口江滩的芦苇荡附近。给我规划一个2小时的狗狗友好型Citywalk环线，需要经过有草坪的公园、可以买到咖啡的街角，并且路线最后必须回到我现在的出发点。注意狗狗不能进室内商场。)", height=100)

    if st.button("开始规划", type="primary"):
        if not query.strip():
            st.warning("请输入你的想法！")
            return
        
        status_placeholder = st.empty()
        with st.expander("🛠️ 执行日志", expanded=False) as log_expander:
            pass
            
        result = asyncio.run(run_planner(query, st.session_state.history, status_placeholder, log_expander))

        final_output = result.get("final_output", {})

        if final_output.get("need_clarification"):
            st.info("🤔 AI 需要更多信息:")
            st.write(final_output.get("question"))
            st.session_state.history.append({"role": "user", "content": query})
            st.session_state.history.append({"role": "assistant", "content": final_output.get("question")})
        else:
            st.success("✅ 规划完成！")
            
            # Show Reasoning First
            st.subheader("💡 规划思路与说明")
            route_summary = final_output.get("route_summary")
            if route_summary:
                st.write(route_summary)
            else:
                st.write("AI 没有提供详细的文字总结。")
            
            st.subheader("🗺️ 路线地图")
            viz_data = format_for_visualization(result)
            
            if viz_data.get("final_output", {}).get("route_polyline"):
                amap_js_key, amap_security_code = load_amap_js_config()
                if not amap_js_key or not amap_security_code:
                    st.warning("⚠️ 缺少高德地图 API Key (AMAP_JS_KEY / AMAP_SECURITY_CODE)。将无法正常显示地图，请检查环境配置。")
                
                html_code = generate_amap_html(viz_data, amap_js_key, amap_security_code)
                
                # Render using components.html
                st.components.v1.html(html_code, width=1000, height=600, scrolling=True)
            else:
                st.warning("结果中没有生成有效的路线坐标，无法渲染地图。")

            with st.expander("查看原始返回结果 (JSON)"):
                st.json(viz_data)

            # Keep history updated (Optional, maybe reset if successful)
            # st.session_state.history = [] 

if __name__ == "__main__":
    main()
