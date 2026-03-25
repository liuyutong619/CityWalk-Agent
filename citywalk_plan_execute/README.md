# CityWalk Plan and Execute Agent

基于 Plan and Execute 模式的精简城市漫步路线规划 Agent。

## 架构设计

```
用户输入 → Clarify Intent (意图澄清)
              ↓ 不清晰 → 返回澄清问题
              ↓ 清晰
         Parse Intent → Planner → Executor → Evaluator
                                      ↑          ↓
                                      └─ Replanner (如果不满足)
                                                 ↓
                                          Final Output
```

## 文件结构

```
citywalk_plan_execute/
├── __init__.py           # 包入口
├── state.py              # 状态定义（精简）
├── configuration.py      # 配置管理
├── prompts.py            # Prompt 模板
├── utils.py              # 工具函数
├── citywalk_agent.py     # 主图实现
└── example.py            # 使用示例
```

## 核心特性

1. **意图澄清**：自动检测用户输入是否清晰，缺少信息时主动询问
2. **真实地图 API**：集成高德地图工具（地理编码、路线规划、POI 搜索等）
3. **动态规划**：根据用户意图生成不同执行计划
4. **反馈循环**：评估结果 → 不满足则重新规划
5. **精简状态**：只保留必要字段，避免冗余

## 使用方法

```python
from citywalk_plan_execute import build_graph

graph = build_graph()
result = await graph.ainvoke({
    "user_query": "我想从武大骑车去东湖，走绿道",
    "execution_results": [],
    "replan_count": 0
})
```

模型配置默认放在 `citywalk_plan_execute/configuration.py` 的 `ROLE_MODEL_DEFAULTS` 中，并可按角色分别覆盖，例如 `supervisor_model`、`poi_explorer_model`、`route_planner_model`。运行时 `configurable` 仍可临时传入这些字段，或继续使用旧的 `model/max_tokens` 作为全局兜底覆盖。

## 对比旧版本

| 维度 | 旧版 (Rewoo) | 新版 (Plan & Execute) |
|------|-------------|---------------------|
| 代码行数 | 3265 行 | ~300 行 |
| 状态字段 | 20+ 字段 | 7 个核心字段 |
| 流程 | 固定流程 | 动态规划 |
| 规则化代码 | 大量硬编码 | LLM 驱动 |

## 可用工具

- `get_coordinates`: 地理编码（地址 → 经纬度）
- `search_nearby_places`: 周边 POI 搜索
- `calculate_walking_route`: 步行路线规划
- `get_walking_route_text`: 适合 LLM 理解的步行路线摘要
- `search_along_route_text`: 适合 LLM 理解的沿线亮点摘要
- `get_detailed_walking_route`: 详细步行路线（含 polyline）
- `search_along_route`: 沿路线搜索 POI
- `search_candidate_corridors`: 搜索候选漫步廊道

## 示例

**清晰的意图**：
```
输入: "我现在在武汉市的武汉大学，想要去东湖骑车，给我推荐一条骑车不错的路线，最好走东湖绿道，不要绕路"
输出: 生成完整路线规划
```

**不清晰的意图**：
```
输入: "我想骑车，给我推荐一条路线"
输出: "请问您在哪个城市？起点和终点分别是哪里？"
```

## TODO

- [ ] 完善评估逻辑
- [ ] 添加日志和可视化
- [ ] 支持骑行路线（当前仅步行）
