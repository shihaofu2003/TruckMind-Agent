"""模型决策服务：依赖 `simkit.ports.SimulationApiPort`，由评测进程注入具体环境。

使用 AOP (Active-sensing Adaptive Opportunity Planner) 替代 LLM 直接决策。
LLM 仅用于初始化时解析偏好文本 → 结构化约束，核心决策逻辑完全确定性。
"""

from __future__ import annotations

import logging
from typing import Any

from simkit.ports import SimulationApiPort

from .planner import Planner


class ModelDecisionService:
    """基于 AOP 规划器的单步决策。

    - LLM 仅用于初始化时解析偏好文本 → 结构化约束
    - 核心决策逻辑完全确定性：精确仿真 + 机会成本评分 + 偏好硬规则过滤
    - 主动查询扩大感受野
    - 短 wait + 重新 query 优于长 wait
    """

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._planner = Planner(api)
        self._logger = logging.getLogger("agent.decision_service")

    def decide(self, driver_id: str) -> dict[str, Any]:
        action = self._planner.decide(driver_id)
        self._logger.info(
            "decision output driver_id=%s action=%s params=%s",
            driver_id,
            action.get("action"),
            action.get("params"),
        )
        return action
