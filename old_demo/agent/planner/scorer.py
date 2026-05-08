"""评分模块：utility = 净收益 - λ×时长 + β×目的地价值。"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from .cargo_math import OrderSimulation, haversine_km
from .cargo_cache import CachedCargo


logger = logging.getLogger("agent.scorer")


@dataclass
class ScoredCandidate:
    """评分后的候选订单。"""

    cargo_id: str
    cargo: dict[str, Any]
    simulation: OrderSimulation
    utility: float
    net_profit: float
    duration_hours: float
    destination_value: float
    opportunity_cost: float
    distance_km: float  # 到起点的距离


# ── 评分参数 ──────────────────────────────────────────────────
PRIOR_LAMBDA = 80.0  # 元/小时，先验机会成本（从150降低）
LAMBDA_MIN = 20.0
LAMBDA_MAX = 300.0
DESTINATION_VALUE_BETA = 0.3  # 目的地价值权重
MIN_PROFIT_PER_HOUR = 0.0  # 最低时薪阈值（从20改为0，只要净收益>0就考虑）


def compute_lambda(
    observed_profits_per_hour: list[float],
    current_time_minutes: int,
    horizon_minutes: int,
) -> float:
    """计算当前每小时机会成本 λ。

    使用贝叶斯先验 + 经验更新，并考虑时段和剩余时间调整。
    """
    # 基础 λ：先验-经验混合
    if len(observed_profits_per_hour) < 10:
        base_lambda = PRIOR_LAMBDA
    else:
        sorted_profits = sorted(observed_profits_per_hour)
        idx = int(len(sorted_profits) * 0.4)  # 用40分位而非60分位，更宽松
        empirical = sorted_profits[min(idx, len(sorted_profits) - 1)]
        alpha = min(len(observed_profits_per_hour) / 80.0, 0.6)  # 更慢收敛，最大权重0.6
        base_lambda = (1 - alpha) * PRIOR_LAMBDA + alpha * empirical

    # 时段调整
    current_hour = (current_time_minutes % 1440) / 60.0
    if 8 <= current_hour < 20:
        time_multiplier = 1.0
    else:
        time_multiplier = 0.5  # 夜间机会成本更低（从0.7改为0.5）

    # 剩余时间调整
    remaining_ratio = max(0, (horizon_minutes - current_time_minutes)) / max(horizon_minutes, 1)
    remaining_multiplier = 0.5 + 0.5 * remaining_ratio  # 从0.7+0.6改为0.5+0.5，更宽松

    lambda_hour = base_lambda * time_multiplier * remaining_multiplier
    return max(LAMBDA_MIN, min(LAMBDA_MAX, lambda_hour))


def score_candidate(
    simulation: OrderSimulation,
    lambda_hour: float,
    destination_value: float = 0.0,
    beta: float = DESTINATION_VALUE_BETA,
) -> float:
    """计算候选订单的 utility。"""
    if not simulation.feasible:
        return -float("inf")

    net_profit = simulation.net_profit
    duration_hours = simulation.total_minutes / 60.0
    opportunity_cost = lambda_hour * duration_hours
    dest_value = beta * destination_value

    utility = net_profit - opportunity_cost + dest_value
    return utility


def score_all_candidates(
    candidates: list[tuple[CachedCargo, OrderSimulation]],
    lambda_hour: float,
    destination_values: dict[str, float] | None = None,
    beta: float = DESTINATION_VALUE_BETA,
) -> list[ScoredCandidate]:
    """批量评分候选订单。"""
    if destination_values is None:
        destination_values = {}

    scored: list[ScoredCandidate] = []
    for cached, sim in candidates:
        if not sim.feasible:
            continue

        cargo_id = cached.cargo.get("cargo_id", "")
        dest_value = destination_values.get(cargo_id, 0.0)
        utility = score_candidate(sim, lambda_hour, dest_value, beta)

        scored.append(ScoredCandidate(
            cargo_id=cargo_id,
            cargo=cached.cargo,
            simulation=sim,
            utility=utility,
            net_profit=sim.net_profit,
            duration_hours=sim.total_minutes / 60.0,
            destination_value=dest_value,
            opportunity_cost=lambda_hour * sim.total_minutes / 60.0,
            distance_km=cached.distance_km,
        ))

    # 按 utility 降序
    scored.sort(key=lambda x: x.utility, reverse=True)
    return scored


def filter_candidates(
    scored: list[ScoredCandidate],
    current_time_minutes: int,
    horizon_minutes: int,
    remaining_buffer_minutes: int = 60,
    min_profit_per_hour: float = MIN_PROFIT_PER_HOUR,
) -> list[ScoredCandidate]:
    """过滤不可行或明显不合理的候选。"""
    filtered: list[ScoredCandidate] = []
    for s in scored:
        # 1. 必须可行
        if not s.simulation.feasible:
            continue

        # 2. 完单不能超过仿真 horizon（留余量）
        if s.simulation.finish_time > horizon_minutes - remaining_buffer_minutes:
            continue

        # 2b. 收入不计入评分的订单不接（income_eligible=False）
        if not s.simulation.income_eligible:
            continue

        # 3. 净收益必须为正（这是核心判断）
        if s.net_profit <= 0:
            continue

        # 4. 时薪检查（仅在阈值>0时生效）
        if min_profit_per_hour > 0 and s.duration_hours > 0:
            profit_per_hour = s.net_profit / s.duration_hours
            if profit_per_hour < min_profit_per_hour:
                continue

        filtered.append(s)

    return filtered


def estimate_destination_value(
    end_lat: float,
    end_lng: float,
    finish_time_minutes: int,
    nearby_cargos: list[dict[str, Any]],
    lambda_hour: float,
    cost_per_km: float = 1.5,
    speed_kmph: float = 60.0,
    top_k: int = 5,
) -> float:
    """估计订单终点的目的地价值。

    基于终点附近可接货源的 top-k utility 平均值。
    """
    if not nearby_cargos:
        return 0.0

    utilities: list[float] = []
    for item in nearby_cargos:
        cargo = item.get("cargo", item)
        dist = float(item.get("distance_km", 0))

        # 简化估计：假设从终点出发到该货源
        price = float(cargo.get("price", 0))  # 元
        cost_time = int(cargo.get("cost_time_minutes", 0))

        # 干线距离：从起点到终点
        start_info = cargo.get("start", {})
        end_info = cargo.get("end", {})
        if isinstance(start_info, dict) and isinstance(end_info, dict):
            haul_km = haversine_km(
                float(start_info.get("lat", 0)),
                float(start_info.get("lng", 0)),
                float(end_info.get("lat", 0)),
                float(end_info.get("lng", 0)),
            )
        else:
            haul_km = float(cargo.get("distance_km", 0))

        # 简化净收益
        pickup_km = dist
        net = price - cost_per_km * (pickup_km + haul_km)
        total_minutes = max(1, math.ceil(pickup_km / speed_kmph * 60)) + cost_time
        duration_hours = total_minutes / 60.0

        util = net - lambda_hour * duration_hours
        if util > 0:
            utilities.append(util)

    if not utilities:
        return 0.0

    # 取 top-k 平均
    utilities.sort(reverse=True)
    top = utilities[:top_k]
    return sum(top) / len(top)


def compute_reposition_value(
    target_lat: float,
    target_lng: float,
    current_lat: float,
    current_lng: float,
    current_time_minutes: int,
    lambda_hour: float,
    cost_per_km: float,
    speed_kmph: float,
    hotspot_value: float,
    expected_active_hours: float = 4.0,
) -> float:
    """计算空驶到目标点的价值。

    move_value = hotspot_value × expected_active_hours
                - cost_per_km × distance
                - λ × travel_hours
    """
    distance = haversine_km(current_lat, current_lng, target_lat, target_lng)
    travel_hours = distance / speed_kmph
    travel_minutes = math.ceil(travel_hours * 60)

    move_value = (
        hotspot_value * expected_active_hours
        - cost_per_km * distance
        - lambda_hour * travel_hours
    )

    return move_value
