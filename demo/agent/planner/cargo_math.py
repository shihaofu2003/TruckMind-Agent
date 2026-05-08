"""精确复刻 take_order 全流程的本地仿真计算模块。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


# ── 仿真纪元 ──────────────────────────────────────────────────
_SIMULATION_EPOCH_STR = "2026-03-01 00:00:00"


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """球面大圆距离（公里）。"""
    radius_km = 6371.0
    p1 = math.radians(lat1)
    l1 = math.radians(lng1)
    p2 = math.radians(lat2)
    l2 = math.radians(lng2)
    dp = p2 - p1
    dl = l2 - l1
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    h = min(1.0, max(0.0, h))
    return 2.0 * radius_km * math.asin(math.sqrt(h))


def distance_to_minutes(distance_km: float, speed_km_per_hour: float) -> int:
    """距离 → 分钟（ceil 向上取整，最少 1 分钟）。"""
    if distance_km <= 0:
        return 1
    return max(1, math.ceil((distance_km / speed_km_per_hour) * 60))


def wall_time_to_simulation_minutes(text: str) -> int:
    """墙钟时间 → 仿真分钟偏移。"""
    from datetime import datetime
    epoch = datetime(2026, 3, 1, 0, 0, 0)
    dt = datetime.strptime(text.strip(), "%Y-%m-%d %H:%M:%S")
    return int((dt - epoch).total_seconds() // 60)


def parse_load_window_minutes(cargo: dict[str, Any]) -> tuple[int, int] | None:
    """解析 load_time 为仿真分钟 [start, end]；无字段返回 None。"""
    raw = cargo.get("load_time")
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    a = str(raw[0]).strip()
    b = str(raw[1]).strip()
    if not a or not b:
        return None
    start_m = wall_time_to_simulation_minutes(a)
    end_m = wall_time_to_simulation_minutes(b)
    if end_m < start_m:
        return None
    return (start_m, end_m)


@dataclass
class OrderSimulation:
    """单票订单本地精确仿真结果。"""

    cargo_id: str
    # 距离
    pickup_km: float
    haul_km: float
    total_km: float
    # 位置
    end_lat: float = 0.0
    end_lng: float = 0.0
    # 时间
    pickup_minutes: int = 0
    wait_at_pickup_minutes: int = 0
    haul_minutes: int = 0
    total_minutes: int = 0  # 总耗时（含空驶+等装+干线）
    finish_time: int = 0  # 完单仿真分钟
    # 收益
    price_yuan: float = 0.0
    cost_yuan: float = 0.0
    net_profit: float = 0.0
    profit_per_hour: float = 0.0
    # 可行性
    feasible: bool = False
    infeasible_reason: str | None = None
    income_eligible: bool = True


def simulate_take_order(
    cargo: dict[str, Any],
    current_time_minutes: int,
    current_lat: float,
    current_lng: float,
    cost_per_km: float,
    reposition_speed_km_per_hour: float,
    simulation_horizon_minutes: int | None = None,
) -> OrderSimulation:
    """精确模拟接单全过程，返回 OrderSimulation 结果。

    逻辑与 simkit/simulation_actions.take_order 完全对齐。
    """
    cargo_id = cargo.get("cargo_id", "")
    start = cargo.get("start", {})
    end = cargo.get("end", {})
    start_lat = float(start.get("lat", 0))
    start_lng = float(start.get("lng", 0))
    end_lat = float(end.get("lat", 0))
    end_lng = float(end.get("lng", 0))

    # 原始价格（元）
    price_yuan = float(cargo.get("price", 0))

    # 干线运输时间
    try:
        haul_minutes = int(cargo.get("cost_time_minutes", 0))
    except (ValueError, TypeError):
        return OrderSimulation(
            cargo_id=cargo_id, pickup_km=0, haul_km=0, total_km=0,
            end_lat=end_lat, end_lng=end_lng,
            price_yuan=price_yuan, cost_yuan=0, net_profit=0,
            profit_per_hour=0, feasible=False,
            infeasible_reason="cost_time_minutes_invalid",
        )

    # 空驶到装货点
    pickup_km = haversine_km(current_lat, current_lng, start_lat, start_lng)
    pickup_minutes = distance_to_minutes(pickup_km, reposition_speed_km_per_hour) if pickup_km > 1e-6 else 0

    arrival_time = current_time_minutes + pickup_minutes

    # ── 关键修复：检查 remove_time ──
    # 如果货源在司机到达前就被系统移除，接单必然失败
    # 加 5 分钟安全余量，防止仿真时间推进的微小差异导致失效
    remove_time_str = cargo.get("remove_time")
    if remove_time_str:
        try:
            remove_minutes = wall_time_to_simulation_minutes(str(remove_time_str))
            if remove_minutes < arrival_time + 5:
                return OrderSimulation(
                    cargo_id=cargo_id, pickup_km=pickup_km, haul_km=0, total_km=pickup_km,
                    end_lat=end_lat, end_lng=end_lng,
                    pickup_minutes=pickup_minutes, wait_at_pickup_minutes=0, haul_minutes=0,
                    total_minutes=pickup_minutes, finish_time=arrival_time,
                    price_yuan=price_yuan, cost_yuan=0, net_profit=0,
                    profit_per_hour=0, feasible=False,
                    infeasible_reason="cargo_expired_before_arrival",
                )
        except (ValueError, TypeError):
            pass

    # 装货时间窗
    window = parse_load_window_minutes(cargo)
    wait_at_pickup = 0
    if window is not None:
        load_start_m, load_end_m = window
        if arrival_time > load_end_m:
            return OrderSimulation(
                cargo_id=cargo_id, pickup_km=pickup_km, haul_km=0, total_km=pickup_km,
                end_lat=end_lat, end_lng=end_lng,
                pickup_minutes=pickup_minutes, wait_at_pickup_minutes=0, haul_minutes=0,
                total_minutes=pickup_minutes, finish_time=arrival_time,
                price_yuan=price_yuan, cost_yuan=0, net_profit=0,
                profit_per_hour=0, feasible=False,
                infeasible_reason="load_time_window_expired",
            )
        if arrival_time < load_start_m:
            wait_at_pickup = load_start_m - arrival_time
        ready_time = max(arrival_time, load_start_m)
    else:
        ready_time = arrival_time

    finish_time = ready_time + haul_minutes

    # 干线距离
    haul_km = haversine_km(start_lat, start_lng, end_lat, end_lng)
    total_km = pickup_km + haul_km

    # 总耗时
    total_duration = pickup_minutes + wait_at_pickup + haul_minutes

    # 成本
    cost_yuan = cost_per_km * total_km
    net_profit = price_yuan - cost_yuan

    # 时薪
    duration_hours = total_duration / 60.0
    profit_per_hour = net_profit / duration_hours if duration_hours > 0 else 0

    # 是否在仿真范围内
    income_eligible = True
    if simulation_horizon_minutes is not None and finish_time > simulation_horizon_minutes:
        income_eligible = False

    return OrderSimulation(
        cargo_id=cargo_id,
        pickup_km=round(pickup_km, 2),
        haul_km=round(haul_km, 2),
        total_km=round(total_km, 2),
        end_lat=end_lat,
        end_lng=end_lng,
        pickup_minutes=pickup_minutes,
        wait_at_pickup_minutes=wait_at_pickup,
        haul_minutes=haul_minutes,
        total_minutes=total_duration,
        finish_time=finish_time,
        price_yuan=round(price_yuan, 2),
        cost_yuan=round(cost_yuan, 2),
        net_profit=round(net_profit, 2),
        profit_per_hour=round(profit_per_hour, 2),
        feasible=True,
        income_eligible=income_eligible,
    )
