"""状态追踪模块：DriverMemory、区域热度、偏好状态追踪。"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .cargo_math import haversine_km
from .preference_parser import PreferenceConstraint


logger = logging.getLogger("agent.state_tracker")


@dataclass
class DayRecord:
    """单天活动记录。"""

    day_index: int  # 0-based 天索引
    had_take_order: bool = False
    had_reposition: bool = False
    rest_minutes: int = 0  # 当天累计 wait 分钟
    max_continuous_rest_minutes: int = 0  # 当天最长连续 wait
    current_continuous_rest_start: int | None = None  # 当前连续 wait 开始时间
    ended_near_target: bool = False  # 当天结束时是否在目标点附近
    ended_near_home: bool = False  # 当天结束时是否在家附近
    position_at_day_end: tuple[float, float] | None = None  # 当天结束位置


@dataclass
class DriverMemory:
    """司机记忆：跨轮次持久化状态。"""

    driver_id: str
    cost_per_km: float = 1.5

    # ── 收益追踪 ──
    observed_profits_per_hour: list[float] = field(default_factory=list)
    completed_order_count: int = 0
    total_net_profit: float = 0.0

    # ── 已接货源 ──
    taken_cargo_ids: set[str] = field(default_factory=set)

    # ── 区域热度 ──
    # grid_key -> {count, total_profit_per_hour, top_values, last_query_time}
    heatmap: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)

    # ── 偏好状态追踪 ──
    day_records: dict[int, DayRecord] = field(default_factory=dict)
    current_day_index: int = 0

    # ── 连续无好货轮次 ──
    consecutive_no_good_order_rounds: int = 0

    # ── 偏好约束 ──
    constraints: PreferenceConstraint | None = None

    # ── 当前轮次状态快照 ──
    last_status: dict[str, Any] | None = None

    def record_order_completed(
        self,
        net_profit: float,
        duration_hours: float,
        cargo_id: str,
    ) -> None:
        """记录完成一单。"""
        self.taken_cargo_ids.add(cargo_id)
        self.completed_order_count += 1
        self.total_net_profit += net_profit
        if duration_hours > 0:
            self.observed_profits_per_hour.append(net_profit / duration_hours)

        # 更新当天记录
        day = self._get_or_create_day(self.current_day_index)
        day.had_take_order = True
        # 接单打断连续休息
        day.current_continuous_rest_start = None

    def record_reposition(self) -> None:
        """记录空驶。"""
        day = self._get_or_create_day(self.current_day_index)
        day.had_reposition = True
        day.current_continuous_rest_start = None

    def record_wait(self, duration_minutes: int, current_time_minutes: int) -> None:
        """记录等待/休息。"""
        day = self._get_or_create_day(self.current_day_index)
        day.rest_minutes += duration_minutes

        # 追踪连续休息
        if day.current_continuous_rest_start is None:
            day.current_continuous_rest_start = current_time_minutes
        else:
            # 继续连续休息
            pass

        continuous = duration_minutes  # 简化：本次 wait 时长
        if day.current_continuous_rest_start is not None:
            # 从开始到现在的总连续休息
            total_continuous = current_time_minutes - day.current_continuous_rest_start + duration_minutes
            continuous = total_continuous
        day.max_continuous_rest_minutes = max(day.max_continuous_rest_minutes, continuous)

    def break_continuous_rest(self) -> None:
        """打断连续休息（接单或空驶时调用）。"""
        day = self._get_or_create_day(self.current_day_index)
        day.current_continuous_rest_start = None

    def update_day_index(self, current_time_minutes: int) -> None:
        """根据当前仿真时间更新天索引。"""
        new_day = current_time_minutes // 1440
        if new_day != self.current_day_index:
            # 结算前一天
            self._finalize_day(self.current_day_index)
            self.current_day_index = new_day

    def _finalize_day(self, day_index: int) -> None:
        """结算一天结束时的状态。"""
        day = self.day_records.get(day_index)
        if day is None:
            return
        # 可以在这里做日终统计
        logger.debug("Driver %s Day %d: take_order=%s, reposition=%s, rest=%d min, max_continuous=%d min",
                     self.driver_id, day_index, day.had_take_order, day.had_reposition,
                     day.rest_minutes, day.max_continuous_rest_minutes)

    def _get_or_create_day(self, day_index: int) -> DayRecord:
        """获取或创建天记录。"""
        if day_index not in self.day_records:
            self.day_records[day_index] = DayRecord(day_index=day_index)
        return self.day_records[day_index]

    # ── 偏好查询 ──

    def get_off_days_count(self) -> int:
        """获取本月完全不出车的天数。"""
        count = 0
        for day_idx, day in self.day_records.items():
            if not day.had_take_order and not day.had_reposition:
                count += 1
        return count

    def get_visit_days_count(self, target_lat: float, target_lng: float, radius_km: float = 1.0) -> int:
        """获取本月到访目标点的天数。"""
        count = 0
        for day_idx, day in self.day_records.items():
            if day.ended_near_target and day.position_at_day_end is not None:
                lat, lng = day.position_at_day_end
                if haversine_km(lat, lng, target_lat, target_lng) <= radius_km:
                    count += 1
        return count

    def get_today_rest_minutes(self) -> int:
        """获取当天累计休息分钟。"""
        day = self.day_records.get(self.current_day_index)
        return day.rest_minutes if day else 0

    def get_today_max_continuous_rest(self) -> int:
        """获取当天最长连续休息分钟。"""
        day = self.day_records.get(self.current_day_index)
        return day.max_continuous_rest_minutes if day else 0

    def should_force_off_day(self, min_off_days: int, remaining_days: int) -> bool:
        """判断今天是否必须强制休息（不出车）。"""
        off_days = self.get_off_days_count()
        if off_days >= min_off_days:
            return False
        # 剩余天数（含今天）刚好等于还需休息的天数
        needed = min_off_days - off_days
        if remaining_days <= needed:
            return True
        return False

    # ── 区域热度 ──

    def update_heatmap(
        self,
        items: list[dict[str, Any]],
        query_lat: float,
        query_lng: float,
        current_time_minutes: int,
    ) -> None:
        """更新区域热度图。"""
        for item in items:
            cargo = item.get("cargo", item)
            price = float(cargo.get("price", 0))  # query_cargo 已将分转为元
            cost_time = int(cargo.get("cost_time_minutes", 1))
            profit_per_hour = price / max(cost_time / 60.0, 0.1)

            # 用货源起点坐标做 grid
            start_info = cargo.get("start", {})
            start_lat = float(start_info.get("lat", query_lat)) if isinstance(start_info, dict) else query_lat
            start_lng = float(start_info.get("lng", query_lng)) if isinstance(start_info, dict) else query_lng
            grid_key = (round(start_lat, 1), round(start_lng, 1))

            if grid_key not in self.heatmap:
                self.heatmap[grid_key] = {
                    "count": 0,
                    "total_profit_per_hour": 0.0,
                    "top_values": [],
                    "last_query_time": current_time_minutes,
                    "center_lat": start_lat,
                    "center_lng": start_lng,
                }

            cell = self.heatmap[grid_key]
            cell["count"] += 1
            cell["total_profit_per_hour"] += profit_per_hour
            cell["last_query_time"] = current_time_minutes

            # 维护 top-5
            top = cell["top_values"]
            top.append(profit_per_hour)
            top.sort(reverse=True)
            if len(top) > 5:
                top.pop()

    def get_hotspot_value(self, lat: float, lng: float, radius_grids: float = 0.2) -> float:
        """获取某位置附近的区域热度值。"""
        values: list[float] = []
        for grid_key, cell in self.heatmap.items():
            dist = haversine_km(lat, lng, cell.get("center_lat", 0), cell.get("center_lng", 0))
            if dist <= radius_grids * 111:  # 粗略：0.1度 ≈ 11km
                if cell["top_values"]:
                    values.append(cell["top_values"][0])
        if not values:
            return 0.0
        return sum(values) / len(values)

    def get_best_hotspot(self, current_lat: float, current_lng: float, min_distance_km: float = 10.0) -> tuple[float, float, float] | None:
        """获取最佳热区（距离当前位置至少 min_distance_km）。"""
        best = None
        best_value = 0.0
        for grid_key, cell in self.heatmap.items():
            center_lat = cell.get("center_lat", 0)
            center_lng = cell.get("center_lng", 0)
            dist = haversine_km(current_lat, current_lng, center_lat, center_lng)
            if dist < min_distance_km:
                continue
            if cell["top_values"]:
                value = cell["top_values"][0]
                if value > best_value:
                    best_value = value
                    best = (center_lat, center_lng, value)
        return best


class StateTracker:
    """全局状态追踪器，管理所有司机的记忆。"""

    def __init__(self) -> None:
        self._drivers: dict[str, DriverMemory] = {}

    def get_memory(self, driver_id: str) -> DriverMemory:
        """获取司机记忆，不存在则创建。"""
        if driver_id not in self._drivers:
            self._drivers[driver_id] = DriverMemory(driver_id=driver_id)
        return self._drivers[driver_id]

    def set_constraints(self, driver_id: str, constraints: PreferenceConstraint) -> None:
        """设置司机偏好约束。"""
        mem = self.get_memory(driver_id)
        mem.constraints = constraints

    def update_after_action(
        self,
        driver_id: str,
        action: dict[str, Any],
        current_time_minutes: int,
        net_profit: float = 0.0,
        duration_hours: float = 0.0,
        cargo_id: str = "",
    ) -> None:
        """动作执行后更新状态。"""
        mem = self.get_memory(driver_id)
        action_name = str(action.get("action", "")).strip().lower()

        if action_name == "take_order":
            mem.record_order_completed(net_profit, duration_hours, cargo_id)
            mem.break_continuous_rest()
        elif action_name == "reposition":
            mem.record_reposition()
            mem.break_continuous_rest()
        elif action_name == "wait":
            duration = int(action.get("params", {}).get("duration_minutes", 0))
            mem.record_wait(duration, current_time_minutes)

        mem.update_day_index(current_time_minutes)
