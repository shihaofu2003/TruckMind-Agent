"""货源缓存与有效性校验模块。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .cargo_math import haversine_km, wall_time_to_simulation_minutes


logger = logging.getLogger("agent.cargo_cache")


@dataclass
class CachedCargo:
    """缓存中的一条货源。"""

    cargo: dict[str, Any]
    query_time_minutes: int  # 查询时的仿真时间
    query_point_lat: float
    query_point_lng: float
    distance_km: float  # 查询点到货源起点的距离


class CargoCache:
    """按 driver_id 隔离的货源缓存。

    - 每轮 query_cargo 返回的货源都缓存下来
    - 候选集来自本轮 query + 缓存中仍然有效的货源
    """

    def __init__(self) -> None:
        self._cache: dict[str, CachedCargo] = {}  # cargo_id -> CachedCargo
        self._taken_cargo_ids: set[str] = set()

    def mark_taken(self, cargo_id: str) -> None:
        """标记已接单的货源。"""
        self._taken_cargo_ids.add(cargo_id)
        self._cache.pop(cargo_id, None)

    def update(self, items: list[dict[str, Any]], current_time_minutes: int, query_lat: float, query_lng: float) -> int:
        """将 query_cargo 返回的货源更新到缓存。返回新增条数。"""
        added = 0
        for item in items:
            cargo = item.get("cargo", {})
            cargo_id = cargo.get("cargo_id", "")
            if not cargo_id:
                continue
            if cargo_id in self._taken_cargo_ids:
                continue
            dist = float(item.get("distance_km", 0))
            self._cache[cargo_id] = CachedCargo(
                cargo=cargo,
                query_time_minutes=current_time_minutes,
                query_point_lat=query_lat,
                query_point_lng=query_lng,
                distance_km=dist,
            )
            added += 1
        return added

    def get_valid_candidates(self, current_time_minutes: int) -> list[CachedCargo]:
        """获取当前仍然有效的缓存货源。"""
        valid: list[CachedCargo] = []
        for cargo_id, cached in list(self._cache.items()):
            if cargo_id in self._taken_cargo_ids:
                continue
            if self._is_cache_valid(cached, current_time_minutes):
                valid.append(cached)
            else:
                del self._cache[cargo_id]
        return valid

    def get_by_id(self, cargo_id: str) -> CachedCargo | None:
        """按 ID 获取缓存货源。"""
        return self._cache.get(cargo_id)

    def _is_cache_valid(self, cached: CachedCargo, current_time_minutes: int) -> bool:
        """缓存货源是否仍然有效。

        检查：
        1. remove_time 未过
        2. load_time 窗口未过（如果有）
        3. 缓存年龄不超过 30 分钟（货源可能已被其他司机接走）
        """
        cargo = cached.cargo

        # 检查 remove_time
        remove_time_str = cargo.get("remove_time")
        if remove_time_str:
            try:
                remove_minutes = wall_time_to_simulation_minutes(str(remove_time_str))
                if remove_minutes < current_time_minutes:
                    return False
            except (ValueError, TypeError):
                pass

        # 检查 load_time 窗口
        load_time = cargo.get("load_time")
        if load_time and isinstance(load_time, list) and len(load_time) == 2:
            try:
                end_str = str(load_time[1]).strip()
                if end_str:
                    load_end_minutes = wall_time_to_simulation_minutes(end_str)
                    # 如果装货窗结束时间已过，则无效
                    if load_end_minutes < current_time_minutes:
                        return False
            except (ValueError, TypeError):
                pass

        # 检查缓存年龄：超过 30 分钟的缓存视为不可靠
        cache_age = current_time_minutes - cached.query_time_minutes
        if cache_age > 30:
            return False

        return True

    @property
    def size(self) -> int:
        return len(self._cache)

    def get_nearby(self, lat: float, lng: float, radius_km: float = 50.0) -> list[dict[str, Any]]:
        """获取指定位置附近的缓存货源（用于目的地价值评估）。

        返回格式与 query_cargo items 一致：{"cargo": ..., "distance_km": ...}
        """
        nearby: list[dict[str, Any]] = []
        for cached in self._cache.values():
            cargo = cached.cargo
            start = cargo.get("start", {})
            if not isinstance(start, dict):
                continue
            start_lat = float(start.get("lat", 0))
            start_lng = float(start.get("lng", 0))
            dist = haversine_km(lat, lng, start_lat, start_lng)
            if dist <= radius_km:
                nearby.append({"cargo": cargo, "distance_km": round(dist, 2)})
        return nearby

    def clear(self) -> None:
        self._cache.clear()
        self._taken_cargo_ids.clear()
