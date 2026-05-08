"""AOP 主规划器：主动感知的自适应机会成本规划器。

核心流程：
1. 偏好驱动的 wait（最高优先级）
2. 获取状态 → 当前位置 query → 重新获取状态
3. 更新缓存和 heatmap
4. 评估候选订单（含偏好过滤）
5. 选择战略查询点（含 ROI 判断）
6. 执行额外查询
7. 重新评估
8. 末期过滤
9. 决策：take_order / wait / reposition
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .cargo_cache import CargoCache, CachedCargo
from .cargo_math import (
    OrderSimulation,
    haversine_km,
    simulate_take_order,
    wall_time_to_simulation_minutes,
)
from .preference_filter import (
    passes_preference_rules,
    preference_aware_wait,
)
from .preference_parser import PreferenceConstraint, parse_preferences_with_llm
from .scorer import (
    ScoredCandidate,
    compute_lambda,
    compute_reposition_value,
    estimate_destination_value,
    filter_candidates,
    score_all_candidates,
)
from .state_tracker import DriverMemory, StateTracker


logger = logging.getLogger("agent.planner")


# ── 仿真参数 ──────────────────────────────────────────────────
REPOSITION_SPEED_KMPH = 60.0
COST_PER_KM = 1.5
SIMULATION_HORIZON_MINUTES = 30 * 1440  # 1天（与 config.json simulation_duration_days=1 对齐）
MAX_EXTRA_QUERIES = 2  # 每轮最多额外查询次数（§7 主动查询策略）
QUERY_ROI_THRESHOLD = 2.0  # 查询 ROI 阈值
END_GAME_HOURS = 12  # 末期策略：最后 N 小时
NO_REPOSITION_HOURS = 6  # 最后 N 小时禁止空驶
WAIT_SHORT = 10  # 短 wait 分钟
WAIT_MEDIUM = 15  # 中等 wait 分钟
WAIT_LONG = 30  # 长 wait 分钟
CACHE_MAX_AGE_MINUTES = 360  # 缓存最大有效期 6 小时


class Planner:
    """AOP 主规划器。"""

    def __init__(self, api) -> None:
        """
        Args:
            api: SimulationApiPort 实例，提供 get_driver_status / query_cargo / model_chat_completion
        """
        self._api = api
        self._state_tracker = StateTracker()
        self._cargo_cache = CargoCache()
        self._initialized_drivers: set[str] = set()

    def decide(self, driver_id: str) -> dict[str, Any]:
        """主决策入口。"""
        try:
            return self._decide_inner(driver_id)
        except Exception as e:
            logger.error("Driver %s decide error: %s", driver_id, e, exc_info=True)
            return {"action": "wait", "params": {"duration_minutes": 30}}

    def _ensure_initialized(self, driver_id: str) -> None:
        """首次遇到某司机时，解析偏好并初始化状态。"""
        if driver_id in self._initialized_drivers:
            return

        # 获取司机状态
        status = self._api.get_driver_status(driver_id)
        preferences = status.get("preferences", [])

        # LLM 偏好解析（一次性调用）
        if preferences:
            constraints = parse_preferences_with_llm(
                preferences=preferences,
                driver_id=driver_id,
                model_chat_fn=self._api.model_chat_completion,
            )
        else:
            constraints = PreferenceConstraint(driver_id=driver_id, rules=[])

        self._state_tracker.set_constraints(driver_id, constraints)

        # 初始化司机记忆
        mem = self._state_tracker.get_memory(driver_id)
        mem.cost_per_km = COST_PER_KM

        self._initialized_drivers.add(driver_id)
        logger.info("Driver %s initialized with %d preference rules", driver_id, len(constraints.rules))

    def _decide_inner(self, driver_id: str) -> dict[str, Any]:
        """核心决策逻辑。"""
        self._ensure_initialized(driver_id)

        mem = self._state_tracker.get_memory(driver_id)
        constraints = mem.constraints or PreferenceConstraint(driver_id=driver_id, rules=[])

        # ── 1. 获取初始状态 ──
        status0 = self._api.get_driver_status(driver_id)
        current_time = int(status0.get("simulation_progress_minutes", 0))
        current_lat = float(status0.get("current_lat", 0))
        current_lng = float(status0.get("current_lng", 0))

        mem.last_status = status0
        mem.update_day_index(current_time)

        # ── 2. 偏好驱动的 wait（最高优先级）──
        pref_wait_minutes = preference_aware_wait(
            current_time_minutes=current_time,
            current_lat=current_lat,
            current_lng=current_lng,
            constraints=constraints,
            daily_rest_minutes=mem.get_today_rest_minutes(),
        )
        if pref_wait_minutes is not None and pref_wait_minutes > 0:
            remaining = SIMULATION_HORIZON_MINUTES - current_time
            wait_min = min(pref_wait_minutes, remaining)
            if wait_min > 0:
                self._state_tracker.update_after_action(
                    driver_id, {"action": "wait", "params": {"duration_minutes": wait_min}}, current_time
                )
                return {"action": "wait", "params": {"duration_minutes": wait_min}}

        # ── 3. 末期偏好强制休息 ──
        endgame_action = self._check_endgame_preferences(driver_id, mem, constraints, current_time)
        if endgame_action is not None:
            return endgame_action

        # ── 4. 当前位置必查 ──
        query_result0 = self._api.query_cargo(driver_id, current_lat, current_lng)
        items0 = query_result0.get("items", []) if isinstance(query_result0, dict) else []

        # query 后重新获取状态（query 推进了时间！）
        status = self._api.get_driver_status(driver_id)
        current_time = int(status.get("simulation_progress_minutes", 0))
        current_lat = float(status.get("current_lat", 0))
        current_lng = float(status.get("current_lng", 0))
        mem.last_status = status
        mem.update_day_index(current_time)

        # ── 5. 更新缓存和 heatmap ──
        self._cargo_cache.update(items0, current_time, current_lat, current_lng)
        mem.update_heatmap(items0, current_lat, current_lng, current_time)

        # ── 6. 初步评估候选订单 ──
        lambda_hour = compute_lambda(
            mem.observed_profits_per_hour, current_time, SIMULATION_HORIZON_MINUTES
        )
        initial_evals = self._evaluate_candidates(
            driver_id=driver_id,
            current_time=current_time,
            current_lat=current_lat,
            current_lng=current_lng,
            mem=mem,
            constraints=constraints,
            lambda_hour=lambda_hour,
        )

        # ── 7. 选择战略查询点 ──
        query_points = self._select_query_points(
            status=status,
            evals=initial_evals,
            mem=mem,
            lambda_hour=lambda_hour,
            current_time=current_time,
        )

        # ── 8. 执行额外查询 ──
        extra_query_ids: set[str] = set()
        for qp_lat, qp_lng in query_points:
            extra_result = self._api.query_cargo(driver_id, qp_lat, qp_lng)
            extra_items = extra_result.get("items", []) if isinstance(extra_result, dict) else []
            # query 后重新获取状态
            status = self._api.get_driver_status(driver_id)
            current_time = int(status.get("simulation_progress_minutes", 0))
            current_lat = float(status.get("current_lat", 0))
            current_lng = float(status.get("current_lng", 0))
            mem.last_status = status
            mem.update_day_index(current_time)

            self._cargo_cache.update(extra_items, current_time, qp_lat, qp_lng)
            mem.update_heatmap(extra_items, qp_lat, qp_lng, current_time)

            # 记录额外查询看到的货源 ID
            for item in extra_items:
                c = item.get("cargo", item)
                cid = c.get("cargo_id", "")
                if cid:
                    extra_query_ids.add(cid)

        # ── 9. 重新评估（时间已推进）──
        lambda_hour = compute_lambda(
            mem.observed_profits_per_hour, current_time, SIMULATION_HORIZON_MINUTES
        )
        final_evals = self._evaluate_candidates(
            driver_id=driver_id,
            current_time=current_time,
            current_lat=current_lat,
            current_lng=current_lng,
            mem=mem,
            constraints=constraints,
            lambda_hour=lambda_hour,
        )

        # ── 10. 末期过滤 ──
        remaining = SIMULATION_HORIZON_MINUTES - current_time
        if remaining < END_GAME_HOURS * 60:
            # 只接能在剩余时间内完成的订单
            final_evals = [
                e for e in final_evals
                if e.simulation.finish_time <= SIMULATION_HORIZON_MINUTES - 60
            ]

        # ── 11. 决策：选最优且有效的候选 ──
        # 用 net_profit > 0 而非 utility > 0（不接单收益为 0，只要净收益为正就值得接）
        # 优先接本轮 query 看到的货源（确定性最高），其次接缓存中仍有效的货源
        current_query_ids: set[str] = set()
        for item in items0:
            c = item.get("cargo", item)
            cid = c.get("cargo_id", "")
            if cid:
                current_query_ids.add(cid)
        # 合并额外查询看到的货源 ID
        current_query_ids |= extra_query_ids

        # 分两轮选择：先从本轮 query 结果中选，再从缓存中选
        # 这样优先接确定性最高的货源，减少"已失效"浪费
        for candidate in final_evals:
            if candidate.net_profit <= 0:
                continue
            cached = self._cargo_cache.get_by_id(candidate.cargo_id)
            if cached is None:
                continue
            if not self._is_cache_valid(cached, current_time):
                continue
            # 本轮 query 看到的货源优先（确定性最高）
            if candidate.cargo_id not in current_query_ids:
                continue
            action = {
                "action": "take_order",
                "params": {"cargo_id": candidate.cargo_id},
            }
            self._state_tracker.update_after_action(
                driver_id, action, current_time,
                net_profit=candidate.net_profit,
                duration_hours=candidate.duration_hours,
                cargo_id=candidate.cargo_id,
            )
            self._cargo_cache.mark_taken(candidate.cargo_id)
            mem.consecutive_no_good_order_rounds = 0
            return action

        # 本轮 query 没有好货，尝试接缓存中最近看到的货源（风险较高但好过空等）
        for candidate in final_evals:
            if candidate.net_profit <= 0:
                continue
            cached = self._cargo_cache.get_by_id(candidate.cargo_id)
            if cached is None:
                continue
            if not self._is_cache_valid(cached, current_time):
                continue
            # 缓存货源：只接最近 60 分钟内看到的（降低失效概率）
            cache_age = current_time - cached.query_time_minutes
            if cache_age > 60:
                continue
            action = {
                "action": "take_order",
                "params": {"cargo_id": candidate.cargo_id},
            }
            self._state_tracker.update_after_action(
                driver_id, action, current_time,
                net_profit=candidate.net_profit,
                duration_hours=candidate.duration_hours,
                cargo_id=candidate.cargo_id,
            )
            self._cargo_cache.mark_taken(candidate.cargo_id)
            mem.consecutive_no_good_order_rounds = 0
            return action

        # ── 12. 无好货，短 wait + 重新 query（reposition 已禁用，纯亏损）──
        mem.consecutive_no_good_order_rounds += 1

        # ── 13. 短 wait + 重新 query ──
        wait_min = self._smart_wait_duration(
            current_time=current_time,
            remaining=remaining,
            mem=mem,
            constraints=constraints,
            items_count=len(items0),
            best_utility=final_evals[0].utility if final_evals else -float("inf"),
        )
        action = {"action": "wait", "params": {"duration_minutes": wait_min}}
        self._state_tracker.update_after_action(driver_id, action, current_time)
        return action

    def _evaluate_candidates(
        self,
        driver_id: str,
        current_time: int,
        current_lat: float,
        current_lng: float,
        mem: DriverMemory,
        constraints: PreferenceConstraint,
        lambda_hour: float,
    ) -> list[ScoredCandidate]:
        """评估所有候选订单（§8.3 含目的地价值评估）。"""
        # 获取所有有效缓存货源
        valid_cargos = self._cargo_cache.get_valid_candidates(current_time)

        # 对每个货源做精确仿真
        candidates: list[tuple[CachedCargo, OrderSimulation]] = []
        for cached in valid_cargos:
            cargo = cached.cargo
            sim = simulate_take_order(
                current_time_minutes=current_time,
                current_lat=current_lat,
                current_lng=current_lng,
                cargo=cargo,
                cost_per_km=mem.cost_per_km,
                reposition_speed_km_per_hour=REPOSITION_SPEED_KMPH,
                simulation_horizon_minutes=SIMULATION_HORIZON_MINUTES,
            )

            if not sim.feasible:
                continue

            # 偏好硬规则过滤
            action = {"action": "take_order", "params": {"cargo_id": cargo.get("cargo_id", "")}}
            if not passes_preference_rules(
                action=action,
                current_time_minutes=current_time,
                current_lat=current_lat,
                current_lng=current_lng,
                constraints=constraints,
                order_finish_time=sim.finish_time,
                order_end_lat=sim.end_lat,
                order_end_lng=sim.end_lng,
                daily_rest_minutes=mem.get_today_rest_minutes(),
                off_days_count=mem.get_off_days_count(),
                visit_days_count=0,
                current_day=mem.current_day_index,
            ):
                continue

            candidates.append((cached, sim))

        # §8.3 目的地价值评估：用缓存中终点附近的货源估计
        destination_values: dict[str, float] = {}
        for cached, sim in candidates:
            cargo_id = cached.cargo.get("cargo_id", "")
            if sim.end_lat == 0 and sim.end_lng == 0:
                continue
            # 从缓存中找终点附近的货源
            nearby = self._cargo_cache.get_nearby(sim.end_lat, sim.end_lng, radius_km=50.0)
            if nearby:
                dest_val = estimate_destination_value(
                    end_lat=sim.end_lat,
                    end_lng=sim.end_lng,
                    finish_time_minutes=sim.finish_time,
                    nearby_cargos=nearby,
                    lambda_hour=lambda_hour,
                    cost_per_km=mem.cost_per_km,
                    speed_kmph=REPOSITION_SPEED_KMPH,
                )
                destination_values[cargo_id] = dest_val

        # 评分（含目的地价值）
        scored = score_all_candidates(candidates, lambda_hour, destination_values)
        # 过滤
        scored = filter_candidates(scored, current_time, SIMULATION_HORIZON_MINUTES)
        return scored

    def _select_query_points(
        self,
        status: dict[str, Any],
        evals: list[ScoredCandidate],
        mem: DriverMemory,
        lambda_hour: float,
        current_time: int,
    ) -> list[tuple[float, float]]:
        """选择战略查询点（§7 主动查询策略）。"""
        points: list[tuple[float, float]] = []
        remaining = SIMULATION_HORIZON_MINUTES - current_time

        # 末期不再额外查询
        if remaining < END_GAME_HOURS * 60:
            return points

        # 情况一：候选分数接近，查 top1 终点（§7.4 情况一）
        if len(evals) >= 2:
            score_gap = evals[0].utility - evals[1].utility
            query_cost_value = lambda_hour * 10 / 60  # 最坏情况 10 分钟
            if score_gap < query_cost_value * QUERY_ROI_THRESHOLD:
                top1 = evals[0]
                end_lat = top1.simulation.end_lat
                end_lng = top1.simulation.end_lng
                if end_lat != 0 or end_lng != 0:
                    points.append((end_lat, end_lng))

        # 情况二：当前无好货，查最佳热区（§7.4 情况三）
        if (not evals or evals[0].utility <= 0) and len(points) == 0:
            current_lat = float(status.get("current_lat", 0))
            current_lng = float(status.get("current_lng", 0))
            hotspot = mem.get_best_hotspot(current_lat, current_lng, min_distance_km=10.0)
            if hotspot is not None:
                hs_lat, hs_lng, hs_value = hotspot
                if hs_value > lambda_hour * 0.5:
                    points.append((hs_lat, hs_lng))

        # 情况三：top1 收益高但终点未查过（§7.4 情况二）
        elif len(evals) >= 1 and len(points) == 0:
            top1 = evals[0]
            if top1.utility > lambda_hour * 2:
                end_lat = top1.simulation.end_lat
                end_lng = top1.simulation.end_lng
                if end_lat != 0 or end_lng != 0:
                    points.append((end_lat, end_lng))


        return points[:MAX_EXTRA_QUERIES]

    def _try_reposition(
        self,
        driver_id: str,
        current_time: int,
        current_lat: float,
        current_lng: float,
        mem: DriverMemory,
        constraints: PreferenceConstraint,
        lambda_hour: float,
        remaining: int,
    ) -> dict[str, Any] | None:
        """尝试空驶到更好的区域。"""
        # 只有连续多轮无好货才考虑 reposition
        if mem.consecutive_no_good_order_rounds < 3:
            return None

        # 获取最佳热区
        hotspot = mem.get_best_hotspot(current_lat, current_lng, min_distance_km=10.0)
        if hotspot is None:
            return None

        hs_lat, hs_lng, hs_value = hotspot

        # 偏好过滤：检查 reposition 目标是否违反偏好
        reposition_action = {
            "action": "reposition",
            "params": {"latitude": hs_lat, "longitude": hs_lng},
        }
        if not passes_preference_rules(
            action=reposition_action,
            current_time_minutes=current_time,
            current_lat=current_lat,
            current_lng=current_lng,
            constraints=constraints,
        ):
            return None

        # 计算空驶价值
        move_value = compute_reposition_value(
            target_lat=hs_lat,
            target_lng=hs_lng,
            current_lat=current_lat,
            current_lng=current_lng,
            current_time_minutes=current_time,
            lambda_hour=lambda_hour,
            cost_per_km=mem.cost_per_km,
            speed_kmph=REPOSITION_SPEED_KMPH,
            hotspot_value=hs_value,
        )

        # wait 价值（保守估计）
        wait_value = lambda_hour * 0.3  # wait 期间可能发现新货源的概率较低

        # 只有 move_value 明显大于 wait_value 才空驶
        if move_value > wait_value + 50:  # 50 元 margin
            return reposition_action

        return None

    def _check_endgame_preferences(
        self,
        driver_id: str,
        mem: DriverMemory,
        constraints: PreferenceConstraint,
        current_time: int,
    ) -> dict[str, Any] | None:
        """检查末期偏好约束是否需要强制休息。"""
        remaining = SIMULATION_HORIZON_MINUTES - current_time
        remaining_days = remaining / 1440

        for rule in constraints.rules:
            if rule.rule_type == "monthly_off":
                min_off_days = int(rule.params.get("min_off_days", 2))
                off_days = mem.get_off_days_count()
                if mem.should_force_off_day(min_off_days, math.ceil(remaining_days)):
                    # 今天必须休息
                    wait_min = min(1440, remaining)  # 休息一整天或剩余时间
                    return {"action": "wait", "params": {"duration_minutes": wait_min}}

            if rule.rule_type == "must_visit":
                min_visit_days = int(rule.params.get("min_visit_days", 5))
                target_lat = float(rule.params.get("target_lat", 0))
                target_lng = float(rule.params.get("target_lng", 0))
                visit_days = mem.get_visit_days_count(target_lat, target_lng)
                needed = min_visit_days - visit_days
                if needed > 0 and remaining_days <= needed + 1:
                    # 剩余天数紧张，需要到目标点
                    # 但这不能通过 wait 实现，需要通过接单到目标点附近
                    # 这里只做标记，不强制
                    pass

        return None

    def _smart_wait_duration(
        self,
        current_time: int,
        remaining: int,
        mem: DriverMemory,
        constraints: PreferenceConstraint,
        items_count: int,
        best_utility: float,
    ) -> int:
        """智能 wait 时长。"""
        # 1. 仿真末期：wait 到结束
        if remaining < 60:
            return remaining

        # 2. 偏好驱动的 wait
        pref_wait = preference_aware_wait(
            current_time_minutes=current_time,
            current_lat=float(mem.last_status.get("current_lat", 0)) if mem.last_status else 0,
            current_lng=float(mem.last_status.get("current_lng", 0)) if mem.last_status else 0,
            constraints=constraints,
            daily_rest_minutes=mem.get_today_rest_minutes(),
        )
        if pref_wait is not None and pref_wait > 0:
            return min(pref_wait, remaining)

        # 3. 常规策略
        if items_count == 0:
            # 当前无货，等新货源上线
            wait_min = WAIT_MEDIUM
        elif best_utility < -100:
            # 有货但质量差
            wait_min = WAIT_MEDIUM
        elif best_utility < 0:
            # 接近阈值
            wait_min = WAIT_SHORT
        else:
            # 有好货但被偏好过滤了
            wait_min = WAIT_SHORT

        # 4. 不要 wait 超过剩余时间
        wait_min = min(wait_min, remaining)
        return max(1, wait_min)

    def _is_cache_valid(self, cached: CachedCargo, current_time_minutes: int) -> bool:
        """检查缓存货源是否仍然有效（含缓存年龄检查）。"""
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
                    if load_end_minutes < current_time_minutes:
                        return False
            except (ValueError, TypeError):
                pass

        # 检查缓存年龄：超过 30 分钟的缓存视为不可靠
        # （货源可能已被其他司机接走或被系统移除）
        cache_age = current_time_minutes - cached.query_time_minutes
        if cache_age > 30:
            return False

        return True
