"""偏好硬规则过滤模块：违反偏好的动作直接拒绝，不进入评分。"""

from __future__ import annotations

import logging
import math
from typing import Any

from .cargo_math import haversine_km
from .preference_parser import PreferenceConstraint, PreferenceRule


logger = logging.getLogger("agent.preference_filter")


def passes_preference_rules(
    action: dict[str, Any],
    current_time_minutes: int,
    current_lat: float,
    current_lng: float,
    constraints: PreferenceConstraint,
    # 仿真参数
    reposition_speed_km_per_hour: float = 60.0,
    simulation_horizon_minutes: int | None = None,
    # 订单仿真信息（用于 take_order）
    order_finish_time: int | None = None,
    order_end_lat: float | None = None,
    order_end_lng: float | None = None,
    # 状态追踪信息
    daily_rest_minutes: int = 0,  # 当天已累计休息分钟
    off_days_count: int = 0,  # 已休息天数
    visit_days_count: int = 0,  # 已到访目标点天数
    current_day: int = 0,  # 当前天索引
) -> bool:
    """硬规则：违反偏好的动作直接拒绝，不进入评分。

    返回 True 表示动作通过偏好检查。
    """
    action_name = str(action.get("action", "")).strip().lower()
    params = action.get("params", {})

    for rule in constraints.rules:
        if not _check_rule(
            rule=rule,
            action_name=action_name,
            params=params,
            current_time_minutes=current_time_minutes,
            current_lat=current_lat,
            current_lng=current_lng,
            reposition_speed_km_per_hour=reposition_speed_km_per_hour,
            order_finish_time=order_finish_time,
            order_end_lat=order_end_lat,
            order_end_lng=order_end_lng,
            daily_rest_minutes=daily_rest_minutes,
            off_days_count=off_days_count,
            visit_days_count=visit_days_count,
            current_day=current_day,
            simulation_horizon_minutes=simulation_horizon_minutes,
        ):
            return False

    return True


def _check_rule(
    rule: PreferenceRule,
    action_name: str,
    params: dict[str, Any],
    current_time_minutes: int,
    current_lat: float,
    current_lng: float,
    reposition_speed_km_per_hour: float,
    order_finish_time: int | None,
    order_end_lat: float | None,
    order_end_lng: float | None,
    daily_rest_minutes: int,
    off_days_count: int,
    visit_days_count: int,
    current_day: int,
    simulation_horizon_minutes: int | None,
) -> bool:
    """检查单条规则。"""
    rt = rule.rule_type

    if rt == "night_rest":
        return _check_night_rest(rule, action_name, current_time_minutes, order_finish_time)

    if rt == "daily_rest":
        # daily_rest 不直接过滤动作，而是在评分中考虑
        # 但如果接单后导致当天无法满足休息要求，可以提前警告
        return True

    if rt == "monthly_off":
        # monthly_off 不直接过滤动作，由末期策略处理
        return True

    if rt == "home_before":
        return _check_home_before(rule, action_name, current_time_minutes, current_lat, current_lng,
                                  order_finish_time, order_end_lat, order_end_lng)

    if rt == "must_visit":
        # must_visit 不直接过滤动作，由末期策略处理
        return True

    if rt == "forbidden_area":
        return _check_forbidden_area(rule, action_name, params, current_lat, current_lng,
                                     order_end_lat, order_end_lng)

    return True


def _check_night_rest(
    rule: PreferenceRule,
    action_name: str,
    current_time_minutes: int,
    order_finish_time: int | None,
) -> bool:
    """夜间休息规则：禁止在指定时段接单或空驶。"""
    if action_name not in ("take_order", "reposition"):
        return True

    start_hour = rule.params.get("start_hour", 23)
    end_hour = rule.params.get("end_hour", 4)

    current_hour_minutes = current_time_minutes % 1440
    current_hour = current_hour_minutes / 60.0

    # 判断当前时间是否在夜间休息时段
    in_night = False
    if start_hour > end_hour:
        # 跨午夜，如 23:00-04:00
        in_night = current_hour >= start_hour or current_hour < end_hour
    else:
        in_night = start_hour <= current_hour < end_hour

    if in_night:
        return False

    # 如果是接单，还要检查接单后是否会进入夜间时段
    if action_name == "take_order" and order_finish_time is not None:
        # 检查接单执行期间是否会跨越夜间时段
        # 简化：如果当前时间接近夜间开始（1小时内），且订单持续时间较长，则拒绝
        if start_hour > end_hour:
            night_start_minutes = (current_time_minutes // 1440) * 1440 + start_hour * 60
            if current_time_minutes < night_start_minutes <= order_finish_time:
                # 订单执行期间会进入夜间
                return False

    return True


def _check_home_before(
    rule: PreferenceRule,
    action_name: str,
    current_time_minutes: int,
    current_lat: float,
    current_lng: float,
    order_finish_time: int | None,
    order_end_lat: float | None,
    order_end_lng: float | None,
) -> bool:
    """回家规则：每天指定时间前必须到家，且到家后到次日指定时间不再接单/空驶。"""
    home_hour = rule.params.get("home_hour", 23)
    home_lat = rule.params.get("home_lat", 23.12)
    home_lng = rule.params.get("home_lng", 113.28)
    quiet_until_hour = rule.params.get("quiet_until_hour", 8)

    current_hour_minutes = current_time_minutes % 1440
    current_hour = current_hour_minutes / 60.0

    # 如果当前时间在安静时段（23:00-08:00），禁止接单和空驶
    if home_hour > quiet_until_hour:
        in_quiet = current_hour >= home_hour or current_hour < quiet_until_hour
    else:
        in_quiet = home_hour <= current_hour < quiet_until_hour

    if in_quiet and action_name in ("take_order", "reposition"):
        return False

    # 如果接单后完成时间超过当天回家时间，检查终点是否在家附近
    if action_name == "take_order" and order_finish_time is not None and order_end_lat is not None:
        finish_day = order_finish_time // 1440
        home_deadline = finish_day * 1440 + home_hour * 60
        if order_finish_time > home_deadline:
            # 接单完成时已超过回家时间，检查终点是否在家附近
            dist_to_home = haversine_km(order_end_lat, order_end_lng, home_lat, home_lng)
            if dist_to_home > rule.params.get("home_radius_km", 1.0):
                return False

    return True


def _check_forbidden_area(
    rule: PreferenceRule,
    action_name: str,
    params: dict[str, Any],
    current_lat: float,
    current_lng: float,
    order_end_lat: float | None,
    order_end_lng: float | None,
) -> bool:
    """禁入区域规则：禁止进入指定区域。"""
    center_lat = rule.params.get("center_lat", 0)
    center_lng = rule.params.get("center_lng", 0)
    radius_km = rule.params.get("radius_km", 2.0)

    # 检查当前位置是否在禁入区域
    if haversine_km(current_lat, current_lng, center_lat, center_lng) <= radius_km:
        return False

    # 检查 reposition 目标是否在禁入区域
    if action_name == "reposition":
        target_lat = float(params.get("latitude", 0))
        target_lng = float(params.get("longitude", 0))
        if haversine_km(target_lat, target_lng, center_lat, center_lng) <= radius_km:
            return False

    # 检查接单终点是否在禁入区域
    if action_name == "take_order" and order_end_lat is not None and order_end_lng is not None:
        if haversine_km(order_end_lat, order_end_lng, center_lat, center_lng) <= radius_km:
            return False

    return True


def preference_aware_wait(
    current_time_minutes: int,
    current_lat: float,
    current_lng: float,
    constraints: PreferenceConstraint,
    daily_rest_minutes: int,
) -> int | None:
    """根据偏好约束决定是否需要主动 wait 来满足偏好。

    返回 wait 时长（分钟），或 None 表示无需偏好驱动的 wait。
    """
    for rule in constraints.rules:
        if rule.rule_type == "night_rest":
            # 如果当前即将进入夜间休息时段，或已在夜间休息时段，主动 wait 到休息结束
            start_hour = rule.params.get("start_hour", 23)
            end_hour = rule.params.get("end_hour", 4)
            current_hour_minutes = current_time_minutes % 1440
            current_hour = current_hour_minutes / 60.0

            if start_hour > end_hour:
                # 跨午夜（如 23:00-04:00）
                # 已在夜间休息时段
                if current_hour >= start_hour or current_hour < end_hour:
                    # 计算夜间结束时间
                    if current_hour < end_hour:
                        # 午夜后，夜间结束在今天
                        night_end = (current_time_minutes // 1440) * 1440 + end_hour * 60
                    else:
                        # 午夜前，夜间结束在明天
                        night_end = (current_time_minutes // 1440 + 1) * 1440 + end_hour * 60
                    wait_minutes = night_end - current_time_minutes
                    if wait_minutes > 0:
                        return min(wait_minutes, 600)  # 最多 wait 10 小时

                # 距离夜间开始不到30分钟，直接 wait 到夜间结束
                night_start = (current_time_minutes // 1440) * 1440 + start_hour * 60
                if 0 < night_start - current_time_minutes <= 30:
                    night_end = (current_time_minutes // 1440 + 1) * 1440 + end_hour * 60
                    wait_minutes = night_end - current_time_minutes
                    return min(wait_minutes, 600)  # 最多 wait 10 小时

        if rule.rule_type == "home_before":
            home_hour = rule.params.get("home_hour", 23)
            home_lat = rule.params.get("home_lat", 23.12)
            home_lng = rule.params.get("home_lng", 113.28)
            quiet_until_hour = rule.params.get("quiet_until_hour", 8)

            current_hour_minutes = current_time_minutes % 1440
            current_hour = current_hour_minutes / 60.0

            # 如果当前在安静时段，wait 到安静结束
            if home_hour > quiet_until_hour:
                in_quiet = current_hour >= home_hour or current_hour < quiet_until_hour
            else:
                in_quiet = home_hour <= current_hour < quiet_until_hour

            if in_quiet:
                # 计算今天安静时段结束时间
                today_quiet_end = (current_time_minutes // 1440) * 1440 + quiet_until_hour * 60
                if today_quiet_end > current_time_minutes:
                    # 今天的安静结束时间还没到
                    wait_minutes = today_quiet_end - current_time_minutes
                else:
                    # 已经过了今天的安静结束时间，等到明天
                    tomorrow_quiet_end = today_quiet_end + 1440
                    wait_minutes = tomorrow_quiet_end - current_time_minutes
                return min(wait_minutes, 600)

    return None
