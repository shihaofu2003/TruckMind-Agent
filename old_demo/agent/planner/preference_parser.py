"""偏好解析模块：将自然语言偏好文本转为结构化约束模板。

使用 LLM 一次性解析，结果缓存供后续使用。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger("agent.preference_parser")


@dataclass
class PreferenceRule:
    """单条偏好约束规则。"""

    rule_type: str  # "night_rest" | "daily_rest" | "monthly_off" | "home_before" | "must_visit" | "forbidden_area"
    params: dict[str, Any] = field(default_factory=dict)
    penalty_per_violation: float = 0.0
    penalty_cap: float = 0.0
    penalty_lump_sum: float = 0.0  # 一次性罚分（如 D008/D010）
    raw_text: str = ""


@dataclass
class PreferenceConstraint:
    """司机的完整偏好约束。"""

    driver_id: str
    rules: list[PreferenceRule] = field(default_factory=list)


# ── LLM 解析 Prompt ──────────────────────────────────────────
PREFERENCE_PARSE_PROMPT = """你是一个货运司机偏好规则解析器。

给定司机的自然语言偏好描述，请将其转换为结构化 JSON 约束模板。

约束类型包括：
- night_rest: 夜间禁止特定动作（如不接单、不空驶）
- daily_rest: 每天最少连续休息时长
- monthly_off: 每月最少休息天数
- home_before: 每天必须回家的时间
- must_visit: 每月必须到访某地的天数
- forbidden_area: 禁止进入的区域

请严格按以下格式输出：
```json
{
  "rules": [
    {
      "rule_type": "...",
      "params": { ... },
      "penalty_per_violation": 0,
      "penalty_cap": 0,
      "penalty_lump_sum": 0
    }
  ]
}
```

注意：
- penalty_per_violation: 每次违反的罚分
- penalty_cap: 累计罚分上限
- penalty_lump_sum: 一次性罚分（不满足条件时一次性扣除）

司机偏好描述：
{preferences_text}
"""


def parse_preferences_with_llm(
    preferences: list[str],
    driver_id: str,
    model_chat_fn,  # callable: payload -> dict
) -> PreferenceConstraint:
    """使用 LLM 解析偏好文本为结构化约束。"""
    if not preferences:
        return PreferenceConstraint(driver_id=driver_id, rules=[])

    text = "\n".join(preferences)
    # 使用字符串拼接而非 format()，避免 prompt 模板中的 JSON 花括号被误解析
    prompt = PREFERENCE_PARSE_PROMPT.replace("{preferences_text}", text, 1)

    try:
        resp = model_chat_fn({
            "messages": [
                {
                    "role": "system",
                    "content": "你是货运司机偏好规则解析器，只输出JSON，不输出其他内容。",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        })
        choices = resp.get("choices", [])
        if not choices:
            raise ValueError("LLM 返回为空")
        content = choices[0].get("message", {}).get("content", "")
        # 提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(content)

        rules_data = parsed.get("rules", [])
        rules = []
        for r in rules_data:
            rules.append(PreferenceRule(
                rule_type=r.get("rule_type", ""),
                params=r.get("params", {}),
                penalty_per_violation=float(r.get("penalty_per_violation", 0)),
                penalty_cap=float(r.get("penalty_cap", 0)),
                penalty_lump_sum=float(r.get("penalty_lump_sum", 0)),
                raw_text=text,
            ))
        return PreferenceConstraint(driver_id=driver_id, rules=rules)
    except Exception as e:
        logger.warning("LLM 偏好解析失败 driver_id=%s: %s, 回退到规则解析", driver_id, e)
        return _fallback_parse_preferences(preferences, driver_id)


def _fallback_parse_preferences(preferences: list[str], driver_id: str) -> PreferenceConstraint:
    """基于关键词的规则解析回退方案。"""
    rules: list[PreferenceRule] = []
    text = " ".join(preferences)

    # D006: 每天至少连续休息5小时
    if "连续" in text and "休息" in text and ("5小时" in text or "5小时以上" in text):
        rules.append(PreferenceRule(
            rule_type="daily_rest",
            params={"min_rest_minutes": 300},
            penalty_per_violation=400.0,
            penalty_cap=4000.0,
            raw_text=text,
        ))

    # D007: 23:00-04:00 不接单不空驶
    if "23:00" in text and "04:00" in text and ("不接单" in text or "不空驶" in text):
        rules.append(PreferenceRule(
            rule_type="night_rest",
            params={"start_hour": 23, "end_hour": 4, "forbidden_actions": ["take_order", "reposition"]},
            penalty_per_violation=500.0,
            penalty_cap=5000.0,
            raw_text=text,
        ))

    # D008: 每月至少2天完全不出车
    if "至少" in text and "天" in text and "不出车" in text:
        # 提取天数
        day_match = re.search(r'至少(\d+)天', text)
        min_days = int(day_match.group(1)) if day_match else 2
        rules.append(PreferenceRule(
            rule_type="monthly_off",
            params={"min_off_days": min_days},
            penalty_lump_sum=3000.0,
            raw_text=text,
        ))

    # D009: 每天23:00前回家
    if "23:00前回家" in text or ("23:00" in text and "回家" in text):
        # 提取家坐标
        coord_match = re.search(r'(\d+\.\d+)\s*,\s*(\d+\.\d+)', text)
        home_lat = float(coord_match.group(1)) if coord_match else 23.12
        home_lng = float(coord_match.group(2)) if coord_match else 113.28
        rules.append(PreferenceRule(
            rule_type="home_before",
            params={"home_hour": 23, "home_lat": home_lat, "home_lng": home_lng, "home_radius_km": 1.0, "quiet_until_hour": 8},
            penalty_per_violation=600.0,
            penalty_cap=6000.0,
            raw_text=text,
        ))

    # D010: 每月至少5天到目标点
    if "至少" in text and "次" in text and ("固定地点" in text or "目标点" in text):
        count_match = re.search(r'至少(\d+)次', text)
        min_visits = int(count_match.group(1)) if count_match else 5
        coord_matches = re.findall(r'(\d+\.\d+)\s*,\s*(\d+\.\d+)', text)
        target_lat = float(coord_matches[0][0]) if coord_matches else 23.13
        target_lng = float(coord_matches[0][1]) if coord_matches else 113.26
        rules.append(PreferenceRule(
            rule_type="must_visit",
            params={"min_visit_days": min_visits, "target_lat": target_lat, "target_lng": target_lng, "radius_km": 1.0},
            penalty_lump_sum=3000.0,
            raw_text=text,
        ))

    # D010: 禁止进入区域
    if "不想进入" in text or "禁止进入" in text or "禁入" in text:
        coord_matches = re.findall(r'(\d+\.\d+)\s*,\s*(\d+\.\d+)', text)
        radius_match = re.search(r'半径\s*(\d+)\s*km', text)
        if coord_matches:
            area_lat = float(coord_matches[-1][0])
            area_lng = float(coord_matches[-1][1])
            area_radius = float(radius_match.group(1)) if radius_match else 2.0
            rules.append(PreferenceRule(
                rule_type="forbidden_area",
                params={"center_lat": area_lat, "center_lng": area_lng, "radius_km": area_radius},
                penalty_lump_sum=3000.0,
                raw_text=text,
            ))

    return PreferenceConstraint(driver_id=driver_id, rules=rules)
