#!/usr/bin/env python3
"""
打包提交脚本 — 按照 04-提交方式.md 的要求，将必要文件打包为 ZIP。

用法:
    python pack_submission.py [--team 队名] [--round 初赛|复赛]

默认:
    --team  TruckMind-Agent
    --round 初赛

初赛: 打包 demo/agent/ + demo/results/
复赛: 仅打包 demo/agent/（不含 results/）
"""

import argparse
import os
import zipfile
from pathlib import Path

# ── 项目根目录（脚本所在目录） ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_DIR = PROJECT_ROOT / "demo"

# ── 需要排除的目录 / 文件模式 ──────────────────────────────────
EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    "node_modules",
}

EXCLUDE_FILES = {
    "calc_monthly_income.py",   # 仅本地自测，不必提交
}

EXCLUDE_FILE_PATTERNS = {
    "monthly_income_",          # 本地收益 JSON，官方会重算
}

# server/ 目录整体排除（评测使用赛方统一代码，且含 data/ 大体积数据）
EXCLUDE_SUBDIRS_IN_DEMO = {
    "server",
}


def should_include(rel_path: str, round_type: str) -> bool:
    """判断文件是否应纳入 ZIP"""
    parts = Path(rel_path).parts

    # 排除 __pycache__ 等
    for part in parts:
        if part in EXCLUDE_DIRS:
            return False

    # 排除 demo/ 下的特定子目录
    if len(parts) >= 2 and parts[0] == "demo" and parts[1] in EXCLUDE_SUBDIRS_IN_DEMO:
        return False

    # 复赛排除 results/
    if round_type == "复赛" and len(parts) >= 2 and parts[0] == "demo" and parts[1] == "results":
        return False

    # 排除特定文件
    filename = parts[-1]
    if filename in EXCLUDE_FILES:
        return False

    # 排除匹配前缀的文件
    for pat in EXCLUDE_FILE_PATTERNS:
        if filename.startswith(pat):
            return False

    return True


def pack(round_type: str, team_name: str) -> str:
    zip_name = f"{team_name}_赛题提交.zip"
    zip_path = PROJECT_ROOT / zip_name

    if zip_path.exists():
        os.remove(zip_path)

    included = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(DEMO_DIR):
            # 就地过滤目录，避免进入不需要的目录
            dirnames[:] = [
                d for d in dirnames
                if d not in EXCLUDE_DIRS
                and (os.path.relpath(os.path.join(dirpath, d), DEMO_DIR).split(os.sep)[0] not in EXCLUDE_SUBDIRS_IN_DEMO)
            ]
            # 复赛排除 results 目录
            if round_type == "复赛":
                dirnames[:] = [d for d in dirnames if d != "results"]

            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, PROJECT_ROOT)

                if not should_include(rel_path, round_type):
                    continue

                arcname = rel_path  # ZIP 内以 demo/ 为根
                zf.write(full_path, arcname)
                included += 1
                size_kb = os.path.getsize(full_path) / 1024
                print(f"  + {arcname}  ({size_kb:.1f} KB)")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ 打包完成: {zip_path}")
    print(f"   文件数: {included}")
    print(f"   压缩包大小: {size_mb:.2f} MB")
    print(f"   赛段: {round_type}")
    if round_type == "初赛":
        print("   包含: demo/agent/ + demo/results/")
    else:
        print("   包含: demo/agent/（不含 results/）")
    return str(zip_path)


def main():
    parser = argparse.ArgumentParser(description="打包 TruckMind-Agent 提交文件")
    parser.add_argument("--team", default="ShihaoFu", help="队伍名称（用于 ZIP 文件名）")
    parser.add_argument("--round", choices=["初赛", "复赛"], default="初赛", help="赛段（初赛/复赛）")
    args = parser.parse_args()

    print(f"📦 开始打包 — 队伍: {args.team}  赛段: {args.round}\n")
    pack(args.round, args.team)


if __name__ == "__main__":
    main()
