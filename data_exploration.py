"""
data_exploration.py
-------------------
先行数据探索脚本：读取 cfps2022childproxy_202410.dta，检查关键变量的
存在性与覆盖率，并自动生成变量保留/剔除建议。

运行方法：
    python data_exploration.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import pyreadstat

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DATA_FILE = "cfps2022childproxy_202410.dta"
COVERAGE_THRESHOLD = 0.40  # 覆盖率阈值（40%）
NEGATIVE_CODES = {-1, -2, -8, -9, -10, 79}  # CFPS 通用缺失/不适用编码

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 关键变量分组
# ---------------------------------------------------------------------------
KEY_VARS = {
    "慢性病诊断": ["qp4001", "wc4_1", "wc5ncode", "ill"],
    "抽样权重": ["child_weight", "rswt_natcs22n", "rswt_natpn1022n"],
    "学业成绩": ["wf501", "wf502"],
    "情绪行为(we3xx)": [f"we3{str(i).zfill(2)}" for i in range(1, 13)],
    "情绪行为(wn4xx)": ["wn401"],
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def load_data(path: str):
    """读取 Stata 数据文件，返回 (DataFrame, meta)。"""
    logger.info("读取数据文件：%s", path)
    if not Path(path).exists():
        logger.error("文件不存在：%s", path)
        sys.exit(1)
    df, meta = pyreadstat.read_dta(path)
    logger.info("数据维度：%d 行 × %d 列", *df.shape)
    return df, meta


def replace_negative_codes(series: pd.Series) -> pd.Series:
    """将 CFPS 通用缺失/不适用编码替换为 NaN。"""
    return series.replace({c: float("nan") for c in NEGATIVE_CODES})


def compute_coverage(df: pd.DataFrame, col: str) -> float:
    """计算某列去除负值编码后的有效覆盖率。"""
    clean = replace_negative_codes(df[col])
    return 1.0 - clean.isna().mean()


def analyze_column(df: pd.DataFrame, meta, col: str) -> dict:
    """返回单列的统计摘要字典。"""
    label = meta.column_names_to_labels.get(col, "")
    raw_missing = df[col].isna().sum()
    coverage = compute_coverage(df, col)
    valid_n = int(round(coverage * len(df)))
    return {
        "variable": col,
        "label": label,
        "raw_missing": raw_missing,
        "coverage_rate": round(coverage, 4),
        "valid_n": valid_n,
        "recommend": "保留" if coverage >= COVERAGE_THRESHOLD else "剔除",
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    df, meta = load_data(DATA_FILE)
    all_cols = set(df.columns)

    results = []
    print("\n" + "=" * 70)
    print("CFPS 2022 儿童代答问卷 — 关键变量探索报告")
    print("=" * 70)

    for group, vars_ in KEY_VARS.items():
        print(f"\n【{group}】")
        group_found = False
        for var in vars_:
            if var in all_cols:
                group_found = True
                info = analyze_column(df, meta, var)
                results.append({**info, "group": group})
                flag = "✓" if info["recommend"] == "保留" else "✗"
                print(
                    f"  {flag} {var:25s} | 覆盖率={info['coverage_rate']:.1%}"
                    f" | 有效N={info['valid_n']:5d} | {info['label']}"
                )
            else:
                print(f"  - {var:25s} | 【变量不存在】")
        if not group_found:
            print(f"  （本组所有候选变量均不存在）")

    # -----------------------------------------------------------------------
    # 额外扫描：其他 we3xx / wn4xx 条目
    # -----------------------------------------------------------------------
    extra_emotion = sorted(
        c for c in all_cols
        if (c.startswith("we3") or c.startswith("wn4"))
        and c not in KEY_VARS["情绪行为(we3xx)"]
        and c not in KEY_VARS["情绪行为(wn4xx)"]
    )
    if extra_emotion:
        print("\n【其他情绪条目（额外发现）】")
        for var in extra_emotion:
            info = analyze_column(df, meta, var)
            results.append({**info, "group": "情绪行为(额外)"})
            flag = "✓" if info["recommend"] == "保留" else "✗"
            print(
                f"  {flag} {var:25s} | 覆盖率={info['coverage_rate']:.1%}"
                f" | 有效N={info['valid_n']:5d} | {info['label']}"
            )

    # -----------------------------------------------------------------------
    # 权重变量推断建议
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("权重变量建议")
    print("=" * 70)
    weight_candidates = {
        "child_weight": "问题声明中的候选权重变量（未找到）",
        "rswt_natcs22n": "CFPS2022 个人横截面权数（标准化）- 推荐用于截面分析",
        "rswt_natpn1022n": "CFPS2022 个人面板权数（标准化）- 仅适用于面板分析",
    }
    selected_weight = None
    for wv, desc in weight_candidates.items():
        if wv in all_cols:
            cov = compute_coverage(df, wv)
            print(f"  ✓ {wv}: {desc}  [覆盖率={cov:.1%}]")
            if selected_weight is None and cov >= COVERAGE_THRESHOLD:
                selected_weight = wv
        else:
            print(f"  ✗ {wv}: {desc}  【不存在】")

    if selected_weight:
        print(f"\n  → 建议使用权重变量：{selected_weight}")
    else:
        print("\n  → 未找到可用权重变量，分析将以未加权方式进行")

    # -----------------------------------------------------------------------
    # 慢性病变量建议
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("慢性病/健康状态变量建议")
    print("=" * 70)
    chronic_candidates = {
        "qp4001": "过去半年医生诊断的慢性病（问题声明中的目标变量，未找到）",
        "wc4_1": "孩子过去12个月是否因病就医（替代指标）",
        "wc0": "过去一月孩子是否生病（替代指标）",
        "ill": "加载变量：孩子出生以来患过最严重的疾病",
    }
    selected_chronic = None
    for cv, desc in chronic_candidates.items():
        if cv in all_cols:
            cov = compute_coverage(df, cv)
            flag = "✓" if cov >= COVERAGE_THRESHOLD else "✗"
            print(f"  {flag} {cv}: {desc}  [覆盖率={cov:.1%}]")
            if selected_chronic is None and cov >= COVERAGE_THRESHOLD:
                selected_chronic = cv
        else:
            print(f"  ✗ {cv}: {desc}  【不存在】")

    if selected_chronic and selected_chronic != "qp4001":
        print(
            f"\n  → qp4001 不存在，建议用 {selected_chronic} 作为健康状态代理变量"
        )
    elif selected_chronic == "qp4001":
        print("\n  → 建议使用 qp4001")

    # -----------------------------------------------------------------------
    # 汇总建议
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("自动建议汇总")
    print("=" * 70)
    keep = [r for r in results if r["recommend"] == "保留"]
    drop = [r for r in results if r["recommend"] == "剔除"]

    print(f"  应保留变量（覆盖率 ≥ {COVERAGE_THRESHOLD:.0%}）：")
    for r in keep:
        print(f"    • {r['variable']:25s} [{r['coverage_rate']:.1%}] {r['label']}")

    if drop:
        print(f"\n  应剔除变量（覆盖率 < {COVERAGE_THRESHOLD:.0%}）：")
        for r in drop:
            print(
                f"    • {r['variable']:25s} [{r['coverage_rate']:.1%}] {r['label']}"
            )
    else:
        print(f"\n  无需剔除的变量（所有关键变量覆盖率均 ≥ {COVERAGE_THRESHOLD:.0%}）")

    print("\n" + "=" * 70)
    print(f"探索完成。数据集总行数：{len(df)}，总列数：{len(df.columns)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
