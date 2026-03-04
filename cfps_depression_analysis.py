"""
cfps_depression_analysis.py
----------------------------
基于 CFPS 2022 儿童代答问卷的儿童情绪健康分析脚本。
实现随机森林分类模型，支持抽样权重、自适应变量过滤与详细日志输出。

主要特性：
  - 自适应变量过滤（覆盖率阈值 COVERAGE_THRESHOLD）
  - 动态检测慢性病/健康状态代理变量（qp4001 优先，不存在则回退至 wc4_1）
  - 自动检测并应用抽样权重（child_weight → rswt_natcs22n → 无权重）
  - CSV 输出新增 coverage_rate 列
  - PNG 标题注明加权状态与有效因子数

运行方法：
    python cfps_depression_analysis.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt

# 注册 WenQuanYi Zen Hei 字体（如存在），以支持中文显示
_WQY_FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
if __import__("pathlib").Path(_WQY_FONT).exists():
    _fm.fontManager.addfont(_WQY_FONT)
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
DATA_FILE = "cfps2022childproxy_202410.dta"
OUTPUT_CSV = "depression_analysis_results.csv"
OUTPUT_PNG = "depression_analysis_plot.png"
COVERAGE_THRESHOLD = 0.40      # 变量有效覆盖率阈值（40%）
RANDOM_STATE = 42

# CFPS 通用负值编码（缺失/不适用）
NEGATIVE_CODES = [-1, -2, -8, -9, -10, 79]

# 抑郁风险得分使用的情绪行为条目（we3xx）
EMOTION_ITEMS = [f"we3{str(i).zfill(2)}" for i in range(1, 13)]
# 逆向计分条目列表（若某条目需取反使高分=更高风险，填入列名）
# 当前所有 we3xx 条目均为正向（高分=更佳心理健康），列表留空
REVERSED_ITEMS: list[str] = []

# ---------------------------------------------------------------------------
# 候选预测因子映射（label -> column）
# 说明：qp4001（慢性病诊断）如存在优先使用，否则动态选择替代变量
# ---------------------------------------------------------------------------
FACTOR_MAP: dict[str, str] = {
    "年龄(age)": "age",
    "性别(gender)": "gender",
    "城乡(urban)": "urban22",
    "民族(ethnicity)": "minzu",
    "语文成绩(chinese_score)": "wf501",
    "数学成绩(math_score)": "wf502",
    "过去一月生病(recent_illness)": "wc0",
    "父母关心教育(parental_edu_care)": "wz301",
    "父母主动沟通(parental_communication)": "wz302",
    "BMI指数(bmi)": "bmi",
    # 慢性病诊断变量：优先 qp4001，动态回退（见 resolve_chronic_disease_var）
    "慢性病诊断(chronic_disease)": "qp4001",
}

# 权重候选变量（按优先级排序）
WEIGHT_CANDIDATES = ["child_weight", "rswt_natcs22n", "rswt_natpn1022n"]

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 数据加载与预处理
# ===========================================================================

def load_data(path: str) -> tuple[pd.DataFrame, object]:
    """读取 Stata 文件，返回 (DataFrame, meta)。"""
    logger.info("读取数据：%s", path)
    if not Path(path).exists():
        logger.error("文件不存在：%s", path)
        sys.exit(1)
    df, meta = pyreadstat.read_dta(path)
    logger.info("数据维度：%d 行 × %d 列", *df.shape)
    return df, meta


def clean_negative_codes(series: pd.Series) -> pd.Series:
    """将 CFPS 负值/不适用编码替换为 NaN。"""
    return series.replace({c: np.nan for c in NEGATIVE_CODES})


def compute_coverage(series: pd.Series) -> float:
    """返回去除负值编码后的有效覆盖率（0~1）。"""
    clean = clean_negative_codes(series)
    return 1.0 - clean.isna().mean()


# ===========================================================================
# 变量动态解析
# ===========================================================================

def resolve_chronic_disease_var(df: pd.DataFrame) -> tuple[str | None, str]:
    """
    动态确定慢性病/健康状态代理变量。
    优先级：qp4001 > wc4_1 > wc0 > ill
    返回 (列名或None, 说明字符串)。
    """
    candidates = [
        ("qp4001", "过去半年医生诊断的慢性病（首选）"),
        ("wc4_1", "过去12个月是否因病就医（替代）"),
        ("wc0", "过去一月是否生病（替代）"),
        ("ill", "孩子出生以来患过最严重的疾病（替代）"),
    ]
    for col, desc in candidates:
        if col in df.columns:
            cov = compute_coverage(df[col])
            if cov >= COVERAGE_THRESHOLD:
                logger.info("慢性病变量：使用 %s（%s，覆盖率=%.1f%%）", col, desc, cov * 100)
                return col, desc
            else:
                logger.warning("慢性病候选变量 %s 覆盖率过低（%.1f%%），跳过", col, cov * 100)
    logger.warning("未找到可用的慢性病/健康状态变量，该因子将被排除")
    return None, "无可用变量"


def resolve_weight_var(df: pd.DataFrame) -> tuple[str | None, str]:
    """
    按优先级检测可用抽样权重变量。
    返回 (列名或None, 说明字符串)。
    """
    for col in WEIGHT_CANDIDATES:
        if col in df.columns:
            cov = compute_coverage(df[col])
            logger.info("权重变量：检测到 %s，覆盖率=%.1f%%", col, cov * 100)
            return col, col
    logger.warning("未找到权重变量（%s），将使用未加权分析", WEIGHT_CANDIDATES)
    return None, "无权重"


# ===========================================================================
# 结局变量构建（情绪健康得分 → 二分类）
# ===========================================================================

def build_outcome(df: pd.DataFrame) -> pd.Series:
    """
    基于 we3xx 情绪行为条目构建情绪健康得分，并二值化为"低健康/高风险"标签。
    - 将负值编码替换为 NaN
    - 对逆向条目取反（6 - score，使高分代表更高风险）
    - 行均值 < 中位数 → label=1（高风险）；否则 → label=0
    """
    available = [c for c in EMOTION_ITEMS if c in df.columns]
    logger.info("情绪条目（%d 个）：%s", len(available), available)

    score_df = df[available].copy()
    for col in available:
        score_df[col] = clean_negative_codes(score_df[col])
        if col in REVERSED_ITEMS:
            score_df[col] = 6 - score_df[col]

    # 行均值得分（正向：越高越健康）
    wellbeing = score_df.mean(axis=1)
    # 低于中位数 → 高风险（label=1）
    threshold = wellbeing.median()
    labels = (wellbeing < threshold).astype(int)
    valid_mask = wellbeing.notna()
    logger.info(
        "情绪健康得分：均值=%.2f，中位数=%.2f；高风险样本比=%.1f%%",
        wellbeing[valid_mask].mean(),
        threshold,
        labels[valid_mask].mean() * 100,
    )
    return labels


# ===========================================================================
# 自适应变量过滤
# ===========================================================================

def filter_factors(
    df: pd.DataFrame,
    factor_map: dict[str, str],
    threshold: float = COVERAGE_THRESHOLD,
) -> tuple[dict[str, str], list[dict]]:
    """
    过滤覆盖率低于阈值的预测因子。
    返回 (有效因子映射, 所有因子的覆盖率明细列表)。
    """
    valid_map: dict[str, str] = {}
    coverage_records: list[dict] = []
    dropped: list[str] = []

    for label, col in factor_map.items():
        if col not in df.columns:
            logger.warning("变量不存在，跳过：%s (%s)", col, label)
            coverage_records.append(
                {"factor_label": label, "variable": col, "coverage_rate": 0.0, "status": "不存在"}
            )
            dropped.append(f"{col}（不存在）")
            continue

        cov = compute_coverage(df[col])
        status = "保留" if cov >= threshold else "剔除"
        coverage_records.append(
            {"factor_label": label, "variable": col, "coverage_rate": round(cov, 4), "status": status}
        )
        if cov >= threshold:
            valid_map[label] = col
        else:
            dropped.append(f"{col}（覆盖率={cov:.1%}）")

    if dropped:
        logger.warning("以下变量因覆盖率 < %.0f%% 被剔除：%s", threshold * 100, "；".join(dropped))
    else:
        logger.info("所有候选变量覆盖率均满足阈值要求（≥ %.0f%%）", threshold * 100)

    logger.info("有效预测因子数：%d / %d", len(valid_map), len(factor_map))
    return valid_map, coverage_records


# ===========================================================================
# 相关性分析
# ===========================================================================

def correlation_analysis(
    df_analysis: pd.DataFrame,
    valid_map: dict[str, str],
    outcome_col: str,
    coverage_records: list[dict],
) -> pd.DataFrame:
    """计算各预测因子与结局变量的 Spearman 相关系数，附加覆盖率列。"""
    cov_lookup = {r["variable"]: r["coverage_rate"] for r in coverage_records}
    rows = []
    for label, col in valid_map.items():
        x = clean_negative_codes(df_analysis[col])
        y = df_analysis[outcome_col]
        mask = x.notna() & y.notna()
        if mask.sum() < 10:
            logger.warning("变量 %s 有效样本不足（n=%d），跳过相关性计算", col, mask.sum())
            continue
        rho = x[mask].corr(y[mask], method="spearman")
        rows.append(
            {
                "factor_label": label,
                "variable": col,
                "spearman_rho": round(rho, 4),
                "valid_n": int(mask.sum()),
                "coverage_rate": cov_lookup.get(col, np.nan),
            }
        )
    corr_df = pd.DataFrame(rows).sort_values("spearman_rho", key=abs, ascending=False)
    return corr_df


# ===========================================================================
# 随机森林模型
# ===========================================================================

def train_random_forest(
    df_analysis: pd.DataFrame,
    valid_map: dict[str, str],
    outcome_col: str,
    weight_col: str | None,
) -> tuple[Pipeline, pd.DataFrame, float]:
    """
    训练随机森林模型，支持抽样权重。
    返回 (pipeline, feature_importance_df, mean_auc)。
    """
    feature_cols = list(valid_map.values())
    X_raw = df_analysis[feature_cols].copy()
    y = df_analysis[outcome_col].copy()

    # 清洗特征中的负值编码
    for col in feature_cols:
        X_raw[col] = clean_negative_codes(X_raw[col])

    # 同步权重
    if weight_col and weight_col in df_analysis.columns:
        w = clean_negative_codes(df_analysis[weight_col])
    else:
        w = pd.Series(np.ones(len(df_analysis)), index=df_analysis.index)

    # 删除结局缺失行
    valid_mask = y.notna()
    X_raw = X_raw[valid_mask]
    y = y[valid_mask]
    w = w[valid_mask]

    # 缺失权重补 1
    w = w.fillna(1.0)

    logger.info("模型训练样本：n=%d，特征数=%d", len(y), len(feature_cols))

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )

    # 交叉验证 AUC（不传权重，仅评估泛化性）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = cross_val_score(pipeline, X_raw, y, cv=cv, scoring="roc_auc")
    mean_auc = float(np.mean(auc_scores))
    logger.info("5 折交叉验证 AUC：%.4f ± %.4f", mean_auc, np.std(auc_scores))

    # 全量训练（含抽样权重）
    if weight_col:
        logger.info("应用抽样权重（%s）进行全量训练", weight_col)
        pipeline.fit(X_raw, y, rf__sample_weight=w.values)
    else:
        logger.info("未加权全量训练")
        pipeline.fit(X_raw, y)

    # 特征重要性
    rf_model: RandomForestClassifier = pipeline.named_steps["rf"]
    importance_df = pd.DataFrame(
        {
            "variable": feature_cols,
            "factor_label": list(valid_map.keys()),
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return pipeline, importance_df, mean_auc


# ===========================================================================
# 可视化
# ===========================================================================

def plot_results(
    corr_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    weight_col: str | None,
    n_valid_factors: int,
    output_path: str,
) -> None:
    """生成相关性与特征重要性双图，标注加权状态与因子数。"""
    weight_label = f"加权（{weight_col}）" if weight_col else "未加权"
    title_suffix = f"[{weight_label} | 有效因子数={n_valid_factors}]"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- 相关性条形图 ---
    ax1 = axes[0]
    colors = ["#e74c3c" if r > 0 else "#3498db" for r in corr_df["spearman_rho"]]
    ax1.barh(corr_df["factor_label"], corr_df["spearman_rho"], color=colors)
    ax1.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Spearman ρ", fontsize=11)
    ax1.set_title(f"预测因子与情绪健康风险相关性\n{title_suffix}", fontsize=12)
    ax1.invert_yaxis()

    # --- 特征重要性条形图 ---
    ax2 = axes[1]
    top_imp = importance_df.head(15)
    ax2.barh(top_imp["factor_label"], top_imp["importance"], color="#2ecc71")
    ax2.set_xlabel("特征重要性（Gini）", fontsize=11)
    ax2.set_title(f"随机森林特征重要性（Top {len(top_imp)}）\n{title_suffix}", fontsize=12)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("图表已保存：%s", output_path)
    plt.close()


# ===========================================================================
# 主流程
# ===========================================================================

def main() -> None:
    # 1. 加载数据
    df, meta = load_data(DATA_FILE)

    # 2. 解析权重变量
    weight_col, weight_desc = resolve_weight_var(df)
    if weight_col:
        logger.info("抽样权重状态：已启用（%s）", weight_desc)
    else:
        logger.info("抽样权重状态：未加权（无可用权重变量）")

    # 3. 解析慢性病变量并动态更新 FACTOR_MAP
    factor_map = dict(FACTOR_MAP)
    chronic_col, chronic_desc = resolve_chronic_disease_var(df)
    if chronic_col is None:
        # 无可用健康变量，移除该条目
        factor_map.pop("慢性病诊断(chronic_disease)", None)
    elif chronic_col != "qp4001":
        # qp4001 不存在，替换为实际可用变量
        factor_map["慢性病诊断(chronic_disease)"] = chronic_col
        logger.info("慢性病变量替换：qp4001 → %s（%s）", chronic_col, chronic_desc)

    # 4. 构建结局变量
    outcome_col = "depression_risk"
    df[outcome_col] = build_outcome(df)

    # 5. 自适应变量过滤
    valid_map, coverage_records = filter_factors(df, factor_map)
    if not valid_map:
        logger.error("所有预测因子均被过滤，无法继续分析")
        sys.exit(1)

    # 6. 相关性分析
    logger.info("开始相关性分析（n 因子=%d）...", len(valid_map))
    corr_df = correlation_analysis(df, valid_map, outcome_col, coverage_records)
    logger.info("相关性分析完成")

    # 7. 随机森林建模
    logger.info("开始随机森林建模...")
    pipeline, importance_df, mean_auc = train_random_forest(
        df, valid_map, outcome_col, weight_col
    )
    logger.info("建模完成，交叉验证 AUC=%.4f", mean_auc)

    # 8. 合并输出
    result_df = corr_df.merge(
        importance_df[["variable", "importance"]],
        on="variable",
        how="left",
    )
    # 附加覆盖率明细（包含被过滤变量）
    full_coverage = pd.DataFrame(coverage_records)
    result_df = result_df.merge(
        full_coverage[["variable", "status"]].rename(columns={"status": "filter_status"}),
        on="variable",
        how="left",
    )
    result_df["weight_var"] = weight_col if weight_col else "无"
    result_df["cv_auc"] = round(mean_auc, 4)

    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logger.info("结果已保存：%s（%d 行）", OUTPUT_CSV, len(result_df))

    # 同时保存覆盖率全表（含被过滤变量）
    coverage_csv = OUTPUT_CSV.replace(".csv", "_coverage.csv")
    full_coverage.to_csv(coverage_csv, index=False, encoding="utf-8-sig")
    logger.info("覆盖率明细已保存：%s", coverage_csv)

    # 9. 可视化
    plot_results(
        corr_df=corr_df,
        importance_df=importance_df,
        weight_col=weight_col,
        n_valid_factors=len(valid_map),
        output_path=OUTPUT_PNG,
    )

    # 10. 控制台摘要
    print("\n" + "=" * 60)
    print("分析摘要")
    print("=" * 60)
    print(f"  数据文件：{DATA_FILE}")
    print(f"  总样本量：{len(df)}")
    print(f"  抽样权重：{weight_col or '未加权'}")
    print(f"  有效预测因子：{len(valid_map)} / {len(factor_map)}")
    print(f"  覆盖率阈值：{COVERAGE_THRESHOLD:.0%}")
    print(f"  交叉验证 AUC：{mean_auc:.4f}")
    print(f"  输出 CSV：{OUTPUT_CSV}")
    print(f"  输出 PNG：{OUTPUT_PNG}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
