"""
cfps_ml_pipeline.py
--------------------
基于 CFPS 2022 儿童代答问卷数据的可复现机器学习 Pipeline。

功能：
  A) 加权逻辑回归（sample_weight，不使用 class_weight）
  B) XGBoost 分类器（binary:logistic，sample_weight，5 折 CV 调参）
  评估指标：ROC-AUC、PR-AUC、Brier Score（加权 & 未加权）、
            校准曲线、最佳阈值下混淆矩阵/灵敏度/特异度/F1
  SHAP：全局 mean|SHAP| 条形图 + BMI dependence plot（年龄叠加）+ 自动文字总结

注意事项（已知限制）：
  - RandomizedSearchCV 内部 CV 评估使用未加权 AUC（sklearn 对 CV 内 sample_weight
    路由需要元数据路由 API，配置复杂）；最终模型拟合使用完整 sample_weight。
  - WELLBEING_CUTOFF 默认为 None，使用样本中位数作为截断（仅具探索意义），
    请查阅 CFPS 用户手册后填入临床划界分。
  - sleep_duration / screen_time 的 CFPS 列名为推测值（wc2 / wn401），
    如果在数据中不存在，相应特征会被自动排除。

运行方法：
    python cfps_ml_pipeline.py
"""

# ===========================================================================
# 导入
# ===========================================================================
import logging
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import randint, uniform
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    warnings.warn("xgboost 未安装，XGBoost 模型将被跳过。请运行：pip install xgboost")

try:
    import shap

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    warnings.warn("shap 未安装，SHAP 分析将被跳过。请运行：pip install shap")


# ===========================================================================
# 全局配置
# ===========================================================================
DATA_FILE = "cfps2022childproxy_202410.dta"
RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5
NEGATIVE_CODES = [-1, -2, -8, -9, -10, 79]

# 情绪条目（用于构建目标变量 high_risk）
EMOTION_ITEMS = [f"we3{str(i).zfill(2)}" for i in range(1, 13)]
REVERSED_ITEMS: list[str] = []  # 逆向计分条目（当前留空）

# 情绪健康得分截断阈值（None = 样本中位数，仅具探索意义）
WELLBEING_CUTOFF: float | None = None

# 覆盖率阈值（低于此值的特征被自动排除）
COVERAGE_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# 特征与 CFPS 列名映射（feature_name -> CFPS 实际列名）
# ---------------------------------------------------------------------------
COLUMN_MAP: dict[str, str] = {
    "age": "age",                    # 问卷变量 ✅
    "gender": "gender",              # 问卷变量 ✅
    "urban": "urban22",              # 数据集派生变量，非问卷题目
    "sleep_duration": "wc2",         # 问卷中存在 ✅，需核实具体题目描述
    "screen_time": "wn401",          # 需核实问卷中实际题号（验证未找到）
    "chronic_disease": "qp4001",     # 可能属于成人问卷模块，动态回退
    "recent_illness": "wc0",         # 需核实问卷中实际题号（验证未找到）
    "chinese_score": "wf501",        # 问卷变量 ✅
    "math_score": "wf502",           # 问卷变量 ✅
    "bmi": "bmi",                    # 问卷变量 ✅
    "weight": "rswt_natcs22n",       # 数据集派生变量（抽样权重）
}

# 慢性病变量动态回退候选（按优先级）
CHRONIC_FALLBACKS = ["qp4001", "wc4_1", "wc0", "ill"]

# 数值特征与类别特征（按 feature_name）
NUMERIC_FEATURES = ["age", "sleep_duration", "screen_time",
                    "chinese_score", "math_score", "bmi"]
CATEGORICAL_FEATURES = ["gender", "urban", "chronic_disease", "recent_illness"]

# 输出文件名
OUT_CSV = "cfps_pipeline_results.csv"
OUT_ROC_PNG = "cfps_pipeline_roc_prc.png"
OUT_CALIB_PNG = "cfps_pipeline_calibration.png"
OUT_SHAP_IMP = "cfps_pipeline_shap_importance.png"
OUT_SHAP_DEP = "cfps_pipeline_shap_bmi.png"


# ===========================================================================
# 日志配置
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansMonoCJKsc-Regular.otf",
]
for _font_path in _FONT_CANDIDATES:
    if Path(_font_path).exists():
        _fm.fontManager.addfont(_font_path)
        _prop = _fm.FontProperties(fname=_font_path)
        plt.rcParams["font.family"] = _prop.get_name()
        break


# ===========================================================================
# 数据加载与目标变量构建
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


def clean_negative(series: pd.Series) -> pd.Series:
    """将 CFPS 负值/不适用编码替换为 NaN。"""
    return series.replace({c: np.nan for c in NEGATIVE_CODES})


def build_high_risk(df: pd.DataFrame) -> pd.Series:
    """
    基于 we3xx 情绪行为条目构建二值化目标变量 high_risk。
      - 行均分 < WELLBEING_CUTOFF（或样本中位数）→ high_risk=1（高风险）
      - 所有条目均 NaN 的行 → NaN（将从训练集排除）
    """
    available = [c for c in EMOTION_ITEMS if c in df.columns]
    if not available:
        logger.error("未找到情绪行为条目（we3xx），无法构建目标变量")
        sys.exit(1)
    logger.info("情绪条目（%d 个）：%s", len(available), available)

    score_df = df[available].apply(clean_negative)
    for col in available:
        if col in REVERSED_ITEMS:
            score_df[col] = 6 - score_df[col]

    wellbeing = score_df.mean(axis=1)

    if WELLBEING_CUTOFF is not None:
        threshold = WELLBEING_CUTOFF
        logger.info("使用临床划界分阈值：%.2f", threshold)
    else:
        threshold = wellbeing.median()
        logger.warning(
            "WELLBEING_CUTOFF 未设置，使用样本中位数 %.2f 作为截断阈值（仅探索意义）。"
            "建议查阅 CFPS 用户手册后将 WELLBEING_CUTOFF 设为官方临床划界分。",
            threshold,
        )

    labels = pd.Series(
        np.where(wellbeing.notna(), (wellbeing < threshold).astype(float), np.nan),
        index=wellbeing.index,
    )
    valid_n = int(wellbeing.notna().sum())
    logger.info(
        "high_risk：阈值=%.2f，高风险率=%.1f%%，有效样本 n=%d",
        threshold, float(labels[wellbeing.notna()].mean()) * 100, valid_n,
    )
    return labels


# ===========================================================================
# 特征准备
# ===========================================================================

def _resolve_chronic_col(df: pd.DataFrame) -> str | None:
    """动态选择慢性病变量（按优先级，要求覆盖率 ≥ COVERAGE_THRESHOLD）。"""
    for col in CHRONIC_FALLBACKS:
        if col in df.columns:
            cov = 1.0 - clean_negative(df[col]).isna().mean()
            if cov >= COVERAGE_THRESHOLD:
                logger.info("慢性病变量：%s（覆盖率 %.1f%%）", col, cov * 100)
                return col
            logger.warning("慢性病变量 %s 覆盖率 %.1f%% < %.0f%%，跳过",
                           col, cov * 100, COVERAGE_THRESHOLD * 100)
    logger.warning("未找到可用慢性病变量，chronic_disease 将被排除")
    return None


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    将 CFPS 原始列映射到模型特征名称，过滤不存在或覆盖率不足的列。
    返回 (feature_df, numeric_cols, categorical_cols)。
    """
    col_map = dict(COLUMN_MAP)

    # 动态解析慢性病列
    chronic_col = _resolve_chronic_col(df)
    if chronic_col is None:
        col_map.pop("chronic_disease", None)
        cat_feats = [c for c in CATEGORICAL_FEATURES if c != "chronic_disease"]
    else:
        col_map["chronic_disease"] = chronic_col
        cat_feats = list(CATEGORICAL_FEATURES)

    num_feats = list(NUMERIC_FEATURES)
    all_feats = num_feats + cat_feats

    feat_dict: dict[str, pd.Series] = {}
    for feat in all_feats:
        cfps_col = col_map.get(feat)
        if cfps_col is None or cfps_col not in df.columns:
            logger.warning("特征 '%s'（CFPS列：%s）不存在，已排除", feat, cfps_col)
            continue
        s = clean_negative(df[cfps_col])
        cov = 1.0 - s.isna().mean()
        if cov < COVERAGE_THRESHOLD:
            logger.warning(
                "特征 '%s' 覆盖率 %.1f%% < %.0f%%，已排除",
                feat, cov * 100, COVERAGE_THRESHOLD * 100,
            )
            continue
        feat_dict[feat] = s
        logger.info("特征 '%s'（%s）：覆盖率 %.1f%%", feat, cfps_col, cov * 100)

    final_num = [f for f in num_feats if f in feat_dict]
    final_cat = [f for f in cat_feats if f in feat_dict]
    logger.info(
        "有效特征：数值 %d 个 %s，类别 %d 个 %s",
        len(final_num), final_num, len(final_cat), final_cat,
    )
    return pd.DataFrame(feat_dict), final_num, final_cat


# ===========================================================================
# 预处理 Pipeline
# ===========================================================================

def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    构建 ColumnTransformer（fit 仅在训练集上调用，确保无数据泄漏）。

    scale_numeric=True  → 数值列：中位数填补 + StandardScaler（用于 LR）
    scale_numeric=False → 数值列：中位数填补，不缩放（用于 XGBoost + SHAP）
    类别列均使用众数填补 + OneHotEncoder。
    """
    if scale_numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        [
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


# ===========================================================================
# 评估函数
# ===========================================================================

def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Youden's J 统计量（TPR − FPR 最大化）在 ROC 曲线上求最佳阈值。"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[best_idx])


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict:
    """
    计算并返回完整评估指标字典。
    加权 AUC/PR-AUC：通过 sklearn 的 sample_weight 参数实现。
    若 sample_weight 为 None，则加权指标填 "N/A"。
    """
    roc_uw = roc_auc_score(y_true, y_prob)
    prc_uw = average_precision_score(y_true, y_prob)
    bri_uw = brier_score_loss(y_true, y_prob)

    if sample_weight is not None:
        roc_w = roc_auc_score(y_true, y_prob, sample_weight=sample_weight)
        prc_w = average_precision_score(y_true, y_prob, sample_weight=sample_weight)
        bri_w = brier_score_loss(y_true, y_prob, sample_weight=sample_weight)
    else:
        roc_w = prc_w = bri_w = np.nan

    thresh = _find_best_threshold(y_true, y_prob)
    y_pred = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    def _fmt(v: float) -> object:
        return round(v, 4) if not np.isnan(v) else "N/A"

    metrics = {
        "model": name,
        "roc_auc_unweighted": round(roc_uw, 4),
        "pr_auc_unweighted": round(prc_uw, 4),
        "brier_unweighted": round(bri_uw, 4),
        "roc_auc_weighted": _fmt(roc_w),
        "pr_auc_weighted": _fmt(prc_w),
        "brier_weighted": _fmt(bri_w),
        "best_threshold": round(thresh, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }

    logger.info("=== %s（测试集）===", name)
    logger.info(
        "  ROC-AUC    未加权/加权：%.4f / %s", roc_uw,
        f"{roc_w:.4f}" if not np.isnan(roc_w) else "N/A",
    )
    logger.info(
        "  PR-AUC     未加权/加权：%.4f / %s", prc_uw,
        f"{prc_w:.4f}" if not np.isnan(prc_w) else "N/A",
    )
    logger.info(
        "  Brier      未加权/加权：%.4f / %s", bri_uw,
        f"{bri_w:.4f}" if not np.isnan(bri_w) else "N/A",
    )
    logger.info(
        "  最佳阈值=%.4f | Sensitivity=%.4f | Specificity=%.4f | F1=%.4f",
        thresh, sensitivity, specificity, f1,
    )
    logger.info("  混淆矩阵：TP=%d FP=%d TN=%d FN=%d", tp, fp, tn, fn)
    return metrics


# ===========================================================================
# 绘图函数
# ===========================================================================

def plot_roc_prc(
    y_true: np.ndarray,
    models: dict[str, np.ndarray],
    output_path: str,
) -> None:
    """绘制 ROC 曲线与 PR 曲线（多模型对比）。"""
    fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=(14, 6))

    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax_roc.set_xlabel("假阳性率（FPR）")
    ax_roc.set_ylabel("真阳性率（TPR）")
    ax_roc.set_title("ROC 曲线")

    baseline_prev = float(y_true.mean())
    ax_prc.axhline(y=baseline_prev, color="k", linestyle="--", lw=0.8,
                   label=f"基线（患病率={baseline_prev:.2f}）")
    ax_prc.set_xlabel("召回率（Recall）")
    ax_prc.set_ylabel("精确率（Precision）")
    ax_prc.set_title("PR 曲线")

    for name, y_prob in models.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_val:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_val = average_precision_score(y_true, y_prob)
        ax_prc.plot(recall, precision, lw=2, label=f"{name} (AP={ap_val:.3f})")

    ax_roc.legend(loc="lower right")
    ax_prc.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("ROC/PRC 图已保存：%s", output_path)
    plt.close()


def plot_calibration(
    y_true: np.ndarray,
    models: dict[str, np.ndarray],
    output_path: str,
    n_bins: int = 10,
) -> None:
    """绘制校准曲线（多模型对比）。"""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="完美校准")

    for name, y_prob in models.items():
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        ax.plot(mean_pred, frac_pos, "s-", lw=2, label=name)

    ax.set_xlabel("平均预测概率")
    ax.set_ylabel("实际正例比例")
    ax.set_title("校准曲线（Calibration Curve）")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("校准曲线已保存：%s", output_path)
    plt.close()


# ===========================================================================
# SHAP 分析
# ===========================================================================

def run_shap_analysis(
    xgb_pipeline: "Pipeline",
    X_test: pd.DataFrame,
    feature_names: list[str],
    bmi_feature_name: str | None = None,
    age_feature_name: str | None = None,
) -> None:
    """
    SHAP 全局重要性条形图（替代 Gini）+ BMI dependence plot（年龄叠加）。
    XGBoost Pipeline 使用未缩放预处理器，SHAP 值可在原始特征尺度下解读。
    """
    if not _HAS_SHAP:
        logger.warning("shap 未安装，跳过 SHAP 分析")
        return

    logger.info("开始 SHAP 分析...")
    preprocessor = xgb_pipeline.named_steps["preprocessor"]
    xgb_model = xgb_pipeline.named_steps["xgb"]
    X_test_transformed = preprocessor.transform(X_test)

    # TreeExplainer（高效，专为树模型设计）
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_transformed)

    # 兼容旧版 shap API：二分类可能返回 list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # -----------------------------------------------------------------------
    # 全局重要性（mean |SHAP|）条形图
    # -----------------------------------------------------------------------
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    imp_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    logger.info("SHAP 全局重要性（Top 10）：\n%s",
                imp_df.head(10).to_string(index=False))

    top_n = min(20, len(imp_df))
    top_df = imp_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    ax.barh(top_df["feature"].iloc[::-1].values,
            top_df["mean_abs_shap"].iloc[::-1].values, color="#2ecc71")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"XGBoost SHAP 全局特征重要性（Top {top_n}）")
    plt.tight_layout()
    plt.savefig(OUT_SHAP_IMP, dpi=150, bbox_inches="tight")
    logger.info("SHAP 重要性图已保存：%s", OUT_SHAP_IMP)
    plt.close()

    # -----------------------------------------------------------------------
    # BMI dependence plot（叠加年龄作为 interaction）
    # -----------------------------------------------------------------------
    if bmi_feature_name is None or bmi_feature_name not in feature_names:
        logger.warning("BMI 特征 '%s' 未在特征列表中找到，跳过 dependence plot",
                       bmi_feature_name)
        return

    bmi_idx = feature_names.index(bmi_feature_name)
    interaction_idx = (
        feature_names.index(age_feature_name)
        if age_feature_name and age_feature_name in feature_names
        else "auto"
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.dependence_plot(
        ind=bmi_idx,
        shap_values=shap_values,
        features=X_test_transformed,
        feature_names=feature_names,
        interaction_index=interaction_idx,
        ax=ax,
        show=False,
    )
    ax.set_title("BMI 的 SHAP Dependence Plot（颜色 = 年龄）")
    plt.tight_layout()
    plt.savefig(OUT_SHAP_DEP, dpi=150, bbox_inches="tight")
    logger.info("BMI dependence plot 已保存：%s", OUT_SHAP_DEP)
    plt.close()

    # -----------------------------------------------------------------------
    # BMI 非线性 / U 形关联自动文字总结
    # -----------------------------------------------------------------------
    _summarize_bmi_shap(X_test_transformed[:, bmi_idx], shap_values[:, bmi_idx])


def _summarize_bmi_shap(bmi_vals: np.ndarray, bmi_shap: np.ndarray) -> None:
    """
    自动分析 BMI 与 SHAP 值的关联形态：
    将 BMI 按三分位数分为低/中/高三段，判断 U 形、单调或复杂模式，
    输出文字总结（仅描述模型统计关联，不做因果表述）。
    """
    mask = ~np.isnan(bmi_vals) & ~np.isnan(bmi_shap)
    bmi_clean = bmi_vals[mask]
    shap_clean = bmi_shap[mask]

    if len(bmi_clean) < 30:
        logger.warning("BMI 有效样本不足（n=%d），跳过文字总结", len(bmi_clean))
        return

    p33 = float(np.percentile(bmi_clean, 33))
    p67 = float(np.percentile(bmi_clean, 67))

    low_shap = shap_clean[bmi_clean < p33].mean()
    mid_shap = shap_clean[(bmi_clean >= p33) & (bmi_clean < p67)].mean()
    high_shap = shap_clean[bmi_clean >= p67].mean()

    def _dir(v: float) -> str:
        if v > 0.01:
            return "推高风险（SHAP > 0）"
        if v < -0.01:
            return "降低风险（SHAP < 0）"
        return "中性（SHAP ≈ 0）"

    vals = [low_shap, mid_shap, high_shap]
    is_u = vals[1] < vals[0] and vals[1] < vals[2]
    is_inv_u = vals[1] > vals[0] and vals[1] > vals[2]
    is_up = vals[0] < vals[1] < vals[2]
    is_down = vals[0] > vals[1] > vals[2]

    lines = [
        "=" * 62,
        "【BMI 与 high_risk 的 SHAP 关联自动总结】",
        "（注：以下描述仅为模型统计关联，不代表因果关系）",
        f"  低 BMI 段（< {p33:.1f}）   ：均值 SHAP = {low_shap:+.4f}  → {_dir(low_shap)}",
        f"  中 BMI 段（{p33:.1f}–{p67:.1f}）：均值 SHAP = {mid_shap:+.4f}  → {_dir(mid_shap)}",
        f"  高 BMI 段（≥ {p67:.1f}）   ：均值 SHAP = {high_shap:+.4f}  → {_dir(high_shap)}",
    ]

    if is_u:
        lines.append(
            f"  → 呈 U 形关联：中段 BMI（{p33:.1f}–{p67:.1f}）与高风险关联最低；"
            f"低段（< {p33:.1f}）和高段（≥ {p67:.1f}）均与更高风险相关。"
        )
    elif is_inv_u:
        lines.append(
            f"  → 呈倒 U 形关联：中段 BMI（{p33:.1f}–{p67:.1f}）与高风险关联最强。"
        )
    elif is_up:
        lines.append("  → 单调上升关联：BMI 越高，模型预测高风险倾向越大。")
    elif is_down:
        lines.append("  → 单调下降关联：BMI 越高，模型预测高风险倾向越小。")
    else:
        lines.append("  → 无明显单调或 U 形特征，关联模式较为复杂。")

    lines.append("=" * 62)
    print("\n" + "\n".join(lines) + "\n")


# ===========================================================================
# 主流程
# ===========================================================================

def main() -> None:
    # ------------------------------------------------------------------
    # 1. 加载数据，构建目标变量 high_risk
    # ------------------------------------------------------------------
    df, _meta = load_data(DATA_FILE)
    df["high_risk"] = build_high_risk(df)

    # ------------------------------------------------------------------
    # 2. 准备特征矩阵（过滤缺失/低覆盖率列）
    # ------------------------------------------------------------------
    feature_df, numeric_cols, categorical_cols = prepare_features(df)
    if not numeric_cols and not categorical_cols:
        logger.error("无有效特征，退出")
        sys.exit(1)

    # 对齐行：将 high_risk 和权重合并进来，过滤 high_risk 为 NaN 的行
    work_df = feature_df.copy()
    work_df["high_risk"] = df["high_risk"].values

    weight_col = COLUMN_MAP["weight"]
    if weight_col in df.columns:
        w_raw = clean_negative(df[weight_col]).fillna(1.0)
        work_df["__weight__"] = w_raw.values
        logger.info("抽样权重：使用 %s", weight_col)
    else:
        logger.warning("权重变量 %s 不存在，使用等权（1.0）", weight_col)
        work_df["__weight__"] = 1.0

    work_df = work_df[work_df["high_risk"].notna()].copy()
    work_df["high_risk"] = work_df["high_risk"].astype(int)
    logger.info("有效分析行数：%d", len(work_df))

    X = work_df[numeric_cols + categorical_cols]
    y = work_df["high_risk"].values
    w = work_df["__weight__"].values

    # ------------------------------------------------------------------
    # 3. 训练/测试集划分（70/30，stratify，random_state=42）
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info("训练集 n=%d，测试集 n=%d", len(y_train), len(y_test))

    print("\n" + "=" * 62)
    print("【数据划分 & 类别比例验证】（Preprocessor 仅在 train 上 fit）")
    print(f"  训练集 high_risk=1 比例：{y_train.mean():.4f}")
    print(f"  测试集  high_risk=1 比例：{y_test.mean():.4f}")
    print("=" * 62 + "\n")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_metrics: list[dict] = []
    all_probs: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 4. 模型 A：加权逻辑回归
    #    - 不使用 class_weight（class_weight=None）
    #    - 通过 lr__sample_weight 传入抽样权重
    # ------------------------------------------------------------------
    logger.info("训练加权逻辑回归（Model A）...")
    lr_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(numeric_cols, categorical_cols,
                                            scale_numeric=True)),
        ("lr", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
            class_weight=None,  # 不使用 class_weight，改用 sample_weight
        )),
    ])
    # fit 仅在训练集上执行（含 sample_weight）
    lr_pipeline.fit(X_train, y_train, lr__sample_weight=w_train)
    lr_probs_test = lr_pipeline.predict_proba(X_test)[:, 1]

    # 5 折交叉验证 AUC（含 sample_weight，sklearn >= 1.4 params= API）
    lr_cv_auc = cross_val_score(
        lr_pipeline, X_train, y_train,
        cv=cv, scoring="roc_auc",
        params={"lr__sample_weight": w_train},
    )
    logger.info("LR 5-fold CV AUC（加权）: %.4f ± %.4f",
                lr_cv_auc.mean(), lr_cv_auc.std())

    lr_metrics = evaluate_model(
        "LogisticRegression", y_test, lr_probs_test, sample_weight=w_test
    )
    all_metrics.append(lr_metrics)
    all_probs["LogisticRegression"] = lr_probs_test

    # ------------------------------------------------------------------
    # 5. 模型 B：XGBoost + 5 折 RandomizedSearchCV 调参 + sample_weight
    #
    #    已知限制：RandomizedSearchCV 内部 CV 评估使用未加权 AUC；
    #    最终模型拟合（refit）使用完整 sample_weight，加权 CV AUC
    #    通过 cross_val_score + params= 单独报告。
    # ------------------------------------------------------------------
    xgb_pipeline = None
    xgb_probs_test = None

    if not _HAS_XGB:
        logger.warning("xgboost 未安装，跳过模型 B")
    else:
        logger.info("XGBoost 5-fold RandomizedSearchCV 调参（Model B）...")

        # XGBoost 使用不缩放的预处理器（保留原始特征尺度，便于 SHAP 解读）
        xgb_base_pipeline = Pipeline([
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols,
                                                scale_numeric=False)),
            ("xgb", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )),
        ])

        param_distributions = {
            "xgb__max_depth": randint(3, 9),
            "xgb__learning_rate": uniform(0.01, 0.29),
            "xgb__n_estimators": randint(100, 400),
            "xgb__subsample": uniform(0.6, 0.4),
            "xgb__colsample_bytree": uniform(0.6, 0.4),
        }

        # refit=False：手动用最佳参数 + sample_weight 重新拟合，避免 CV 内泄漏
        search = RandomizedSearchCV(
            xgb_base_pipeline,
            param_distributions,
            n_iter=30,
            cv=cv,
            scoring="roc_auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=False,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            search.fit(X_train, y_train)

        best_params = search.best_params_
        logger.info("XGBoost 最佳参数：%s", best_params)
        logger.info("XGBoost CV best AUC（未加权）: %.4f", search.best_score_)

        # 以最佳参数重建并拟合（含 sample_weight），确保最终模型加权训练
        xgb_pipeline = Pipeline([
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols,
                                                scale_numeric=False)),
            ("xgb", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
                **{k.replace("xgb__", ""): v for k, v in best_params.items()},
            )),
        ])
        xgb_pipeline.fit(X_train, y_train, xgb__sample_weight=w_train)
        xgb_probs_test = xgb_pipeline.predict_proba(X_test)[:, 1]

        # 加权 CV AUC（独立评估，供参考）
        xgb_cv_auc = cross_val_score(
            xgb_pipeline, X_train, y_train,
            cv=cv, scoring="roc_auc",
            params={"xgb__sample_weight": w_train},
        )
        logger.info("XGBoost 5-fold CV AUC（加权）: %.4f ± %.4f",
                    xgb_cv_auc.mean(), xgb_cv_auc.std())

        xgb_metrics = evaluate_model(
            "XGBoost", y_test, xgb_probs_test, sample_weight=w_test
        )
        all_metrics.append(xgb_metrics)
        all_probs["XGBoost"] = xgb_probs_test

    # ------------------------------------------------------------------
    # 6. 绘图：ROC/PRC + 校准曲线
    # ------------------------------------------------------------------
    plot_roc_prc(y_test, all_probs, OUT_ROC_PNG)
    plot_calibration(y_test, all_probs, OUT_CALIB_PNG)

    # ------------------------------------------------------------------
    # 7. SHAP 分析（仅 XGBoost）
    # ------------------------------------------------------------------
    if xgb_pipeline is not None and _HAS_SHAP:
        feature_names_out = list(
            xgb_pipeline.named_steps["preprocessor"].get_feature_names_out()
        )
        bmi_name = next(
            (n for n in feature_names_out if "bmi" in n.lower()), None
        )
        age_name = next(
            (n for n in feature_names_out if n.lower().endswith("age") or
             "num__age" in n.lower()), None
        )
        run_shap_analysis(
            xgb_pipeline, X_test, feature_names_out,
            bmi_feature_name=bmi_name,
            age_feature_name=age_name,
        )

    # ------------------------------------------------------------------
    # 8. 保存 CSV 结果
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    logger.info("结果已保存：%s", OUT_CSV)

    # ------------------------------------------------------------------
    # 9. 控制台摘要
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("最终评估摘要（测试集）")
    print("=" * 62)
    for m in all_metrics:
        print(f"\n  【{m['model']}】")
        print(f"    ROC-AUC   ：{m['roc_auc_unweighted']} (未加权) / "
              f"{m['roc_auc_weighted']} (加权)")
        print(f"    PR-AUC    ：{m['pr_auc_unweighted']} (未加权) / "
              f"{m['pr_auc_weighted']} (加权)")
        print(f"    Brier     ：{m['brier_unweighted']} (未加权) / "
              f"{m['brier_weighted']} (加权)")
        print(f"    最佳阈值  ：{m['best_threshold']}")
        print(f"    Sensitivity：{m['sensitivity']}  "
              f"Specificity：{m['specificity']}  F1：{m['f1']}")
        print(f"    混淆矩阵  ：TP={m['TP']}  FP={m['FP']}  "
              f"TN={m['TN']}  FN={m['FN']}")
    print("\n" + "=" * 62)


if __name__ == "__main__":
    main()
