from __future__ import annotations

"""
cfps_depression_analysis.py
----------------------------
基于 CFPS 2022 儿童代答问卷的儿童社会情绪发展得分顶刊级分析脚本。
对标 The Lancet Digital Health 方法学规范，实现：
  - 多模型 Benchmarking（RF / XGBoost / SVR / MLP）
  - Stacking 集成（RidgeCV 元学习器）
  - 5 折交叉验证（KFold），报告 *R*² / RMSE / MAE
  - SHAP Beeswarm + Dependence Plot（BMI × 年龄交互）
  - 模型 *R*² 分布箱线图
  - 自动检测输出路径（桌面 → 工作目录）
  - 导出 socioemotional_pro_analysis.csv（预测得分 + SHAP 贡献值）

背景说明：
  CFPS 2022 we3xx 条目（we301–we312）测量儿童社会情绪发展中的积极行为
  （乐观、自我调节、同伴关系、亲社会行为等），采用 1–5 点 Likert 正向计分。
  高分代表更高的社会情绪发展水平；所有模型以行均值作为连续结局变量 Y。

运行方法：
    python cfps_depression_analysis.py
"""

from __future__ import annotations

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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
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
# 中文字体注册（优先 WenQuanYi Zen Hei，其次 Noto Sans CJK SC）
# ===========================================================================
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

# 全局绘图参数（出版级）
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.size": 11,
})

# ===========================================================================
# 全局配置
# ===========================================================================
DATA_FILE = "cfps2022childproxy_202410.dta"
RANDOM_STATE = 42
CV_FOLDS = 5
COVERAGE_THRESHOLD = 0.40      # 变量有效覆盖率阈值


# 社会情绪发展得分条目（we3xx，1–5 点 Likert 正向计分）
EMOTION_ITEMS = [f"we3{str(i).zfill(2)}" for i in range(1, 13)]
REVERSED_ITEMS: list[str] = []   # 当前所有 we3xx 条目均为正向

# ---------------------------------------------------------------------------
# 候选预测因子映射（显示标签 → CFPS 列名）
# 问题陈述要求：年龄、性别、城乡、民族、语文/数学成绩、BMI、近期生病及就医情况
# ---------------------------------------------------------------------------
FACTOR_MAP: dict[str, str] = {
    "年龄":         "age",
    "性别":         "gender",
    "城乡":         "urban22",
    "民族":         "minzu",
    "语文成绩":     "wf501",
    "数学成绩":     "wf502",
    "BMI":          "bmi",
    "近期生病":     "wc0",
    "就医情况":     "wc4_1",
    # 慢性病/健康代理：qp4001 优先，动态回退
    "慢性病诊断":   "qp4001",
}

# 权重候选变量（按优先级）
WEIGHT_CANDIDATES = ["child_weight", "rswt_natcs22n", "rswt_natpn1022n"]

# ===========================================================================
# 输出路径自动检测（桌面 → 工作目录）
# ===========================================================================
def _resolve_output_dir() -> Path:
    """优先输出至桌面，否则使用当前工作目录。"""
    desk = Path.home() / "Desktop"
    if desk.exists() and desk.is_dir():
        return desk
    return Path.cwd()

OUT_DIR = _resolve_output_dir()
OUT_CSV_PRO   = str(OUT_DIR / "socioemotional_pro_analysis.csv")
OUT_PNG_BOX   = str(OUT_DIR / "socioemotional_r2_boxplot.png")
OUT_PNG_SHAP_BEE  = str(OUT_DIR / "socioemotional_shap_beeswarm.png")
OUT_PNG_SHAP_DEP  = str(OUT_DIR / "socioemotional_shap_bmi_dependence.png")
OUT_PNG_CORR  = str(OUT_DIR / "socioemotional_spearman_corr.png")

# ===========================================================================
# 日志配置
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 工具函数
# ===========================================================================

def clean_negative_codes(
    series: pd.Series,

    if codes is None:
        codes = NEGATIVE_CODES
    return series.replace({c: np.nan for c in codes})


def compute_coverage(series: pd.Series) -> float:
    """返回去除负值编码后的有效覆盖率（0~1）。"""
    return 1.0 - clean_negative_codes(series, codes=EXTENDED_NEGATIVE_CODES).isna().mean()


# ===========================================================================
# 数据加载
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


# ===========================================================================
# 变量动态解析
# ===========================================================================

def resolve_chronic_disease_var(df: pd.DataFrame) -> tuple[str | None, str]:
    """动态确定慢性病/健康状态代理变量（qp4001 > wc4_1 > wc0 > ill）。"""
    candidates = [
        ("qp4001", "过去半年医生诊断慢性病（首选）"),
        ("wc4_1",  "过去12个月是否因病就医（替代）"),
        ("wc0",    "过去一月是否生病（替代）"),
        ("ill",    "出生以来最严重疾病（替代）"),
    ]
    for col, desc in candidates:
        if col in df.columns:
            cov = compute_coverage(df[col])
            if cov >= COVERAGE_THRESHOLD:
                logger.info("慢性病变量：%s（%s，覆盖率 %.1f%%）", col, desc, cov * 100)
                return col, desc
            logger.warning("慢性病候选 %s 覆盖率 %.1f%% < %.0f%%，跳过",
                           col, cov * 100, COVERAGE_THRESHOLD * 100)
    logger.warning("未找到可用慢性病变量，该因子将被排除")
    return None, "无可用变量"


def resolve_weight_var(df: pd.DataFrame) -> str | None:
    """按优先级检测可用抽样权重变量。"""
    for col in WEIGHT_CANDIDATES:
        if col in df.columns:
            cov = compute_coverage(df[col])
            if cov >= COVERAGE_THRESHOLD:
                logger.info("权重变量：%s（覆盖率 %.1f%%）", col, cov * 100)
                return col
            logger.warning("权重变量 %s 覆盖率 %.1f%% 过低，跳过", col, cov * 100)
    logger.warning("未找到可用权重变量，将使用未加权分析")
    return None


# ===========================================================================
# 结局变量构建（社会情绪发展得分，连续变量 Y）
# ===========================================================================

def build_socioemotional_score(df: pd.DataFrame) -> pd.Series:
    """
    基于 we3xx 条目构建儿童社会情绪发展得分（连续变量，均值 1–5 分）。
    - 将负值编码替换为 NaN
    - 对逆向计分条目取反（6 – score）
    - 计算行均值作为结局变量 *Y*（高分 = 更高社会情绪发展水平）
    """
    available = [c for c in EMOTION_ITEMS if c in df.columns]
    logger.info("社会情绪发展条目（%d 个）：%s", len(available), available)

    score_df = df[available].copy()
    for col in available:
        score_df[col] = clean_negative_codes(score_df[col], codes=EXTENDED_NEGATIVE_CODES)
        if col in REVERSED_ITEMS:
            score_df[col] = 6 - score_df[col]

    socioemotional = score_df.mean(axis=1)
    valid_mask = socioemotional.notna()
    nan_count = (~valid_mask).sum()

    if nan_count > 0:
        logger.warning("we3xx 全部缺失行：%d（%.1f%%），将从训练集排除",
                       nan_count, nan_count / len(df) * 100)

    partial_count = int((score_df[valid_mask].notna().sum(axis=1) < len(available)).sum())
    if partial_count > 0:
        logger.info("we3xx 部分缺失行：%d（%.1f%%），使用可用条目均值计算",
                    partial_count, partial_count / len(df) * 100)

    logger.info("社会情绪发展得分：均值=%.3f，SD=%.3f，n=%d",
                socioemotional[valid_mask].mean(),
                socioemotional[valid_mask].std(),
                valid_mask.sum())
    return socioemotional


# ===========================================================================
# 自适应变量过滤
# ===========================================================================

def filter_factors(
    df: pd.DataFrame,
    factor_map: dict[str, str],
) -> tuple[dict[str, str], list[dict]]:
    """过滤覆盖率低于阈值的预测因子，返回 (有效映射, 覆盖率明细)。"""
    valid_map: dict[str, str] = {}
    records: list[dict] = []

    for label, col in factor_map.items():
        if col not in df.columns:
            records.append({"label": label, "variable": col,
                            "coverage": 0.0, "status": "不存在"})
            logger.warning("变量不存在：%s (%s)", col, label)
            continue
        cov = compute_coverage(df[col])
        status = "保留" if cov >= COVERAGE_THRESHOLD else "剔除"
        records.append({"label": label, "variable": col,
                        "coverage": round(cov, 4), "status": status})
        if cov >= COVERAGE_THRESHOLD:
            valid_map[label] = col
        else:
            logger.warning("变量 %s（%s）覆盖率 %.1f%% < %.0f%%，剔除",
                           col, label, cov * 100, COVERAGE_THRESHOLD * 100)

    logger.info("有效预测因子：%d / %d", len(valid_map), len(factor_map))
    return valid_map, records


# ===========================================================================
# 特征矩阵构建（带 StandardScaler + SimpleImputer Pipeline）
# ===========================================================================

def build_feature_matrix(
    df: pd.DataFrame,
    valid_map: dict[str, str],
    outcome_col: str,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Index]:
    """
    构建清洗后的特征矩阵 X 和结局向量 y。

    """
    feature_labels = list(valid_map.keys())

    X_raw = pd.DataFrame(index=df.index)
    for label, col in valid_map.items():

        X_raw[label] = clean_negative_codes(df[col], codes=codes)

    y = df[outcome_col]
    mask = y.notna()

    X_clean = X_raw[mask].reset_index(drop=True)
    y_clean = y[mask].reset_index(drop=True)

    logger.info("特征矩阵：%d 行 × %d 列，特征：%s",
                len(y_clean), len(feature_labels), feature_labels)



# ===========================================================================
# 构建单模型 Pipeline（StandardScaler + SimpleImputer）
# ===========================================================================

def make_pipeline(estimator: object) -> Pipeline:
    """封装 SimpleImputer → StandardScaler → 模型 的标准 Pipeline。"""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   estimator),
    ])


# ===========================================================================
# 模型定义
# ===========================================================================

def get_base_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """返回各基础模型 Pipeline 字典（均含 Imputer + Scaler）。"""
    models: dict[str, Pipeline] = {
        "随机森林 (RF)": make_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        "SVR": make_pipeline(
            SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
        ),
        "MLP 神经网络": make_pipeline(
            MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                max_iter=500,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1,
            )
        ),
    }
    if _HAS_XGB:
        models["XGBoost"] = make_pipeline(
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
        )
    return models


def get_stacking_model(
    base_models: dict[str, Pipeline],
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """
    构建 Stacking 集成模型（RidgeCV 元学习器）。
    base_estimators 为已包含 Pipeline 的基模型；
    StackingRegressor 对输入额外封装一层 Imputer+Scaler（cv=5）。
    """
    estimators = [(name, pipeline) for name, pipeline in base_models.items()]
    stacker = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    # Stacking 本身不需要额外缩放（基模型 Pipeline 已各自缩放）
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("stacking", stacker),
    ])


# ===========================================================================
# 5 折交叉验证
# ===========================================================================

def run_cv(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    cv: KFold,
) -> dict[str, dict[str, np.ndarray]]:
    """
    对所有模型执行 5 折 CV，记录每折的 R²、RMSE、MAE。
    返回 {model_name: {"r2": arr, "rmse": arr, "mae": arr}}。
    """
    results: dict[str, dict[str, np.ndarray]] = {}
    for name, pipeline in models.items():
        logger.info("CV 评估模型：%s", name)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores = cross_validate(
                pipeline, X, y, cv=cv,
                scoring=["r2", "neg_mean_squared_error", "neg_mean_absolute_error"],
                n_jobs=-1,
            )
        r2_scores = scores["test_r2"]

        rmse_scores = np.sqrt(-neg_mse)
        mae_scores  = -neg_mae
        results[name] = {
            "r2":   r2_scores,
            "rmse": rmse_scores,
            "mae":  mae_scores,
        }
        logger.info(
            "  *R*²=%.4f±%.4f | RMSE=%.4f±%.4f | MAE=%.4f±%.4f",
            r2_scores.mean(), r2_scores.std(),
            rmse_scores.mean(), rmse_scores.std(),
            mae_scores.mean(), mae_scores.std(),
        )
    return results


# ===========================================================================
# Spearman 相关性分析
# ===========================================================================

def spearman_analysis(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """计算各特征与结局变量的 Spearman 相关系数（含 p 值）。"""
    rows = []
    for feat in X.columns:
        x_col = X[feat]
        mask = x_col.notna() & y.notna()
        if mask.sum() < 10:
            continue
        rho, pval = stats.spearmanr(x_col[mask], y[mask])
        rows.append({
            "特征": feat,
            "Spearman_ρ": round(float(rho), 4),
            "p 值": round(float(pval), 6),
            "有效 n": int(mask.sum()),
        })
    corr_df = pd.DataFrame(rows)
    corr_df = corr_df.sort_values("Spearman_ρ", key=abs, ascending=False)
    return corr_df.reset_index(drop=True)


# ===========================================================================
# SHAP 分析（XGBoost 优先；回退至随机森林）
# ===========================================================================

def run_shap_analysis(
    model_pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_labels: list[str],
    model_name: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    SHAP 全局 Beeswarm + BMI Dependence Plot（叠加年龄交互）。
    返回 (shap_values, X_test_transformed) 供后续导出。
    """
    if not _HAS_SHAP:
        logger.warning("shap 未安装，跳过 SHAP 分析")
        return None, None

    logger.info("开始 SHAP 分析（%s）...", model_name)

    # 提取 Pipeline 中 imputer+scaler 变换后的矩阵
    # 防御性检查：仅支持含 'imputer'/'scaler'/'model' 三步的 Pipeline（基础模型）
    steps = model_pipeline.named_steps
    if "imputer" not in steps or "scaler" not in steps or "model" not in steps:
        logger.warning(
            "SHAP 分析仅支持含 imputer/scaler/model 三步的 Pipeline，"
            "当前 Pipeline（%s）不符合要求，跳过 SHAP",
            model_name,
        )
        return None, None

    imputer = steps["imputer"]
    scaler  = steps["scaler"]
    model   = steps["model"]

    X_test_imp  = imputer.transform(X_test)
    X_test_trans = scaler.transform(X_test_imp)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_trans)
    except Exception:
        try:
            explainer   = shap.Explainer(model, shap.maskers.Independent(
                X_test_trans, max_samples=200))
            sv = explainer(X_test_trans)
            shap_values = sv.values
        except Exception as exc:
            logger.warning("SHAP 计算失败：%s，跳过 SHAP 输出", exc)
            return None, None

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # -----------------------------------------------------------------------
    # Beeswarm Plot
    # -----------------------------------------------------------------------
    try:
        shap.summary_plot(
            shap_values,
            X_test_trans,
            feature_names=feature_labels,
            show=False,
            plot_type="dot",
        )
        fig = plt.gcf()

        plt.tight_layout()
        fig.savefig(OUT_PNG_SHAP_BEE, dpi=300, bbox_inches="tight")
        logger.info("SHAP Beeswarm 图已保存：%s", OUT_PNG_SHAP_BEE)
        plt.close(fig)
    except Exception as exc:
        logger.warning("SHAP Beeswarm 绘制失败：%s", exc)
        plt.close("all")

    # -----------------------------------------------------------------------
    # BMI Dependence Plot（叠加年龄交互）
    # -----------------------------------------------------------------------
    bmi_idx = next(
        (i for i, lbl in enumerate(feature_labels) if "bmi" in lbl.lower()),
        None,
    )
    age_idx = next(
        (i for i, lbl in enumerate(feature_labels) if "年龄" in lbl or "age" in lbl.lower()),
        None,
    )

    if bmi_idx is not None:
        try:
            fig, ax = plt.subplots(figsize=(9, 6))
            shap.dependence_plot(
                ind=bmi_idx,
                shap_values=shap_values,
                features=X_test_trans,
                feature_names=feature_labels,
                interaction_index=age_idx if age_idx is not None else "auto",
                ax=ax,
                show=False,
            )
            ax.set_title(
                "BMI 对社会情绪得分的 SHAP Dependence Plot\n"
                r"（颜色 = 年龄 $age$，展示 BMI $\times$ 年龄交互作用）",
                fontsize=12,
            )
            ax.set_xlabel(r"BMI（$kg/m^2$）", fontsize=11)
            ax.set_ylabel(r"SHAP 值（对 $Y$ 的贡献）", fontsize=11)
            plt.tight_layout()
            plt.savefig(OUT_PNG_SHAP_DEP, dpi=300, bbox_inches="tight")
            logger.info("SHAP Dependence Plot 已保存：%s", OUT_PNG_SHAP_DEP)
            plt.close()
        except Exception as exc:
            logger.warning("SHAP Dependence Plot 绘制失败：%s", exc)
            plt.close("all")
    else:
        logger.warning("未找到 BMI 特征，跳过 Dependence Plot")

    return shap_values, X_test_trans


# ===========================================================================
# 可视化：模型 R² 分布箱线图
# ===========================================================================

def plot_r2_boxplot(
    cv_results: dict[str, dict[str, np.ndarray]],
) -> None:
    """
    绘制各模型 5 折 *R*² 分布箱线图（出版级）。
    图注说明使用 LaTeX 格式的统计量符号。
    """
    model_names = list(cv_results.keys())
    r2_data = [cv_results[n]["r2"] for n in model_names]

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.8), 6))
    bp = ax.boxplot(
        r2_data,
        patch_artist=True,
        notch=False,
        widths=0.5,
        medianprops={"color": "#e74c3c", "linewidth": 2.5},
        boxprops={"linewidth": 1.5},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 5, "alpha": 0.5},
    )


    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # 均值点标注
    for i, r2_arr in enumerate(r2_data, start=1):
        mu = r2_arr.mean()
        ax.scatter(i, mu, color="white", edgecolors="black",
                   zorder=5, s=60, linewidths=1.5, label=None)
        ax.text(i, mu + 0.01, f"{mu:.3f}", ha="center", va="bottom",
                fontsize=9, color="black")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(range(1, len(model_names) + 1))
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel(r"$R^2$（5 折交叉验证）", fontsize=12)
    ax.set_title(
        r"各模型预测性能对比（$R^2$ 分布，5-fold CV）",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(bottom=min(-0.1, min(v.min() for v in r2_data) - 0.05))

    # 图注
    note = (r"注：箱线图展示 $R^2$ 的 5 折 CV 分布；"
            r"白点（$\bar{R}^2$）为均值；红线为中位数。")
    fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=9,
             color="dimgray", style="italic")

    plt.tight_layout()
    plt.savefig(OUT_PNG_BOX, dpi=300, bbox_inches="tight")
    logger.info("R² 箱线图已保存：%s", OUT_PNG_BOX)
    plt.close()


# ===========================================================================
# 可视化：Spearman 相关性条形图
# ===========================================================================

def plot_spearman_bar(corr_df: pd.DataFrame) -> None:
    """绘制特征与结局变量的 Spearman ρ 条形图（含显著性标记）。"""
    df_plot = corr_df.sort_values("Spearman_ρ", ascending=True)
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in df_plot["Spearman_ρ"]]

    fig, ax = plt.subplots(figsize=(9, max(5, len(df_plot) * 0.55)))
    bars = ax.barh(df_plot["特征"], df_plot["Spearman_ρ"],
                   color=colors, alpha=0.85, height=0.6)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    # 显著性星号（p < 0.05 = *，p < 0.01 = **，p < 0.001 = ***）
    for bar, (_, row) in zip(bars, df_plot.iterrows()):
        p = row["p 值"]
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        x_pos = row["Spearman_ρ"]
        offset = 0.005 if x_pos >= 0 else -0.005
        ha = "left" if x_pos >= 0 else "right"
        if star:
            ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                    star, va="center", ha=ha, fontsize=10, color="black")

    ax.set_xlabel(r"Spearman $\rho$", fontsize=12)
    ax.set_title(
        r"预测因子与社会情绪发展得分（$Y$）的 Spearman 相关系数",
        fontsize=12, fontweight="bold",
    )
    fig.text(0.01, -0.03,
             r"注：* $p$ < 0.05，** $p$ < 0.01，*** $p$ < 0.001（未校正）",
             ha="left", fontsize=9, style="italic", color="dimgray")
    plt.tight_layout()
    plt.savefig(OUT_PNG_CORR, dpi=300, bbox_inches="tight")
    logger.info("Spearman 相关图已保存：%s", OUT_PNG_CORR)
    plt.close()


# ===========================================================================
# 导出 Pro CSV（预测得分 + SHAP 贡献值）
# ===========================================================================

def export_pro_csv(
    df_orig: pd.DataFrame,
    X_clean: pd.DataFrame,
    y_clean: pd.Series,
    orig_indices: pd.Index,
    best_pipeline: Pipeline,
    best_name: str,
    shap_values: np.ndarray | None,
    feature_labels: list[str],
    weight_col: str | None,
    orig_index: pd.Index,
) -> None:
    """
    导出 socioemotional_pro_analysis.csv：
    包含原始 ID（若有）、特征值、真实得分、模型预测得分、SHAP 贡献值（若可用）。

    ``orig_indices`` 为 ``build_feature_matrix`` 返回的原始行号索引，
    确保从 ``df_orig`` 中安全提取受访者 ID 和抽样权重，不会产生行错位。
    """
    out = X_clean.copy()


            break

    out["真实社会情绪发展得分_Y"] = y_clean.values

    # 模型预测
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        y_pred = best_pipeline.predict(X_clean)
    out[f"预测得分_{best_name}"] = y_pred


    if shap_values is not None and shap_values.shape[0] == len(out):
        for j, lbl in enumerate(feature_labels):
            col_name = f"SHAP_{lbl}"
            if j < shap_values.shape[1]:
                out[col_name] = shap_values[:, j]
    elif shap_values is not None:
        logger.warning(
            "SHAP values 行数 (%d) 与输出行数 (%d) 不一致，跳过 SHAP 列导出",
            shap_values.shape[0], len(out),
        )



    out.to_csv(OUT_CSV_PRO, index=False, encoding="utf-8-sig")
    logger.info("Pro CSV 已保存：%s（%d 行 × %d 列）",
                OUT_CSV_PRO, *out.shape)


# ===========================================================================
# 控制台摘要（LaTeX 风格符号）
# ===========================================================================

def print_summary(
    cv_results: dict[str, dict[str, np.ndarray]],
    weight_col: str | None,
    n_total: int,
    n_valid: int,
) -> None:
    """打印顶刊级统计摘要（LaTeX 格式符号）。"""
    print("\n" + "=" * 68)
    print("  社会情绪发展得分多模型预测 — 统计摘要")
    print("  （对标 The Lancet Digital Health 方法学规范）")
    print("=" * 68)
    print(f"  数据文件  ：cfps2022childproxy_202410.dta")
    print(f"  总样本量  ：N = {n_total}")
    print(f"  有效分析量：n = {n_valid}")
    print(f"  抽样权重  ：{weight_col or '未加权（等权）'}")
    print(f"  CV 设置   ：{CV_FOLDS}-fold KFold（random_state={RANDOM_STATE}）")
    print()
    print(f"  {'模型':<20} {'R²（均±SD）':>18} {'RMSE（均±SD）':>18} {'MAE（均±SD）':>18}")
    print("  " + "-" * 76)
    for name, res in cv_results.items():
        r2_m,  r2_s  = res["r2"].mean(),   res["r2"].std()
        rmse_m, rmse_s = res["rmse"].mean(), res["rmse"].std()
        mae_m,  mae_s  = res["mae"].mean(),  res["mae"].std()
        print(f"  {name:<20} {r2_m:+.4f} ± {r2_s:.4f}  "
              f"{rmse_m:.4f} ± {rmse_s:.4f}  "
              f"{mae_m:.4f} ± {mae_s:.4f}")
    print()
    # 最优模型
    best_name = max(cv_results, key=lambda n: cv_results[n]["r2"].mean())
    best_r2 = cv_results[best_name]["r2"].mean()
    print(f"  最优模型：{best_name}（R² = {best_r2:.4f}）")
    print()
    print("  输出文件：")
    print(f"    • {OUT_CSV_PRO}")
    print(f"    • {OUT_PNG_BOX}")
    print(f"    • {OUT_PNG_SHAP_BEE}")
    print(f"    • {OUT_PNG_SHAP_DEP}")
    print(f"    • {OUT_PNG_CORR}")
    print("=" * 68 + "\n")


# ===========================================================================
# 主流程
# ===========================================================================

def main() -> None:
    # ------------------------------------------------------------------
    # 1. 加载数据
    # ------------------------------------------------------------------
    df, meta = load_data(DATA_FILE)

    # ------------------------------------------------------------------
    # 2. 解析权重与慢性病变量，动态更新 FACTOR_MAP
    # ------------------------------------------------------------------
    weight_col = resolve_weight_var(df)
    factor_map = dict(FACTOR_MAP)
    chronic_col, chronic_desc = resolve_chronic_disease_var(df)
    if chronic_col is None:
        factor_map.pop("慢性病诊断", None)
    elif chronic_col != "qp4001":
        factor_map["慢性病诊断"] = chronic_col
        logger.info("慢性病变量替换：qp4001 → %s（%s）", chronic_col, chronic_desc)

    # ------------------------------------------------------------------
    # 3. 构建结局变量 Y（连续社会情绪发展得分）
    # ------------------------------------------------------------------
    outcome_col = "socioemotional_score"
    df[outcome_col] = build_socioemotional_score(df)

    # ------------------------------------------------------------------
    # 4. 自适应变量过滤
    # ------------------------------------------------------------------
    valid_map, coverage_records = filter_factors(df, factor_map)
    if not valid_map:
        logger.error("所有预测因子均被过滤，无法继续分析")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. 构建特征矩阵 X 和结局向量 y
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # 6. Spearman 相关性分析 + 绘图
    # ------------------------------------------------------------------
    logger.info("执行 Spearman 相关性分析...")
    corr_df = spearman_analysis(X_clean, y_clean)
    logger.info("相关性结果（Top 5）：\n%s", corr_df.head(5).to_string(index=False))
    plot_spearman_bar(corr_df)

    # ------------------------------------------------------------------
    # 7. 定义基础模型 + Stacking 集成
    # ------------------------------------------------------------------
    base_models = get_base_models(random_state=RANDOM_STATE)
    stacking_pipeline = get_stacking_model(base_models, random_state=RANDOM_STATE)
    all_models = {**base_models, "Stacking (RidgeCV)": stacking_pipeline}

    # ------------------------------------------------------------------
    # 8. 5 折交叉验证
    # ------------------------------------------------------------------
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    logger.info("开始 %d 折交叉验证（%d 个模型）...", CV_FOLDS, len(all_models))
    cv_results = run_cv(all_models, X_clean, y_clean, cv)

    # ------------------------------------------------------------------
    # 9. 模型 R² 箱线图
    # ------------------------------------------------------------------
    plot_r2_boxplot(cv_results)

    # ------------------------------------------------------------------
    # 10. 选出最优基模型（非 Stacking）用于 SHAP 分析
    # ------------------------------------------------------------------
    # 优先 XGBoost（TreeExplainer 最快），否则随机森林
    shap_candidate_order = ["XGBoost", "随机森林 (RF)", "MLP 神经网络", "SVR"]
    shap_model_name = next(
        (n for n in shap_candidate_order if n in base_models), list(base_models.keys())[0]
    )
    shap_pipeline = base_models[shap_model_name]

    # 全量训练 SHAP 目标模型
    logger.info("全量训练 SHAP 目标模型：%s", shap_model_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        shap_pipeline.fit(X_clean, y_clean)

    # 使用全部样本作为测试集（全局 SHAP）
    shap_values, X_test_trans = run_shap_analysis(
        shap_pipeline, X_clean, X_clean, feature_labels, shap_model_name
    )

    # ------------------------------------------------------------------
    # 11. 全量训练最优模型（用于预测列导出）
    # ------------------------------------------------------------------
    best_name = max(cv_results, key=lambda n: cv_results[n]["r2"].mean())
    best_pipeline = all_models[best_name]
    logger.info("全量训练最优模型：%s", best_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        best_pipeline.fit(X_clean, y_clean)

    # ------------------------------------------------------------------
    # 12. 导出 Pro CSV
    # ------------------------------------------------------------------
    export_pro_csv(
        df_orig=df,
        X_clean=X_clean,
        y_clean=y_clean,
        orig_indices=orig_indices,
        best_pipeline=best_pipeline,
        best_name=best_name,
        shap_values=shap_values,
        feature_labels=feature_labels,
        weight_col=weight_col,
        orig_index=orig_index,
    )

    # ------------------------------------------------------------------
    # 13. 控制台摘要
    # ------------------------------------------------------------------
    print_summary(cv_results, weight_col, len(df), len(y_clean))


if __name__ == "__main__":
    main()
