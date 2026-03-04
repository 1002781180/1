"""
CFPS 2022 儿童抑郁数据分析脚本
数据集：cfps2022childproxy_202410.dta（儿童代答问卷）
功能：自动化清洗、建模与结果保存
兼容 Python 3.10+

说明：
  CFPS 2022 儿童代答问卷中未包含 wn401 系列 CES-D 成人自评题目。
  本脚本优先查找 wn401 系列，若不存在则自动使用 we301-we312
  行为情绪健康条目（孩子乐观、不冲动、容易克服烦躁等正向特质评分）。
  由于所有 we3xx 条目均为正向特质（得分越高代表心理健康越好），
  计算"情绪困扰风险分"时对全部条目做反向计分（6 - 原始分），
  使最终 depression_score 方向与 CES-D 一致（得分越高 = 抑郁风险越高）。
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面环境下使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. 路径自动化
# ──────────────────────────────────────────────
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
os.makedirs(DESKTOP, exist_ok=True)

DATA_FILE = os.path.join(DESKTOP, "cfps2022childproxy_202410.dta")

OUTPUT_CSV = os.path.join(DESKTOP, "depression_factors_correlation.csv")
OUTPUT_PNG = os.path.join(DESKTOP, "depression_feature_importance.png")

print(f"[信息] 桌面路径：{DESKTOP}")
print(f"[信息] 数据文件：{DATA_FILE}")

# ──────────────────────────────────────────────
# 2. 读取数据
# ──────────────────────────────────────────────
print("[步骤 1] 读取 .dta 数据文件...")
try:
    df = pd.read_stata(DATA_FILE, convert_categoricals=False)
    print(f"  原始数据形状：{df.shape}")
except FileNotFoundError:
    raise FileNotFoundError(
        f"未找到数据文件：{DATA_FILE}\n"
        "请将 cfps2022childproxy_202410.dta 放置到桌面后重新运行。"
    )

# 统一列名小写，便于后续匹配
df.columns = df.columns.str.lower().str.strip()

# ──────────────────────────────────────────────
# 3. 全变量扫描
# ──────────────────────────────────────────────
print("[步骤 2] 扫描变量...")

# 3-1 优先查找 wn401 系列抑郁条目（CFPS 成人自评 CES-D）
depression_items = [c for c in df.columns if c.startswith("wn401")]

# 若 wn401 系列不存在，则回退到 we3xx 行为情绪健康条目
if not depression_items:
    depression_items = sorted([c for c in df.columns if c.startswith("we3")])
    ITEM_TYPE = "we3xx"  # 正向特质条目，全部需要反向计分
    print(f"  未找到 wn401 系列，使用 we3xx 行为情绪条目 {len(depression_items)} 个")
    print(f"  条目列表：{depression_items}")
else:
    ITEM_TYPE = "wn401"
    print(f"  找到 wn401 系列抑郁条目 {len(depression_items)} 个：{depression_items}")

# 3-2 影响因子变量映射（基于 CFPS 2022 代答问卷实际列名）
#
# 说明：
#   - 睡眠时长：由起床时间（wf309a 小时 / wf309b 分钟）与
#              睡觉时间（wf310a 小时 / wf310b 分钟）计算得到
#   - 亲子互动：wg301（讲故事）wg302（买书）wg303（出游）取均值
#   - 更多变量见各分组注释
FACTOR_MAP = {
    # 人口学
    "性别(gender)":           "gender",       # 0=女, 1=男
    "年龄(age)":              "age",
    "城乡(urban)":            "urban22",      # 1=城镇, 0=农村
    "户口状态(hukou)":        "wa301",        # 1=农业, 2=非农业
    # 行为习惯（睡眠时长单独计算，见下方）
    "看电视时长/周(tv_hour)": "wb9",          # 小时/周
    "上网时长(internet_min)": "wu401",        # 分钟/天
    "使用电子设备(device)":   "wu1",          # 0=否, 1=是
    # 家庭环境
    "父亲同住月数(co_dad)":   "wb401",        # 过去12个月
    "母亲同住月数(co_mom)":   "wb402",
    "夜间监护人(guardian)":   "wb203",        # 1=父, 2=母, 3=祖父母…
    "讲故事频率(story)":      "wg301",        # 1~5 频率
    "带孩子出游(outing)":     "wg303",        # 1~5 频率
    "父母关心教育(parent_edu_care)": "wz301", # 1~5
    "父母主动沟通(parent_comm)":     "wz302", # 1~5
    # 学校/学习
    "语文成绩(chinese)":      "wf501",        # 1=优, 2=良, 3=中, 4=差
    "数学成绩(math)":         "wf502",
    "期望成绩(expected_score)": "wf701",      # 百分制
    # 健康
    "BMI指数(bmi)":           "bmi",
}

# 只保留数据集中实际存在的因子列
factor_columns = {
    name: col for name, col in FACTOR_MAP.items()
    if col in df.columns
}

print(f"  有效因子变量 {len(factor_columns)} 个：")
for name, col in factor_columns.items():
    print(f"    {name} → {col}")

# ──────────────────────────────────────────────
# 4. 深度清洗（CFPS 专属）
# ──────────────────────────────────────────────
print("[步骤 3] 深度清洗...")

MISSING_CODES = [-1, -2, -8, -9]
# CFPS 最小合法缺失码为 -9；任何更小的值均视为数据录入错误
_CFPS_MIN_VALID_CODE = -9

# 儿童每日睡眠时长合理区间（小时）：低于 4 或高于 16 视为异常值
_SLEEP_MIN_HOURS = 4
_SLEEP_MAX_HOURS = 16

# 相关性分析所需的最小有效样本量（低于此值统计意义不足）
_MIN_CORR_SAMPLE = 10


def replace_missing(series: pd.Series) -> pd.Series:
    """将 CFPS 负数缺失码（-1/-2/-8/-9）及其他极端负值替换为 NaN。"""
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace(MISSING_CODES, np.nan)
    s[s < _CFPS_MIN_VALID_CODE] = np.nan  # 比最小缺失码更小的值视为录入异常
    return s


# 清洗抑郁/情绪条目
for col in depression_items:
    df[col] = replace_missing(df[col])

# 清洗因子列
for col in factor_columns.values():
    df[col] = replace_missing(df[col])

# ──────────────────────────────────────────────
# 计算睡眠时长（小时）
# 起床：wf309a（点，范围 4-12）+ wf309b（分，范围 0-59）
# 睡觉：wf310a（点，范围 18-24）+ wf310b（分，范围 0-59）
# 公式使用模运算处理跨午夜情形：sleep_hours = (wake_time - sleep_time) % 24
#   示例：睡觉 21:30，起床 7:30 → (7.5 - 21.5) % 24 = 10.0 小时 ✓
#   示例：睡觉 23:00，起床 6:00 → (6.0 - 23.0) % 24 = 7.0 小时  ✓
# ──────────────────────────────────────────────
sleep_cols = ["wf309a", "wf309b", "wf310a", "wf310b"]
if all(c in df.columns for c in sleep_cols):
    for c in sleep_cols:
        df[c] = replace_missing(df[c])
    wake  = df["wf309a"] + df["wf309b"].fillna(0) / 60.0
    sleep = df["wf310a"] + df["wf310b"].fillna(0) / 60.0
    df["sleep_hours"] = (wake - sleep) % 24.0
    # 过滤儿童睡眠时长合理范围之外的异常值
    df.loc[
        (df["sleep_hours"] < _SLEEP_MIN_HOURS) | (df["sleep_hours"] > _SLEEP_MAX_HOURS),
        "sleep_hours",
    ] = np.nan
    factor_columns["睡眠时长/小时(sleep_hours)"] = "sleep_hours"
    valid_sleep = df["sleep_hours"].notna().sum()
    print(f"  睡眠时长：计算完成，有效 {valid_sleep} 条，"
          f"均值 {df['sleep_hours'].mean():.1f} 小时")

# ──────────────────────────────────────────────
# CES-D / 行为情绪自动计分
#
# wn401 系列（标准 CES-D，量表范围 0-3）：
#   正向题（悲伤/抑郁方向）直接计分；
#   反向题（以 e/h/f/g 结尾，表示正性情绪）做翻转：翻转值 = 3 - 原始值
#
# we3xx 系列（正向特质量表，代答版本，量表范围 1-5）：
#   全部条目均为正向特质（乐观/耐心/不冲动…），统一反向计分：
#   翻转值 = 6 - 原始值
#   翻转后得分越高 = 正向特质越少 = 情绪困扰风险越高
#
# 注：量表范围均基于 CFPS 2022 官方说明及实际数据验证（we3xx: 1-5, wn401: 0-3）
# ──────────────────────────────────────────────
# CES-D 反向计分题后缀（仅对 wn401 系列生效）
_CESD_REVERSE_SUFFIX = {"e", "h", "f", "g"}
# CES-D (wn401) 量表最高分（0-3 四点量表）
_CESD_MAX = 3
# we3xx 正向特质量表最高分（1-5 五点量表）
_WE3_MAX = 5


def score_depression(
    row: pd.Series,
    items: list[str],
    item_type: str,
) -> float:
    """
    计算情绪困扰得分（depression_score）。

    Parameters
    ----------
    row       : 数据行
    items     : 情绪/抑郁条目列名列表
    item_type : 'wn401'（CES-D 标准版）或 'we3xx'（代答正向特质版）

    Returns
    -------
    float 或 NaN（缺失题目超过 20% 时返回 NaN）
    """
    values = row[items]
    missing_ratio = values.isna().mean()
    if missing_ratio > 0.2:
        return np.nan

    total = 0.0
    valid_count = 0

    for col in items:
        v = values[col]
        if pd.isna(v):
            continue

        if item_type == "wn401":
            # CES-D 反向计分（量表 0-3）：以 e/h/f/g 结尾的题目为正性情绪题，需翻转
            suffix = col.replace("wn401", "").strip("_").lower()
            if suffix in _CESD_REVERSE_SUFFIX:
                v = float(_CESD_MAX) - v
        else:
            # we3xx：全部正向特质条目统一反向计分（量表 1-5，翻转后 1→5, 5→1）
            v = _WE3_MAX + 1 - v  # = 6 - v

        total += v
        valid_count += 1

    if valid_count == 0:
        return np.nan

    # 按有效条目数等比推算总分（处理少量随机缺失，保持量级与满分一致）
    # 注：此方法假设缺失条目与已答条目具有相同的平均分布（随机缺失假设）
    return total * len(items) / valid_count


print(f"  计算情绪困扰得分（共 {len(depression_items)} 个条目，类型={ITEM_TYPE}）...")
df["depression_score"] = df.apply(
    score_depression, axis=1, items=depression_items, item_type=ITEM_TYPE
)
valid_n = df["depression_score"].notna().sum()
print(f"  有效样本量：{valid_n} / {len(df)}")
print(f"  得分均值={df['depression_score'].mean():.2f}，"
      f"标准差={df['depression_score'].std():.2f}")

# ──────────────────────────────────────────────
# 5. 相关性分析（Pearson r + P 值）
# ──────────────────────────────────────────────
print("[步骤 4] 相关性分析...")

target_col = "depression_score"
analysis_cols = list(factor_columns.values())

df_analysis = df[analysis_cols + [target_col]].copy()
df_analysis = df_analysis.dropna(subset=[target_col])
print(f"  有效样本（目标变量非缺失）：{len(df_analysis)}")

# Pearson 相关系数 + P 值
corr_results = []
for factor_name, col in factor_columns.items():
    sub = df_analysis[[col, target_col]].dropna()
    if len(sub) < _MIN_CORR_SAMPLE:
        corr_results.append(
            {"factor": factor_name, "column": col,
             "correlation": np.nan, "p_value": np.nan,
             "sample_size": len(sub)}
        )
        continue
    r, p = stats.pearsonr(sub[col], sub[target_col])
    corr_results.append(
        {"factor": factor_name, "column": col,
         "correlation": round(r, 4), "p_value": round(p, 6),
         "sample_size": len(sub)}
    )

corr_df = pd.DataFrame(corr_results).sort_values(
    "correlation", key=abs, ascending=False
)
print(corr_df.to_string(index=False))

# ──────────────────────────────────────────────
# Random Forest 特征重要性
# ──────────────────────────────────────────────
print("  训练 Random Forest 模型...")

# 只保留数据集中实际存在的列
valid_factor_cols = [c for c in analysis_cols if c in df_analysis.columns]
X = df_analysis[valid_factor_cols].copy()
y = df_analysis[target_col].copy()

# 管道：中位数填充缺失值 → 随机森林回归
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])
pipeline.fit(X, y)

importances = pipeline.named_steps["rf"].feature_importances_
importance_df = pd.DataFrame({
    "column":     valid_factor_cols,
    "importance": importances,
})

# 将列名映射回因子中文名
col_to_factor = {v: k for k, v in factor_columns.items()}
importance_df["factor"] = importance_df["column"].map(col_to_factor)
importance_df = importance_df.sort_values("importance", ascending=False)

# 合并相关性与重要性
final_df = corr_df.merge(
    importance_df[["column", "importance"]],
    on="column", how="left"
).sort_values("importance", ascending=False, na_position="last")

print(f"\n  特征重要性 TOP 10：")
print(importance_df.head(10).to_string(index=False))

# ──────────────────────────────────────────────
# 6. 保存结果
# ──────────────────────────────────────────────
print("[步骤 5] 保存结果...")

# 6-1 CSV 表格：所有变量相关系数 + P 值 + 特征重要性
final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"  已保存 CSV：{OUTPUT_CSV}")

# 6-2 可视化图表：前 20 个核心影响因素
top_n = min(20, len(importance_df))
plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

# 尝试加载中文字体；若不可用则使用 column 名称（英文）
CN_FONTS = [
    "SimHei", "STHeiti", "WenQuanYi Micro Hei",
    "Noto Sans CJK SC", "Microsoft YaHei",
]
available_fonts = {f.name for f in fm.fontManager.ttflist}
used_font = next((f for f in CN_FONTS if f in available_fonts), None)
if used_font:
    plt.rcParams["font.family"] = used_font
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.48)))

# 颜色映射：由深红（高重要性）到浅绿（低重要性）
colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, top_n))[::-1]

# 图例标签：优先用中文因子名，否则用列名
y_labels = plot_df["factor"].fillna(plot_df["column"]).tolist()
bars = ax.barh(y_labels, plot_df["importance"].tolist(), color=colors)

ax.set_xlabel("Feature Importance (Random Forest)", fontsize=12)
ax.set_title(
    "Top Factors Associated with Children's Emotional Distress Score\n"
    f"(CFPS 2022 Proxy Survey, N={len(df_analysis):,})",
    fontsize=13, fontweight="bold",
)
ax.xaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

# 条形末端标注数值
for bar, val in zip(bars, plot_df["importance"].tolist()):
    ax.text(
        bar.get_width() + max(plot_df["importance"]) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center", ha="left", fontsize=9,
    )

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"  已保存图表：{OUTPUT_PNG}")

print("\n[完成] 所有结果已保存到桌面。")
