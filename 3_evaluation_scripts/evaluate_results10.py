# ============================================================
# ANALYSIS OF LEGAL ENTITY RETRIEVAL RESULTS
# ============================================================
#
# Input:
#     results.tsv
#
# Main dimensions:
#     mbe_id
#     idx
#     biased_category
#     difficulty
#     model
#     threshold
#     temperature
#     run
#
# Main metrics:
#     precision
#     recall
#     f1
#     tp
#     fp
#     fn
#     gt_count
#     pred_count
#
# Outputs:
#     analysis_output/
#         tables/
#         figures/
#         summaries/
#
# ============================================================

import os
import re
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "final_results/combined_results.tsv"

OUTPUT_DIR = "final_results/analysis_output"
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summaries")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading results...")

df = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    dtype=str
)

# Remove accidental whitespace from column names
df.columns = df.columns.str.strip()

print("\nColumns found:")
print(df.columns.tolist())

# ------------------------------------------------------------
# Clean column names if necessary
# ------------------------------------------------------------

required_columns = [
    "mbe_id",
    "idx",
    "biased_category",
    "difficulty",
    "model",
    "threshold",
    "temperature",
    "run",
    "gt_count",
    "pred_count",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "gt_entities",
    "pred_entities"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        f"\nMissing columns: {missing}\n"
        f"Columns found: {df.columns.tolist()}"
    )


# ============================================================
# 3. CLEAN DATA TYPES
# ============================================================

numeric_columns = [
    "threshold",
    "temperature",
    "run",
    "gt_count",
    "pred_count",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in ["mbe_id", "idx", "biased_category", "difficulty", "model"]:
    df[col] = df[col].astype(str).str.strip()

df["difficulty"] = df["difficulty"].str.lower()
df["biased_category"] = df["biased_category"].str.lower()

# Count difference
df["count_difference"] = df["pred_count"] - df["gt_count"]

# Count ratio
df["count_ratio"] = np.where(
    df["gt_count"] > 0,
    df["pred_count"] / df["gt_count"],
    np.nan
)

# Reference divergence
#
# 0 = perfect F1 agreement with reference
# 1 = complete disagreement
#
df["reference_divergence"] = 1 - df["f1"]

print(f"\nRows: {len(df):,}")
print(f"Questions: {df['mbe_id'].nunique():,}")
print(f"Variants: {df['idx'].nunique():,}")
print(f"Models: {df['model'].nunique()}")
print(f"Thresholds: {sorted(df['threshold'].dropna().unique())}")
print(f"Runs: {sorted(df['run'].dropna().unique())}")


# ============================================================
# 4. BASIC DATA QUALITY CHECK
# ============================================================

print("\nChecking data consistency...")

expected_variants = 10
expected_runs = 5
expected_thresholds = 5

questions_with_wrong_variants = (
    df.groupby("mbe_id")["idx"]
    .nunique()
)

wrong_variants = questions_with_wrong_variants[
    questions_with_wrong_variants != expected_variants
]

print(
    f"Questions not having exactly {expected_variants} variants: "
    f"{len(wrong_variants)}"
)

print(
    f"Duplicate rows: "
    f"{df.duplicated().sum():,}"
)

# ============================================================
# FIXED MODEL COLORS AND ORDER
# ============================================================

MODEL_ORDER = [
    "claude-haiku-4-5",
    "gemini-3.1-flash-lite",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Llama-4-Scout-17B-16E-Instruct",
    "gpt-5-mini",
]

MODEL_COLORS = {
    "claude-haiku-4-5": "#1f77b4",
    "gemini-3.1-flash-lite": "#ff7f0e",
    "Llama-4-Maverick-17B-128E-Instruct-FP8": "#2ca02c",
    "Llama-4-Scout-17B-16E-Instruct": "#d62728",
    "gpt-5-mini": "#9467bd",
}

# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def macro_metrics(data):
    """
    Macro-average of per-row precision, recall and F1.
    """
    return pd.Series({
        "precision": data["precision"].mean(),
        "recall": data["recall"].mean(),
        "f1": data["f1"].mean(),
        "n": len(data)
    })


def micro_metrics(data):
    """
    Micro precision, recall and F1 calculated from TP/FP/FN.
    """
    tp = data["tp"].sum()
    fp = data["fp"].sum()
    fn = data["fn"].sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = np.nan

    return pd.Series({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n": len(data)
    })


def save_table(data, filename):
    path = os.path.join(TABLE_DIR, filename)
    data.to_csv(path, index=False)
    print(f"Saved table: {path}")


def save_figure(filename):
    path = os.path.join(FIGURE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {path}")


# ============================================================
# 6. OVERALL PERFORMANCE
# ============================================================

print("\n============================================================")
print("OVERALL PERFORMANCE")
print("============================================================")

overall_macro = (
    df.groupby("model")
    .apply(macro_metrics, include_groups=False)
    .reset_index()
)

overall_micro = (
    df.groupby("model")
    .apply(micro_metrics, include_groups=False)
    .reset_index()
)

overall = overall_macro.merge(
    overall_micro,
    on="model",
    suffixes=("_macro", "_micro")
)

save_table(overall, "01_overall_model_performance.csv")

print(overall)

# ============================================================
# 6A. NON-ZERO F1 AND ZERO-F1 ANALYSIS
# ============================================================
#
# Two complementary metrics:
#
# 1. Mean F1 among non-zero cases:
#       Mean F1 calculated only for cases where F1 > 0.
#
#    This tells us how well the model performs when it retrieves
#    at least some overlap with the reference entities.
#
# 2. Percentage of zero-F1 cases:
#       Percentage of cases where F1 == 0.
#
#    This measures how frequently the model has no overlap with
#    the reference entities.
#
# These are calculated at the row level:
# question × demographic variant × model × threshold × run.
#
# ============================================================

print("\n============================================================")
print("NON-ZERO F1 AND ZERO-F1 ANALYSIS")
print("============================================================")


# ------------------------------------------------------------
# Overall by model
# ------------------------------------------------------------

nonzero_f1_results = (
    df.groupby("model")
    .apply(
        lambda data: pd.Series({
            "mean_f1_nonzero": data.loc[
                data["f1"] > 0, "f1"
            ].mean(),

            "n_nonzero_f1": (
                data["f1"] > 0
            ).sum(),

            "n_zero_f1": (
                data["f1"] == 0
            ).sum(),

            "total_cases": len(data),

            "pct_f1_zero": (
                (data["f1"] == 0).mean() * 100
            )
        }),
        include_groups=False
    )
    .reset_index()
)

save_table(
    nonzero_f1_results,
    "24_nonzero_f1_and_zero_f1_by_model.csv"
)

print("\nF1 among non-zero cases and percentage of zero-F1 cases:")
print(nonzero_f1_results)


# ------------------------------------------------------------
# Model × demographic category
# ------------------------------------------------------------

nonzero_f1_category = (
    df.groupby(
        ["model", "biased_category"]
    )
    .apply(
        lambda data: pd.Series({
            "mean_f1_nonzero": data.loc[
                data["f1"] > 0, "f1"
            ].mean(),

            "n_nonzero_f1": (
                data["f1"] > 0
            ).sum(),

            "n_zero_f1": (
                data["f1"] == 0
            ).sum(),

            "total_cases": len(data),

            "pct_f1_zero": (
                (data["f1"] == 0).mean() * 100
            )
        }),
        include_groups=False
    )
    .reset_index()
)

save_table(
    nonzero_f1_category,
    "25_nonzero_f1_and_zero_f1_by_model_category.csv"
)

print("\nNon-zero F1 and zero-F1 percentage by model and category:")
print(nonzero_f1_category)


# ============================================================
# 7. PERFORMANCE BY MODEL AND THRESHOLD
# ============================================================

model_threshold = (
    df.groupby(["model", "threshold"])
    .apply(macro_metrics, include_groups=False)
    .reset_index()
)

save_table(
    model_threshold,
    "02_model_threshold_performance.csv"
)


# ============================================================
# 8. DEMOGRAPHIC CATEGORY ANALYSIS
# ============================================================

print("\n============================================================")
print("DEMOGRAPHIC CATEGORY ANALYSIS")
print("============================================================")

category_results = (
    df.groupby("biased_category")
    .apply(macro_metrics, include_groups=False)
    .reset_index()
)

save_table(
    category_results,
    "03_demographic_category_performance.csv"
)

# Model × demographic category
model_category = (
    df.groupby(["model", "biased_category"])
    .apply(macro_metrics, include_groups=False)
    .reset_index()
)

save_table(
    model_category,
    "04_model_by_demographic_category.csv"
)


# ============================================================
# 9. DEMOGRAPHIC PERFORMANCE GAP
# ============================================================

gap_rows = []

for model, group in model_category.groupby("model"):

    max_f1 = group["f1"].max()
    min_f1 = group["f1"].min()

    max_category = group.loc[
        group["f1"].idxmax(), "biased_category"
    ]

    min_category = group.loc[
        group["f1"].idxmin(), "biased_category"
    ]

    gap_rows.append({
        "model": model,
        "maximum_f1": max_f1,
        "minimum_f1": min_f1,
        "f1_gap": max_f1 - min_f1,
        "category_max_f1": max_category,
        "category_min_f1": min_category,
        "category_f1_std": group["f1"].std()
    })

demographic_gap = pd.DataFrame(gap_rows)

save_table(
    demographic_gap,
    "05_demographic_performance_gap.csv"
)


# ============================================================
# 10. MODEL × CATEGORY HEATMAP
# ============================================================

heatmap_data = model_category.pivot(
    index="model",
    columns="biased_category",
    values="f1"
)

plt.figure(figsize=(12, 6))

plt.imshow(
    heatmap_data.values,
    aspect="auto"
)

plt.colorbar(label="Mean F1")

plt.xticks(
    range(len(heatmap_data.columns)),
    heatmap_data.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(heatmap_data.index)),
    heatmap_data.index
)

plt.xlabel("Demographic category")
plt.ylabel("Model")
plt.title("Mean Entity Retrieval F1 by Model and Demographic Category")

save_figure("01_model_demographic_heatmap.png")


# ============================================================
# 11. DIFFICULTY ANALYSIS
# ============================================================

print("\n============================================================")
print("DIFFICULTY ANALYSIS")
print("============================================================")

difficulty_results = (
    df.groupby("difficulty")
    .apply(macro_metrics, include_groups=False)
    .reset_index()
)

save_table(
    difficulty_results,
    "06_difficulty_performance.csv"
)

model_difficulty = (
    df.groupby(["model", "difficulty"])
    .apply(macro_metrics, include_groups=False)
    .reset_index()
)

save_table(
    model_difficulty,
    "07_model_by_difficulty.csv"
)


# ============================================================
# 12. DIFFICULTY FIGURE
# ============================================================

difficulty_order = ["easy", "medium", "hard"]

available_difficulties = [
    x for x in difficulty_order
    if x in df["difficulty"].unique()
]

plt.figure(figsize=(9, 6))

for model in MODEL_ORDER:

    temp = (
        model_difficulty[
            model_difficulty["model"] == model
        ]
        .set_index("difficulty")
        .reindex(available_difficulties)
    )

    plt.plot(
        available_difficulties,
        temp["f1"],
        marker="o",
        color=MODEL_COLORS[model],
        label=model
    )

plt.xlabel("Question difficulty")
plt.ylabel("Mean F1")
plt.title("Entity Retrieval Performance by Question Difficulty")
plt.legend()

save_figure("02_difficulty_effect.png")


# ============================================================
# 13. THRESHOLD SENSITIVITY
# ============================================================

threshold_pivot = model_threshold.pivot(
    index="threshold",
    columns="model",
    values="f1"
)

save_table(
    threshold_pivot.reset_index(),
    "08_threshold_sensitivity.csv"
)

plt.figure(figsize=(9, 6))

for model in MODEL_ORDER:
    if model not in threshold_pivot.columns:
        continue

    plt.plot(
        threshold_pivot.index,
        threshold_pivot[model],
        marker="o",
        color=MODEL_COLORS[model],
        label=model
    )

plt.xlabel("Hungarian matching threshold")
plt.ylabel("Mean F1")
plt.title("Sensitivity to Semantic Matching Threshold")
plt.legend()

save_figure("03_threshold_sensitivity.png")


# ============================================================
# 14. THRESHOLD RANGE
# ============================================================

threshold_range = (
    model_threshold
    .groupby("model")
    .agg(
        min_f1=("f1", "min"),
        max_f1=("f1", "max")
    )
    .reset_index()
)

threshold_range["f1_range"] = (
    threshold_range["max_f1"]
    - threshold_range["min_f1"]
)

save_table(
    threshold_range,
    "09_threshold_f1_range.csv"
)


# ============================================================
# 15. RUN-TO-RUN VARIABILITY
# ============================================================

print("\n============================================================")
print("RUN VARIABILITY")
print("============================================================")

# Each row here represents:
#
# question × demographic variant × model × threshold × run
#
# We summarize the five runs for each experimental condition.

run_variability = (
    df.groupby(
        ["mbe_id", "idx", "biased_category",
         "difficulty", "model", "threshold"]
    )
    .agg(
        mean_f1=("f1", "mean"),
        sd_f1=("f1", "std"),
        min_f1=("f1", "min"),
        max_f1=("f1", "max"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean")
    )
    .reset_index()
)

run_variability["f1_range"] = (
    run_variability["max_f1"]
    - run_variability["min_f1"]
)

run_variability["cv_f1"] = (
    run_variability["sd_f1"]
    / run_variability["mean_f1"]
)

save_table(
    run_variability,
    "10_question_run_variability.csv"
)


# Aggregate variability by model
model_run_variability = (
    run_variability
    .groupby("model")
    .agg(
        mean_sd_f1=("sd_f1", "mean"),
        median_sd_f1=("sd_f1", "median"),
        mean_f1_range=("f1_range", "mean"),
        mean_cv=("cv_f1", "mean")
    )
    .reset_index()
)

save_table(
    model_run_variability,
    "11_model_run_variability.csv"
)


# ============================================================
# 16. RUN VARIABILITY FIGURE
# ============================================================

plt.figure(figsize=(10, 6))

data_for_plot = [
    run_variability[
        run_variability["model"] == model
    ]["mean_f1"].dropna().values
    for model in MODEL_ORDER
]

bp = plt.boxplot(
    data_for_plot,
    tick_labels=MODEL_ORDER,
    patch_artist=True
)

for patch, model in zip(bp["boxes"], MODEL_ORDER):
    patch.set_facecolor(MODEL_COLORS[model])

plt.ylabel("Mean F1 across repeated runs")
plt.xlabel("Model")
plt.title("Run-to-Run Variability")

plt.xticks(rotation=30)

save_figure("04_run_variability.png")


# ============================================================
# 17. ENTITY COUNT ANALYSIS
# ============================================================

count_analysis = (
    df.groupby("biased_category")
    .agg(
        mean_gt_count=("gt_count", "mean"),
        mean_pred_count=("pred_count", "mean"),
        mean_count_difference=("count_difference", "mean"),
        median_count_difference=("count_difference", "median"),
        mean_count_ratio=("count_ratio", "mean")
    )
    .reset_index()
)

save_table(
    count_analysis,
    "12_entity_count_analysis.csv"
)


# ============================================================
# 18. COUNT DIFFERENCE BY MODEL AND CATEGORY
# ============================================================

model_count = (
    df.groupby(["model", "biased_category"])
    .agg(
        mean_count_difference=("count_difference", "mean"),
        mean_count_ratio=("count_ratio", "mean"),
        mean_gt_count=("gt_count", "mean"),
        mean_pred_count=("pred_count", "mean")
    )
    .reset_index()
)

save_table(
    model_count,
    "13_model_demographic_entity_counts.csv"
)


# ============================================================
# 19. COUNT DIFFERENCE FIGURE
# ============================================================

categories = sorted(df["biased_category"].unique())

plt.figure(figsize=(12, 6))

for model in MODEL_ORDER:

    temp = (
        model_count[
            model_count["model"] == model
        ]
        .set_index("biased_category")
        .reindex(categories)
    )

    plt.plot(
        categories,
        temp["mean_count_difference"],
        marker="o",
        color=MODEL_COLORS[model],
        label=model
    )

plt.axhline(0, linestyle="--")

plt.xlabel("Demographic category")
plt.ylabel("Predicted count − reference count")
plt.title("Entity Count Difference by Demographic Category")
plt.legend()

plt.xticks(rotation=45, ha="right")

save_figure("05_entity_count_difference.png")


# ============================================================
# 20. QUESTION-LEVEL DEMOGRAPHIC VARIABILITY
# ============================================================
#
# This is particularly important.
#
# For every mbe_id we examine how much F1 changes across its
# ten demographic variants.
#
# This answers:
#
# "How sensitive is a particular legal question to demographic
# perturbation?"
#
# ============================================================

question_category = (
    df.groupby(
        ["mbe_id", "biased_category", "model", "threshold"]
    )
    .agg(
        mean_f1=("f1", "mean")
    )
    .reset_index()
)

question_variability = (
    question_category
    .groupby(["mbe_id", "model", "threshold"])
    .agg(
        mean_variant_f1=("mean_f1", "mean"),
        sd_variant_f1=("mean_f1", "std"),
        min_variant_f1=("mean_f1", "min"),
        max_variant_f1=("mean_f1", "max")
    )
    .reset_index()
)

question_variability["variant_f1_range"] = (
    question_variability["max_variant_f1"]
    - question_variability["min_variant_f1"]
)

save_table(
    question_variability,
    "14_question_demographic_variability.csv"
)


# ============================================================
# 21. DISTRIBUTION OF DEMOGRAPHIC EFFECT
# ============================================================

plt.figure(figsize=(9, 6))

plt.hist(
    question_variability["variant_f1_range"].dropna(),
    bins=30
)

plt.xlabel("F1 range across demographic variants")
plt.ylabel("Number of questions")
plt.title(
    "Distribution of Demographic Sensitivity Across Legal Questions"
)

save_figure("06_question_demographic_variability.png")


# ============================================================
# 22. EXTREME QUESTION CASES
# ============================================================

# Questions with the largest demographic variation

extreme_questions = (
    question_variability
    .sort_values("variant_f1_range", ascending=False)
    .head(100)
)

save_table(
    extreme_questions,
    "15_highest_demographic_variability_questions.csv"
)


# ============================================================
# 23. TP / FP / FN ANALYSIS
# ============================================================

error_analysis = (
    df.groupby("biased_category")
    .agg(
        mean_tp=("tp", "mean"),
        mean_fp=("fp", "mean"),
        mean_fn=("fn", "mean")
    )
    .reset_index()
)

save_table(
    error_analysis,
    "16_error_analysis_by_demographic_category.csv"
)


# Model × category error analysis
model_error_analysis = (
    df.groupby(["model", "biased_category"])
    .agg(
        mean_tp=("tp", "mean"),
        mean_fp=("fp", "mean"),
        mean_fn=("fn", "mean")
    )
    .reset_index()
)

save_table(
    model_error_analysis,
    "17_model_error_analysis.csv"
)


# ============================================================
# 24. FP / FN FIGURE
# ============================================================

fp_fn = (
    df.groupby("biased_category")
    .agg(
        fp=("fp", "mean"),
        fn=("fn", "mean")
    )
)

plt.figure(figsize=(12, 6))

x = np.arange(len(fp_fn.index))
width = 0.35

plt.bar(
    x - width / 2,
    fp_fn["fp"],
    width,
    label="False positives"
)

plt.bar(
    x + width / 2,
    fp_fn["fn"],
    width,
    label="False negatives"
)

plt.xticks(
    x,
    fp_fn.index,
    rotation=45,
    ha="right"
)

plt.ylabel("Mean number of entities")
plt.xlabel("Demographic category")
plt.title("False Positives and False Negatives")
plt.legend()

save_figure("07_fp_fn_by_demographic_category.png")


# ============================================================
# 25. LEXICAL COUNTERFACTUAL CONSISTENCY
# ============================================================
#
# IMPORTANT:
#
# This is a lexical Jaccard measure, NOT semantic matching.
#
# It is included because pred_entities are available as text.
#
# It gives an additional measure of whether different demographic
# variants produce the same entity strings.
#
# ============================================================

def normalize_entity(entity):
    entity = str(entity).strip().lower()

    # Normalize whitespace
    entity = re.sub(r"\s+", " ", entity)

    return entity


def parse_entities(text):
    if pd.isna(text):
        return set()

    entities = str(text).split("|")

    return {
        normalize_entity(e)
        for e in entities
        if normalize_entity(e)
    }


def jaccard_similarity(a, b):

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


# Create entity sets
df["pred_entity_set"] = df["pred_entities"].apply(parse_entities)

# We need one prediction per:
# mbe_id × demographic category × model × threshold
#
# Because there are 5 runs, use the union of predicted entities
# across runs as one exploratory representation.

prediction_sets = (
    df.groupby(
        ["mbe_id", "biased_category", "model", "threshold"]
    )["pred_entity_set"]
    .apply(
        lambda sets: set().union(*sets)
    )
    .reset_index(name="pred_entity_set")
)

# Calculate pairwise Jaccard across the 10 demographic variants

consistency_rows = []

for (mbe_id, model, threshold), group in prediction_sets.groupby(
    ["mbe_id", "model", "threshold"]
):

    sets = list(group["pred_entity_set"])

    categories_here = list(group["biased_category"])

    pairwise_scores = []

    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            pairwise_scores.append(
                jaccard_similarity(
                    sets[i],
                    sets[j]
                )
            )

    if pairwise_scores:

        consistency_rows.append({
            "mbe_id": mbe_id,
            "model": model,
            "threshold": threshold,
            "mean_pairwise_jaccard": np.mean(pairwise_scores),
            "sd_pairwise_jaccard": np.std(pairwise_scores),
            "min_pairwise_jaccard": np.min(pairwise_scores),
            "max_pairwise_jaccard": np.max(pairwise_scores)
        })

counterfactual_consistency = pd.DataFrame(
    consistency_rows
)

save_table(
    counterfactual_consistency,
    "18_counterfactual_lexical_consistency.csv"
)


# ============================================================
# 26. MODEL-LEVEL COUNTERFACTUAL CONSISTENCY
# ============================================================

model_consistency = (
    counterfactual_consistency
    .groupby("model")
    .agg(
        mean_jaccard=("mean_pairwise_jaccard", "mean"),
        sd_jaccard=("mean_pairwise_jaccard", "std")
    )
    .reset_index()
)

save_table(
    model_consistency,
    "19_model_counterfactual_consistency.csv"
)


# ============================================================
# 27. PAIRED DEMOGRAPHIC COMPARISONS
# ============================================================
#
# This section compares demographic categories within the SAME
# mbe_id.
#
# This is much more appropriate than treating all questions as
# independent.
#
# ============================================================

category_list = sorted(
    df["biased_category"].dropna().unique()
)

paired_results = []

# Use mean F1 across runs/thresholds for each question-category.
question_category_f1 = (
    df.groupby(
        ["mbe_id", "biased_category"]
    )["f1"]
    .mean()
    .reset_index()
)

pivot = question_category_f1.pivot(
    index="mbe_id",
    columns="biased_category",
    values="f1"
)

for i in range(len(category_list)):

    for j in range(i + 1, len(category_list)):

        c1 = category_list[i]
        c2 = category_list[j]

        if c1 not in pivot.columns or c2 not in pivot.columns:
            continue

        paired = pivot[[c1, c2]].dropna()

        if len(paired) < 2:
            continue

        # Wilcoxon signed-rank test
        try:
            statistic, p_value = stats.wilcoxon(
                paired[c1],
                paired[c2]
            )
        except ValueError:
            statistic = np.nan
            p_value = np.nan

        mean_difference = (
            paired[c1] - paired[c2]
        ).mean()

        paired_results.append({
            "category_1": c1,
            "category_2": c2,
            "n_questions": len(paired),
            "mean_f1_category_1": paired[c1].mean(),
            "mean_f1_category_2": paired[c2].mean(),
            "mean_paired_difference": mean_difference,
            "wilcoxon_statistic": statistic,
            "p_value": p_value
        })

paired_tests = pd.DataFrame(paired_results)

save_table(
    paired_tests,
    "20_paired_demographic_tests.csv"
)


# ============================================================
# 28. FRIEDMAN TEST
# ============================================================
#
# Tests whether F1 differs systematically across the demographic
# categories while keeping mbe_id as the repeated-measures unit.
#
# ============================================================

friedman_rows = []

for model in MODEL_ORDER:

    temp = (
        df[df["model"] == model]
        .groupby(
            ["mbe_id", "biased_category"]
        )["f1"]
        .mean()
        .reset_index()
        .pivot(
            index="mbe_id",
            columns="biased_category",
            values="f1"
        )
    )

    # Only complete questions
    temp = temp.dropna()

    if temp.shape[1] >= 3 and len(temp) >= 2:

        try:

            statistic, p_value = stats.friedmanchisquare(
                *[
                    temp[col].values
                    for col in temp.columns
                ]
            )

            friedman_rows.append({
                "model": model,
                "n_questions": len(temp),
                "n_categories": temp.shape[1],
                "friedman_statistic": statistic,
                "p_value": p_value
            })

        except Exception as e:

            friedman_rows.append({
                "model": model,
                "n_questions": len(temp),
                "n_categories": temp.shape[1],
                "friedman_statistic": np.nan,
                "p_value": np.nan
            })

friedman_results = pd.DataFrame(friedman_rows)

save_table(
    friedman_results,
    "21_friedman_demographic_test.csv"
)


# ============================================================
# 29. EFFECT SIZE FOR PAIRED COMPARISONS
# ============================================================
#
# We calculate paired Cohen's dz for every demographic pair.
#
# dz = mean(difference) / SD(difference)
#
# ============================================================

effect_rows = []

for i in range(len(category_list)):

    for j in range(i + 1, len(category_list)):

        c1 = category_list[i]
        c2 = category_list[j]

        if c1 not in pivot.columns or c2 not in pivot.columns:
            continue

        paired = pivot[[c1, c2]].dropna()

        if len(paired) < 2:
            continue

        differences = paired[c1] - paired[c2]

        sd_difference = differences.std()

        if sd_difference > 0:

            dz = differences.mean() / sd_difference

        else:

            dz = np.nan

        effect_rows.append({
            "category_1": c1,
            "category_2": c2,
            "n_questions": len(paired),
            "cohens_dz": dz
        })

effect_sizes = pd.DataFrame(effect_rows)

save_table(
    effect_sizes,
    "22_paired_effect_sizes.csv"
)


# ============================================================
# 30. SUMMARY STATISTICS
# ============================================================

summary = {
    "number_of_questions": df["mbe_id"].nunique(),
    "number_of_variants": df["idx"].nunique(),
    "number_of_demographic_categories":
        df["biased_category"].nunique(),
    "number_of_models":
        df["model"].nunique(),
    "number_of_thresholds":
        df["threshold"].nunique(),
    "number_of_runs":
        df["run"].nunique(),
    "total_rows":
        len(df)
}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["measure", "value"]
)

summary_df.to_csv(
    os.path.join(
        SUMMARY_DIR,
        "dataset_summary.csv"
    ),
    index=False
)


# ============================================================
# 31. CREATE A MAIN RESULTS TABLE
# ============================================================
#
# This is intended to be close to something you could adapt
# directly for the paper.
#
# ============================================================

main_results = (
    df.groupby("model")
    .agg(
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        sd_f1=("f1", "std"),
        mean_tp=("tp", "mean"),
        mean_fp=("fp", "mean"),
        mean_fn=("fn", "mean"),
        mean_gt_count=("gt_count", "mean"),
        mean_pred_count=("pred_count", "mean")
    )
    .reset_index()
)

save_table(
    main_results,
    "23_main_results_table.csv"
)


# ============================================================
# 32. WRITE HUMAN-READABLE SUMMARY
# ============================================================

summary_file = os.path.join(
    SUMMARY_DIR,
    "results_summary.txt"
)

with open(summary_file, "w", encoding="utf-8") as f:

    f.write("RESULTS ANALYSIS SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        f"Questions: {df['mbe_id'].nunique():,}\n"
    )

    f.write(
        f"Variants: {df['idx'].nunique():,}\n"
    )

    f.write(
        f"Demographic categories: "
        f"{df['biased_category'].nunique()}\n"
    )

    f.write(
        f"Models: {df['model'].nunique()}\n"
    )

    f.write(
        f"Thresholds: {df['threshold'].nunique()}\n"
    )

    f.write(
        f"Runs: {df['run'].nunique()}\n\n"
    )

    f.write("OVERALL MODEL PERFORMANCE\n")
    f.write("-" * 70 + "\n")

    for _, row in overall.iterrows():

        f.write(
            f"{row['model']}: "
            f"Macro P={row['precision_macro']:.4f}, "
            f"Macro R={row['recall_macro']:.4f}, "
            f"Macro F1={row['f1_macro']:.4f}\n"
        )

    f.write("\nDEMOGRAPHIC PERFORMANCE GAPS\n")
    f.write("-" * 70 + "\n")

    for _, row in demographic_gap.iterrows():

        f.write(
            f"{row['model']}: "
            f"F1 gap={row['f1_gap']:.4f}; "
            f"max={row['category_max_f1']}; "
            f"min={row['category_min_f1']}\n"
        )

    f.write("\nTHRESHOLD SENSITIVITY\n")
    f.write("-" * 70 + "\n")

    for _, row in threshold_range.iterrows():

        f.write(
            f"{row['model']}: "
            f"F1 range={row['f1_range']:.4f}\n"
        )

    f.write("\nNON-ZERO F1 AND ZERO-F1 CASES\n")
    f.write("-" * 70 + "\n")

    for _, row in nonzero_f1_results.iterrows():

        mean_nonzero = row["mean_f1_nonzero"]

        if pd.isna(mean_nonzero):
            mean_nonzero_text = "NA"
        else:
            mean_nonzero_text = f"{mean_nonzero:.4f}"

        f.write(
            f"{row['model']}: "
            f"Mean F1 among non-zero cases={mean_nonzero_text}; "
            f"Zero-F1 cases={row['pct_f1_zero']:.2f}% "
            f"({int(row['n_zero_f1'])}/{int(row['total_cases'])})\n"
        )


# ============================================================
# 33. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nAll results saved in: {OUTPUT_DIR}/")

print("\nTables:")
print(f"    {TABLE_DIR}/")

print("\nFigures:")
print(f"    {FIGURE_DIR}/")

print("\nSummaries:")
print(f"    {SUMMARY_DIR}/")

print("\nKey outputs include:")
print("  01_overall_model_performance.csv")
print("  04_model_by_demographic_category.csv")
print("  05_demographic_performance_gap.csv")
print("  07_model_by_difficulty.csv")
print("  08_threshold_sensitivity.csv")
print("  11_model_run_variability.csv")
print("  14_question_demographic_variability.csv")
print("  16_error_analysis_by_demographic_category.csv")
print("  18_counterfactual_lexical_consistency.csv")
print("  20_paired_demographic_tests.csv")
print("  21_friedman_demographic_test.csv")
print("  22_paired_effect_sizes.csv")
print("  24_nonzero_f1_and_zero_f1_by_model.csv")
print("  25_nonzero_f1_and_zero_f1_by_model_category.csv")


print("\nFigures include:")
print("  01_model_demographic_heatmap.png")
print("  02_difficulty_effect.png")
print("  03_threshold_sensitivity.png")
print("  04_run_variability.png")
print("  05_entity_count_difference.png")
print("  06_question_demographic_variability.png")
print("  07_fp_fn_by_demographic_category.png")
