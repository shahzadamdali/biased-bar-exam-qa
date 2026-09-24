# ============================================================
# MIXED-EFFECTS MODEL FOR DEMOGRAPHIC FAIRNESS
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

RESULTS_FILE = "final_results/combined_results.tsv"

df = pd.read_csv(RESULTS_FILE, sep="\t")

# Clean column names
df.columns = df.columns.str.strip()

# Numeric columns
numeric_cols = [
    "gt_count",
    "pred_count",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "threshold",
    "temperature",
    "run"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# 2. BASIC CHECKS
# ------------------------------------------------------------

print("=" * 70)
print("DATASET")
print("=" * 70)

print("Rows:", len(df))
print("Unique mbe_id:", df["mbe_id"].nunique())
print("Unique idx:", df["idx"].nunique())
print("Models:", df["model"].unique())
print("Thresholds:", sorted(df["threshold"].unique()))
print("Categories:", df["biased_category"].unique())
print("Runs:", sorted(df["run"].unique()))
print("Difficulties:", df["difficulty"].unique())

# ------------------------------------------------------------
# 3. VERIFY EXPERIMENTAL STRUCTURE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENTAL STRUCTURE")
print("=" * 70)

# Each mbe_id should have 10 demographic variants
variants_per_question = df.groupby("mbe_id")["idx"].nunique()

print("\nVariants per mbe_id:")
print(variants_per_question.value_counts().sort_index())

# Each idx should have 5 models
models_per_variant = df.groupby("idx")["model"].nunique()

print("\nModels per idx:")
print(models_per_variant.value_counts().sort_index())

# Each idx-model combination should have 5 thresholds
thresholds_per_condition = (
    df.groupby(["idx", "model"])["threshold"]
      .nunique()
)

print("\nThresholds per idx-model:")
print(thresholds_per_condition.value_counts().sort_index())

# Each idx-model-threshold combination should have 5 runs
runs_per_condition = (
    df.groupby(["idx", "model", "threshold"])["run"]
      .nunique()
)

print("\nRuns per idx-model-threshold:")
print(runs_per_condition.value_counts().sort_index())

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
# 4. AVERAGE ACROSS THE 5 RUNS
# ============================================================
#
# This is important.
#
# The 5 runs are repeated measurements of the SAME experimental
# condition. We do not want to treat them as 5 independent
# observations.
#
# Therefore, for the primary mixed-effects model, we average
# F1 across the 5 runs.
# ============================================================

group_cols = [
    "mbe_id",
    "idx",
    "biased_category",
    "difficulty",
    "model",
    "threshold"
]

df_avg = (
    df.groupby(group_cols, as_index=False)
      .agg(
          f1_mean=("f1", "mean"),
          precision_mean=("precision", "mean"),
          recall_mean=("recall", "mean"),
          tp_mean=("tp", "mean"),
          fp_mean=("fp", "mean"),
          fn_mean=("fn", "mean"),
          run_count=("run", "nunique")
      )
)

print("\n" + "=" * 70)
print("RUN-AVERAGED DATA")
print("=" * 70)

print("Rows before averaging:", len(df))
print("Rows after averaging:", len(df_avg))

print("\nExpected rows:")
print("1000 questions × 10 variants × 5 models × 5 thresholds")
print("= 250,000 condition-level observations")


# ============================================================
# 5. CONVERT VARIABLES TO CATEGORICAL
# ============================================================

df_avg["mbe_id"] = df_avg["mbe_id"].astype(str)
df_avg["idx"] = df_avg["idx"].astype(str)

df_avg["biased_category"] = df_avg["biased_category"].astype("category")
df_avg["difficulty"] = df_avg["difficulty"].astype("category")
df_avg["model"] = df_avg["model"].astype("category")
df_avg["threshold"] = df_avg["threshold"].astype("category")


# ============================================================
# 6. PRIMARY MIXED-EFFECTS MODEL
# ============================================================
#
# Fixed effects:
#
#   biased_category
#   model
#   threshold
#   difficulty
#
# Interactions:
#
#   biased_category × model
#   biased_category × threshold
#
# Random effects:
#
#   mbe_id  = shared underlying legal question
#   idx     = demographic variant of that question
#
# The idx random effect accounts for the fact that the same
# demographic variant is evaluated repeatedly across models
# and thresholds.
# ============================================================

formula = """
f1_mean ~
C(biased_category)
+ C(model)
+ C(threshold)
+ C(difficulty)
+ C(biased_category):C(model)
+ C(biased_category):C(threshold)
"""

print("\n" + "=" * 70)
print("FITTING MIXED-EFFECTS MODEL")
print("=" * 70)

print("\nFormula:")
print(formula)

# mbe_id = main grouping factor
#
# idx = variance component nested within mbe_id
#
# Since idx is globally unique, using C(idx) here gives each
# demographic variant its own random intercept.

model = smf.mixedlm(
    formula=formula,
    data=df_avg,
    groups=df_avg["mbe_id"],
    vc_formula={
        "variant": "0 + C(idx)"
    },
    re_formula="1"
)

result = model.fit(
    method="lbfgs",
    reml=False,
    maxiter=500,
    disp=True
)


# ============================================================
# 7. MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MIXED-EFFECTS MODEL RESULTS")
print("=" * 70)

print(result.summary())


# ============================================================
# 8. SAVE FULL MODEL SUMMARY
# ============================================================

OUTPUT_DIR = Path("final_results/analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

with open(
    OUTPUT_DIR / "mixed_effects_model_summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(str(result.summary()))


# ============================================================
# 9. FIXED-EFFECT COEFFICIENT TABLE
# ============================================================

fixed_effects = pd.DataFrame({
    "term": result.fe_params.index,
    "coefficient": result.fe_params.values,
    "std_error": result.bse_fe.values,
    "z": result.tvalues[:len(result.fe_params)].values,
    "p_value": result.pvalues[:len(result.fe_params)].values
})

# 95% confidence intervals
conf = result.conf_int()

fixed_effects["ci_lower"] = [
    conf.loc[x, 0]
    for x in result.fe_params.index
]

fixed_effects["ci_upper"] = [
    conf.loc[x, 1]
    for x in result.fe_params.index
]

fixed_effects.to_csv(
    OUTPUT_DIR / "mixed_effects_fixed_effects.tsv",
    sep="\t",
    index=False
)

print("\nFixed effects:")
print(fixed_effects.to_string(index=False))


# ============================================================
# 10. RANDOM-EFFECT VARIANCES
# ============================================================

print("\n" + "=" * 70)
print("RANDOM EFFECTS")
print("=" * 70)

print("\nQuestion-level random-effect variance:")
print(result.cov_re)

print("\nVariant-level variance components:")
print(result.vcomp)


# ============================================================
# 11. INTERPRET THE MAIN DEMOGRAPHIC EFFECTS
# ============================================================

print("\n" + "=" * 70)
print("DEMOGRAPHIC EFFECTS")
print("=" * 70)

demographic_terms = fixed_effects[
    fixed_effects["term"].str.contains("biased_category")
]

print(
    demographic_terms.to_string(index=False)
)


# ============================================================
# 12. MODEL × DEMOGRAPHIC INTERACTION
# ============================================================

print("\n" + "=" * 70)
print("DEMOGRAPHIC × MODEL INTERACTIONS")
print("=" * 70)

interaction_model = fixed_effects[
    fixed_effects["term"].str.contains(
        "biased_category.*model|model.*biased_category"
    )
]

print(
    interaction_model.to_string(index=False)
)


# ============================================================
# 13. DEMOGRAPHIC × THRESHOLD INTERACTION
# ============================================================

print("\n" + "=" * 70)
print("DEMOGRAPHIC × THRESHOLD INTERACTIONS")
print("=" * 70)

interaction_threshold = fixed_effects[
    fixed_effects["term"].str.contains(
        "biased_category.*threshold|threshold.*biased_category"
    )
]

print(
    interaction_threshold.to_string(index=False)
)


# ============================================================
# 14. MODEL DIAGNOSTICS
# ============================================================

print("\n" + "=" * 70)
print("MODEL DIAGNOSTICS")
print("=" * 70)

print("\nConverged:", result.converged)

print("\nLog-likelihood:", result.llf)

print("\nAIC:", result.aic)

print("\nBIC:", result.bic)

# Residuals
df_avg["fitted_f1"] = result.fittedvalues
df_avg["residual"] = (
    df_avg["f1_mean"] - df_avg["fitted_f1"]
)

print("\nResidual mean:", df_avg["residual"].mean())
print("Residual SD:", df_avg["residual"].std())


# ============================================================
# 15. SAVE DATA WITH MODEL PREDICTIONS
# ============================================================

df_avg.to_csv(
    OUTPUT_DIR / "mixed_effects_data_with_predictions.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 16. SIMPLE INTERPRETATION TABLE
# ============================================================

summary_rows = []

for _, row in fixed_effects.iterrows():

    term = row["term"]

    summary_rows.append({
        "term": term,
        "coefficient": row["coefficient"],
        "p_value": row["p_value"],
        "significant_0.05": row["p_value"] < 0.05,
        "ci_lower": row["ci_lower"],
        "ci_upper": row["ci_upper"]
    })

interpretation_df = pd.DataFrame(summary_rows)

interpretation_df.to_csv(
    OUTPUT_DIR / "mixed_effects_interpretation.tsv",
    sep="\t",
    index=False
)

print("\nSaved:")
print(OUTPUT_DIR / "mixed_effects_model_summary.txt")
print(OUTPUT_DIR / "mixed_effects_fixed_effects.tsv")
print(OUTPUT_DIR / "mixed_effects_interpretation.tsv")
print(OUTPUT_DIR / "mixed_effects_data_with_predictions.tsv")

# ============================================================
# PUBLICATION TABLES AND FIGURES
# ============================================================

FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)


# ============================================================
# TABLE 1
# MIXED-EFFECTS FIXED EFFECTS
# ============================================================

table1 = fixed_effects.copy()

table1["coefficient"] = table1["coefficient"].round(4)
table1["std_error"] = table1["std_error"].round(4)
table1["z"] = table1["z"].round(3)
table1["p_value"] = table1["p_value"].round(4)
table1["ci_lower"] = table1["ci_lower"].round(4)
table1["ci_upper"] = table1["ci_upper"].round(4)

table1.to_csv(
    TABLE_DIR / "Table_1_Mixed_Effects_Fixed_Effects.tsv",
    sep="\t",
    index=False
)

# Also save as HTML for easy viewing
table1.to_html(
    TABLE_DIR / "Table_1_Mixed_Effects_Fixed_Effects.html",
    index=False
)

print("\nTable 1 saved.")


# ============================================================
# TABLE 2
# DEMOGRAPHIC CATEGORY × MODEL
# ============================================================

table2 = (
    df_avg
    .groupby(["biased_category", "model"], observed=True)
    .agg(
        mean_f1=("f1_mean", "mean"),
        sd_f1=("f1_mean", "std"),
        n=("f1_mean", "count")
    )
    .reset_index()
)

table2["se"] = table2["sd_f1"] / np.sqrt(table2["n"])

table2["ci_lower"] = (
    table2["mean_f1"] - 1.96 * table2["se"]
)

table2["ci_upper"] = (
    table2["mean_f1"] + 1.96 * table2["se"]
)

for col in [
    "mean_f1",
    "sd_f1",
    "se",
    "ci_lower",
    "ci_upper"
]:
    table2[col] = table2[col].round(4)

table2.to_csv(
    TABLE_DIR / "Table_2_Demographic_Model_F1.tsv",
    sep="\t",
    index=False
)

table2.to_html(
    TABLE_DIR / "Table_2_Demographic_Model_F1.html",
    index=False
)

print("Table 2 saved.")


# ============================================================
# TABLE 3
# MODEL × THRESHOLD
# ============================================================

table3 = (
    df_avg
    .groupby(["model", "threshold"], observed=True)
    .agg(
        mean_f1=("f1_mean", "mean"),
        sd_f1=("f1_mean", "std"),
        n=("f1_mean", "count")
    )
    .reset_index()
)

table3["se"] = table3["sd_f1"] / np.sqrt(table3["n"])

table3["ci_lower"] = (
    table3["mean_f1"] - 1.96 * table3["se"]
)

table3["ci_upper"] = (
    table3["mean_f1"] + 1.96 * table3["se"]
)

for col in [
    "mean_f1",
    "sd_f1",
    "se",
    "ci_lower",
    "ci_upper"
]:
    table3[col] = table3[col].round(4)

table3.to_csv(
    TABLE_DIR / "Table_3_Model_Threshold_F1.tsv",
    sep="\t",
    index=False
)

print("Table 3 saved.")


# ============================================================
# FIGURE 1
# F1 BY DEMOGRAPHIC CATEGORY AND MODEL
# ============================================================

plt.figure(figsize=(14, 7))

sns.barplot(
    data=df_avg,
    x="biased_category",
    y="f1_mean",
    hue="model",
    hue_order=MODEL_ORDER,
    palette=MODEL_COLORS,
    errorbar=("ci", 95)
)

plt.xlabel("Demographic category")
plt.ylabel("F1 score")
plt.title("Entity Retrieval F1 by Demographic Category and Model")

plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()

plt.savefig(
    FIG_DIR / "Figure_1_F1_Demographic_Category_Model.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Figure 1 saved.")


# ============================================================
# FIGURE 2
# F1 ACROSS MATCHING THRESHOLDS
# ============================================================

threshold_plot = (
    df_avg
    .groupby(
        ["model", "threshold"],
        observed=True
    )
    .agg(
        mean_f1=("f1_mean", "mean"),
        sd_f1=("f1_mean", "std"),
        n=("f1_mean", "count")
    )
    .reset_index()
)

threshold_plot["se"] = (
    threshold_plot["sd_f1"] /
    np.sqrt(threshold_plot["n"])
)

threshold_plot["ci_lower"] = (
    threshold_plot["mean_f1"] -
    1.96 * threshold_plot["se"]
)

threshold_plot["ci_upper"] = (
    threshold_plot["mean_f1"] +
    1.96 * threshold_plot["se"]
)

plt.figure(figsize=(11, 7))

for model_name in threshold_plot["model"].unique():

    temp = threshold_plot[
        threshold_plot["model"] == model_name
    ].sort_values("threshold")

    plt.plot(
        temp["threshold"],
        temp["mean_f1"],
        marker="o",
        label=model_name
    )

    plt.fill_between(
        temp["threshold"].astype(float),
        temp["ci_lower"],
        temp["ci_upper"],
        alpha=0.10
    )

plt.xlabel("Hungarian matching threshold")
plt.ylabel("Mean F1 score")
plt.title("Entity Retrieval Performance Across Matching Thresholds")

plt.ylim(0, 1)
plt.legend(title="Model")

plt.tight_layout()

plt.savefig(
    FIG_DIR / "Figure_2_F1_Threshold.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Figure 2 saved.")


# ============================================================
# FIGURE 3
# DEMOGRAPHIC × MODEL INTERACTION
# ============================================================

interaction_dm = (
    df_avg
    .groupby(
        ["biased_category", "model"],
        observed=True
    )["f1_mean"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(14, 8))

sns.pointplot(
    data=interaction_dm,
    x="biased_category",
    y="f1_mean",
    hue="model",
    palette=MODEL_COLORS,
    hue_order=MODEL_ORDER,
    dodge=0.3,
    markers="o",
    linestyles="-"
)

plt.xlabel("Demographic category")
plt.ylabel("Mean F1 score")
plt.title("Demographic Category × Model Interaction")

plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)

plt.legend(
    title="Model",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR / "Figure_3_Demographic_Model_Interaction.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Figure 3 saved.")


# ============================================================
# FIGURE 4
# DEMOGRAPHIC × THRESHOLD INTERACTION
# ============================================================

interaction_dt = (
    df_avg
    .groupby(
        ["biased_category", "threshold"],
        observed=True
    )["f1_mean"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(14, 8))

sns.lineplot(
    data=interaction_dt,
    x="threshold",
    y="f1_mean",
    hue="biased_category",
    marker="o"
)

plt.xlabel("Hungarian matching threshold")
plt.ylabel("Mean F1 score")
plt.title("Demographic Category × Matching Threshold")

plt.ylim(0, 1)

plt.legend(
    title="Demographic category",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR / "Figure_4_Demographic_Threshold_Interaction.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Figure 4 saved.")


# ============================================================
# FIGURE 5
# QUESTION-LEVEL DEMOGRAPHIC VARIABILITY
# ============================================================
#
# For each legal question, calculate the range of F1 across
# its demographic variants.
#
# This is especially important for your fairness argument:
#
#     high range = predictions change substantially when
#                  only demographic information changes
#
#     low range = more demographic consistency
#
# ============================================================

question_variability = (
    df_avg
    .groupby(
        ["mbe_id", "model", "threshold"],
        observed=True
    )
    .agg(
        max_f1=("f1_mean", "max"),
        min_f1=("f1_mean", "min"),
        mean_f1=("f1_mean", "mean")
    )
    .reset_index()
)

question_variability["demographic_f1_range"] = (
    question_variability["max_f1"] -
    question_variability["min_f1"]
)

plt.figure(figsize=(11, 7))

sns.boxplot(
    data=question_variability,
    x="model",
    y="demographic_f1_range"
)

plt.xlabel("Model")
plt.ylabel("F1 range across demographic variants")
plt.title(
    "Question-Level Demographic Sensitivity"
)

plt.xticks(rotation=30, ha="right")

plt.tight_layout()

plt.savefig(
    FIG_DIR / "Figure_5_Demographic_Sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Figure 5 saved.")


# ============================================================
# TABLE 4
# QUESTION-LEVEL DEMOGRAPHIC VARIABILITY
# ============================================================

table4 = (
    question_variability
    .groupby("model", observed=True)
    .agg(
        mean_range=("demographic_f1_range", "mean"),
        median_range=("demographic_f1_range", "median"),
        sd_range=("demographic_f1_range", "std"),
        max_range=("demographic_f1_range", "max"),
        n_questions=("demographic_f1_range", "count")
    )
    .reset_index()
)

for col in [
    "mean_range",
    "median_range",
    "sd_range",
    "max_range"
]:
    table4[col] = table4[col].round(4)

table4.to_csv(
    TABLE_DIR / "Table_4_Demographic_Sensitivity.tsv",
    sep="\t",
    index=False
)

table4.to_html(
    TABLE_DIR / "Table_4_Demographic_Sensitivity.html",
    index=False
)

print("Table 4 saved.")


# ============================================================
# FIGURE 6
# DIFFICULTY × MODEL
# ============================================================

difficulty_plot = (
    df_avg
    .groupby(
        ["difficulty", "model"],
        observed=True
    )
    .agg(
        mean_f1=("f1_mean", "mean"),
        sd_f1=("f1_mean", "std"),
        n=("f1_mean", "count")
    )
    .reset_index()
)

difficulty_plot["se"] = (
    difficulty_plot["sd_f1"] /
    np.sqrt(difficulty_plot["n"])
)

plt.figure(figsize=(11, 7))

sns.barplot(
    data=difficulty_plot,
    x="difficulty",
    y="mean_f1",
    hue="model",
    hue_order=MODEL_ORDER,
    palette=MODEL_COLORS,
    errorbar=("ci", 95)
)

plt.xlabel("Question difficulty")
plt.ylabel("Mean F1 score")
plt.title("Entity Retrieval F1 by Question Difficulty and Model")

plt.ylim(0, 1)

plt.legend(
    title="Model",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR / "Figure_6_Difficulty_Model.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Figure 6 saved.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ALL TABLES AND FIGURES CREATED")
print("=" * 70)

print("\nTables:")
for f in sorted(TABLE_DIR.iterdir()):
    print("  ", f.name)

print("\nFigures:")
for f in sorted(FIG_DIR.iterdir()):
    print("  ", f.name)
