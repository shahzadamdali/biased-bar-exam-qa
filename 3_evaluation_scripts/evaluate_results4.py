import pandas as pd
import os

# ============================================================
# Configuration
# ============================================================

GEMINI_3_1_FLASH_LITE = "final_results/gemini_3.1_flash_lite_results.tsv"
GPT_5_MINI = "final_results/gpt_5_mini_results.tsv"
CLAUDE_HAIKU_4_5 = "final_results/claude_haiku_4.5_results.tsv"
LLAMA_4_SCOUT = "final_results/llama_4_scout_results.tsv"
LLAMA_4_MAVERICK = "final_results/llama_4_maverick_results.tsv"

OUTPUT_FILE = "f1_by_biased_category.tsv"

# Map model names to their result files
MODEL_FILES = {
    "Gemini 3.1 Flash Lite": GEMINI_3_1_FLASH_LITE,
    "GPT-5 Mini": GPT_5_MINI,
    "Claude Haiku 4.5": CLAUDE_HAIKU_4_5,
    "Llama 4 Scout": LLAMA_4_SCOUT,
    "Llama 4 Maverick": LLAMA_4_MAVERICK,
}

# ============================================================
# Load all model results
# ============================================================

all_dfs = []

for model_name, input_file in MODEL_FILES.items():

    print(f"Reading: {input_file}")

    if not os.path.exists(input_file):
        print(f"WARNING: File not found, skipping: {input_file}")
        continue

    df = pd.read_csv(input_file, sep="\t")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Add model name
    df["model"] = model_name

    # Clean categorical values
    df["biased_category"] = (
        df["biased_category"]
        .astype(str)
        .str.strip()
    )

    # Make sure F1 is numeric
    df["f1"] = pd.to_numeric(
        df["f1"],
        errors="coerce"
    )

    # Remove rows where F1 or biased category is missing
    df = df.dropna(
        subset=["f1", "biased_category"]
    )

    all_dfs.append(df)

# ============================================================
# Combine all models
# ============================================================

if not all_dfs:
    raise ValueError("No input files were found.")

df = pd.concat(
    all_dfs,
    ignore_index=True
)

# ============================================================
# Remove variants where ALL runs have F1 = 0
#
# IMPORTANT:
# This is done separately for each model.
#
# A variant is identified by:
#   model + mbe_id + biased_category
#
# Example:
#
# Gemini 3.1 Flash Lite
# mbe_0_1:
#   run 1 -> 0
#   run 2 -> 0
#   run 3 -> 0
#   run 4 -> 0
#   run 5 -> 0
#
# -> Remove Gemini's mbe_0_1 variant
#
# But if:
#   0, 0, 0.3, 0, 0
#
# -> Keep the variant and all five runs.
# ============================================================

all_zero_variants = (
    df.groupby(
        ["model", "mbe_id", "biased_category"]
    )["f1"]
    .transform(lambda x: (x == 0).all())
)

# Keep only variants that are NOT all-zero
df = df[~all_zero_variants].copy()

# ============================================================
# Calculate mean F1 for each variant
#
# Each mbe_id + biased_category combination represents
# one variant/question.
# ============================================================

question_means = (
    df.groupby(
        ["model", "mbe_id", "biased_category"]
    )["f1"]
    .mean()
    .reset_index(name="mean_f1")
)

# ============================================================
# Calculate statistics for each model + biased category
# ============================================================

stats = (
    question_means
    .groupby(["model", "biased_category"])["mean_f1"]
    .agg(
        mean_f1="mean",
        std_f1="std",
        min_f1="min",
        max_f1="max",
        n="count"
    )
    .reset_index()
)

# ============================================================
# Append results to TSV
# ============================================================

# If the file already exists, append without writing the header.
# Otherwise, create a new file with the header.

file_exists = os.path.exists(OUTPUT_FILE)

stats.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False,
    mode="a",
    header=not file_exists
)

# ============================================================
# Print results
# ============================================================

print("\nF1 Statistics by Model and Biased Category")
print("=" * 80)
print(stats.to_string(index=False))

print(f"\nResults appended to: {OUTPUT_FILE}")
