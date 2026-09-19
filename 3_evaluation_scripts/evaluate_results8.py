import pandas as pd
import os

# ============================================================
# Configuration
# ============================================================

GEMINI_3_1_FLASH_LITE = "final_results/gemini_3.1_flash_lite_results_categorized.tsv"
GPT_5_MINI = "final_results/gpt_5_mini_results_categorized.tsv"
CLAUDE_HAIKU_4_5 = "final_results/claude_haiku_4.5_results_categorized.tsv"
LLAMA_4_SCOUT = "final_results/llama_4_scout_results_categorized.tsv"
LLAMA_4_MAVERICK = "final_results/llama_4_maverick_results_categorized.tsv"

OUTPUT_FILE = "f1_by_biased_category_and_difficulty.tsv"

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

    df["difficulty"] = (
        df["difficulty"]
        .astype(str)
        .str.strip()
    )

    # Make sure F1 is numeric
    df["f1"] = pd.to_numeric(
        df["f1"],
        errors="coerce"
    )

    # Remove rows with missing values
    df = df.dropna(
        subset=["f1", "biased_category", "difficulty"]
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
# Variant = model + mbe_id + biased_category + difficulty
# ============================================================

all_zero_variants = (
    df.groupby(
        ["model", "mbe_id", "biased_category", "difficulty"]
    )["f1"]
    .transform(lambda x: (x == 0).all())
)

df = df[~all_zero_variants].copy()

# ============================================================
# Calculate mean F1 for each variant/question
# ============================================================

question_means = (
    df.groupby(
        ["model", "mbe_id", "biased_category", "difficulty"]
    )["f1"]
    .mean()
    .reset_index(name="mean_f1")
)

# ============================================================
# Calculate statistics
# ============================================================

stats = (
    question_means
    .groupby(
        ["model", "biased_category", "difficulty"]
    )["mean_f1"]
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
# Sort results
# ============================================================

difficulty_order = {
    "easy": 0,
    "medium": 1,
    "hard": 2
}

stats["_difficulty_order"] = (
    stats["difficulty"]
    .str.lower()
    .map(difficulty_order)
    .fillna(99)
)

stats = (
    stats
    .sort_values(
        ["model", "biased_category", "_difficulty_order"]
    )
    .drop(columns="_difficulty_order")
)

# ============================================================
# Save results
# ============================================================

stats.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

# ============================================================
# Print results
# ============================================================

print("\nF1 Statistics by Model, Biased Category, and Difficulty")
print("=" * 100)
print(stats.to_string(index=False))

print(f"\nResults saved to: {OUTPUT_FILE}")
