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

OUTPUT_FILE = "f1_by_difficulty.tsv"

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
    df.columns = df.columns.str.strip()

    df["model"] = model_name

    df["difficulty"] = (
        df["difficulty"]
        .astype(str)
        .str.strip()
    )

    df["f1"] = pd.to_numeric(
        df["f1"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["f1", "difficulty"]
    )

    all_dfs.append(df)

if not all_dfs:
    raise ValueError("No input files were found.")

df = pd.concat(all_dfs, ignore_index=True)

# ============================================================
# Remove variants where ALL runs have F1 = 0
# ============================================================

all_zero_variants = (
    df.groupby(
        ["model", "mbe_id", "difficulty"]
    )["f1"]
    .transform(lambda x: (x == 0).all())
)

df = df[~all_zero_variants].copy()

# ============================================================
# Calculate mean F1 for each variant/question
# ============================================================

question_means = (
    df.groupby(
        ["model", "mbe_id", "difficulty"]
    )["f1"]
    .mean()
    .reset_index(name="mean_f1")
)

# ============================================================
# Calculate statistics by model + difficulty
# ============================================================

stats = (
    question_means
    .groupby(["model", "difficulty"])["mean_f1"]
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
# Write results
# ============================================================

stats.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

print("\nF1 Statistics by Model and Difficulty")
print("=" * 80)
print(stats.to_string(index=False))

print(f"\nResults saved to: {OUTPUT_FILE}")
