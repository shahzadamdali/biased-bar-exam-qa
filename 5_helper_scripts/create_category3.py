import pandas as pd

tsv = pd.read_csv("final_results/llama_4_scout_results.tsv", sep="\t")
csv = pd.read_csv("create_category2.csv")

difficulty_map = csv.set_index("idx")["difficulty"]
tsv["difficulty"] = tsv["mbe_id"].map(difficulty_map)

# Move difficulty to 4th column, after biased_category
cols = list(tsv.columns)
cols.remove("difficulty")
cols.insert(cols.index("biased_category") + 1, "difficulty")
tsv = tsv[cols]

tsv.to_csv("final_results/llama_4_scout_results_categorized.tsv", sep="\t", index=False)

matched = tsv["difficulty"].notna().sum()
unmatched = tsv["difficulty"].isna().sum()

print(f"Total rows: {len(tsv)}")
print(f"Matched: {matched}")
print(f"Unmatched: {unmatched}")
