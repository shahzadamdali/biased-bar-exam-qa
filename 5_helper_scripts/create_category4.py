import pandas as pd

df = pd.read_csv("final_results/llama_4_scout_results_categorized.tsv", sep="\t")
df.insert(df.columns.get_loc("model") + 1, "threshold", 0.9)
df.to_csv("final_results/llama_4_scout_th0.9_categorized.tsv", sep="\t", index=False)
