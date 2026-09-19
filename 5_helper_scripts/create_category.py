import pandas as pd

df = pd.read_csv("qa.csv")

df["modified_prompt"] = df["prompt"].fillna("") + " " + df["question"].fillna("")

df = df.drop(columns=["prompt", "question", "dataset", "example_id", "prompt_id", "source", "subject", "question_number", "choice_a", "choice_b", "choice_c", "choice_d", "answer", "gold_passage", "gold_idx"])
df.to_csv("create_category.csv", index=False)
