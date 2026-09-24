INPUT_FILE = "final_results/combined_results.tsv"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = f.read()

data = data.replace("models/gemini-3.1-flash-lite", "gemini-3.1-flash-lite")
data = data.replace("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "Llama-4-Maverick-17B-128E-Instruct-FP8")
data = data.replace("meta-llama/Llama-4-Scout-17B-16E-Instruct", "Llama-4-Scout-17B-16E-Instruct")

with open(INPUT_FILE, "w", encoding="utf-8") as f:
    f.write(data)
