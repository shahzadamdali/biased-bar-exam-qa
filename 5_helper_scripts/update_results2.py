import json

# Configuration
MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
TEMPERATURE = 0.7

INPUT_FILE = "retrieval_results14.jsonl"
OUTPUT_FILE = "retrieval_results14_updated.jsonl"

# Bias category mapping
bias_mapping = {
    1: "black",
    2: "white",
    3: "arab",
    4: "asian",
    5: "hispanic",
    6: "wealthy",
    7: "low-income",
    8: "homeless",
    9: "immigrant",
    10: "female"
}

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in infile:
        if not line.strip():
            continue

        # Read one JSONL record
        row = json.loads(line)

        # Get idx, e.g. "mbe_0_10"
        idx = row["idx"]

        # Extract the last number from idx
        # e.g. mbe_0_10 -> 10
        bias_id = int(idx.split("_")[-1])

        # Look up the corresponding bias category
        biased_category = bias_mapping.get(bias_id, None)

        # Convert results into numbered runs
        runs = []

        for run_number, result in enumerate(
            row.get("results", []),
            start=1
        ):
            runs.append({
                "run": run_number,
                "entities": result.get("entities", [])
            })

        # Build the new structure
        updated_row = {
            "idx": idx,
            "metadata": {
                "model": MODEL,
                "temperature": TEMPERATURE,
                "biased_category": biased_category
            },
            "runs": runs
        }

        # Write one JSON object per line
        outfile.write(
            json.dumps(updated_row, ensure_ascii=False) + "\n"
        )

print(f"Converted JSONL saved to: {OUTPUT_FILE}")