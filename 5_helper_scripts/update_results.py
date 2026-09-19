import json

# Configuration
MODEL = "models/gemini-3.1-flash-lite"
TEMPERATURE = 0.7

INPUT_FILE = "retrieval_results10.jsonl"
OUTPUT_FILE = "retrieval_results10_updated.jsonl"


with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in infile:
        if not line.strip():
            continue

        # Read one JSONL record
        row = json.loads(line)

        # Convert results into numbered runs
        runs = []

        for run_number, result in enumerate(row.get("results", []), start=1):
            runs.append({
                "run": run_number,
                "entities": result.get("entities", [])
            })

        # Build the new structure
        updated_row = {
            "idx": row["idx"],
            "metadata": {
                "model": MODEL,
                "temperature": TEMPERATURE
            },
            "runs": runs
        }

        # Write one JSON object per line
        outfile.write(
            json.dumps(updated_row, ensure_ascii=False) + "\n"
        )

print(f"Converted JSONL saved to: {OUTPUT_FILE}")