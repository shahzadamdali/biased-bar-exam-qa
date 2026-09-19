import json

input_file = "retrieval_results10.json"
output_file = "retrieval_results10.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as f:
    for key, value in data.items():
        record = {
            "idx": key,
            "results": value
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved JSONL to {output_file}")