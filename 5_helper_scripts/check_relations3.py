import json

file_path = "retrieval_results10.jsonl"

bad_rows = []
total_rows = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
        if not line.strip():
            continue

        total_rows += 1
        row = json.loads(line)

        result_count = len(row.get("results", []))

        if result_count != 5:
            bad_rows.append({
                "line": line_num,
                "idx": row.get("idx"),
                "result_count": result_count
            })

print(f"Total rows: {total_rows}")
print(f"Rows with exactly 5 results: {total_rows - len(bad_rows)}")
print(f"Rows with != 5 results: {len(bad_rows)}")

if bad_rows:
    print("\nRows that do not have exactly 5 results:")
    for row in bad_rows:
        print(
            f"Line {row['line']} | "
            f"idx={row['idx']} | "
            f"results={row['result_count']}"
        )
else:
    print("\nAll rows have exactly 5 results.")