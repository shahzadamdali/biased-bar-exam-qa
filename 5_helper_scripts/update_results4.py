from pathlib import Path

input_dir = Path("final_results_tsv_only")
output_file = Path("final_results/combined_results.tsv")

files = sorted(input_dir.glob("*.tsv"))

with output_file.open("w", encoding="utf-8", newline="") as outfile:
    for i, file in enumerate(files):
        with file.open("r", encoding="utf-8") as infile:
            header = infile.readline()

            if i == 0:
                outfile.write(header)

            for line in infile:
                outfile.write(line)

print(f"Combined {len(files)} files into {output_file}")
