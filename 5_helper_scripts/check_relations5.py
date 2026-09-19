import json

with open("validation_results/validation_summary.json", "r") as f:
    data = json.load(f)

for file_number, file_data in data.items():
    print(f"\nFile: {file_number}")

    invalid_reasons = file_data.get("invalid_reasons", {})

    for mbe_number, reasons in invalid_reasons.items():
        for reason in reasons:
            if isinstance(reason, str):
                print(f"{mbe_number} - {reason}")

