import csv
import json
import re
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

QA_FILE = Path("qa.csv")

# Directory containing your retrieval files
RESULTS_DIR = Path(".")

# Where validation outputs will be saved
OUTPUT_DIR = Path("validation_results")


# ============================================================
# LOAD QA INDICES
# ============================================================

def get_qa_indices(path):
    """Return all idx values from qa.csv."""

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        return {
            row["idx"].strip()
            for row in reader
            if row.get("idx") and row["idx"].strip()
        }


# ============================================================
# LOAD GT INDICES
# ============================================================

def get_gt_indices(path):
    """Return all top-level idx values from GT JSON."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return set(data.keys())


# ============================================================
# LOAD UPDATED JSONL
# ============================================================

def get_updated_data(path):
    """
    Read updated JSONL.

    Expected format:

        mbe_0_1
        mbe_0_2
        ...
        mbe_0_10

    These are treated as variants of:

        mbe_0
    """

    base_indices = set()
    raw_indices = set()

    # base_idx -> set of variant numbers
    variants = {}

    invalid_json_lines = []

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as e:

                invalid_json_lines.append({
                    "line": line_number,
                    "error": str(e)
                })

                continue

            idx = record.get("idx")

            if not idx:
                continue

            raw_indices.add(idx)

            # ------------------------------------------------
            # Expected:
            #
            # mbe_0_1
            # mbe_0_2
            # mbe_0_10
            # ------------------------------------------------

            match = re.match(r"^(.+)_([0-9]+)$", idx)

            if match:

                base_idx = match.group(1)
                variant_number = int(match.group(2))

                base_indices.add(base_idx)

                variants.setdefault(
                    base_idx,
                    set()
                ).add(variant_number)

            else:

                # Unexpected index with no variant number
                base_indices.add(idx)

                variants.setdefault(
                    idx,
                    set()
                )

    return {
        "base_indices": base_indices,
        "raw_indices": raw_indices,
        "variants": variants,
        "invalid_json_lines": invalid_json_lines,
    }


# ============================================================
# FIND INVALID INDICES
# ============================================================

def find_invalid_indices(
    qa_indices,
    gt_indices,
    updated_data,
):
    """
    Determine which indices need to be re-processed.

    An index is considered invalid if:

    1. It exists in QA but not GT
       OR

    2. It exists in GT but not updated JSONL
       OR

    3. It has anything other than variants 1..10
       in updated JSONL.

    The goal is to produce the indices that need work.
    """

    updated_indices = updated_data["base_indices"]
    variants = updated_data["variants"]

    invalid = set()

    reasons = {}

    # --------------------------------------------------------
    # 1. QA -> GT
    #
    # If QA has an idx that GT doesn't have, it cannot be
    # recovered from the updated file.
    # --------------------------------------------------------

    qa_missing_from_gt = qa_indices - gt_indices

    for idx in qa_missing_from_gt:

        invalid.add(idx)

        reasons.setdefault(idx, []).append(
            "missing_from_gt"
        )

    # --------------------------------------------------------
    # 2. GT -> UPDATED
    #
    # This is the main check for missing examples.
    # --------------------------------------------------------

    gt_missing_from_updated = gt_indices - updated_indices

    for idx in gt_missing_from_updated:

        invalid.add(idx)

        reasons.setdefault(idx, []).append(
            "missing_from_updated"
        )

    # --------------------------------------------------------
    # 3. Check variants
    #
    # Every GT idx should have:
    #
    #     _1
    #     _2
    #     ...
    #     _10
    # --------------------------------------------------------

    expected_variants = set(range(1, 11))

    for idx in gt_indices:

        actual_variants = variants.get(idx, set())

        missing_variants = (
            expected_variants - actual_variants
        )

        extra_variants = (
            actual_variants - expected_variants
        )

        if missing_variants or extra_variants:

            invalid.add(idx)

            reason = {
                "type": "incorrect_variants",
                "actual_count": len(actual_variants),
                "actual_variants": sorted(actual_variants),
                "missing_variants": sorted(
                    missing_variants
                ),
                "extra_variants": sorted(
                    extra_variants
                ),
            }

            reasons.setdefault(
                idx,
                []
            ).append(reason)

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return sorted(invalid), reasons


# ============================================================
# VALIDATE ONE RETRIEVAL FILE
# ============================================================

def validate_retrieval(
    retrieval_number,
    qa_indices,
    gt_file,
    updated_file,
):

    gt_indices = get_gt_indices(
        gt_file
    )

    updated_data = get_updated_data(
        updated_file
    )

    updated_indices = updated_data[
        "base_indices"
    ]

    # --------------------------------------------------------
    # Basic set checks
    # --------------------------------------------------------

    qa_missing_from_gt = sorted(
        qa_indices - gt_indices
    )

    gt_missing_from_updated = sorted(
        gt_indices - updated_indices
    )

    gt_not_in_qa = sorted(
        gt_indices - qa_indices
    )

    updated_not_in_gt = sorted(
        updated_indices - gt_indices
    )

    # --------------------------------------------------------
    # Find invalid indices
    # --------------------------------------------------------

    invalid_indices, invalid_reasons = (
        find_invalid_indices(
            qa_indices=qa_indices,
            gt_indices=gt_indices,
            updated_data=updated_data,
        )
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    valid_indices = sorted(
        gt_indices - set(invalid_indices)
    )

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    overall_pass = len(invalid_indices) == 0

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {

        "retrieval_number": retrieval_number,

        "files": {
            "qa": str(QA_FILE),
            "gt": str(gt_file),
            "updated": str(updated_file),
        },

        "counts": {

            "qa_indices": len(
                qa_indices
            ),

            "gt_indices": len(
                gt_indices
            ),

            "updated_base_indices": len(
                updated_indices
            ),

            "updated_raw_variants": len(
                updated_data["raw_indices"]
            ),

            "valid_indices": len(
                valid_indices
            ),

            "invalid_indices": len(
                invalid_indices
            ),
        },

        "checks": {

            "qa_subset_of_gt": {
                "passed": (
                    len(qa_missing_from_gt) == 0
                ),
                "missing_indices": (
                    qa_missing_from_gt
                ),
            },

            "gt_subset_of_updated": {
                "passed": (
                    len(gt_missing_from_updated) == 0
                ),
                "missing_indices": (
                    gt_missing_from_updated
                ),
            },

            "gt_subset_of_qa": {
                "passed": (
                    len(gt_not_in_qa) == 0
                ),
                "extra_indices": (
                    gt_not_in_qa
                ),
            },

            "updated_subset_of_gt": {
                "passed": (
                    len(updated_not_in_gt) == 0
                ),
                "extra_indices": (
                    updated_not_in_gt
                ),
            },

            "invalid_json_lines": {
                "passed": (
                    len(
                        updated_data[
                            "invalid_json_lines"
                        ]
                    ) == 0
                ),
                "lines": (
                    updated_data[
                        "invalid_json_lines"
                    ]
                ),
            },
        },

        "overall_pass": overall_pass,

        # ----------------------------------------------------
        # MOST IMPORTANT FOR YOUR NEXT STEP
        # ----------------------------------------------------

        "invalid_indices": invalid_indices,

        "invalid_reasons": invalid_reasons,

        "valid_indices": valid_indices,
    }

    return report


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load QA once
    # --------------------------------------------------------

    qa_indices = get_qa_indices(
        QA_FILE
    )

    # --------------------------------------------------------
    # Find all GT files
    # --------------------------------------------------------

    gt_files = sorted(
        RESULTS_DIR.glob(
            "retrieval_results*_gt.json"
        )
    )

    if not gt_files:

        print(
            "No retrieval_results*_gt.json files found."
        )

        return

    all_reports = {}

    # This will become:
    #
    # {
    #     "12": ["mbe_5", "mbe_18"],
    #     "13": ["mbe_2"],
    #     "14": ["mbe_7", "mbe_9"]
    # }

    all_invalid_indices = {}

    # --------------------------------------------------------
    # Process every retrieval number
    # --------------------------------------------------------

    for gt_file in gt_files:

        match = re.match(
            r"retrieval_results(.+)_gt\.json$",
            gt_file.name
        )

        if not match:
            continue

        retrieval_number = match.group(1)

        updated_file = (
            RESULTS_DIR
            / f"retrieval_results"
              f"{retrieval_number}"
              f"_updated.jsonl"
        )

        # ----------------------------------------------------
        # Missing updated file
        # ----------------------------------------------------

        if not updated_file.exists():

            print(
                f"SKIP {retrieval_number}: "
                f"{updated_file.name} not found"
            )

            all_reports[
                retrieval_number
            ] = {
                "retrieval_number":
                    retrieval_number,

                "overall_pass":
                    False,

                "error":
                    f"Missing file: "
                    f"{updated_file.name}",
            }

            # No individual indices can be reliably
            # determined without the updated file.
            all_invalid_indices[
                retrieval_number
            ] = []

            continue

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        print(
            f"Checking retrieval_results"
            f"{retrieval_number}..."
        )

        report = validate_retrieval(
            retrieval_number=(
                retrieval_number
            ),

            qa_indices=qa_indices,

            gt_file=gt_file,

            updated_file=updated_file,
        )

        all_reports[
            retrieval_number
        ] = report

        all_invalid_indices[
            retrieval_number
        ] = report[
            "invalid_indices"
        ]

        # ----------------------------------------------------
        # Save individual report
        # ----------------------------------------------------

        individual_file = (
            OUTPUT_DIR
            / f"retrieval_results"
              f"{retrieval_number}"
              f"_validation.json"
        )

        with open(
            individual_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"  Invalid indices: "
            f"{len(report['invalid_indices'])}"
        )

    # ========================================================
    # SAVE COMBINED VALIDATION SUMMARY
    # ========================================================

    combined_file = (
        OUTPUT_DIR
        / "validation_summary.json"
    )

    with open(
        combined_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_reports,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # SAVE INVALID INDICES
    # ========================================================

    invalid_file = (
        OUTPUT_DIR
        / "invalid_indices.json"
    )

    with open(
        invalid_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_invalid_indices,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # SAVE A FLAT LIST
    # ========================================================
    #
    # Useful if you want to inspect all problematic examples
    # across all retrieval runs.

    flat_invalid = []

    for retrieval_number, indices in (
        all_invalid_indices.items()
    ):

        for idx in indices:

            flat_invalid.append({
                "retrieval_number":
                    retrieval_number,

                "idx":
                    idx,
            })

    flat_invalid_file = (
        OUTPUT_DIR
        / "invalid_indices_flat.json"
    )

    with open(
        flat_invalid_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            flat_invalid,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Processed: {len(all_reports)} retrieval runs"
    )

    print(
        f"Invalid-index file: "
        f"{invalid_file}"
    )

    print(
        f"Full validation summary: "
        f"{combined_file}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
