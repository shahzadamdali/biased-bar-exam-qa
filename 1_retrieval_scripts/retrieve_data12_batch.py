import pandas as pd
import json
import anthropic
from dotenv import load_dotenv
import os
import time
import re
import sys


# ============================================================
# CONFIG
# ============================================================

load_dotenv(".env", override=True)

API_KEY = os.getenv("ANTHROPIC_API_KEY")

CSV_FILE = "biased_final2.csv"
OUTPUT_FILE = "retrieval_results12.jsonl"

# Used to keep track of a batch that has been submitted
# but has not yet been processed by this script.
STATE_FILE = "batch_state.json"

MODEL = "claude-haiku-4-5"

N_RUNS = 5

# Number of independent API requests submitted in one batch.
#
# 5,000 requests = approximately 1,000 questions × 5 runs.
#
# Anthropic currently allows up to 100,000 requests per batch,
# but smaller batches make recovery/debugging easier.
BATCH_SIZE = 5000

# How often to check whether a batch has finished.
POLL_SECONDS = 60

MAX_RETRY_BATCHES = 20


# ============================================================
# VALIDATE CONFIG
# ============================================================

if not API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY was not found in your .env file."
    )


# ============================================================
# LOAD DATA
# ============================================================

print("Loading CSV...")

df = pd.read_csv(CSV_FILE)

print(f"Loaded {len(df)} rows.")


# ============================================================
# ANTHROPIC CLIENT
# ============================================================

client = anthropic.Anthropic(
    api_key=API_KEY
)


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
You are generating retrieval queries for a legal knowledge graph.

- The knowledge graph was built from explanatory legal text, not fact patterns.
- Given a fact pattern, infer the canonical legal nodes that are most likely to appear in the relevant knowledge graph.
- Do not merely extract nouns or people mentioned in the fact pattern.
- Instead, identify the governing legal concepts, doctrines, rules, standards, tests, elements, procedures, rights, duties, defenses, exceptions, statutes, cases, courts, parties, and other legally meaningful nodes.
- Use the same node types as the knowledge graph.
- Return the 5–10 most relevant nodes.
- Prioritize the canonical nodes most likely to match the knowledge graph.
- Do not list every potentially related legal concept.
"""


# ============================================================
# JSON SCHEMA
# ============================================================

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string"
                    },
                    "type": {
                        "type": "string",
                        "enum": [
                            "LEGAL_DOCTRINE",
                            "LEGAL_RULE",
                            "LEGAL_CONCEPT",
                            "LEGAL_CRITERION",
                            "LEGAL_STANDARD",
                            "LEGAL_DUTY",
                            "LEGAL_RIGHT",
                            "LEGAL_EXCEPTION",
                            "LEGAL_DEFENSE",
                            "LEGAL_PROCEDURE",
                            "LEGAL_EVENT",
                            "LEGAL_PARTY",
                            "LEGAL_ACTOR",
                            "LEGAL_CITATION",
                            "LEGAL_CASE",
                            "LEGAL_STATUTE",
                            "LEGAL_TEST",
                            "LEGAL_ELEMENT",
                            "LEGAL_REMEDY",
                            "LEGAL_OFFER",
                            "LEGAL_CONTRACT_TERM",
                            "LEGAL_JURISDICTION",
                            "LEGAL_COURT",
                            "LEGAL_REGULATION"
                        ]
                    }
                },
                "required": [
                    "name",
                    "type"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": [
        "entities"
    ],
    "additionalProperties": False
}


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_existing_results():

    results = {}

    if not os.path.exists(OUTPUT_FILE):
        return results

    print(f"Loading existing results from {OUTPUT_FILE}...")

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:

                record = json.loads(line)

                idx = str(record["idx"])

                results[idx] = record["results"]

            except Exception as e:

                print(
                    f"WARNING: Could not parse line "
                    f"{line_number}: {e}"
                )

    print(
        f"Loaded {len(results)} existing question results."
    )

    return results


all_results = load_existing_results()


# ============================================================
# WRITE COMPLETE RESULTS FILE
# ============================================================

def save_all_results():

    """
    Rewrite the JSONL file so that each idx occurs exactly once.

    This is safer than repeatedly appending duplicate idx records.
    """

    temp_file = OUTPUT_FILE + ".tmp"

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:

        for idx, results in all_results.items():

            record = {
                "idx": idx,
                "results": results
            }

            json.dump(
                record,
                f,
                ensure_ascii=False
            )

            f.write("\n")

    # Atomic replacement on normal local filesystems.
    os.replace(
        temp_file,
        OUTPUT_FILE
    )


# ============================================================
# CUSTOM ID
# ============================================================

def make_custom_id(idx, run_number):

    """
    Creates a unique custom_id for the Batch API.

    Anthropic custom_id must be <= 64 characters and use
    alphanumeric characters, hyphens, and underscores.
    """

    idx_string = str(idx)

    # Replace anything outside the allowed character set.
    idx_string = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        idx_string
    )

    custom_id = f"{idx_string}__run_{run_number}"

    if len(custom_id) > 64:
        raise ValueError(
            f"custom_id is too long: {custom_id}"
        )

    return custom_id


# ============================================================
# PARSE CUSTOM ID
# ============================================================

def parse_custom_id(custom_id):

    """
    Converts:

        mbe_117_2__run_3

    into:

        ("mbe_117_2", 3)
    """

    match = re.match(
        r"^(.*)__run_(\d+)$",
        custom_id
    )

    if not match:
        raise ValueError(
            f"Could not parse custom_id: {custom_id}"
        )

    idx = match.group(1)

    run_number = int(
        match.group(2)
    )

    return idx, run_number


# ============================================================
# BUILD MISSING REQUESTS
# ============================================================

def build_missing_requests():

    requests = []

    question_lookup = {}

    # Create lookup table for CSV rows.
    for _, row in df.iterrows():

        idx = str(row["idx"])

        question_lookup[idx] = row["biased_question"]

    # Check every question.
    for idx, question in question_lookup.items():

        existing = all_results.get(
            idx,
            []
        )

        existing_count = len(existing)

        # ----------------------------------------------------
        # Already complete
        # ----------------------------------------------------

        if existing_count == N_RUNS:

            continue

        # ----------------------------------------------------
        # More than 5 is unexpected.
        # Don't silently destroy data.
        # ----------------------------------------------------

        if existing_count > N_RUNS:

            raise RuntimeError(
                f"{idx} has {existing_count} results. "
                f"Expected at most {N_RUNS}. "
                f"Please inspect the file before continuing."
            )

        # ----------------------------------------------------
        # Generate only missing runs.
        #
        # Example:
        #
        # existing_count = 3
        #
        # → create run 3
        # → create run 4
        # ----------------------------------------------------

        for run_number in range(
            existing_count,
            N_RUNS
        ):

            custom_id = make_custom_id(
                idx,
                run_number
            )

            full_prompt = f"""
{PROMPT}

Fact Pattern:
{question}
"""

            request = {
                "custom_id": custom_id,

                "params": {
                    "model": MODEL,

                    "max_tokens": 1000,

                    "temperature": 0.7,

                    "output_config": {
                        "format": {
                            "type": "json_schema",
                            "schema": OUTPUT_SCHEMA
                        }
                    },

                    "messages": [
                        {
                            "role": "user",
                            "content": full_prompt
                        }
                    ]
                }
            }

            requests.append(request)

    return requests


# ============================================================
# BATCH STATE
# ============================================================

def save_batch_state(
    batch_id,
    requests,
    retry_number
):

    state = {
        "batch_id": batch_id,
        "retry_number": retry_number,
        "requests": requests
    }

    temp_file = STATE_FILE + ".tmp"

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            state,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(
        temp_file,
        STATE_FILE
    )


def load_batch_state():

    if not os.path.exists(STATE_FILE):
        return None

    with open(
        STATE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def delete_batch_state():

    if os.path.exists(STATE_FILE):

        os.remove(
            STATE_FILE
        )


# ============================================================
# PROCESS ONE COMPLETED BATCH
# ============================================================

def process_completed_batch(
    batch_id,
    submitted_requests
):

    print()
    print("=" * 70)
    print(f"Downloading results for batch {batch_id}")
    print("=" * 70)

    # Map custom_id → request.
    request_map = {
        request["custom_id"]: request
        for request in submitted_requests
    }

    successful = {}
    failed_requests = []

    total_results = 0

    for result in client.messages.batches.results(
        batch_id
    ):

        total_results += 1

        custom_id = result.custom_id

        result_type = result.result.type

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if result_type == "succeeded":

            try:

                message = result.result.message

                # Find the first text block.
                text = None

                for block in message.content:

                    if getattr(
                        block,
                        "type",
                        None
                    ) == "text":

                        text = block.text
                        break

                if text is None:

                    raise ValueError(
                        "No text block found in response."
                    )

                parsed = json.loads(text)

                # Validate the expected top-level structure.
                if not isinstance(
                    parsed,
                    dict
                ):

                    raise ValueError(
                        "Response is not a JSON object."
                    )

                if "entities" not in parsed:

                    raise ValueError(
                        "Response does not contain 'entities'."
                    )

                if not isinstance(
                    parsed["entities"],
                    list
                ):

                    raise ValueError(
                        "'entities' is not a list."
                    )

                idx, run_number = parse_custom_id(
                    custom_id
                )

                if idx not in successful:
                    successful[idx] = {}

                successful[idx][run_number] = parsed

            except Exception as e:

                print(
                    f"ERROR parsing {custom_id}: {e}"
                )

                if custom_id in request_map:

                    failed_requests.append(
                        request_map[custom_id]
                    )

        # ----------------------------------------------------
        # FAILED REQUEST
        # ----------------------------------------------------

        else:

            print(
                f"Request failed: "
                f"{custom_id} "
                f"(type={result_type})"
            )

            if custom_id in request_map:

                failed_requests.append(
                    request_map[custom_id]
                )

    print()
    print(
        f"Received {total_results} results."
    )

    print(
        f"Successful: "
        f"{sum(len(v) for v in successful.values())}"
    )

    print(
        f"Failed: "
        f"{len(failed_requests)}"
    )

    # ========================================================
    # MERGE SUCCESSFUL RESULTS
    # ========================================================

    for idx, run_results in successful.items():

        if idx not in all_results:

            all_results[idx] = []

        current = all_results[idx]

        # Add results according to their run number.
        #
        # Normally:
        #
        # current length = 3
        # run_number = 3
        #
        # so append the fourth result.
        #

        for run_number in sorted(run_results):

            parsed = run_results[run_number]

            # We expect the existing results to correspond
            # to runs 0,1,2,... in order.
            #
            # If the result already exists at that position,
            # don't overwrite it.
            if run_number < len(current):

                continue

            if run_number == len(current):

                current.append(
                    parsed
                )

            else:

                # This should not normally happen.
                # Fill the gap with None so that we don't
                # accidentally misalign runs.
                while len(current) < run_number:

                    current.append(None)

                current.append(
                    parsed
                )

    # ========================================================
    # CLEAN UP NONE VALUES
    # ========================================================

    for idx in list(all_results.keys()):

        all_results[idx] = [
            result
            for result in all_results[idx]
            if result is not None
        ]

    # ========================================================
    # SAVE
    # ========================================================

    save_all_results()

    print()
    print(
        f"Saved updated results to {OUTPUT_FILE}"
    )

    return failed_requests


# ============================================================
# CHECK EXISTING DATA
# ============================================================

print()
print("=" * 70)
print("CHECKING EXISTING RESULTS")
print("=" * 70)

complete = 0
partial = 0
missing = 0
invalid = 0

for _, row in df.iterrows():

    idx = str(row["idx"])

    count = len(
        all_results.get(
            idx,
            []
        )
    )

    if count == N_RUNS:

        complete += 1

    elif count == 0:

        missing += 1

    elif 0 < count < N_RUNS:

        partial += 1

    else:

        invalid += 1

print(f"Total CSV rows:       {len(df)}")
print(f"Complete (5/5):       {complete}")
print(f"Partial (1-4/5):      {partial}")
print(f"Missing (0/5):        {missing}")
print(f"Invalid (>5):         {invalid}")

if invalid > 0:

    raise RuntimeError(
        "Some questions have more than 5 results. "
        "Inspect your JSONL before continuing."
    )


# ============================================================
# RESUME AN EXISTING BATCH IF ONE EXISTS
# ============================================================

existing_batch_state = load_batch_state()

if existing_batch_state is not None:

    print()
    print("=" * 70)
    print("FOUND AN UNFINISHED BATCH")
    print("=" * 70)

    batch_id = existing_batch_state["batch_id"]

    submitted_requests = existing_batch_state["requests"]

    retry_number = existing_batch_state.get(
        "retry_number",
        0
    )

    print(
        f"Batch ID: {batch_id}"
    )

    print(
        f"Requests: {len(submitted_requests)}"
    )

    print(
        f"Retry number: {retry_number}"
    )

else:

    batch_id = None
    submitted_requests = None
    retry_number = 0

#Dry Run
print("\nDRY RUN")
requests = build_missing_requests()

print(f"Requests that would be submitted: {len(requests)}")

for request in requests[:20]:
    print(request["custom_id"])

#sys.exit()

# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # If there is an existing batch, wait for it.
    # --------------------------------------------------------

    if batch_id is not None:

        print()
        print("=" * 70)
        print("WAITING FOR BATCH")
        print("=" * 70)

        print(
            f"Batch ID: {batch_id}"
        )

        while True:

            try:

                batch = client.messages.batches.retrieve(
                    batch_id
                )

            except Exception as e:

                print(
                    f"Could not retrieve batch: {e}"
                )

                print(
                    f"Retrying in {POLL_SECONDS} seconds..."
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            counts = batch.request_counts

            print(
                f"Status: {batch.processing_status} | "
                f"Processing: {counts.processing} | "
                f"Succeeded: {counts.succeeded} | "
                f"Errored: {counts.errored} | "
                f"Canceled: {counts.canceled} | "
                f"Expired: {counts.expired}"
            )

            if batch.processing_status == "ended":

                break

            time.sleep(
                POLL_SECONDS
            )

        # ----------------------------------------------------
        # Batch finished.
        # ----------------------------------------------------

        failed_requests = process_completed_batch(
            batch_id,
            submitted_requests
        )

        # Batch is now fully processed.
        delete_batch_state()

        batch_id = None
        submitted_requests = None

        # ----------------------------------------------------
        # Retry failed requests.
        # ----------------------------------------------------

        if failed_requests:

            retry_number += 1

            if retry_number > MAX_RETRY_BATCHES:

                print()
                print(
                    "Maximum retry batches exceeded."
                )

                print(
                    f"Still failed: "
                    f"{len(failed_requests)}"
                )

                # Save failed requests so they are not lost.
                with open(
                    "failed_requests.json",
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        failed_requests,
                        f,
                        ensure_ascii=False,
                        indent=2
                    )

                raise RuntimeError(
                    "Too many retry failures. "
                    "See failed_requests.json"
                )

            print()
            print(
                f"{len(failed_requests)} requests "
                f"need to be retried."
            )

            requests_to_submit = failed_requests

        else:

            requests_to_submit = None

        # ----------------------------------------------------
        # If there are failed requests, submit them.
        # Otherwise rebuild the missing request list.
        # ----------------------------------------------------

        if requests_to_submit is None:

            requests_to_submit = build_missing_requests()

    else:

        # ----------------------------------------------------
        # No active batch.
        # Build requests for missing runs.
        # ----------------------------------------------------

        requests_to_submit = build_missing_requests()

    # --------------------------------------------------------
    # EVERYTHING COMPLETE?
    # --------------------------------------------------------

    if not requests_to_submit:

        print()
        print("=" * 70)
        print("NO REQUESTS REMAINING")
        print("=" * 70)

        break

    # --------------------------------------------------------
    # Take only BATCH_SIZE requests.
    # --------------------------------------------------------

    current_batch_requests = requests_to_submit[
        :BATCH_SIZE
    ]

    remaining_requests = requests_to_submit[
        BATCH_SIZE:
    ]

    print()
    print("=" * 70)
    print("CREATING NEW BATCH")
    print("=" * 70)

    print(
        f"Requests in this batch: "
        f"{len(current_batch_requests)}"
    )

    print(
        f"Requests remaining after this batch: "
        f"{len(remaining_requests)}"
    )

    # --------------------------------------------------------
    # CREATE BATCH
    # --------------------------------------------------------

    try:

        batch = client.messages.batches.create(
            requests=current_batch_requests
        )

    except Exception as e:

        print()
        print(
            f"ERROR creating batch: {e}"
        )

        print(
            "The requests have NOT been discarded."
        )

        print(
            "Fix the problem and restart the script."
        )

        raise

    batch_id = batch.id
    submitted_requests = current_batch_requests

    print()
    print(
        f"Batch successfully created:"
    )

    print(
        f"Batch ID: {batch_id}"
    )

    print(
        f"Status: {batch.processing_status}"
    )

    # --------------------------------------------------------
    # SAVE STATE IMMEDIATELY
    # --------------------------------------------------------

    save_batch_state(
        batch_id=batch_id,
        requests=current_batch_requests,
        retry_number=retry_number
    )

    print(
        f"Batch state saved to {STATE_FILE}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We now go back to the top of the while loop.
    # The script will wait for this batch to finish.
    # --------------------------------------------------------


# ============================================================
# FINAL VERIFICATION
# ============================================================

print()
print("=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

final_counts = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0
}

more_than_five = []

for _, row in df.iterrows():

    idx = str(row["idx"])

    count = len(
        all_results.get(
            idx,
            []
        )
    )

    if count > 5:

        more_than_five.append(
            (idx, count)
        )

    elif count in final_counts:

        final_counts[count] += 1


print()
print(
    f"0/5 results: {final_counts[0]}"
)

print(
    f"1/5 results: {final_counts[1]}"
)

print(
    f"2/5 results: {final_counts[2]}"
)

print(
    f"3/5 results: {final_counts[3]}"
)

print(
    f"4/5 results: {final_counts[4]}"
)

print(
    f"5/5 results: {final_counts[5]}"
)

if more_than_five:

    print()
    print(
        "WARNING: Some questions have more than 5 results:"
    )

    for idx, count in more_than_five[:20]:

        print(
            f"  {idx}: {count}"
        )

    if len(more_than_five) > 20:

        print(
            f"  ... and "
            f"{len(more_than_five) - 20} more."
        )


# ============================================================
# LIST ANY INCOMPLETE QUESTIONS
# ============================================================

incomplete = []

for _, row in df.iterrows():

    idx = str(row["idx"])

    count = len(
        all_results.get(
            idx,
            []
        )
    )

    if count != N_RUNS:

        incomplete.append(
            (idx, count)
        )


if incomplete:

    print()
    print("=" * 70)
    print("INCOMPLETE QUESTIONS")
    print("=" * 70)

    for idx, count in incomplete[:100]:

        print(
            f"{idx}: {count}/5"
        )

    if len(incomplete) > 100:

        print(
            f"... and "
            f"{len(incomplete) - 100} more."
        )

    # Save a machine-readable report.
    with open(
        "incomplete_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            [
                {
                    "idx": idx,
                    "results": count
                }
                for idx, count in incomplete
            ],
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(
        "Incomplete-question report saved to "
        "incomplete_results.json"
    )

else:

    print()
    print(
        "SUCCESS: Every question has exactly "
        "5 results."
    )


print()
print("=" * 70)
print("FINISHED")
print("=" * 70)

