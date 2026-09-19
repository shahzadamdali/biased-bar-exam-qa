# openai_gpt_5_mini

import pandas as pd
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import re


# ============================================================
# CONFIG
# ============================================================

load_dotenv(".env", override=True)

API_KEY = os.getenv("OPENAI_API_KEY")

CSV_FILE = "biased_final2.csv"
OUTPUT_FILE = "retrieval_results11.jsonl"

# Tracks a submitted batch that has not yet been processed.
STATE_FILE = "openai_batch_state.json"

MODEL = "gpt-5-mini"

N_RUNS = 5

# Number of individual requests in one OpenAI batch.
BATCH_SIZE = 5000

# How often to check batch status.
POLL_SECONDS = 60

MAX_RETRY_BATCHES = 20


# ============================================================
# VALIDATE CONFIG
# ============================================================

if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY was not found in your .env file."
    )


# ============================================================
# LOAD DATA
# ============================================================

print("Loading CSV...")

df = pd.read_csv(CSV_FILE)

print(f"Loaded {len(df)} rows.")


# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI(
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
- Instead, identify the governing:

LEGAL_DOCTRINE
LEGAL_RULE
LEGAL_CONCEPT
LEGAL_CRITERION
LEGAL_STANDARD
LEGAL_DUTY
LEGAL_RIGHT
LEGAL_EXCEPTION
LEGAL_DEFENSE
LEGAL_PROCEDURE
LEGAL_EVENT
LEGAL_PARTY
LEGAL_ACTOR
LEGAL_CITATION
LEGAL_CASE
LEGAL_STATUTE
LEGAL_TEST
LEGAL_ELEMENT
LEGAL_REMEDY
LEGAL_OFFER
LEGAL_CONTRACT_TERM
LEGAL_JURISDICTION
LEGAL_COURT
LEGAL_REGULATION

- Use the same node types as the knowledge graph.

Return the 5–10 most relevant canonical legal nodes.
Prioritize nodes most likely to match the knowledge graph.
Do not list every potentially related legal concept.
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

    print(
        f"Loading existing results from {OUTPUT_FILE}..."
    )

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            if not line.strip():
                continue

            try:

                record = json.loads(line)

                idx = str(
                    record["idx"]
                )

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
# SAVE COMPLETE RESULTS FILE
# ============================================================

def save_all_results():

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

    os.replace(
        temp_file,
        OUTPUT_FILE
    )


# ============================================================
# CUSTOM ID
# ============================================================

def make_custom_id(idx, run_number):

    idx_string = str(idx)

    idx_string = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        idx_string
    )

    custom_id = (
        f"{idx_string}__run_{run_number}"
    )

    if len(custom_id) > 64:

        raise ValueError(
            f"custom_id is too long: {custom_id}"
        )

    return custom_id


# ============================================================
# PARSE CUSTOM ID
# ============================================================

def parse_custom_id(custom_id):

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

    # --------------------------------------------------------
    # Create lookup table
    # --------------------------------------------------------

    for _, row in df.iterrows():

        idx = str(
            row["idx"]
        )

        question_lookup[idx] = (
            row["biased_question"]
        )

    # --------------------------------------------------------
    # Find missing runs
    # --------------------------------------------------------

    for idx, question in question_lookup.items():

        existing = all_results.get(
            idx,
            []
        )

        existing_count = len(existing)

        # Already has all 5 runs.
        if existing_count == N_RUNS:
            continue

        # More than 5 is invalid.
        if existing_count > N_RUNS:

            raise RuntimeError(
                f"{idx} has {existing_count} results. "
                f"Expected at most {N_RUNS}."
            )

        # ----------------------------------------------------
        # Generate only missing runs.
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

                "method": "POST",

                "url": "/v1/responses",

                "body": {

                    "model": MODEL,

                    "input": full_prompt,

                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "legal_entities",
                            "strict": True,
                            "schema": OUTPUT_SCHEMA
                        }
                    }
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
# CREATE JSONL INPUT FILE FOR OPENAI
# ============================================================

def create_batch_input_file(
    requests,
    filename="openai_batch_input.jsonl"
):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        for request in requests:

            json.dump(
                request,
                f,
                ensure_ascii=False
            )

            f.write("\n")

    return filename


# ============================================================
# PROCESS COMPLETED BATCH
# ============================================================

def process_completed_batch(
    batch,
    submitted_requests
):

    print()
    print("=" * 70)
    print(
        f"Downloading results for batch {batch.id}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Map custom_id -> original request
    # --------------------------------------------------------

    request_map = {
        request["custom_id"]: request
        for request in submitted_requests
    }

    successful = {}

    failed_requests = []

    total_results = 0

    # --------------------------------------------------------
    # Download output file
    # --------------------------------------------------------

    output_file_id = batch.output_file_id

    if not output_file_id:

        raise RuntimeError(
            "Batch ended without an output_file_id."
        )

    output_content = client.files.content(
        output_file_id
    )

    # Convert file response to text.
    output_text = output_content.text

    # --------------------------------------------------------
    # Process each JSONL result
    # --------------------------------------------------------

    for line in output_text.splitlines():

        if not line.strip():
            continue

        total_results += 1

        result = json.loads(line)

        custom_id = result["custom_id"]

        response = result.get(
            "response"
        )

        error = result.get(
            "error"
        )

        # ----------------------------------------------------
        # FAILED REQUEST
        # ----------------------------------------------------

        if error is not None:

            print(
                f"Request failed: "
                f"{custom_id}: {error}"
            )

            if custom_id in request_map:

                failed_requests.append(
                    request_map[custom_id]
                )

            continue

        # ----------------------------------------------------
        # HTTP ERROR
        # ----------------------------------------------------

        if response is None:

            print(
                f"Request failed with no response: "
                f"{custom_id}"
            )

            if custom_id in request_map:

                failed_requests.append(
                    request_map[custom_id]
                )

            continue

        status_code = response.get(
            "status_code"
        )

        if status_code != 200:

            print(
                f"Request failed: "
                f"{custom_id} "
                f"(status={status_code})"
            )

            if custom_id in request_map:

                failed_requests.append(
                    request_map[custom_id]
                )

            continue

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        try:

            body = response["body"]

            output = body.get(
                "output",
                []
            )

            text = None

            for output_item in output:

                if output_item.get(
                    "type"
                ) != "message":

                    continue

                for content_item in output_item.get(
                    "content",
                    []
                ):

                    if content_item.get(
                        "type"
                    ) == "output_text":

                        text = content_item.get(
                            "text"
                        )

                        break

                if text is not None:
                    break

            if text is None:

                raise ValueError(
                    "No output_text found."
                )

            parsed = json.loads(
                text
            )

            if not isinstance(
                parsed,
                dict
            ):

                raise ValueError(
                    "Response is not a JSON object."
                )

            if "entities" not in parsed:

                raise ValueError(
                    "Response does not contain "
                    "'entities'."
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
                f"ERROR parsing "
                f"{custom_id}: {e}"
            )

            if custom_id in request_map:

                failed_requests.append(
                    request_map[custom_id]
                )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    successful_count = sum(
        len(v)
        for v in successful.values()
    )

    print()
    print(
        f"Received {total_results} results."
    )

    print(
        f"Successful: {successful_count}"
    )

    print(
        f"Failed: {len(failed_requests)}"
    )

    # ========================================================
    # MERGE RESULTS
    # ========================================================

    for idx, run_results in successful.items():

        if idx not in all_results:

            all_results[idx] = []

        current = all_results[idx]

        for run_number in sorted(
            run_results
        ):

            parsed = run_results[
                run_number
            ]

            # Already exists.
            if run_number < len(current):
                continue

            # Correct next position.
            if run_number == len(current):

                current.append(
                    parsed
                )

            else:

                # Fill unexpected gap.
                while len(current) < run_number:

                    current.append(None)

                current.append(
                    parsed
                )

    # ========================================================
    # REMOVE NONE VALUES
    # ========================================================

    for idx in list(
        all_results.keys()
    ):

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
        f"Saved updated results to "
        f"{OUTPUT_FILE}"
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

    idx = str(
        row["idx"]
    )

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


print(
    f"Total CSV rows:       {len(df)}"
)

print(
    f"Complete (5/5):       {complete}"
)

print(
    f"Partial (1-4/5):      {partial}"
)

print(
    f"Missing (0/5):        {missing}"
)

print(
    f"Invalid (>5):         {invalid}"
)

if invalid > 0:

    raise RuntimeError(
        "Some questions have more than "
        "5 results."
    )


# ============================================================
# RESUME EXISTING BATCH
# ============================================================

existing_batch_state = load_batch_state()

if existing_batch_state is not None:

    print()
    print("=" * 70)
    print("FOUND AN UNFINISHED BATCH")
    print("=" * 70)

    batch_id = (
        existing_batch_state["batch_id"]
    )

    submitted_requests = (
        existing_batch_state["requests"]
    )

    retry_number = (
        existing_batch_state.get(
            "retry_number",
            0
        )
    )

    print(
        f"Batch ID: {batch_id}"
    )

    print(
        f"Requests: "
        f"{len(submitted_requests)}"
    )

    print(
        f"Retry number: "
        f"{retry_number}"
    )

else:

    batch_id = None

    submitted_requests = None

    retry_number = 0


# ============================================================
# DRY RUN
# ============================================================

print()
print("DRY RUN")

requests = build_missing_requests()

print(
    f"Requests that would be submitted: "
    f"{len(requests)}"
)

for request in requests[:20]:

    print(
        request["custom_id"]
    )


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # Existing batch
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

                batch = client.batches.retrieve(
                    batch_id
                )

            except Exception as e:

                print(
                    f"Could not retrieve batch: {e}"
                )

                print(
                    f"Retrying in "
                    f"{POLL_SECONDS} seconds..."
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            print(
                f"Status: {batch.status}"
            )

            # OpenAI Batch status can be:
            #
            # validating
            # in_progress
            # finalizing
            # completed
            # failed
            # expired
            # cancelling
            # cancelled

            if batch.status in [
                "completed",
                "failed",
                "expired",
                "cancelled"
            ]:

                break

            time.sleep(
                POLL_SECONDS
            )

        # ----------------------------------------------------
        # Handle completed batch
        # ----------------------------------------------------

        if batch.status == "completed":

            failed_requests = (
                process_completed_batch(
                    batch,
                    submitted_requests
                )
            )

        else:

            print()
            print(
                f"Batch ended with status: "
                f"{batch.status}"
            )

            # If the entire batch failed/expired/cancelled,
            # retry all submitted requests.

            failed_requests = (
                submitted_requests
            )

        # ----------------------------------------------------
        # Batch finished
        # ----------------------------------------------------

        delete_batch_state()

        batch_id = None

        submitted_requests = None

        # ----------------------------------------------------
        # Retry failed requests
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

            requests_to_submit = (
                failed_requests
            )

        else:

            requests_to_submit = None

        # ----------------------------------------------------
        # Rebuild missing requests if no failures
        # ----------------------------------------------------

        if requests_to_submit is None:

            requests_to_submit = (
                build_missing_requests()
            )

    else:

        # ----------------------------------------------------
        # No active batch
        # ----------------------------------------------------

        requests_to_submit = (
            build_missing_requests()
        )

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
    # Take BATCH_SIZE requests
    # --------------------------------------------------------

    current_batch_requests = (
        requests_to_submit[:BATCH_SIZE]
    )

    remaining_requests = (
        requests_to_submit[BATCH_SIZE:]
    )

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
    # CREATE INPUT JSONL
    # --------------------------------------------------------

    input_filename = (
        "openai_batch_input.jsonl"
    )

    create_batch_input_file(
        current_batch_requests,
        input_filename
    )

    print(
        f"Created input file: "
        f"{input_filename}"
    )

    # --------------------------------------------------------
    # UPLOAD INPUT FILE
    # --------------------------------------------------------

    print(
        "Uploading batch input file..."
    )

    with open(
        input_filename,
        "rb"
    ) as f:

        uploaded_file = client.files.create(
            file=f,
            purpose="batch"
        )

    print(
        f"Uploaded file: "
        f"{uploaded_file.id}"
    )

    # --------------------------------------------------------
    # CREATE BATCH
    # --------------------------------------------------------

    print(
        "Creating OpenAI batch..."
    )

    try:

        batch = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/responses",
            completion_window="24h"
        )

    except Exception as e:

        print()
        print(
            f"ERROR creating batch: {e}"
        )

        print(
            "The input file has NOT been discarded."
        )

        raise

    batch_id = batch.id

    submitted_requests = (
        current_batch_requests
    )

    print()
    print(
        "Batch successfully created:"
    )

    print(
        f"Batch ID: {batch_id}"
    )

    print(
        f"Status: {batch.status}"
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
        f"Batch state saved to "
        f"{STATE_FILE}"
    )

    # Loop goes back to top and waits.


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

    idx = str(
        row["idx"]
    )

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
        "WARNING: Some questions have "
        "more than 5 results:"
    )

    for idx, count in more_than_five[:20]:

        print(
            f"  {idx}: {count}"
        )


# ============================================================
# LIST INCOMPLETE QUESTIONS
# ============================================================

incomplete = []

for _, row in df.iterrows():

    idx = str(
        row["idx"]
    )

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