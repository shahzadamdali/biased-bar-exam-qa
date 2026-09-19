# Llama_4_scout

import pandas as pd
import json
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ============================================================
# CONFIG
# ============================================================

load_dotenv(".env", override=True)

HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

CSV_FILE = "biased_final2.csv"
OUTPUT_FILE = "retrieval_results13.jsonl"

MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# Hugging Face provider.
#
# Recommended options for Llama 4 Scout:
#
#   "nscale"
#   "deepinfra"
#
# You can also use:
#
#   "auto"
#
# but for structured JSON output I recommend explicitly
# selecting a provider that supports the required format.
#
PROVIDER = "nscale"

# Number of independent generations per question.
N_RUNS = 5

# Maximum number of requests running concurrently.
#
# IMPORTANT:
# Do not set this to 5000.
# That was appropriate for Anthropic's Batch API,
# but here we are making normal inference requests.
#
MAX_WORKERS = 20

# Retry settings.
MAX_RETRIES = 5

INITIAL_RETRY_DELAY = 5

# Save after this many successful requests.
SAVE_EVERY = 50


# ============================================================
# VALIDATE CONFIG
# ============================================================

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN was not found in your .env file."
    )


# ============================================================
# LOAD DATA
# ============================================================

print("Loading CSV...")

df = pd.read_csv(CSV_FILE)

print(f"Loaded {len(df)} rows.")


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    provider=PROVIDER,
    api_key=HF_TOKEN,
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

    return f"{idx_string}__run_{run_number}"


# ============================================================
# BUILD MISSING REQUESTS
# ============================================================

def build_missing_requests():

    requests = []

    question_lookup = {}

    for _, row in df.iterrows():

        idx = str(row["idx"])

        question_lookup[idx] = row["biased_question"]

    for idx, question in question_lookup.items():

        existing = all_results.get(
            idx,
            []
        )

        existing_count = len(existing)

        # Already complete.
        if existing_count == N_RUNS:
            continue

        # Unexpected.
        if existing_count > N_RUNS:

            raise RuntimeError(
                f"{idx} has {existing_count} results. "
                f"Expected at most {N_RUNS}."
            )

        # Generate missing runs.
        for run_number in range(
            existing_count,
            N_RUNS
        ):

            custom_id = make_custom_id(
                idx,
                run_number
            )

            requests.append(
                {
                    "custom_id": custom_id,
                    "idx": idx,
                    "run_number": run_number,
                    "question": question
                }
            )

    return requests


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
# VALIDATE MODEL RESPONSE
# ============================================================

def validate_result(parsed):

    if not isinstance(parsed, dict):

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

    # Validate every entity.
    for entity in parsed["entities"]:

        if not isinstance(entity, dict):

            raise ValueError(
                "Entity is not an object."
            )

        if "name" not in entity:

            raise ValueError(
                "Entity is missing 'name'."
            )

        if "type" not in entity:

            raise ValueError(
                "Entity is missing 'type'."
            )

        if not isinstance(
            entity["name"],
            str
        ):

            raise ValueError(
                "Entity name is not a string."
            )

        if not isinstance(
            entity["type"],
            str
        ):

            raise ValueError(
                "Entity type is not a string."
            )

    return parsed


# ============================================================
# ONE MODEL REQUEST
# ============================================================

def run_single_request(request):

    custom_id = request["custom_id"]
    idx = request["idx"]
    run_number = request["run_number"]
    question = request["question"]

    full_prompt = f"""
{PROMPT}

Fact Pattern:
{question}
"""

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = client.chat.completions.create(

                model=MODEL,

                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],

                temperature=0.7,

                max_tokens=1000,

                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "legal_entities",
                        "description": (
                            "Legal knowledge graph entities "
                            "relevant to the fact pattern."
                        ),
                        "schema": OUTPUT_SCHEMA,
                        "strict": True
                    }
                }
            )

            content = response.choices[0].message.content

            if not content:

                raise ValueError(
                    "Model returned empty content."
                )

            # Usually content is already a string
            # containing JSON.
            parsed = json.loads(content)

            parsed = validate_result(parsed)

            return {
                "custom_id": custom_id,
                "idx": idx,
                "run_number": run_number,
                "result": parsed,
                "error": None
            }

        except Exception as e:

            last_error = e

            print(
                f"[{custom_id}] "
                f"Attempt {attempt}/{MAX_RETRIES} failed: "
                f"{e}"
            )

            if attempt < MAX_RETRIES:

                delay = (
                    INITIAL_RETRY_DELAY
                    * (2 ** (attempt - 1))
                )

                time.sleep(delay)

    return {
        "custom_id": custom_id,
        "idx": idx,
        "run_number": run_number,
        "result": None,
        "error": str(last_error)
    }


# ============================================================
# PROCESS REQUESTS
# ============================================================

def process_requests(requests):

    successful = 0
    failed = []

    total = len(requests)

    print()
    print("=" * 70)
    print("PROCESSING REQUESTS")
    print("=" * 70)

    print(
        f"Total requests: {total}"
    )

    print(
        f"Workers: {MAX_WORKERS}"
    )

    print(
        f"Model: {MODEL}"
    )

    print(
        f"Provider: {PROVIDER}"
    )

    print()

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_to_request = {
            executor.submit(
                run_single_request,
                request
            ): request
            for request in requests
        }

        for completed_number, future in enumerate(
            as_completed(future_to_request),
            start=1
        ):

            request = future_to_request[future]

            try:

                result = future.result()

            except Exception as e:

                print(
                    f"FATAL request error "
                    f"{request['custom_id']}: {e}"
                )

                failed.append(request)

                continue

            if result["result"] is not None:

                idx = result["idx"]
                run_number = result["run_number"]

                if idx not in all_results:
                    all_results[idx] = []

                current = all_results[idx]

                # Normally this is exactly the next position.
                if run_number < len(current):

                    # Already exists.
                    pass

                elif run_number == len(current):

                    current.append(
                        result["result"]
                    )

                else:

                    # Fill missing positions if necessary.
                    while len(current) < run_number:

                        current.append(None)

                    current.append(
                        result["result"]
                    )

                successful += 1

            else:

                failed.append(request)

            # Progress.
            if (
                completed_number % SAVE_EVERY == 0
                or completed_number == total
            ):

                # Remove temporary None values only if
                # they are not needed.
                for idx in list(all_results.keys()):

                    all_results[idx] = [
                        x
                        for x in all_results[idx]
                        if x is not None
                    ]

                save_all_results()

                print(
                    f"Progress: "
                    f"{completed_number}/{total} | "
                    f"Successful: {successful} | "
                    f"Failed: {len(failed)}"
                )

    # Clean None values.
    for idx in list(all_results.keys()):

        all_results[idx] = [
            x
            for x in all_results[idx]
            if x is not None
        ]

    save_all_results()

    print()
    print(
        f"Finished batch: "
        f"{successful} successful, "
        f"{len(failed)} failed."
    )

    return failed


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
        "Some questions have more than 5 results."
    )


# ============================================================
# DRY RUN
# ============================================================

print()
print("=" * 70)
print("DRY RUN")
print("=" * 70)

requests = build_missing_requests()

print(
    f"Requests that would be submitted: "
    f"{len(requests)}"
)

print()

for request in requests[:20]:

    print(
        request["custom_id"]
    )

# Uncomment this if you want to inspect
# before making any API calls.
#
# raise SystemExit()


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

retry_number = 0

pending_requests = requests


while pending_requests:

    print()
    print("=" * 70)
    print("STARTING REQUEST GROUP")
    print("=" * 70)

    print(
        f"Requests: {len(pending_requests)}"
    )

    failed_requests = process_requests(
        pending_requests
    )

    # --------------------------------------------------------
    # Retry failures.
    # --------------------------------------------------------

    if not failed_requests:

        print(
            "No failed requests."
        )

        break

    retry_number += 1

    print()
    print(
        f"{len(failed_requests)} requests failed."
    )

    print(
        f"Retry round: {retry_number}"
    )

    if retry_number > MAX_RETRIES:

        print()
        print(
            "Maximum retry rounds exceeded."
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

    pending_requests = failed_requests


# ============================================================
# REBUILD ANY STILL-MISSING REQUESTS
# ============================================================

remaining_requests = build_missing_requests()

if remaining_requests:

    print()
    print(
        f"Some requests are still missing: "
        f"{len(remaining_requests)}"
    )

    # Run the missing requests.
    retry_number = 0

    pending_requests = remaining_requests

    while pending_requests:

        failed_requests = process_requests(
            pending_requests
        )

        if not failed_requests:
            break

        retry_number += 1

        if retry_number > MAX_RETRIES:

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

        pending_requests = failed_requests


# ============================================================
# FINAL SAVE
# ============================================================

save_all_results()


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
        "WARNING: Some questions have more than "
        "5 results:"
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
