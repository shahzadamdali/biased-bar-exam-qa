import os
import json
import time
import random
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "create_category.csv"
OUTPUT_FILE = "create_category2.csv"

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(api_key=API_KEY)

MODEL = "models/gemini-3.5-flash-lite"

# Exactly 5 concurrent workers
MAX_WORKERS = 5

# Process 5 prompts per batch
BATCH_SIZE = 5

# Retry configuration
MAX_RETRIES = 10

# Initial retry delay
INITIAL_RETRY_DELAY = 2

# Maximum retry delay
MAX_RETRY_DELAY = 60

# Delay between completed batches
BATCH_DELAY_SECONDS = 0.5


# ============================================================
# 2. DIFFICULTY CLASSIFICATION INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTION = """
You are an expert legal-education evaluator.

Your task is to classify the difficulty of a legal question into exactly
one of three categories:

- easy
- medium
- hard

The questions consist of legal scenarios, fact patterns, and a question.

Judge difficulty based on the amount and complexity of LEGAL REASONING
required to reach the correct answer, NOT merely on the length of the
question, vocabulary, or number of facts.

Use these criteria:

EASY:
- Tests one clear and well-established legal rule or doctrine.
- Requires little interpretation of the facts.
- The correct answer follows relatively directly from the applicable rule.
- Usually requires only one or two straightforward reasoning steps.
- Few materially relevant or competing facts.

MEDIUM:
- Requires applying a legal rule to several facts.
- Requires two or more meaningful reasoning steps.
- May require distinguishing between related doctrines or rules.
- May contain distracting or conflicting facts.
- Requires some interpretation, but the applicable legal framework is
  reasonably identifiable.

HARD:
- Requires integrating multiple legal rules or doctrines.
- Requires several sequential or sophisticated reasoning steps.
- Involves subtle exceptions, competing legal principles, ambiguous facts,
  or difficult distinctions.
- Requires identifying which facts are legally significant among substantial
  distractions.
- A knowledgeable law student would likely need substantial analysis rather
  than straightforward rule application.

IMPORTANT:
- Do NOT classify a question as hard simply because it is long.
- Do NOT classify a question as easy simply because the legal rule is
  familiar.
- Focus on the reasoning required to arrive at the correct answer.
- Assume the reader is a law student with basic familiarity with the
  relevant area of law.

Return ONLY valid JSON in this exact format:

{
  "difficulty": "easy"
}

The value of "difficulty" must be exactly one of:
"easy", "medium", "hard".
"""


# ============================================================
# 3. CLASSIFY ONE QUESTION WITH RETRIES
# ============================================================

def classify_prompt(index, prompt):
    """
    Send one prompt to Gemini.

    Retries temporary API failures using exponential backoff.
    The request will be attempted up to MAX_RETRIES times.
    """

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )

            result = json.loads(response.text)

            difficulty = (
                result
                .get("difficulty", "")
                .lower()
                .strip()
            )

            if difficulty not in {"easy", "medium", "hard"}:
                raise ValueError(
                    f"Invalid classification returned: {difficulty}"
                )

            return {
                "index": index,
                "difficulty": difficulty,
                "error": None,
            }

        except Exception as e:

            error_message = str(e)

            print(
                f"[{index + 1}] "
                f"Attempt {attempt}/{MAX_RETRIES} failed: "
                f"{error_message}"
            )

            # ------------------------------------------------
            # If this was the final attempt, return failure
            # ------------------------------------------------

            if attempt == MAX_RETRIES:

                return {
                    "index": index,
                    "difficulty": None,
                    "error": error_message,
                }

            # ------------------------------------------------
            # Exponential backoff
            #
            # Attempt 1 -> ~2 sec
            # Attempt 2 -> ~4 sec
            # Attempt 3 -> ~8 sec
            # Attempt 4 -> ~16 sec
            # ...
            #
            # Random jitter prevents all workers from retrying
            # at exactly the same time.
            # ------------------------------------------------

            delay = min(
                INITIAL_RETRY_DELAY * (2 ** (attempt - 1)),
                MAX_RETRY_DELAY,
            )

            jitter = random.uniform(0, 1)

            time.sleep(delay + jitter)

    # Should never reach here
    return {
        "index": index,
        "difficulty": None,
        "error": "Unknown error",
    }


# ============================================================
# 4. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

if "modified_prompt" not in df.columns:
    raise ValueError(
        "The CSV must contain a 'modified_prompt' column."
    )

# Create output column
df["difficulty"] = None


# ============================================================
# 5. PREPARE VALID PROMPTS
# ============================================================

tasks = []

for i, row in df.iterrows():

    prompt = str(row["modified_prompt"])

    if not prompt.strip() or prompt.lower() == "nan":

        print(
            f"[{i + 1}/{len(df)}] "
            f"Empty prompt - skipped"
        )

        continue

    tasks.append(
        (i, prompt)
    )


# ============================================================
# 6. PROCESS IN BATCHES OF 5
# ============================================================

total_tasks = len(tasks)

total_batches = (
    total_tasks + BATCH_SIZE - 1
) // BATCH_SIZE


for batch_start in range(
    0,
    total_tasks,
    BATCH_SIZE
):

    batch = tasks[
        batch_start:
        batch_start + BATCH_SIZE
    ]

    batch_number = (
        batch_start // BATCH_SIZE
    ) + 1

    print("\n" + "=" * 60)

    print(
        f"Batch {batch_number}/{total_batches}"
    )

    print(
        f"Requests in batch: {len(batch)}"
    )

    print("=" * 60)


    # ========================================================
    # 7. RUN 5 REQUESTS CONCURRENTLY
    # ========================================================

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                classify_prompt,
                index,
                prompt
            ): index

            for index, prompt in batch
        }


        # ====================================================
        # 8. COLLECT RESULTS
        # ====================================================

        for future in as_completed(futures):

            index = futures[future]

            try:

                result = future.result()

                result_index = result["index"]
                difficulty = result["difficulty"]
                error = result["error"]


                if difficulty is not None:

                    df.at[
                        result_index,
                        "difficulty"
                    ] = difficulty

                    print(
                        f"[{result_index + 1}/{len(df)}] "
                        f"SUCCESS: "
                        f"{difficulty.upper()}"
                    )

                else:

                    print(
                        f"[{result_index + 1}/{len(df)}] "
                        f"FAILED AFTER {MAX_RETRIES} "
                        f"ATTEMPTS: {error}"
                    )

                    df.at[
                        result_index,
                        "difficulty"
                    ] = None


            except Exception as e:

                print(
                    f"[{index + 1}/{len(df)}] "
                    f"Unexpected worker error: {e}"
                )

                df.at[
                    index,
                    "difficulty"
                ] = None


    # ========================================================
    # 9. SAVE AFTER EVERY BATCH
    #
    # This protects your progress if the script crashes.
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Batch {batch_number} complete."
    )

    print(
        f"Progress saved to {OUTPUT_FILE}"
    )


    # ========================================================
    # 10. DELAY BEFORE NEXT BATCH
    # ========================================================

    if batch_start + BATCH_SIZE < total_tasks:

        time.sleep(
            BATCH_DELAY_SECONDS
        )


# ============================================================
# 11. FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)

print(
    f"Saved classified data to: {OUTPUT_FILE}"
)

print("\nClassification counts:")

print(
    df["difficulty"]
    .value_counts(dropna=False)
)
