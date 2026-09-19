#gemini_3.1_flash_lite

import pandas as pd
import json
from google import genai
from dotenv import load_dotenv
import os
import time

# ==========================
# CONFIG
# ==========================
load_dotenv(".env", override=True)

API_KEY = os.getenv("GEMINI_API_KEY")
CSV_FILE = "biased_final2.csv"
OUTPUT_FILE = "retrieval_results10.jsonl"

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(CSV_FILE)

# ==========================
# GEMINI CLIENT
# ==========================

client = genai.Client(api_key=API_KEY)

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

Return only:

{
  "entities": [
    {
      "name": "...",
      "type": "LEGAL_RULE"
    }
  ]
}
"""

# ==========================
# LOAD EXISTING RESULTS
# ==========================

all_results = {}

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                all_results[record["idx"]] = record["results"]

# ==========================
# PROCESS ALL ROWS
# ==========================

for _, row in df.iterrows():

    query_idx = row["idx"]

    # Skip if already processed
    if query_idx in all_results:
        print(f"Skipping {query_idx}")
        continue

    question = row["biased_question"]

    full_prompt = f"""
{PROMPT}

Fact Pattern:
{question}
"""

    all_results[query_idx] = []

    try:
        for run in range(5):

            for attempt in range(5):
                try:
                    response = client.models.generate_content(
                        model="models/gemini-3.1-flash-lite",
                        contents=full_prompt,
                        config={
                            "response_mime_type": "application/json",
                            "temperature": 0.7,
                        },
                    )

                    result = json.loads(response.text)
                    all_results[query_idx].append(result)

                    print(f"{query_idx} - Run {run+1}/5")
                    time.sleep(5)
                    break

                except Exception as e:
                    print(
                        f"{query_idx}: attempt {attempt+1}/5 failed: {e}"
                    )

                    if attempt == 4:
                        raise

                    time.sleep(15)

        # Save after every completed question
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "idx": query_idx,
                    "results": all_results[query_idx]
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

        print(f"Saved {query_idx}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
        print("Progress saved.")
        raise

print(f"\nFinished! Processed {len(all_results)} questions.")
#After you're done, make sure to run a script which checks that each question has 5 results. If not, you can re-run this script to fill in the missing results.