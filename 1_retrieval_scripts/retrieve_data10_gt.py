import pandas as pd
import json
from google import genai
from dotenv import load_dotenv
import os
import time

# ==========================
# CONFIG
# ==========================
load_dotenv('.env', override=True)

API_KEY = os.getenv('GEMINI_API_KEY')
CSV_FILE = "qa.csv"
OUTPUT_FILE = "retrieval_results10_gt.json"

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

# Load existing results if the file exists
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        all_results = json.load(f)
else:
    all_results = {}

for _, row in df.iterrows():
    query_idx = row["idx"]

    fact_pattern = f"{row['prompt']}\n\n{row['question']}"

    full_prompt = f"""
    {PROMPT}

    Fact Pattern:
    {fact_pattern}
    """

    response = client.models.generate_content(
        model="models/gemini-3.1-flash-lite",
        contents=full_prompt,
        config={
            "response_mime_type": "application/json",
            "temperature": 0.0,
        }
    )

    result = json.loads(response.text)
    all_results[query_idx] = result
    print(f"Processed {query_idx}")

    time.sleep(5)

# Save all results once
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"Saved results for {len(df)} questions to {OUTPUT_FILE}")