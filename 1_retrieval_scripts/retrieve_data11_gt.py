import pandas as pd
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

# ==========================
# CONFIG
# ==========================
load_dotenv(".env", override=True)

API_KEY = os.getenv("OPENAI_API_KEY")
CSV_FILE = "qa.csv"
OUTPUT_FILE = "retrieval_results11_gt.json"

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(CSV_FILE)

# ==========================
# OPENAI CLIENT
# ==========================

client = OpenAI(api_key=API_KEY)

PROMPT = """
You are generating retrieval queries for a legal knowledge graph.

- The knowledge graph was built from explanatory legal text, not fact patterns.
- Given a fact pattern, infer the canonical legal nodes that are most likely to appear in the relevant knowledge graph.
- Do not merely extract nouns or people mentioned in the fact pattern.
- Instead, identify the governing legal doctrine, rules, concepts, criteria, standards, duties, rights, exceptions, defenses, procedures, events, parties, actors, citations, cases, statutes, tests, elements, remedies, offers, contract terms, jurisdictions, courts, and regulations that are relevant to the fact pattern.
- Use the same node types as the knowledge graph.

Return only the entities that are legally relevant to the fact pattern.
"""

# ==========================
# STRUCTURED OUTPUT SCHEMA
# ==========================

ENTITY_SCHEMA = {
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
                "required": ["name", "type"],
                "additionalProperties": False
            }
        }
    },
    "required": ["entities"],
    "additionalProperties": False
}

# ==========================
# LOAD EXISTING RESULTS
# ==========================

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        all_results = json.load(f)
else:
    all_results = {}

# ==========================
# PROCESS QUESTIONS
# ==========================

for _, row in df.iterrows():

    query_idx = str(row["idx"])

    # Skip already processed queries
    if query_idx in all_results:
        print(f"Skipping {query_idx} - already processed")
        continue

    fact_pattern = f"{row['prompt']}\n\n{row['question']}"

    full_prompt = f"""
{PROMPT}

Fact Pattern:
{fact_pattern}
"""

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=full_prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "legal_entities",
                    "strict": True,
                    "schema": ENTITY_SCHEMA
                }
            }
        )

        result = json.loads(response.output_text)

        all_results[query_idx] = result

        print(f"Processed {query_idx}")

        # Save after every successful query so progress is not lost
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(
                all_results,
                f,
                indent=2,
                ensure_ascii=False
            )

    except Exception as e:
        print(f"Error processing {query_idx}: {e}")
        continue

# ==========================
# FINAL SAVE
# ==========================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        all_results,
        f,
        indent=2,
        ensure_ascii=False
    )

print(f"Saved results for {len(all_results)} questions to {OUTPUT_FILE}")
