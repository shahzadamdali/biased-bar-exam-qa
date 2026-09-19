import pandas as pd
import json
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# ==========================
# CONFIG
# ==========================
load_dotenv(".env", override=True)

HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

CSV_FILE = "qa.csv"
OUTPUT_FILE = "retrieval_results13_gt.json"

RERUN_IDS = {
    "mbe_7",
}

MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
PROVIDER = "nscale"

# ==========================
# CHECK API KEY
# ==========================

if not HF_TOKEN:
    raise ValueError(
        "HUGGING_FACE_TOKEN not found. Add HUGGING_FACE_TOKEN=your_huggingface_token to your .env file."
    )

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(CSV_FILE)

# ==========================
# HUGGING FACE CLIENT
# ==========================

client = InferenceClient(
    provider=PROVIDER,
    api_key=HF_TOKEN,
)

# ==========================
# PROMPT
# ==========================

PROMPT = """
You are generating retrieval queries for a legal knowledge graph.

- The knowledge graph was built from explanatory legal text, not fact patterns.
- Given a fact pattern, infer the canonical legal nodes that are most likely to appear in the relevant knowledge graph.
- Do not merely extract nouns or people mentioned in the fact pattern.
- Instead, identify the governing legal doctrine, rules, concepts, criteria, standards, duties, rights, exceptions, defenses, procedures, events, parties, actors, citations, cases, statutes, tests, elements, remedies, offers, contract terms, jurisdictions, courts, and regulations that are relevant to the fact pattern.
- Use the same node types as the knowledge graph.
- Return only legally relevant entities.
- Do not include entities merely because they are mentioned in the fact pattern.
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

    # Only process the specified queries
    if query_idx not in RERUN_IDS:
        continue

    print(f"Rerunning {query_idx}")

    fact_pattern = f"{row['prompt']}\n\n{row['question']}"

    try:

        # ==========================
        # CALL LLAMA 4 SCOUT
        # ==========================

        response = client.chat_completion(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": PROMPT
                },
                {
                    "role": "user",
                    "content": f"""
Fact Pattern:

{fact_pattern}
"""
                }
            ],
            max_tokens=2048,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "legal_entities",
                    "schema": ENTITY_SCHEMA
                }
            }
        )

        # ==========================
        # EXTRACT STRUCTURED OUTPUT
        # ==========================

        result_text = response.choices[0].message.content

        result = json.loads(result_text)

        # ==========================
        # STORE RESULT
        # ==========================

        all_results[query_idx] = result

        print(f"Processed {query_idx}")

        # ==========================
        # SAVE AFTER EACH SUCCESS
        # ==========================

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

print(
    f"Saved results for {len(all_results)} questions "
    f"to {OUTPUT_FILE}"
)
