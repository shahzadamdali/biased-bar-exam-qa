import json
import csv
import os
import pickle

import numpy as np

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

from langchain_core.embeddings import Embeddings
from google import genai
from google.genai import types


# ============================================================
# Environment
# ============================================================

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )


# ============================================================
# Gemini Embedding 2
# ============================================================

class GeminiEmbeddings(Embeddings):

    def __init__(
        self,
        model_name="gemini-embedding-2",
        output_dimensionality=1536,
        batch_size=50
    ):

        self.model_name = model_name
        self.output_dimensionality = output_dimensionality
        self.batch_size = batch_size

        self.client = genai.Client(
            api_key=gemini_api_key
        )


    def embed_documents(self, texts):

        all_embeddings = []

        # ----------------------------------------------------
        # Process exactly batch_size texts at a time
        # ----------------------------------------------------

        for start in range(
            0,
            len(texts),
            self.batch_size
        ):

            batch = texts[
                start:start + self.batch_size
            ]

            print(
                f"Sending embedding batch: "
                f"{start + 1}-{start + len(batch)} "
                f"of {len(texts)}"
            )


            # ------------------------------------------------
            # IMPORTANT:
            #
            # Each text is its own Content object.
            #
            # Therefore Gemini returns one embedding
            # for each individual text.
            # ------------------------------------------------

            contents = [
                types.Content(
                    parts=[
                        types.Part.from_text(
                            text=text
                        )
                    ]
                )
                for text in batch
            ]


            result = self.client.models.embed_content(
                model=self.model_name,
                contents=contents,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.output_dimensionality
                )
            )


            batch_embeddings = [
                embedding.values
                for embedding in result.embeddings
            ]


            if len(batch_embeddings) != len(batch):

                raise RuntimeError(
                    "Gemini returned "
                    f"{len(batch_embeddings)} embeddings "
                    f"for {len(batch)} texts."
                )


            all_embeddings.extend(
                batch_embeddings
            )


        return all_embeddings


    def embed_query(self, text):

        result = self.client.models.embed_content(
            model=self.model_name,
            contents=types.Content(
                parts=[
                    types.Part.from_text(
                        text=text
                    )
                ]
            ),
            config=types.EmbedContentConfig(
                output_dimensionality=self.output_dimensionality
            )
        )

        return result.embeddings[0].values


# ============================================================
# Initialize Embeddings
# ============================================================

embeddings = GeminiEmbeddings(
    model_name="gemini-embedding-2",
    output_dimensionality=1536,
    batch_size=50
)


# ============================================================
# Configuration
# ============================================================

GROUND_TRUTH_FILE = "retrieval_results11_gt.json"
PREDICTIONS_FILE = "retrieval_results11_updated.jsonl"

OUTPUT_FILE = "gemini_embedding_2_results.tsv"
CACHE_FILE = "embedding_cache.pkl"

THRESHOLD = 0.90

BATCH_SIZE = 100


# ============================================================
# Load Embedding Cache
# ============================================================

if os.path.exists(CACHE_FILE):

    print(
        f"Loading embedding cache from "
        f"{CACHE_FILE}..."
    )

    with open(
        CACHE_FILE,
        "rb"
    ) as f:

        embedding_cache = pickle.load(f)


    print(
        f"Loaded {len(embedding_cache)} "
        f"cached embeddings."
    )

else:

    embedding_cache = {}

    print(
        "No existing embedding cache found."
    )


# ============================================================
# Load Ground Truth
# ============================================================

with open(
    GROUND_TRUTH_FILE,
    "r",
    encoding="utf-8"
) as f:

    ground_truth = json.load(f)


# ============================================================
# Load Predictions
# ============================================================

predictions = {}

with open(
    PREDICTIONS_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line_number, line in enumerate(
        f,
        start=1
    ):

        line = line.strip()

        if not line:
            continue

        try:

            obj = json.loads(line)

        except json.JSONDecodeError as e:

            print(
                f"WARNING: Could not parse "
                f"line {line_number}: {e}"
            )

            continue

        idx = obj["idx"]

        predictions[idx] = obj


# ============================================================
# Load Completed TSV Rows
# ============================================================

completed = set()

if os.path.exists(OUTPUT_FILE):

    print(
        f"Reading completed rows from "
        f"{OUTPUT_FILE}..."
    )

    with open(
        OUTPUT_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            completed.add(
                (
                    row["mbe_id"],
                    row["idx"],
                    row["run"]
                )
            )


    print(
        f"Found {len(completed)} "
        f"completed rows."
    )


# ============================================================
# Collect ALL entity names needed by the dataset
#
# This is the important change.
#
# We scan the entire ground truth + predictions first.
# Then we find which names are missing from the cache.
#
# This means batch 1 can contain names from many different
# MBE questions/runs instead of being limited to one run.
# ============================================================

all_required_names = []
seen_names = set()


def add_name(name):

    if name not in seen_names:

        seen_names.add(name)

        all_required_names.append(name)


# ------------------------------------------------------------
# Ground truth names
# ------------------------------------------------------------

for mbe_id, gt_data in ground_truth.items():

    for entity in gt_data["entities"]:

        add_name(
            entity["name"]
        )


# ------------------------------------------------------------
# Prediction names
# ------------------------------------------------------------

for variant_id, prediction_data in predictions.items():

    for run_data in prediction_data["runs"]:

        for entity in run_data["entities"]:

            add_name(
                entity["name"]
            )


print()
print(
    f"Total unique entity names required: "
    f"{len(all_required_names)}"
)


# ============================================================
# Find Missing Names
# ============================================================

missing_names = [
    name
    for name in all_required_names
    if name not in embedding_cache
]


print(
    f"Already cached: "
    f"{len(all_required_names) - len(missing_names)}"
)

print(
    f"Missing embeddings: "
    f"{len(missing_names)}"
)


# ============================================================
# Save Cache Safely
# ============================================================

def save_embedding_cache():

    temp_file = CACHE_FILE + ".tmp"

    with open(
        temp_file,
        "wb"
    ) as f:

        pickle.dump(
            embedding_cache,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    os.replace(
        temp_file,
        CACHE_FILE
    )


# ============================================================
# Generate ALL Missing Embeddings in Full Batches
# ============================================================

if missing_names:

    print()
    print(
        f"Generating {len(missing_names)} "
        f"missing embeddings..."
    )

    total_batches = (
        len(missing_names) + BATCH_SIZE - 1
    ) // BATCH_SIZE


    for batch_number, start in enumerate(
        range(
            0,
            len(missing_names),
            BATCH_SIZE
        ),
        start=1
    ):

        batch = missing_names[
            start:start + BATCH_SIZE
        ]


        print()
        print(
            f"Embedding batch "
            f"{batch_number}/{total_batches} "
            f"({len(batch)} names)"
        )


        # ----------------------------------------------------
        # Create one Content object per name
        # ----------------------------------------------------

        contents = [
            types.Content(
                parts=[
                    types.Part.from_text(
                        text=text
                    )
                ]
            )
            for text in batch
        ]


        # ----------------------------------------------------
        # One Gemini API call for the entire batch
        # ----------------------------------------------------

        result = embeddings.client.models.embed_content(
            model=embeddings.model_name,
            contents=contents,
            config=types.EmbedContentConfig(
                output_dimensionality=(
                    embeddings.output_dimensionality
                )
            )
        )


        batch_embeddings = [
            embedding.values
            for embedding in result.embeddings
        ]


        if len(batch_embeddings) != len(batch):

            raise RuntimeError(
                "Gemini returned "
                f"{len(batch_embeddings)} embeddings "
                f"for {len(batch)} names."
            )


        # ----------------------------------------------------
        # Store embeddings
        # ----------------------------------------------------

        for text, vector in zip(
            batch,
            batch_embeddings
        ):

            embedding_cache[text] = vector


        # ----------------------------------------------------
        # Save immediately after every batch
        #
        # If the program stops after batch 7, batches 1-7
        # are already safely stored.
        # ----------------------------------------------------

        save_embedding_cache()


        print(
            f"Saved batch {batch_number}. "
            f"Cache now contains "
            f"{len(embedding_cache)} embeddings."
        )


else:

    print()
    print(
        "All required embeddings are already cached."
    )


# ============================================================
# Get Embeddings From Cache
# ============================================================

def get_embeddings(texts):

    missing = [
        text
        for text in texts
        if text not in embedding_cache
    ]

    if missing:

        raise RuntimeError(
            "Unexpected missing embeddings: "
            + str(missing[:10])
        )

    return [
        embedding_cache[text]
        for text in texts
    ]


# ============================================================
# TSV Fields
# ============================================================

fieldnames = [
    "mbe_id",
    "idx",
    "biased_category",
    "model",
    "temperature",
    "run",

    "gt_count",
    "pred_count",

    "tp",
    "fp",
    "fn",

    "precision",
    "recall",
    "f1",

    "gt_entities",
    "pred_entities",
]


# ============================================================
# Open TSV in Append Mode
# ============================================================

file_exists = (
    os.path.exists(OUTPUT_FILE)
    and os.path.getsize(OUTPUT_FILE) > 0
)


with open(
    OUTPUT_FILE,
    "a",
    newline="",
    encoding="utf-8"
) as out:

    writer = csv.DictWriter(
        out,
        fieldnames=fieldnames,
        delimiter="\t"
    )


    # --------------------------------------------------------
    # Header only if new file
    # --------------------------------------------------------

    if not file_exists:

        writer.writeheader()

        out.flush()


    # ========================================================
    # Evaluate every MBE
    # ========================================================

    for mbe_id, gt_data in ground_truth.items():

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        gt = [
            entity["name"]
            for entity in gt_data["entities"]
        ]


        # ----------------------------------------------------
        # Prediction variants
        # ----------------------------------------------------

        variants = [
            key
            for key in predictions.keys()
            if key.startswith(mbe_id + "_")
        ]


        if not variants:

            print(
                f"WARNING: No predictions found "
                f"for {mbe_id}"
            )

            continue


        # ====================================================
        # Each variant
        # ====================================================

        for variant_id in variants:

            prediction_data = predictions[
                variant_id
            ]

            metadata = prediction_data.get(
                "metadata",
                {}
            )


            model = metadata.get(
                "model",
                ""
            )

            temperature = metadata.get(
                "temperature",
                ""
            )

            biased_category = metadata.get(
                "biased_category",
                ""
            )


            # =================================================
            # Each run
            # =================================================

            for run_data in prediction_data["runs"]:

                run_number = run_data["run"]


                run_key = (
                    mbe_id,
                    variant_id,
                    str(run_number)
                )


                # ------------------------------------------------
                # Skip completed TSV rows
                # ------------------------------------------------

                if run_key in completed:

                    continue


                # ------------------------------------------------
                # Prediction entity names
                # ------------------------------------------------

                pred = [
                    entity["name"]
                    for entity in run_data["entities"]
                ]


                # ------------------------------------------------
                # Empty prediction
                # ------------------------------------------------

                if len(pred) == 0:

                    tp = 0

                    fp = 0

                    fn = len(gt)

                    precision = 0

                    recall = 0

                    f1 = 0


                else:

                    # ============================================
                    # Retrieve embeddings ONLY from cache
                    # ============================================

                    gt_emb = get_embeddings(gt)

                    pred_emb = get_embeddings(pred)


                    # ============================================
                    # NumPy arrays
                    # ============================================

                    gt_emb = np.asarray(
                        gt_emb,
                        dtype=np.float32
                    )

                    pred_emb = np.asarray(
                        pred_emb,
                        dtype=np.float32
                    )


                    # ============================================
                    # Cosine similarity
                    # ============================================

                    sim = cosine_similarity(
                        pred_emb,
                        gt_emb
                    )


                    # ============================================
                    # Hungarian matching
                    # ============================================

                    cost = 1 - sim

                    pred_idx, gt_idx = (
                        linear_sum_assignment(cost)
                    )


                    # ============================================
                    # True positives
                    # ============================================

                    tp = 0

                    for p, g in zip(
                        pred_idx,
                        gt_idx
                    ):

                        if sim[p, g] >= THRESHOLD:

                            tp += 1


                    # ============================================
                    # False positives / negatives
                    # ============================================

                    fp = len(pred) - tp

                    fn = len(gt) - tp


                    # ============================================
                    # Precision
                    # ============================================

                    precision = (
                        tp / (tp + fp)
                        if (tp + fp) > 0
                        else 0
                    )


                    # ============================================
                    # Recall
                    # ============================================

                    recall = (
                        tp / (tp + fn)
                        if (tp + fn) > 0
                        else 0
                    )


                    # ============================================
                    # F1
                    # ============================================

                    f1 = (
                        2 * precision * recall
                        / (precision + recall)
                        if (precision + recall) > 0
                        else 0
                    )


                # =================================================
                # Write TSV row
                # =================================================

                writer.writerow({

                    "mbe_id": mbe_id,

                    "idx": variant_id,

                    "biased_category": biased_category,

                    "model": model,

                    "temperature": temperature,

                    "run": run_number,

                    "gt_count": len(gt),

                    "pred_count": len(pred),

                    "tp": tp,

                    "fp": fp,

                    "fn": fn,

                    "precision": f"{precision:.4f}",

                    "recall": f"{recall:.4f}",

                    "f1": f"{f1:.4f}",

                    "gt_entities": " | ".join(gt),

                    "pred_entities": " | ".join(pred),
                })


                # ------------------------------------------------
                # Immediately save row
                # ------------------------------------------------

                out.flush()


                # ------------------------------------------------
                # Mark completed
                # ------------------------------------------------

                completed.add(
                    run_key
                )


                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                print(
                    f"Processed {mbe_id} "
                    f"| {variant_id} "
                    f"| run {run_number}"
                )


# ============================================================
# Finished
# ============================================================

print()
print(
    "Evaluation complete."
)

print(
    f"Results saved to: {OUTPUT_FILE}"
)

print(
    f"Total cached embeddings: "
    f"{len(embedding_cache)}"
)

print(
    f"Total completed rows: "
    f"{len(completed)}"
)
