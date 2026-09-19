import json
import csv
import os
import sqlite3
import time
import threading
import random
import pickle

import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

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
# Configuration
# ============================================================

GROUND_TRUTH_FILE = "retrieval_results14_gt.json"
PREDICTIONS_FILE = "retrieval_results14_updated.jsonl"

difficulty_df = pd.read_csv("create_category2.csv")
difficulty_map = difficulty_df.set_index("idx")["difficulty"]

OUTPUT_FILE = "llama_4_maverick_th0.85_categorized.tsv"

# New persistent embedding database
CACHE_DB_FILE = "embedding_cache.db"

# Existing old cache, used only for one-time migration
OLD_CACHE_FILE = "embedding_cache.pkl"

# Cosine similarity threshold
THRESHOLD = 0.85

# ------------------------------------------------------------
# Gemini request configuration
# ------------------------------------------------------------

# Number of entity names per Gemini request.
#
# Larger batches = fewer API requests.
BATCH_SIZE = 100

# Number of simultaneous worker threads.
#
# The global rate limiter below controls the actual request rate,
# so even with multiple workers requests will not be launched
# faster than REQUEST_INTERVAL_SECONDS.
MAX_WORKERS = 5

# ------------------------------------------------------------
# GLOBAL REQUEST RATE LIMIT
# ------------------------------------------------------------
#
# Your quota is 3,000 RPM.
#
# With:
#
#     REQUEST_INTERVAL_SECONDS = 2
#
# the program can start at most approximately:
#
#     60 / 2 = 30 requests/minute
#
# This is intentionally much lower than your quota to avoid
# bursts and accidental quota spikes.
#
# You can increase/decrease this later.
#
# Examples:
#
#   0.5 sec = ~120 RPM
#   1.0 sec = ~60 RPM
#   2.0 sec = ~30 RPM
#   5.0 sec = ~12 RPM
#
REQUEST_INTERVAL_SECONDS = 2.0

# Small random delay added before retrying a failed request.
RETRY_JITTER_SECONDS = 1.0

# Maximum number of attempts per request.
MAX_ATTEMPTS = 5

# Embedding model
MODEL_NAME = "gemini-embedding-2"

OUTPUT_DIMENSIONALITY = 1536


# ============================================================
# Global Gemini request limiter
# ============================================================

request_lock = threading.Lock()

last_request_time = 0.0


def wait_for_request_slot():
    """
    Global rate limiter.

    Guarantees that the START of Gemini API requests is separated
    by at least REQUEST_INTERVAL_SECONDS.

    This works across all worker threads.
    """

    global last_request_time

    with request_lock:

        current_time = time.monotonic()

        elapsed = (
            current_time - last_request_time
        )

        if elapsed < REQUEST_INTERVAL_SECONDS:

            sleep_time = (
                REQUEST_INTERVAL_SECONDS - elapsed
            )

            time.sleep(sleep_time)

        last_request_time = time.monotonic()


# ============================================================
# Gemini Embeddings
# ============================================================

class GeminiEmbeddings(Embeddings):

    def __init__(
        self,
        model_name=MODEL_NAME,
        output_dimensionality=OUTPUT_DIMENSIONALITY
    ):

        self.model_name = model_name
        self.output_dimensionality = output_dimensionality

        self.client = genai.Client(
            api_key=gemini_api_key
        )

    def embed_documents(self, texts):

        contents = [
            types.Content(
                parts=[
                    types.Part.from_text(
                        text=text
                    )
                ]
            )
            for text in texts
        ]

        wait_for_request_slot()

        result = self.client.models.embed_content(
            model=self.model_name,
            contents=contents,
            config=types.EmbedContentConfig(
                output_dimensionality=self.output_dimensionality
            )
        )

        vectors = [
            embedding.values
            for embedding in result.embeddings
        ]

        if len(vectors) != len(texts):

            raise RuntimeError(
                f"Gemini returned {len(vectors)} embeddings "
                f"for {len(texts)} texts."
            )

        return vectors

    def embed_query(self, text):

        wait_for_request_slot()

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
# Initialize Gemini
# ============================================================

embeddings = GeminiEmbeddings()


# ============================================================
# SQLite Database
# ============================================================

print()
print(
    f"Opening embedding database: {CACHE_DB_FILE}"
)

db = sqlite3.connect(
    CACHE_DB_FILE
)

db.execute(
    """
    CREATE TABLE IF NOT EXISTS embeddings (
        name TEXT PRIMARY KEY,
        vector BLOB NOT NULL
    )
    """
)

db.commit()

print("Embedding database ready.")


# ============================================================
# One-time migration from old pickle cache
#
# If embedding_cache.pkl exists, import it into SQLite.
#
# IMPORTANT:
# The pickle is NOT loaded during normal operation after
# migration.
# ============================================================

migration_marker = (
    CACHE_DB_FILE + ".migration_complete"
)


if (
    os.path.exists(OLD_CACHE_FILE)
    and not os.path.exists(migration_marker)
):

    print()
    print(
        "Existing pickle cache detected."
    )

    print(
        "Migrating existing embeddings to SQLite..."
    )

    with open(
        OLD_CACHE_FILE,
        "rb"
    ) as f:

        old_cache = pickle.load(f)

    print(
        f"Loaded {len(old_cache)} embeddings "
        f"from old pickle cache."
    )

    cursor = db.cursor()

    migrated = 0

    for name, vector in old_cache.items():

        vector_array = np.asarray(
            vector,
            dtype=np.float32
        )

        cursor.execute(
            """
            INSERT OR IGNORE INTO embeddings
            (name, vector)
            VALUES (?, ?)
            """,
            (
                name,
                vector_array.tobytes()
            )
        )

        migrated += 1

        if migrated % 10000 == 0:

            db.commit()

            print(
                f"Migrated {migrated} embeddings..."
            )

    db.commit()

    del old_cache

    # --------------------------------------------------------
    # Mark migration complete
    # --------------------------------------------------------

    with open(
        migration_marker,
        "w"
    ) as f:

        f.write(
            "Migration completed."
        )

    print(
        "Pickle cache migration complete."
    )

    print(
        "The pickle file can now be kept as a backup "
        "or deleted manually."
    )


# ============================================================
# Database helper functions
# ============================================================

def embedding_exists(name):

    cursor = db.execute(
        """
        SELECT 1
        FROM embeddings
        WHERE name = ?
        LIMIT 1
        """,
        (name,)
    )

    return cursor.fetchone() is not None


def get_embedding(name):

    cursor = db.execute(
        """
        SELECT vector
        FROM embeddings
        WHERE name = ?
        """,
        (name,)
    )

    row = cursor.fetchone()

    if row is None:

        raise RuntimeError(
            f"Embedding not found in database: {name}"
        )

    return np.frombuffer(
        row[0],
        dtype=np.float32
    )


def get_embeddings(names):

    vectors = []

    for name in names:

        vectors.append(
            get_embedding(name)
        )

    return vectors


def store_embeddings(names, vectors):

    cursor = db.cursor()

    for name, vector in zip(
        names,
        vectors
    ):

        vector_array = np.asarray(
            vector,
            dtype=np.float32
        )

        cursor.execute(
            """
            INSERT OR REPLACE INTO embeddings
            (name, vector)
            VALUES (?, ?)
            """,
            (
                name,
                vector_array.tobytes()
            )
        )

    db.commit()


def count_embeddings():

    cursor = db.execute(
        """
        SELECT COUNT(*)
        FROM embeddings
        """
    )

    return cursor.fetchone()[0]


# ============================================================
# Load Ground Truth
# ============================================================

print()
print(
    "Loading ground truth..."
)

with open(
    GROUND_TRUTH_FILE,
    "r",
    encoding="utf-8"
) as f:

    ground_truth = json.load(f)


# ============================================================
# Load Predictions
# ============================================================

print(
    "Loading predictions..."
)

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

    print()
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
# Collect ALL required entity names
# ============================================================

all_required_names = []
seen_names = set()


def add_name(name):

    if not isinstance(name, str) or not name.strip():
        print(f"WARNING: Skipping invalid entity name: {repr(name)}")
        return

    name = name.strip()

    if name not in seen_names:
        seen_names.add(name)
        all_required_names.append(name)


# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------

for mbe_id, gt_data in ground_truth.items():

    for entity in gt_data["entities"]:

        add_name(
            entity["name"]
        )


# ------------------------------------------------------------
# Predictions
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
# Find Missing Embeddings
#
# IMPORTANT:
# We do NOT load the whole database into RAM.
# SQLite checks each name directly on disk.
# ============================================================

missing_names = []

for name in all_required_names:

    if not embedding_exists(name):

        missing_names.append(name)


cached_count = (
    len(all_required_names)
    - len(missing_names)
)


print(
    f"Already cached: {cached_count}"
)

print(
    f"Missing embeddings: {len(missing_names)}"
)

print(
    f"Total embeddings in database: "
    f"{count_embeddings()}"
)


# ============================================================
# Thread-local Gemini clients
# ============================================================

thread_local = threading.local()


def get_thread_client():

    if not hasattr(
        thread_local,
        "client"
    ):

        thread_local.client = genai.Client(
            api_key=gemini_api_key
        )

    return thread_local.client


# ============================================================
# Embed one batch
#
# This function runs in worker threads.
# ============================================================

def embed_batch(batch):

    client = get_thread_client()

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

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1
    ):

        try:

            # ------------------------------------------------
            # IMPORTANT:
            # Wait BEFORE every API request, including retries.
            #
            # This prevents multiple workers/retries from
            # creating request bursts.
            # ------------------------------------------------

            wait_for_request_slot()

            print(
                f"Sending Gemini embedding request "
                f"({len(batch)} names) "
                f"| attempt {attempt}/{MAX_ATTEMPTS}"
            )

            result = client.models.embed_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.EmbedContentConfig(
                    output_dimensionality=(
                        OUTPUT_DIMENSIONALITY
                    )
                )
            )

            vectors = [
                embedding.values
                for embedding in result.embeddings
            ]

            if len(vectors) != len(batch):

                raise RuntimeError(
                    f"Gemini returned {len(vectors)} "
                    f"embeddings for {len(batch)} names."
                )

            return vectors

        except Exception as e:

            if attempt == MAX_ATTEMPTS:

                print(
                    f"Batch permanently failed after "
                    f"{MAX_ATTEMPTS} attempts."
                )

                raise

            # ------------------------------------------------
            # Exponential backoff + jitter
            #
            # 1st retry: ~2 sec
            # 2nd retry: ~4 sec
            # 3rd retry: ~8 sec
            # 4th retry: ~16 sec
            #
            # Plus small random jitter.
            # ------------------------------------------------

            base_wait = 2 ** attempt

            jitter = random.uniform(
                0,
                RETRY_JITTER_SECONDS
            )

            wait_time = (
                base_wait + jitter
            )

            print(
                f"Batch failed "
                f"(attempt {attempt}/{MAX_ATTEMPTS}): "
                f"{e}"
            )

            print(
                f"Retrying in "
                f"{wait_time:.2f} seconds..."
            )

            time.sleep(
                wait_time
            )


# ============================================================
# Generate Missing Embeddings Concurrently
# ============================================================

if missing_names:

    print()
    print(
        f"Generating {len(missing_names)} "
        f"missing embeddings..."
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Concurrent workers: {MAX_WORKERS}"
    )

    print(
        f"Minimum time between Gemini requests: "
        f"{REQUEST_INTERVAL_SECONDS} seconds"
    )

    estimated_rpm = (
        60 / REQUEST_INTERVAL_SECONDS
    )

    print(
        f"Maximum request rate: "
        f"~{estimated_rpm:.1f} requests/minute"
    )

    batches = [
        missing_names[i:i + BATCH_SIZE]
        for i in range(
            0,
            len(missing_names),
            BATCH_SIZE
        )
    ]

    total_batches = len(batches)

    print(
        f"Total batches: {total_batches}"
    )

    print()

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_to_batch_number = {}

        for batch_number, batch in enumerate(
            batches,
            start=1
        ):

            future = executor.submit(
                embed_batch,
                batch
            )

            future_to_batch_number[
                future
            ] = batch_number

        completed_batches = 0

        for future in as_completed(
            future_to_batch_number
        ):

            batch_number = (
                future_to_batch_number[future]
            )

            batch = batches[
                batch_number - 1
            ]

            vectors = future.result()

            # ------------------------------------------------
            # Main thread writes to SQLite
            # ------------------------------------------------

            store_embeddings(
                batch,
                vectors
            )

            completed_batches += 1

            print(
                f"Completed embedding batch "
                f"{batch_number}/{total_batches} "
                f"({len(batch)} names) "
                f"| {completed_batches}/{total_batches} "
                f"finished "
                f"| database: "
                f"{count_embeddings()} embeddings"
            )


else:

    print()
    print(
        "All required embeddings are already cached."
    )


# ============================================================
# TSV Fields
# ============================================================

fieldnames = [
    "mbe_id",
    "idx",
    "biased_category",
    "difficulty",
    "model",
    "threshold",
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
# Open TSV
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

    if not file_exists:

        writer.writeheader()

        out.flush()


    # ========================================================
    # Evaluate every MBE
    # ========================================================

    for mbe_id, gt_data in ground_truth.items():

        difficulty = difficulty_map.get(mbe_id, "")

        gt = [
            entity["name"].strip()
            for entity in gt_data["entities"]
            if isinstance(entity.get("name"), str) and entity["name"].strip()
        ]



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


                if run_key in completed:

                    continue


                pred = [
                    entity["name"].strip()
                    for entity in run_data["entities"]
                    if isinstance(entity.get("name"), str) and entity["name"].strip()
                ]



                # ------------------------------------------------
                # Find bad ground truth or prediction entries
                # ------------------------------------------------

                if "" in gt or any(not isinstance(x, str) or not x.strip() for x in gt):
                    print("BAD GT:", mbe_id, repr(gt))

                if "" in pred or any(not isinstance(x, str) or not x.strip() for x in pred):
                    print("BAD PRED:", mbe_id, variant_id, run_number, repr(pred))


                # ------------------------------------------------
                # Empty ground truth OR empty prediction
                # ------------------------------------------------

                if len(gt) == 0 or len(pred) == 0:

                    tp = 0

                    fp = len(pred)

                    fn = len(gt)

                    precision = (
                        tp / (tp + fp)
                        if (tp + fp) > 0
                        else 0
                    )

                    recall = (
                        tp / (tp + fn)
                        if (tp + fn) > 0
                        else 0
                    )

                    f1 = (
                        2 * precision * recall
                        / (precision + recall)
                        if (precision + recall) > 0
                        else 0
                    )

                else:

                    # ============================================
                    # Retrieve only the embeddings needed for this
                    # particular evaluation.
                    # ============================================

                    gt_emb = get_embeddings(gt)

                    pred_emb = get_embeddings(pred)


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
                        linear_sum_assignment(
                            cost
                        )
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
                # Write TSV
                # =================================================

                writer.writerow({

                    "mbe_id": mbe_id,

                    "idx": variant_id,

                    "biased_category": biased_category,

                    "difficulty": difficulty,

                    "model": model,

                    "threshold": THRESHOLD,

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


                out.flush()


                completed.add(
                    run_key
                )


                print(
                    f"Processed {mbe_id} "
                    f"| {variant_id} "
                    f"| run {run_number}"
                )


# ============================================================
# Close database
# ============================================================

final_embedding_count = count_embeddings()

db.close()


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
    f"Total embeddings in database: "
    f"{final_embedding_count}"
)

print(
    f"Total completed rows: "
    f"{len(completed)}"
)
