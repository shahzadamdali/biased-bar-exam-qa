import sqlite3

NUM_RECORDS = 50_000  # Number of records to check per batch

db = sqlite3.connect("embedding_cache.db")

# Get total number of records
total_records = db.execute("""
    SELECT COUNT(*)
    FROM embeddings
""").fetchone()[0]

print(f"Total records in database: {total_records:,}")
print(f"Checking in batches of {NUM_RECORDS:,}...\n")

# Check non-overlapping batches
for offset in range(0, total_records, NUM_RECORDS):

    rows = db.execute("""
        SELECT LENGTH(vector) / 4 AS dimension
        FROM embeddings
        LIMIT ? OFFSET ?
    """, (NUM_RECORDS, offset)).fetchall()

    dimensions = [row[0] for row in rows]

    batch_start = offset + 1
    batch_end = offset + len(dimensions)

    if dimensions and all(d == 1536 for d in dimensions):
        print(
            f"Records {batch_start:,}-{batch_end:,}: "
            f"ALL have 1536 dimensions."
        )
    else:
        print(
            f"Records {batch_start:,}-{batch_end:,}: "
            f"NOT all have 1536 dimensions."
        )
        print("Dimensions found:", sorted(set(dimensions)))

db.close()
