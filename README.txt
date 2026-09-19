- The scripts are arranged in a sequential order to make it easier to understand and run the code
- A small description of each script and what it does is given at the end of this document
- embedding_cache.db is a large file so its not uploaded on github, however it can be provided upon request
- Retrieval scripts folder contains all the scripts which were used for extracting legal rules by calling different LLMs
- Retrieval results folder contains the results of the evaluations
- Different threshold extractions are kept in separate .tsv files
- Evaluation scripts folder contains different evaluation files that were used during research
- Evaluation results folder contains the results of evaluation scripts
- Helper scripts were small intermediate scripts that were used for various purposes, like adding a column to the dataset or converting .json to .jsonl etc.
- If a code doesn't run properly, then its probably because one of the helper scripts were not run before
- For example a code might not show biased categories in its results, that is because there is helper script which was used to add biased categories to the results later, after extracting legal rules from the LLMs

Below is a small description of each script and what it does

RETRIEVAL_SCRIPTS:

retrieve_data10_gt.py (Gemini 3.1 Flash Lite ground truth file, temperature=0.0)
retrieve_data10.py (Gemini 3.1 Flash Lite, retrieval file, temperature=0.7)
retrieve_data11_gt.py (GPT 5 mini Ground Truth file)
retrieve_data11_batch.py (GPT 5 mini, retrieval file)
retrieve_data12_gt.py (Claude Haiku 4.5, ground truth file, temperature=0)
retrieve_data12_batch.py (Claude Haiku 4.5, retrieval file, temperature=0.7)
retrieve_data13_gt.py (Meta Llama 4 Scout, ground truth file, temperature=0)
retrieve_data13_batch.py (Meta Llama 4 Scout, retrieval file, temperature=0.7)
retrieve_data14_gt.py (Meta Llama 4 Maverick, ground truth file, temperature=0)
retrieve_data14_batch.py (Meta Llama 4 Maverick, retrieval file, temperature=0.7)

EVALUATION_SCRIPTS:

evaluate_results4.py (Aggregate f1 across biased category)
evaluate_results5.py (Gemini Embeddings 2 instead of BGE for faster inference)
evaluate_results6.py (Added concurrency and converted to .db instead of .pkl)
evaluate_results7.py (Calculated f1_by_difficulty)
evaluate_results8.py (Calculated f1_by_biased_category_and_difficulty)
evaluate_results9.py (evaluated all the results at once)

HELPER_SCRIPTS:

check_relations3.py (Check if each row has exactly 5 results i.e. runs)
check_relations4.py (Check if every qa idx is in gt idx and others etc.)
check_relations5.py (Check all the missing idx numbers)
check_relations6.py (Checking if the embeddings are all of the same dimension)
create_category.py (Get prompt + question in one column and drop all other columns)
create_category2.py (Ask LLM to classify modified prompts into easy, medium and hard categories)
create_category3.py (Add the difficulty column to .tsv files)
create_category4.py (Add threshold = 0.90 in results.tsv)
update_results.py (add LLM, temperature and run info)
update_results2.py (add biased_category and other metadata)
update_results3.py (Convert json to jsonl)

DATA_FILES:

qa.csv (source file)
biased_final2.csv (converted qa.csv into this biased file)
embedding_cache.db (Contains all the embeddings for evaluation, available upon request)