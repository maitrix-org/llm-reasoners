from huggingface_hub import hf_hub_download

def download_all_files(repo_id, file_names, cache_dir=None):
    """
    Download all specified files from a Hugging Face dataset repository.

    Args:
        repo_id (str): The "namespace/repo_name" of the HF dataset (e.g., "your_user/dataset_name").
        file_names (List[str]): List of filenames to download.
        cache_dir (str, optional): Custom cache directory where files will be stored.
    """
    for fname in file_names:
        local_path = hf_hub_download(repo_id=repo_id, filename=fname, cache_dir=cache_dir, repo_type="dataset")
        print(f"✅ Downloaded {fname} → {local_path}")

if __name__ == "__main__":
    # Replace with your actual dataset repo ID
    repo_id = "LLM360/guru_preview"
    # List all files you want to download
    file_names = [
        "codegen__deduped_leetcode2k_1.3k.parquet",
        "codegen__deduped_livecodebench_440.parquet",
        "codegen__deduped_primeintellect_7.5k.parquet",
        "codegen__deduped_taco_8.8k.parquet",
        "logic__arcagi1_111.parquet",
        "logic__arcagi2_190.parquet",
        "logic__barc_1.6k.parquet",
        "logic__graph_logical_dataset_1.2k.parquet",
        "logic__ordering_puzzle_dataset_1.9k.parquet",
        "logic__zebra_puzzle_dataset_1.3k.parquet",
        "math__combined_54.4k.parquet",
        "simulation__codeio_fixed_12.1k_3.7k.parquet",
        "stem__web_3.6k.parquet",
        "table__hitab_4.3k.parquet",
        "table__multihier_1.5k.parquet"
    ]

    # Optionally specify a cache directory, or leave as None to use the default HF cache
    cache_dir = "./hf_cache"

    download_all_files(repo_id, file_names, cache_dir)
