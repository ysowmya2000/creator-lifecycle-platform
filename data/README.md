# Data Provenance

Source: [ageitgey/all-podcasts-dataset](https://github.com/ageitgey/all-podcasts-dataset) on GitHub.

- `data/raw/` holds the downloaded dataset (not tracked in git, see `.gitignore`).
- `data/processed/` holds `creators.parquet` and `creators.db`, the outputs of `src/data/loader.py`
  and `src/data/features.py`.

Run `python src/data/loader.py` to fetch and parse the raw data before running feature engineering.
