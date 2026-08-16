"""Load the SPoRC podcast/episode catalogs and build the creators table.

Data source: blitt/SPoRC on HuggingFace (research-use-only license).
The episode_catalog only covers episodes published in a ~61-day observation
window (2020-04-30 to 2020-06-30) rather than each show's full since-launch
history, so downstream features are defined relative to that window rather
than to a creator's true first-ever episode. See CLAUDE.md for the reframing
rationale.
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/sporc/metadata")
PROCESSED_DIR = Path("data/processed")

PODCAST_CATALOG = RAW_DIR / "podcast_catalog.parquet"
EPISODE_CATALOG = RAW_DIR / "episode_catalog.parquet"


def load_raw_catalogs(con: duckdb.DuckDBPyConnection) -> None:
    """Register the raw parquet catalogs as duckdb views."""
    con.execute(f"CREATE OR REPLACE VIEW podcast_catalog AS SELECT * FROM '{PODCAST_CATALOG}'")
    con.execute(f"CREATE OR REPLACE VIEW episode_catalog AS SELECT * FROM '{EPISODE_CATALOG}'")
    n_podcasts = con.execute("SELECT COUNT(*) FROM podcast_catalog").fetchone()[0]
    n_episodes = con.execute("SELECT COUNT(*) FROM episode_catalog").fetchone()[0]
    logger.info("Loaded %d podcasts, %d episodes", n_podcasts, n_episodes)


def build_creators_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Join podcast + episode catalogs into one row per creator with an
    episode_dates array for downstream feature engineering."""
    query = """
    WITH ep_agg AS (
        SELECT
            podcast_id,
            list_sort(list(to_timestamp(CAST(episode_date AS BIGINT) / 1000.0))) AS episode_dates,
            list(ep_title) AS episode_titles,
            COUNT(*) AS episodes_in_window
        FROM episode_catalog
        WHERE episode_date IS NOT NULL
        GROUP BY podcast_id
    )
    SELECT
        p.podcast_id,
        p.pod_title AS title,
        p.pod_description,
        p.language,
        p.primary_category AS category,
        e.episodes_in_window,
        e.episode_dates,
        e.episode_titles,
        list_min(e.episode_dates) AS first_seen_date,
        list_max(e.episode_dates) AS last_seen_date
    FROM podcast_catalog p
    JOIN ep_agg e ON p.podcast_id = e.podcast_id
    """
    df = con.execute(query).df()
    for col in ("first_seen_date", "last_seen_date"):
        df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)
    logger.info("Joined creators table: %d rows before filtering", len(df))
    return df


def filter_creators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the filtering rules from CLAUDE.md, adapted to the windowed data.

    - keep only creators with a first_seen_date (non-null)
    - keep only creators with at least 2 episodes observed in the window
      (single-episode creators give no cadence signal)
    - keep only English-language podcasts
    """
    n0 = len(df)
    df = df[df["first_seen_date"].notna()]
    n1 = len(df)
    df = df[df["episodes_in_window"] >= 2]
    n2 = len(df)
    df = df[df["language"].str.lower().str.startswith("en", na=False)]
    n3 = len(df)

    logger.info("Filter: %d -> %d (non-null first_seen_date)", n0, n1)
    logger.info("Filter: %d -> %d (>=2 episodes in window)", n1, n2)
    logger.info("Filter: %d -> %d (English language)", n2, n3)

    null_rates = df.isna().mean().sort_values(ascending=False)
    logger.info("Null rates after filtering:\n%s", null_rates[null_rates > 0])

    return df.reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    load_raw_catalogs(con)

    df = build_creators_table(con)
    df = filter_creators(df)

    out_parquet = PROCESSED_DIR / "creators.parquet"
    df.to_parquet(out_parquet, index=False)
    logger.info("Wrote %s (%d rows)", out_parquet, len(df))

    out_db = PROCESSED_DIR / "creators.db"
    con.execute(f"CREATE OR REPLACE TABLE creators AS SELECT * FROM df")
    con.execute(f"ATTACH '{out_db}' AS sqlite_out (TYPE sqlite)")
    con.execute("CREATE OR REPLACE TABLE sqlite_out.creators AS SELECT * EXCLUDE (episode_dates, episode_titles) FROM creators")
    con.execute("DETACH sqlite_out")
    logger.info("Wrote %s", out_db)

    logger.info("Final counts: total=%d, mean episodes_in_window=%.2f", len(df), df["episodes_in_window"].mean())


if __name__ == "__main__":
    main()
