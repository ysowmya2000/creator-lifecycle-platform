"""Validate the feature matrix with Great Expectations.

Note: CLAUDE.md pins great-expectations==0.18.19, but that release predates
Python 3.13 support. This runs against GX 1.20.0's fluent (context/suite/
batch) API instead; requirements.txt reflects the installed version.
"""

import logging
from pathlib import Path

import great_expectations as gx
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PATH = Path("data/processed/features.parquet")
REPORT_PATH = Path("outputs/data_validation_report.html")

CRITICAL_EXPECTATIONS = {
    "podcast_id not null",
    "first_seen_date not null",
    "posting_freq_mean not null",
}


def build_suite() -> "gx.ExpectationSuite":
    suite = gx.ExpectationSuite(name="creator_features_suite")

    not_null_cols = ["podcast_id", "first_seen_date", "posting_freq_mean"]
    for col in not_null_cols:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    range_checks = [
        ("consistency_score_30d", 0, 1),
        ("consistency_score_60d", 0, 1),
        ("cadence_variance_ratio", 0, 100),
    ]
    for col, lo, hi in range_checks:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(column=col, min_value=lo, max_value=hi)
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnMeanToBeBetween(column="posting_freq_mean", min_value=1, max_value=60)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnMedianToBeBetween(column="first_14d_episode_count", min_value=0, max_value=30)
    )

    set_checks = [
        ("event_occurred", [0, 1]),
        ("survived_14d", [0, 1]),
        ("survived_30d", [0, 1]),
        ("event_type", [0, 1, 2]),
    ]
    for col, values in set_checks:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column=col, value_set=values))

    return suite


def run_validation(df: pd.DataFrame) -> "gx.core.expectation_validation_result.ExpectationSuiteValidationResult":
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("pandas")
    data_asset = data_source.add_dataframe_asset(name="features")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(build_suite())
    return batch.validate(suite)


def render_html_report(result, out_path: Path) -> None:
    rows = []
    for r in result.results:
        cfg = r.expectation_config
        exp_type = cfg.type
        col = cfg.kwargs.get("column", "")
        status = "PASS" if r.success else "FAIL"
        color = "#1a7f37" if r.success else "#cf222e"
        unexpected = r.result.get("unexpected_percent")
        detail = f"{unexpected:.2f}% unexpected" if unexpected is not None else ""
        rows.append(
            f"<tr><td>{exp_type}</td><td>{col}</td>"
            f"<td style='color:{color};font-weight:bold'>{status}</td><td>{detail}</td></tr>"
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Data Validation Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d0d7de; padding: 8px 12px; text-align: left; }}
th {{ background: #f6f8fa; }}
h1 {{ font-size: 1.4rem; }}
.summary {{ font-size: 1.1rem; margin-bottom: 1rem; }}
</style></head>
<body>
<h1>Creator Features — Data Validation Report</h1>
<p class="summary">Overall: <strong style="color:{'#1a7f37' if result.success else '#cf222e'}">
{'PASS' if result.success else 'FAIL'}</strong> &mdash; {sum(r.success for r in result.results)}/{len(result.results)} expectations passed</p>
<table>
<tr><th>Expectation</th><th>Column</th><th>Status</th><th>Detail</th></tr>
{''.join(rows)}
</table>
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    logger.info("Wrote %s", out_path)


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    logger.info("Validating %d rows", len(df))

    result = run_validation(df)
    render_html_report(result, REPORT_PATH)

    failed_critical = [
        r for r in result.results
        if not r.success and f"{r.expectation_config.kwargs.get('column')} not null" in CRITICAL_EXPECTATIONS
    ]
    n_pass = sum(r.success for r in result.results)
    logger.info("%d/%d expectations passed", n_pass, len(result.results))

    if failed_critical:
        raise AssertionError(f"Critical expectations failed: {failed_critical}")

    if not result.success:
        logger.warning("Some non-critical expectations failed; see %s", REPORT_PATH)


if __name__ == "__main__":
    main()
