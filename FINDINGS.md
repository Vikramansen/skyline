# Findings log

Use this file for measured results from real data. Do not replace these fields with estimates.

## Run ledger

| Night | Alert rows | Unique objects | Budget | Confirmed in top-k | Precision@k | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Demo fixture | 24 | 5 | 5 | 2 | 40% | Synthetic sanity check only; not a scientific result. |

## To measure on broker data

- Which feature distributions separate later-confirmed transients from known variables?
- Which catalog and object types remove the greatest number of false positives?
- Cross-match latency at p50, p95, and p99 for the actual catalog size.
- End-to-end time from alert ingestion to a frozen ranked receipt.
- Precision@5, precision@10, and precision@20 per night, with the denominator shown.

## Guardrails

- Treat labels as delayed ground truth. Freeze each night’s ranking before querying confirmation sources.
- Inspect light curves manually before changing a score weight.
- Report failures and missing confirmations, not only successful targets.
