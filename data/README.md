# PulseOps Data

This directory separates immutable source files from derived data. Raw NHS England files must be kept exactly as downloaded: do not rename their columns, edit values, or merge them manually.

## Directory layout

- `raw/beds/`: NHS England UEC daily bed SitRep CSV files.
- `raw/discharges/`: NHS England Acute Discharge SitRep CSV files.
- `raw/admissions/`: NHS England A&E Attendances and Emergency Admissions monthly CSV files.
- `interim/`: files created after validation and normalization, before feature engineering.
- `processed/`: model-ready datasets and feature tables.

Raw CSV files are excluded from version control by the repository `.gitignore`. Keep the directory structure and use a data registry or artifact store for large files when sharing the project.

## Sources

### NHS England Bed Availability / UEC Daily SitRep

Source: <https://www.england.nhs.uk/statistics/statistical-work-areas/bed-availability-and-occupancy/critical-care-and-general-acute-beds-urgent-and-emergency-care-daily-situation-reports/>

Coverage: April 2026 - August 2026

Granularity: Daily, with long-format metric records by reporting level and organisation.

Usage: Hospital bed capacity, occupancy, and critical-care operational features.

Schema observed: `Period`, `Level`, `Region`, `ICB`, `Org Code`, `Org Name`, `Metric`, `Type`, `Value`.

Note: the August file is marked as a provisional version by NHS England.

### NHS England Acute Discharge Situation Report

Source: <https://www.england.nhs.uk/statistics/statistical-work-areas/discharge-delays/acute-discharge-situation-report/>

Coverage: April 2026 - August 2026

Granularity: Daily, with long-format metric records by reporting level and organisation.

Usage: Discharge flow, criteria-to-reside, and delayed discharge features.

Schema observed: `Period`, `Level`, `Region`, `ICB`, `Org Code`, `Org Name`, `Metric`, `Metric Type`, `Metric Group`, `Value`.

### NHS England A&E Attendances and Emergency Admissions

Source: <https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/ae-attendances-and-emergency-admissions-2026-27/>

Coverage: April 2026 - August 2026

Granularity: Monthly, by provider organisation.

Usage: Monthly demand and emergency-admission context. This source is not daily and should not be treated as if it were a daily target series.

Schema observed: `Period`, `Org Code`, `Parent Org`, `Org name`, plus A&E attendance, waiting-time, and emergency-admission measures.

## Raw file manifest

The following files were found under `data/raw/` on 2026-09-15. Names are preserved exactly as downloaded.

### Beds

| Coverage month | Source file | Local size |
|---|---|---:|
| 2026-04 | `202604-April-2026-beds-sitrep-data-finalversion.csv` | 1,376,476 bytes |
| 2026-05 | `202605-May-2026-beds-sitrep-data-finalversion.csv` | 1,460,688 bytes |
| 2026-06 | `202606-June-2026-beds-sitrep-data-finalversion.csv` | 1,376,241 bytes |
| 2026-07 | `202607-July-2026-beds-sitrep-data-finalversion.csv` | 1,368,201 bytes |
| 2026-08 | `202608-August-2026-beds-sitrep-data-provisionalversion.csv` | 1,368,181 bytes |

### Discharges

| Coverage month | Source file | Local size |
|---|---|---:|
| 2026-04 | `Daily-discharge-sitrep-monthly-data-webfile-1-CSV-apr26-v2.csv` | 9,067,799 bytes |
| 2026-05 | `Daily-discharge-sitrep-monthly-data-webfile-2-CSV-may26-v2.csv` | 8,939,990 bytes |
| 2026-06 | `Daily-discharge-sitrep-monthly-data-webfile-3-CSV-jun26-v2.csv` | 8,835,230 bytes |
| 2026-07 | `Daily-discharge-sitrep-monthly-data-webfile-04-CSV-jul26.csv` | 9,009,206 bytes |
| 2026-08 | `Daily-discharge-sitrep-monthly-data-webfile-05-CSV-aug26.csv` | 8,882,396 bytes |

### Admissions

| Coverage month | Source file | Local size |
|---|---|---:|
| 2026-04 | `April-2026-CSV-QD21vf.csv` | 27,915 bytes |
| 2026-05 | `May-2026-CSV-F4flrg.csv` | 27,775 bytes |
| 2026-06 | `June-2026-CSV-Wfg38l.csv` | 27,989 bytes |
| 2026-07 | `July-2026-CSV-Trf28h.csv` | 27,845 bytes |
| 2026-08 | `August-2026-CSV-De2k3n.csv` | 28,180 bytes |

## Provenance fields

For every future download, record these fields in the ingestion manifest before processing:

| Field | Meaning |
|---|---|
| `downloaded_at` | UTC timestamp when the file was retrieved |
| `source_url` | NHS England page or direct download URL |
| `source_file` | Original filename, unchanged |
| `coverage_start` | First period represented by the file |
| `coverage_end` | Last period represented by the file |
| `sha256` | Content hash used to detect replacement or corruption |

The timestamps in the manifest above are local filesystem verification timestamps, not authoritative download timestamps. Populate `downloaded_at` and `sha256` when the reproducible downloader is added.

## Processing contract

The planned workflow is:

```text
raw CSV -> validate -> normalize -> interim -> merge -> features -> processed
```

The first inspection has established that beds and discharges can share common dimensions such as `Period` and `Org Code`, but their metric columns differ. A&E is monthly and wide, so the exact join and prediction target must be chosen after profiling periods, organisation coverage, duplicate keys, and metric values.

Do not decide the final feature list until schema validation confirms the available fields. Candidate derived fields include occupancy rate, day of week, weekend flag, lagged occupancy, rolling occupancy, and discharge-flow windows.

## Generated interim artifacts

The first normalization run created these derived, reproducible files under `data/interim/`:

| File | Rows | Columns | Purpose |
|---|---:|---:|---|
| `discharges_normalized.parquet` | 17,958 | 5 | Canonical provider-day discharge table plus `target_next_day` |
| `beds_normalized.parquet` | 603 | 8 | Provider Type 1 exact-date G&A bed context |
| `ae_normalized.parquet` | 956 | 5 | Provider-month A&E demand context |

These files are excluded from source control and can be regenerated with `write_interim_tables` in `src/data/normalize.py`. The raw CSV files remain untouched.
