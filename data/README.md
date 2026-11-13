# Data

## Source
Bao, Ke, Li, Yu & Zhang (2020), *Journal of Accounting Research* 58(1), public at
<https://github.com/JarFraud/FraudDetection> (`data_FraudDetection_JAR2020.csv`, ~48 MB).

- 146,045 U.S. firm-years, fiscal years 1990–2014, identified only by Compustat `gvkey`
- Label `misstate` = 1 if the firm-year was later the subject of an SEC AAER; `p_aaer` = AAER case ID
- 28 raw financial-statement items (Bao et al. Table 1) and 14 Dechow et al. (2011) ratios

The CSV is not committed (see `.gitignore`); `src/data.py` downloads it to this folder on first use.

## Preparation pipeline (all in code)
| Step | Where | What |
|---|---|---|
| Clean | `data.load` | ±inf → missing |
| Split | `data.split`, `data.validation_split` | train 1991–2001 / test 2003–2008; selection 1991–97 / 2000–01; 2-year gaps |
| Serial-fraud fix | `data.temporal_split` | training years of an AAER case that continues into the evaluation window are relabelled 0 |
| Classical preprocessing | `data.preprocess` | median imputation + 1/99% winsorizing, statistics from training rows only |
| Case sampling | `llm_baseline.sample_cases` | balanced later-AAER / no-AAER firm-years that have a prior-year record |
| Anonymisation | `run_agent.build_packet` | no gvkey or calendar year (t−1, t), dollar items × random constant in [0.5, 2] |

| Split | Years | Firm-years | Later-AAER |
|---|---|---|---|
| Fit (model selection) | 1991–1997 | 40,508 | 179 |
| Validation / LLM dev | 2000–2001 | 13,114 | 167 |
| Train (final refit) | 1991–2001 | 67,166 | 400 |
| Test (out of sample) | 2003–2008 | 35,166 | 261 |

## Files in this folder
- `cards/technique_cards.jsonl` — 30 forensic-accounting technique cards written for this project (original wording)
- `samples/case_packets_sample.jsonl` — 4 anonymised test packets exactly as the model sees them
