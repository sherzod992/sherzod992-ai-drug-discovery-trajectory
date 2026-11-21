# Processed data (local regenerate)

대용량 CSV는 `.gitignore` 대상입니다. 로컬에서 스크립트로 다시 만드세요.

| 출력 | 스크립트 |
|---|---|
| `patents_raw.csv`, `patents_preprocessed.csv`, `collection_summary.txt` | `scripts/collect_stage1_data.py` |
| `topic_*.csv/json`, `patent_topic_mapping.csv` | `scripts/step2_topic_modeling.py` |
| `cpc_*.csv/json`, `common_cpcs.csv` | `scripts/step3_cpc_network.py` |
| `assignee_*.csv/json` | `scripts/step4_assignee_comparison.py` |

논문 표·그림용 **소용량 요약**은 `data/results/`에 커밋되어 있습니다.
