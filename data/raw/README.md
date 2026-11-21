# Raw USPTO / PatentsView data

이 폴더의 `.zip` 원본은 **용량이 커서 GitHub에 올리지 않습니다** (약 2.7GB).

## 필요 파일 (PatentsView ODP Q4 2025)

| 파일 | 용도 |
|---|---|
| `g_patent.tsv.zip` | 특허 기본 정보 |
| `g_patent_abstract.tsv.zip` | 초록 |
| `g_application.tsv.zip` | 출원일 |
| `g_cpc_current.tsv.zip` | CPC |
| `g_assignee_disambiguated.tsv.zip` | 출원인 |

## 받는 방법

1. [USPTO Open Data Portal](https://data.uspto.gov/) / PatentsView bulk 제품에서 위 파일을 다운로드
2. 이 디렉터리(`data/raw/`)에 저장
3. (선택) API 사용 시: `export USPTO_API_KEY=발급키`

```bash
python scripts/collect_stage1_data.py
```
