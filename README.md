# Mapping the technological trajectory of AI-assisted drug discovery

USPTO 특허(2016–2025)로 AI–신약 영역의 **토픽 궤적 · CPC 융합 네트워크 · 빅테크 vs 빅파마 포트폴리오**를 분석하는 연구 코드·산출물입니다.

## 폴더 구조

```
├── scripts/           # 분석 파이프라인 (1→4단계)
├── data/
│   ├── raw/           # USPTO zip (로컬만, Git 제외)
│   ├── processed/     # 대용량 CSV (로컬 재생성, Git 제외)
│   └── results/       # 소용량 요약 결과 (Git 포함)
├── figures/           # Figure 1–3
├── paper/             # 국문 MD/PDF, 영문 DOCX, 원 기획 docx
├── requirements.txt
└── README.md
```

## 환경

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API를 쓸 경우:

```bash
export USPTO_API_KEY=your_key
```

## 실행 순서

1. `data/raw/`에 PatentsView ODP zip 배치 ([안내](data/raw/README.md))
2. `python scripts/collect_stage1_data.py`
3. `python scripts/step2_topic_modeling.py`
4. `python scripts/step3_cpc_network.py`
5. `python scripts/step4_assignee_comparison.py`

요약 수치만 보려면 `data/results/`와 `figures/`를 확인하면 됩니다.

## 논문 산출물

| 파일 | 설명 |
|---|---|
| `paper/Mapping_AI_trajectory_KO.md` | 국문 원고 (수치 반영) |
| `paper/AI_drug_discovery_trajectory_paper.pdf` | 국문 PDF |
| `paper/AI_drug_discovery_trajectory_paper_EN.docx` | 영문 원고 |
| `figures/fig1_yearly_patents.png` | 연도별 출원 |
| `figures/fig2_topic_shares.png` | 토픽 점유율 |
| `figures/fig3_cpc_network.png` | CPC 네트워크 |

## 데이터 규모 (참고)

- 분석 집합: **105,954** patents (filing 2016-01-01 ~ 2025-12-31)
- 원본 raw zip ≈ 2.7GB → GitHub 미포함
- 대용량 processed CSV ≈ 350MB → GitHub 미포함 (로컬 재생성)

## 라이선스 / 출처

- 특허 데이터: USPTO / PatentsView ODP
- 코드: 이 저장소의 `scripts/`
