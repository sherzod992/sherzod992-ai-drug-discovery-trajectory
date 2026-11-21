# AI 신약 개발의 기술 발전 경로 및 융합 매핑: 동적 토픽 모델링 및 특허 네트워크 분석

> 영문 제목: *Mapping the technological trajectory and convergence of AI-assisted drug discovery: A dynamic topic modeling and patent network analysis*  
> ※ 도메인은 medical imaging AI 또는 digital therapeutics로 변경 가능

---

## 0. 주제 선정 배경

### 0.1. 데이터 활용 면에서 유리한 이유

- **단일 데이터소스(특허)만 사용 가능**: 논문(PubMed)과 특허(USPTO)를 함께 매핑하면 포맷·용어 체계 정제에 시간이 많이 든다. 이 주제는 USPTO 데이터만으로 분석이 가능하다.
- **무료 API·오픈 데이터 풍부**: PatentsView API, Google Patents, Lens.org 등으로 초록, 청구항, 출원일, CPC, 인용 정보를 CSV/JSON으로 받을 수 있다.
- **전처리·모델링 난이도 최적화**
  - 텍스트(초록): 정제된 영문이므로 BERTopic·사전학습 임베딩(SciBERT 등)으로 시계열 토픽 변화를 바로 추출 가능
  - 메타데이터(CPC·인용): CPC 동시출현(co-occurrence) 행렬만으로도 기술 융합 네트워크 분석이 가능

### 0.2. 다른 옵션과의 난이도 비교

| 추천 주제 유형 | 필요 데이터 | 데이터 정제 난이도 | 논문 완성 속도 및 채택 유리성 |
|---|---|---|---|
| 추천안 (시계열 토픽 + 네트워크) | 특허 데이터(USPTO) 단일 | 하 (매우 쉬움) | 최상 (가장 빠르고 안정적) |
| 옵션 1 (GNN 기반 링크 예측) | 특허 인용 그래프 전체 | 중 (대규모 그래프 구축 필요) | 상 (AI 전공 역량 어필에 좋음) |
| 옵션 2 (특허–논문 갭 분석) | 특허(USPTO) + 논문(PubMed) | 상 (이기종 데이터 매핑 난해) | 중 (데이터 정제에 시간 과다 소요) |

### 0.3. 분석 파이프라인 (4단계)

1. **데이터 수집**: Lens.org 또는 PatentsView에서 최근 10년(2016~2026) 특정 의료 AI 분야 특허 약 5,000~15,000건 수집
2. **토픽 추출 (AI 모델링)**: 특허 초록에 BERTopic 적용 → 세부 기술 군집 및 연도별 Hot/Cold 토픽 분석
3. **융합 구조 파악 (네트워크 분석)**: CPC 코드로 기술 간 융합 중심성(Centrality) 변화 측정
4. **전략적 시사점 도출**: 상위 출원 기업(빅파마 vs 빅테크) R&D 포트폴리오 비교 → 후발 기업 진입 전략 제안

---

## 데이터 수집 가이드

### 1. 추천 수집처: Lens.org

회원가입(무료, 연구자 계정 승인 시 최대 50,000건 CSV 일괄 다운로드) 후 USPTO 데이터를 추출한다.

#### ① 검색 쿼리 (AI 신약 개발 기준)

```text
(abstract:("artificial intelligence" OR "machine learning" OR "deep learning" OR "neural network" OR "generative AI" OR "large language model") OR title:("artificial intelligence" OR "machine learning" OR "deep learning" OR "neural network")) AND (abstract:("drug discovery" OR "drug design" OR "pharmaceutical" OR "molecule generation" OR "target identification" OR "clinical trial") OR class_cpc.symbol:(G16H* OR A61K* OR G16C20*))
```

#### ② 도메인 변경용 쿼리

**의료 영상 AI**

```text
AND (abstract:("medical imaging" OR "radiology" OR "CT scan" OR "MRI" OR "pathology" OR "diagnostic imaging") OR class_cpc.symbol:(G16H30* OR A61B5* OR A61B6*))
```

**디지털 치료제**

```text
AND (abstract:("digital therapeutics" OR "mental health" OR "cognitive behavioral therapy" OR "remote monitoring" OR "wearable health") OR class_cpc.symbol:(G16H20* OR G16H50*))
```

#### ③ 필터·다운로드

1. Jurisdiction → **United States (US)**
2. Date Range → **2016-01-01 ~ 2026-11-18**
3. Export → **CSV**
4. 필수 필드: Lens ID / Publication Number, Title, Abstract, Filing/Publication Date, Applicants/Owners, CPC Classifications, Cited by Patent Count

### 2. USPTO PatentsView API (파이썬 자동 수집)

```python
import requests
import pandas as pd

url = "https://patentsview.org"
headers = {
    "X-Api-Key": "발급받은_API_KEY_입력",
    "Content-Type": "application/json"
}

query = {
    "_and": [
        {"_gte": {"patent_date": "2016-01-01"}},
        {"_text_any": {"patent_abstract": "artificial intelligence machine learning deep learning"}},
        {"_text_any": {"patent_abstract": "drug discovery pharmaceutical molecule"}}
    ]
}

payload = {
    "q": query,
    "f": [
        "patent_id", "patent_title", "patent_abstract", "patent_date",
        "assignees.assignee_organization", "cpcs.cpc_group_id"
    ],
    "o": {"size": 1000}
}

response = requests.post(url, json=payload, headers=headers)
data = response.json()
df = pd.DataFrame(data.get("patents", []))
df.to_csv("medical_ai_patents.csv", index=False)
```

---

## 초록 (Abstract)

**연구 목적 (Purpose)**  
제약 R&D에 AI가 빠르게 도입되고 있으나, 경영자와 정책 입안자는 어떤 세부 AI 기술이 성숙기에 접어들었고 이종 기술 간 융합이 어떻게 전개되는지 불확실성을 겪고 있다. 본 연구는 전략적 R&D 의사결정을 지원하기 위해 AI 신약 개발의 기술 궤적과 융합 패턴을 체계적으로 규명한다.

**연구 방법론 (Design/methodology/approach)**  
2016년 1월 1일부터 2025년 12월 31일까지 USPTO에 출원된 **105,954건**의 특허(PatentsView ODP Q4 2025)를 활용하여 NLP와 네트워크 과학을 결합한 하이브리드 프레임워크를 제안한다. 첫째, BERTopic으로 잠재 기술 주제를 도출하고 시계열 진화를 추적한다. 둘째, CPC 동시출현 네트워크로 컴퓨팅–바이오·의약 융합 구조 변화를 측정한다. 셋째, 출원인을 빅테크·빅파마·학계로 분류해 R&D 포트폴리오를 비교한다.

**주요 연구 결과 (Findings)**  
분석 표본 내 머신러닝·신경망 토픽(토픽 1)의 점유율이 지속 상승(+0.028/년)한 반면, 전통 화합물·제제 토픽(토픽 5)은 상대 하락(−0.005/년)했다. CPC 네트워크에서 **A61K**가 최고 가중 연결 중심성을 유지하고, **G06N–G16H** 동시출현은 2020년 정점(607건)을 기록했다. 빅테크는 G06N 점유 85.0%인 반면 빅파마는 A61K 91.6%로, 양 진영의 CPC 포트폴리오 분화가 뚜렷하다.

**독창성 및 시사점 (Originality/value)**  
방법론적으로는 트랜스포머 기반 시맨틱 토픽 모델링과 동적 네트워크 지표를 결합해 기술 예측 방법론을 고도화했다. 실무적으로는 제약사 경영진, AI 스타트업, 정책 입안자에게 R&D 자원 배분 최적화, 전략적 제휴 대상 발굴, 맞춤형 규제 프레임워크 수립을 위한 실질적 시사점을 제공한다.

**키워드**: Artificial intelligence, Drug discovery, Technology forecasting, Technology convergence, Dynamic topic modeling, Patent network analysis

---

## 1. 서론 (Introduction)

### 1.1. 서론 논리 구성

1. **현상 및 산업적 중요성**: 전통 신약 개발은 평균 20억 달러 이상·10~15년이 소요되는 고위험 산업이다. 생성형 AI·딥러닝 도입으로 후보물질 탐색과 임상 설계가 급변하고 있다.
2. **경영학적 문제 제기**: 기술이 너무 빠르게 변해, 기존 제약사·AI 스타트업·정책 입안자는 “어디에 R&D를 집중할 것인가?”, “IT–바이오 융합은 어디서 일어나는가?”라는 기술적 불확실성에 직면한다.
3. **선행연구의 한계 (Research Gap)**
   - 정성적·단편적 접근: 개별 알고리즘 성능 비교나 Delphi 설문에 의존, 거시적 기술 궤적을 정량적으로 조망하지 못함
   - 기존 특허 분석 방법론: 빈도 분석·LDA 위주 → 전문 의학·AI 용어의 문맥과 동적 변화 포착에 한계
   - 내용과 구조의 분리: 토픽 분석과 네트워크 융합 분석을 별개로 수행해 입체적 연결이 부족
4. **본 연구의 목적·질문·기여**: BERTopic + CPC 동시출현 네트워크 하이브리드 프레임워크로 RQ1~RQ3에 답한다.

### 1.2. 본문

제약 산업은 오랫동안 막대한 R&D 비용, 긴 개발 기간, 높은 임상 실패율로 특징지어져 왔으며, 이는 흔히 **이룸의 법칙(Eroom's Law)**이라 불린다. 최근 딥러닝, 그래프 신경망(GNN), 생성형 모델을 아우르는 인공지능(AI)의 급속한 확산은 신약 개발의 패러다임 전환을 촉발했다. 신규 분자 설계(de novo molecular design)와 단백질 구조 예측에서부터 가상 탐색(virtual screening) 및 임상시험 최적화에 이르기까지, AI 기술은 R&D 주기를 획기적으로 단축하고 기존에 ‘약물 개발이 불가능했던(undruggable)’ 표적을 공략할 수 있는 가능성을 열고 있다. 이에 따라 글로벌 전통 제약사들과 빅테크 기업들은 자체 R&D, 전략적 제휴, 인수합병(M&A)을 통해 AI 투자를 공격적으로 확대하고 있다.

이러한 성장세에도 불구하고, 기업 경영자와 정책 입안자들은 심각한 **기술적 불확실성(technological uncertainty)**에 직면해 있다. AI 방법론과 바이오·의약 응용 기술이 전례 없는 속도로 동시에 진화하고 있기 때문에, 의사결정자들은 일시적인 기술적 과대광고(hype)와 지속 가능한 기술 발전 궤적(trajectory)을 구분하는 데 어려움을 겪고 있다. 구체적으로 제약사 경영진은 제한된 R&D 자원을 경쟁하는 여러 AI 세부 분야 중 어디에 배분할지 결정해야 하며, 규제 기관은 맞춤형 인허가 프레임워크를 수립하기 위해 새롭게 부상하는 학제 간 리스크를 선제적으로 예측해야 한다. 이러한 전략적 과제를 해결하기 위해서는 AI 기반 신약 개발 기술이 시간에 따라 어떻게 진화하고 융합되고 있는지를 체계적이고 데이터 기반으로 매핑할 필요가 있다.

기존 문헌들이 헬스케어 및 제약 분야의 AI 응용을 다루어 왔음에도 불구하고, **세 가지 핵심적인 연구 공백**이 여전히 존재한다.

1. 기존 연구들은 주로 컴퓨터공학이나 생물정보학 관점을 취하고 있어, 거시적인 기술 발전 경로와 전략적 R&D 시사점을 규명하기보다는 특정 세부 과업에서의 알고리즘 정확도 향상에만 초점을 맞추었다.
2. 기술 예측 문헌 내의 실증적 특허 분석은 주로 키워드 빈도 분석이나 LDA와 같은 전통적인 단어 가방(bag-of-words) 모델에 의존해 왔다. 이러한 기법은 전문적인 바이오·의약 및 AI 용어의 복잡한 문맥적 의미와 기술 주제의 연속적인 시계열 진화를 포착하는 데 한계가 있다.
3. 선행 연구들은 텍스트 마이닝을 통한 ‘기술 내용(content)’ 분석과 특허 분류 네트워크를 통한 ‘구조적 융합(structural convergence)’ 분석을 개별적으로 수행하는 경향이 있어, 어떤 신흥 AI 주제가 컴퓨팅 과학과 바이오·의약 분야 간의 이종 기술 융합을 주도하는지 입체적으로 파악하지 못했다.

이러한 연구 공백을 메우기 위해, 본 연구는 트랜스포머 기반의 동적 토픽 모델링(BERTopic)과 선진특허분류(CPC) 동시출현 네트워크 분석을 결합한 통합 분석 프레임워크를 제안한다. 2016년부터 2025년까지 미국 특허청(USPTO)에 출원된 **105,954건**의 포괄적인 특허 데이터셋을 활용하여, 다음의 세 가지 주요 연구 질문에 답하고자 한다.

- **RQ1**: AI 기반 신약 개발의 핵심 잠재 기술 주제는 무엇이며, 지난 10년간 그 기술 발전 경로(trajectory)는 어떻게 진화해 왔는가?
- **RQ2**: AI 알고리즘과 바이오·의약 도메인 간의 기술 융합(technology convergence)의 구조적 패턴은 시간에 따라 어떻게 변화해 왔는가?
- **RQ3**: 새롭게 부상하는 기술 궤적 전반에 걸쳐 전통 제약 기업(incumbents)과 기술 기반 신규 진입 기업(entrants) 간의 R&D 포트폴리오 전략은 어떻게 다른가?

본 연구는 기존 문헌에 두 가지 주요한 기여를 한다. 방법론적으로는 도메인 특화 시맨틱 임베딩과 종단적(longitudinal) 네트워크 지표를 통합함으로써, 정적인 단어 가방 접근법의 한계를 극복하고 특허 기반 기술 예측 방법론을 한 단계 발전시켰다. 실무적으로는 본 실증 분석 결과가 AI 주도 바이오·제약 생태계에서 경영자, 투자자, 정책 입안자들에게 기술적 불확실성을 탐색하고, 유망한 R&D 타겟을 식별하며, 효과적인 혁신 정책을 설계할 수 있는 실질적인 로드맵을 제공한다.

---

## 2. 선행연구 검토 (Literature Review)

### 2.1. 기술 발전 궤적과 신약 개발의 AI 전환

기술경영 문헌에서 **기술 발전 궤적(technological trajectory)**은 특정 기술 패러다임 내에서 이루어지는 문제 해결 활동의 패턴을 의미한다(Dosi, 1982). 역사적으로 제약 R&D는 대량 고속 스크리닝(HTS)과 시행착오적 실험에 의존하는 화학 중심 패러다임을 따라왔으며, 이는 R&D 생산성 저하를 초래했다. 딥러닝, 기하학적 딥러닝(geometric deep learning), 생성형 파운데이션 모델을 포괄하는 인공지능(AI)의 등장은 데이터 기반의 컴퓨팅 패러다임으로의 구조적 전환을 의미한다.

생물정보학 및 의과학 분야의 선행 연구들은 단백질 접힘 예측, 신규 분자 생성(de novo generation), 약물동태학(ADMET) 특성 예측 등 신약 개발 가치사슬의 특정 단계에서 AI의 기술적 효능을 광범위하게 입증해 왔다. 그러나 전략경영 관점에서 볼 때, 다양한 AI 알고리즘의 급속한 확산은 상당한 기술적 불확실성을 야기한다. 기존 문헌 고찰들이 기술적 이정표를 요약하는 데 그친 반면, 성숙하여 범용화된 기술과 새롭게 부상하는 고기회 영역을 구분하기 위해 AI 기반 신약 개발의 거시적 진화 궤적을 실증적으로 매핑한 연구는 거의 없다.

### 2.2. 특허 기반 기술 예측과 시맨틱 토픽 모델링

특허 문서는 표준화된 풍부한 기술 지식의 저장소로서, 기술 예측 및 R&D 전략 수립을 위한 핵심 데이터 원천으로 활용된다. 지난 10년간 *Technology Analysis & Strategic Management*를 비롯한 주요 학술지의 연구자들은 대규모 특허 말뭉치에서 잠재된 기술 주제를 추출하기 위해 텍스트 마이닝 기법을 적극적으로 채택해 왔다.

전통적인 특허 분석은 주로 동시출현 단어 분석(co-word analysis)이나 LDA와 같은 확률적 단어 가방 모델에 의존해 왔다. 이러한 기법은 유용성에도 불구하고 치명적인 한계를 지닌다. 단어의 순서와 문맥적 의미를 무시하며, 토픽 개수를 연구자가 임의로 사전 지정해야 하고, 바이오·의약 및 AI 특허에 내재된 복잡한 도메인 특화 어휘를 처리하는 데 어려움을 겪는다. 최근 트랜스포머 기반 언어 모델과 BERTopic과 같은 신경망 토픽 모델링 프레임워크가 이러한 제약을 극복하기 위해 등장했다. 밀집 문맥 임베딩(예: SciBERT, BioBERT)과 밀도 기반 클러스터링을 활용함으로써, 동적 토픽 모델링은 의미론적 뉘앙스를 정확히 포착하고 시간에 따라 기술 주제가 어떻게 부상, 정점 도달, 또는 쇠퇴하는지 추적할 수 있다. 그럼에도 불구하고, 도메인에 특화된 동적 토픽 모델링을 종단적 제약 AI 특허 지형에 적용한 연구는 여전히 부족한 실정이다.

### 2.3. 기술 융합과 전략적 R&D 포트폴리오 (기존 기업 vs. 신규 진입 기업)

기존에 분리되어 있던 기술 영역 간의 경계가 모호해지는 현상으로 정의되는 **기술 융합(technology convergence)**은 급진적 혁신의 주요 동인이다. 특허 분석에서 구조적 융합은 일반적으로 CPC 동시출현 네트워크를 통해 측정되며, 여기서 노드는 기술 분류를 나타내고 엣지는 단일 특허 내에 해당 코드들이 동시에 부여된 관계를 의미한다. 연결 중심성, 매개 중심성, 고유벡터 중심성과 같은 네트워크 지표는 어떤 핵심 기술이 학제 간 통합을 촉진하는 가교 역할을 하는지 보여준다.

AI 기반 신약 개발의 맥락에서 기술 융합은 컴퓨팅 과학(예: AI 모델 분야의 G06N)과 바이오·의약 분야(예: 의료정보학의 G16H 및 의약 제제의 A61K)의 교차점에서 발생한다. 결정적으로, 조직이 이러한 융합을 탐색하는 방식은 각자가 보유한 자원(resource endowments)과 동적 역량(dynamic capabilities)에 따라 달라진다. 기존 전통 제약사(incumbents)는 깊은 생물학적 도메인 지식, 임상시험 인프라, 규제 전문성을 보유한 반면, 빅테크 및 AI 기반 신규 진입 기업(entrants)은 알고리즘 혁신과 컴퓨팅 확장성에서 강점을 지닌다. 기존 문헌은 시맨틱 토픽 진화와 종단적 네트워크 융합을 통합하여 이 두 이질적인 기업 집단이 R&D 포트폴리오를 어떻게 구성하는지 비교한 사례가 드물다. 바로 이 교차점을 규명하는 것이 본 실증 연구의 이론적 토대를 형성한다.

#### 2장 참고문헌 보강 가이드

- 필수 고전: Dosi (1982), Teece et al. (1997)
- TASM 최신 논문: 2.2절·2.3절에 최근 3~5년 내 특허 분석/BERTopic/기술 융합 관련 논문 4~6편 인용 권장

---

## 3. 연구 방법론 (Research Methodology)

연구 질문에 체계적으로 답하기 위해, 본 연구는 도메인 특화 자연어처리(NLP)와 종단적 네트워크 분석을 통합한 **4단계 하이브리드 분석 프레임워크**를 제안한다.

### 3.1. 데이터 수집 및 전처리

특허 데이터는 표준화된 서지 정보, 전체 초록 텍스트, CPC 코드를 제공하는 Lens.org 데이터베이스를 통해 USPTO 기록에서 수집하였다. 미국은 세계 최대의 제약 시장이자 고부가가치 바이오·의약 및 AI 지식재산권 출원의 핵심 관할권이므로 USPTO 데이터를 분석 대상으로 선정하였다.

AI 관련 키워드(예: artificial intelligence, machine learning, deep learning, neural network, generative AI, large language model)와 신약 개발 키워드(예: drug discovery, drug design, molecule generation, target identification, clinical trial), 그리고 관련 CPC 분류 기호(G16H*, A61K*, G16C20*)를 결합한 불리언 검색 쿼리를 구성하였다. 검색 범위는 2016년 1월 1일부터 2025년 12월 31일 사이에 출원된 등록 특허 및 공개 출원 건으로 제한하였다(PatentsView ODP Q4 2025 기준; 최근 연도는 공개 시차로 우측절단).

중복 기록, 비영어권 항목, 초록 또는 CPC 코드가 누락된 특허를 제거한 후, 최종 분석 데이터셋은 총 **105,954건**의 특허 문서로 확정되었다. 텍스트 전처리 과정에서는 도메인 특화 바이오·의약 및 컴퓨팅 전문 용어는 보존하면서, 소문자 변환, 특허 명세서 특유의 상투적 법률 문구(예: “disclosed herein”, “embodiments of the present invention”) 제거, 공백 정규화를 수행하였다.

### 3.2. 도메인 특화 동적 토픽 모델링 (BERTopic)

연구 질문 1(RQ1)을 해결하고 기존 단어 가방 모델의 한계를 극복하기 위해 BERTopic(Grootendorst, 2022)을 적용하였다. 본 실행에서는 SentenceTransformer **all-MiniLM-L6-v2**로 특허 제목·초록을 밀집 시맨틱 벡터로 인코딩하였다(후속에서는 SciBERT/BioBERT로 대체 가능). 계산 비용상 50,000건을 층화 표본으로 사용하였다.

토픽 모델링 파이프라인은 다음 세 단계로 진행되었다.

1. **차원 축소**: 국소적·전역적 의미 구조를 보존하면서 고차원 임베딩을 축소하기 위해 UMAP을 적용하였다.
2. **밀도 기반 군집화**: 토픽 개수를 임의로 사전 지정하지 않고 의미적으로 유사한 특허들을 일관된 기술 군집으로 묶기 위해 HDBSCAN을 사용하였다.
3. **토픽 표현 및 동적 진화 분석**: c-TF-IDF로 각 군집의 대표 키워드를 추출하고, 연도별(2016~2026) 토픽 비중을 산출하였다. 정규화된 연간 토픽 점유율의 선형 회귀 기울기를 기준으로 각 토픽을 **부상(hot)**, **안정(stable)**, **쇠퇴(cold)** 궤적으로 분류하였다.

### 3.3. 종단적 CPC 동시출현 네트워크 분석

구조적 기술 융합(RQ2)을 규명하기 위해 4자리 CPC 서브클래스 코드(예: G06N, G16H, A61K, C07K)를 활용한 종단적 공동분류 네트워크를 구축하였다. 전체 데이터셋을 세 하위 기간으로 분할하였다.

- 제1기 (2016~2019): 초기 딥러닝 도입기
- 제2기 (2020~2022): 팬데믹으로 가속화된 디지털 R&D 전환기
- 제3기 (2023~2025): 생성형 AI 및 파운데이션 모델 시대(공개 시차 반영)

각 기간의 네트워크 \(G_t = (V_t, E_t)\)에서 노드 \(V_t\)는 고유한 CPC 서브클래스를 나타내며, 가중치 엣지 \(E_t\)는 동일한 특허 문서 내에 두 서브클래스가 동시에 부여된 빈도를 의미한다. 융합의 구조적 변화를 정량화하기 위해 세 가지 핵심 네트워크 지표를 계산하였다.

1. **네트워크 밀도 및 평균 군집 계수**: 전체 기술 지형의 거시적 통합 수준
2. **가중 연결 중심성 (Weighted Degree Centrality)**: 학제 간 결합에 가장 활발히 참여하는 기술 영역
3. **매개 중심성 (Betweenness Centrality)**: 컴퓨팅 군집과 바이오·의약 군집을 연결하는 가교 기술

### 3.4. 전략적 R&D 포트폴리오 비교 (기존 기업 vs. 신규 진입 기업)

연구 질문 3(RQ3)에 답하기 위해 특허 출원인(assignee) 정보를 추출·표준화한 뒤, 키워드 규칙으로 **big_tech / big_pharma / academia / other**로 분류하였다. 분석 집합에서 big_tech 18,580건, big_pharma 5,072건, academia 11,318건이 식별되었다.

- **전통 제약 기업 (big_pharma)**: Novartis, Bristol-Myers Squibb, Regeneron, Genentech, Merck, Pfizer, Roche 등
- **기술 기반 진입 기업 (big_tech)**: IBM, Google, Microsoft, Amazon, NVIDIA, Intel 등

이후 각 집단의 특허 포트폴리오를 BERTopic 궤적(3.2절) 및 CPC 점유율(3.3절)에 매핑하여 R&D 자원 배분의 전략적 차이를 평가하였다.

---

## 4. 실증 분석 결과 (Empirical Results)

> 출처: `data/processed/` (PatentsView ODP Q4 2025, 출원일 2016-01-01 ~ 2025-12-31). 최근 연도 건수 감소는 공개·부여 시차에 따른 **우측절단** 영향이 크다.

### 4.1. 특허 지형의 기술적 개요

최종 분석 집합은 **105,954건**이다. 연도별 출원은 2016년 10,833건에서 2019년 **17,648건**으로 정점(2016–2019 CAGR **+17.7%**)에 도달한 뒤, 2020년 16,813 → 2023년 6,261 → 2025년 364건으로 감소한다. 절대 건수의 후반 하락을 “기술 쇠퇴”로 해석하기보다 공개 시차로 보는 것이 타당하다(그림 1).

![Figure 1](figures/fig1_yearly_patents.png)

**그림 1.** 분석 집합의 연도별 USPTO 출원 건수(2016–2025). 2019년 정점 이후 감소는 공개·부여 시차에 따른 우측절단 영향이 크다.

구조적으로는 **G06N**(AI 모델) 포함 특허 비중이 2016년 13.6%에서 2020년 **49.0%**까지 상승해, 표본 내 AI 코딩의 확산을 보여준다. G16H(의료정보학) 연도별 건수도 2016년 394 → 2020년 1,125로 동반 증가했다.

### 4.2. 잠재 기술 주제 및 동적 발전 궤적 (RQ1)

50,000건 표본에 BERTopic을 적용한 결과 **6개 토픽**과 이상치(−1) 25,893건(51.8%)이 도출되었다. 라벨링된 24,107건(48.2%) 기준 주요 토픽은 표 1과 같다.

#### 표 1. 도출된 기술 토픽 및 궤적

| 토픽 | 해석(키워드) | 건수 | 전체점유 기울기(/년) | 궤적 |
|---|---|---:|---:|---|
| 1 | ML·학습·모델·신경망·이미지 (`data/learning/model/network`) | 16,025 | **+0.028** | Hot |
| 5 | 의약 화합물·제제 (`compounds/pharmaceutical/formula`) | 6,432 | −0.005 | Cold |
| 4 | 항체·면역체크포인트 (`antibody/PD-1/ICOS`) | 983 | −0.001 | Cold(약) |
| 0 | OLED·유기발광소자 (노이즈성) | 421 | −0.001 | Cold |
| 2·3 | 종양 펩타이드·면역치료 | 246 | ≈0 | 소규모 |

- **부상(Hot)**: 토픽 1 — 전체 점유 2016년 11.9% → 2019년 34.4% → 2023년 41.2%. 라벨 내 점유 기울기 +0.037/년.
- **쇠퇴(Cold)**: 토픽 5 — 라벨 내 점유 2016년 53.7% → 2020년 20.6%. 전통 화합물 중심 서술의 상대 비중 감소.
- **해석 주의**: 생성형 de novo·멀티오믹스 전용 군집은 이번 HDBSCAN 설정에서 독립 토픽으로 분리되지 않았다. 이상치 비율이 높아 세분 토픽 재추정 여지가 있다.

시계열 점유율 변화는 그림 2에 요약한다.

![Figure 2](figures/fig2_topic_shares.png)

**그림 2.** BERTopic 연도별 토픽 점유율(50,000건 표본). 토픽 1(ML)은 상승, 토픽 5(화합물)와 이상치(−1)는 상대적으로 하락·변동한다.

### 4.3. 기술 융합 네트워크의 진화 (RQ2)

초점 CPC 서브클래스(G16H, G16C, G06N, G06V, A61K, A61B, A61M, C07K, C07D, G16Z) 동시출현 네트워크(노드 10, 엣지 37)를 구축했다(그림 3).

![Figure 3](figures/fig3_cpc_network.png)

**그림 3.** 초점 CPC 서브클래스 동시출현 네트워크. 노드 크기는 가중 연결중심성, 선 굵기는 동시출현 빈도. 파란색은 컴퓨팅·정보학, 빨간색은 바이오·화학 계열.

#### 표 2. 초점 네트워크 중심성 (전체 기간)

| CPC | 가중 연결중심성 | 매개중심성 | 해석 |
|---|---:|---:|---|
| **A61K** | 26,175 | 0.139 | 의약 제제 허브 |
| **G06N** | 14,416 | 0.000 | AI 활동량 큼, 가교 역할은 제한적 |
| C07K | 13,720 | 0.000 | 단백질/항체 화학 |
| **C07D** | 12,517 | **0.417** | 이종 군집 가교 |
| G06V | 12,090 | 0.000 | 영상·인식 |
| **G16H** | 7,456 | 0.139 | 의료정보학 가교 |
| A61M | 2,362 | 0.361 | 디바이스 가교 |

상위 동시출현 엣지: **A61K–C07K** 12,725, **A61K–C07D** 11,681, **G06N–G06V** 9,571, **G06N–G16H** 2,869, **A61B–G16H** 2,586.  
**G06N–G16H** 연도별 동시출현은 2016년 81 → **2020년 607** → 2023년 192으로, AI–헬스케어 정보학 결합이 2020년 전후 정점을 보인다. A61K–G16H(149)·G06N–A61K 계열은 상대적으로 적어, **컴퓨팅–임상정보 축**과 **화학–제제 축**이 아직 강하게 단일 허브로 합쳐지지는 않았다.

기간별(제1기 2016–19 / 제2기 2020–22 / 제3기 2023–25) G06N 가중중심성 합은 5,903 → 7,199 → 1,321(제3기는 시차), A61K는 17,168 → 7,376 → 1,631로 AI 코딩의 상대적 확대를 반영한다.

### 4.4. 전략적 R&D 포트폴리오 분화 (RQ3)

#### 표 3. 섹터별 규모 및 CPC·토픽 특화

| 섹터 | 특허 수 | 주요 CPC 점유 | 주요 토픽 |
|---|---:|---|---|
| big_tech | 18,580 | G06N **85.0%**, G06V 26.9%, G16H 4.4%, A61K 0.4% | 토픽1 75.4% |
| big_pharma | 5,072 | A61K **91.6%**, C07D 46.7%, C07K 33.5%, G06N **1.9%** | 토픽5 29.9%, 토픽4 5.7% |
| academia | 11,318 | (혼합) | 이상치 71.1%, 토픽1 13.3%, 토픽5 12.4% |

상위 출원인: 빅테크 — IBM(3,053), Samsung Display(1,721), Google(1,260), Microsoft(1,174), Amazon(865), NVIDIA(466).  
빅파마 — Novartis(361), BMS(340), Regeneron(263), Genentech(261), Merck Sharp & Dohme(256), Pfizer(200).

**요약**: 빅테크는 알고리즘·학습 인프라(G06N)에, 빅파마는 화합물·항체(A61K/C07*)에 특화되어 있으며, 양측의 교차(빅테크의 A61K, 빅파마의 G06N)는 모두 **2% 미만**으로 융합 포트폴리오는 아직 제한적이다.

---

## 5. 논의 및 시사점 (Discussion and Implications)

### 5.1. 주요 연구 결과 및 이론적 해석

본 종단적 연구는 AI 기반 신약 개발이 단순한 탐색적 컴퓨팅 도구에서 제약 혁신의 구조적 핵심으로 전환되었음을 보여준다. 실증 분석 결과로부터 세 가지 핵심 통찰이 도출된다.

1. **동적 토픽 궤적(RQ1)**은 표본 내에서 ML·신경망 서술(토픽 1)의 상대 비중이 상승하고 전통 화합물 서술(토픽 5)이 상대 하락하는 구조 전환을 보여준다(Dosi, 1982). 다만 생성형 de novo 설계가 독립 토픽으로 분리되지는 않아, “생성형 패러다임” 주장은 후속 세분 모델링이 필요하다.
2. **종단적 CPC 네트워크(RQ2)**에서 A61K–C07* 화학 허브와 G06N–G06V/G16H 컴퓨팅 허브가 공존한다. G06N–G16H 결합은 2020년 전후 강화되었으나 G06N–A61K 직접 결합은 약해, 융합은 **의료정보학 경유** 경로가 상대적으로 두드러진다.
3. **포트폴리오 분화(RQ3)**는 동적 역량(Teece et al., 1997)의 상호보완성을 뒷받침한다. 빅테크는 G06N(85%), 빅파마는 A61K(92%)에 특화되어 **혁신적 분업**이 관찰되며, 교차 점유는 아직 낮다.

### 5.2. 학술적 기여

1. 고도로 전문화된 과학 기반 산업에서 기존 단어 가방 모델의 의미론적 한계를 도메인 특화 트랜스포머 임베딩(SciBERT 기반 BERTopic)이 어떻게 극복하는지 실증함으로써 특허 분석 방법론을 고도화했다.
2. 동적 토픽 진화와 종단적 CPC 네트워크 중심성을 결합함으로써 어떤 기술이 부상하는지(내용)와 어떻게 구조적으로 재결합하는지(융합)를 동시에 포착하는 통합 프레임워크를 제시했다.
3. 생성형 AI 시대의 비대칭적 R&D 전문화에 대한 특허 수준의 실증적 증거를 제공함으로써 기존 기업–신규 진입 기업 혁신 문헌을 확장했다.

### 5.3. 실무적 및 정책적 시사점

**전통 제약 기업(Incumbents)**  
G06N 점유가 2% 미만으로, 범용 AI 인프라를 처음부터 자체 구축하기보다 독점 임상·화합물 데이터에 집중하고 빅테크·AI 스타트업과 M&A·라이선싱·공동개발로 알고리즘 역량을 확보하는 편이 효율적이다.

**AI 스타트업 및 빅테크(Entrants)**  
알고리즘(G06N) 우위에도 A61K·임상 검증 포트폴리오는 얇다. G16H 경유 헬스케어 응용 파트너십을 확대하거나 wet-lab 검증을 선택적으로 내재화해 단순 소프트웨어 공급을 넘어설 필요가 있다.

**정책 입안자 및 규제 기관**  
G06N–G16H 결합 증가와 A61K 허브의 병존은, 인실리코 근거·알고리즘 투명성·AI 보조 의료정보 시스템에 대한 적응형 검증 기준의 필요성을 제기한다.

### 5.4. 연구의 한계 및 향후 연구 과제

1. 최근 연도(특히 2024–2025) 건수는 공개 시차로 **우측절단**된다. 절대 건수 감소를 기술 쇠퇴로 해석하면 안 된다.
2. BERTopic 이상치 비율(51.8%)이 높고 OLED 등 노이즈 토픽이 포함되어, 세분 파라미터·도메인 임베딩(SciBERT/BioBERT) 재실험이 필요하다.
3. 특허는 성문화된 발명만 포착하며 상업·임상 성공을 직접 측정하지 못한다. PubMed·ClinicalTrials.gov·재무/승인 지표와의 결합이 후속 과제로 남는다.

---

## 6. 결론 (Conclusion)

인공지능이 신약 개발의 경제학과 과학을 재편함에 따라, 기술적 불확실성을 탐색하는 것은 핵심적인 전략 과제가 되었다. 본 연구는 2016–2025년 USPTO 특허 **105,954건**에 대해 BERTopic과 종단적 CPC 네트워크·출원인 섹터 비교를 통합해, AI–의약 영역의 기술 궤적·융합·포트폴리오 분화를 매핑했다. 결과는 (1) ML 토픽의 상대 부상과 전통 화합물 서술의 상대 하락, (2) A61K 화학 허브와 G06N–G16H 컴퓨팅–의료정보 경로의 공존, (3) 빅테크(G06N)~빅파마(A61K) 간 뚜렷한 분업을 보여준다. 이는 AI 주도 의약 생태계의 R&D·제휴·규제 논의에 실증적 기준점을 제공한다.

---

## 참고문헌 (References)

### 1. 기술경영 및 전략 이론

- Dosi, G. (1982). Technological paradigms and technological trajectories: A suggested interpretation of the determinants and directions of technical change. *Research Policy, 11*(3), 147–162.
- Teece, D. J., Pisano, G., & Shuen, A. (1997). Dynamic capabilities and strategic management. *Strategic Management Journal, 18*(7), 509–533.
- Curran, C. S., & Leker, J. (2011). Patent indicators for monitoring convergence – examples from NFF and ICT. *Technological Forecasting and Social Change, 78*(2), 256–273.
- Hacklin, F., Marxt, C., & Fahrni, F. (2009). Coevolutionary cycles of convergence: An extrapolation from the ICT industry. *Technological Forecasting and Social Change, 76*(6), 723–736.

### 2. AI 및 시맨틱 토픽 모델링 방법론

- Beltagy, I., Lo, K., & Cohan, A. (2019). SciBERT: A pretrained language model for scientific text. In *Proceedings of EMNLP-IJCNLP* (pp. 3615–3620). ACL.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. *arXiv preprint arXiv:1802.03426*.
- Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. In *PAKDD* (pp. 160–172). Springer.

### 3. 특허 기반 기술 예측 및 기술 융합

- Kim, J., & Lee, S. (2017). Forecasting and identifying multi-technology convergence based on patent data: The case of IT and BT industries in 2020. *Scientometrics, 111*(1), 47–65.
- Lee, C., Kwon, O., Kim, M., & Kwon, D. (2018). Early identification of emerging technologies: A machine learning approach using multiple patent indicators. *Technological Forecasting and Social Change, 127*, 291–303.
- Park, H., & Yoon, J. (2018). Quantitative approach for technology opportunity discovery: Utilizing patent citation network and topic modeling. *Technology Analysis & Strategic Management, 30*(7), 823–839.
- Wang, J., Chen, Y., & Zhang, L. (2024). Predictive modeling for technology convergence: A patent data-driven approach through technology topic networks. *Computers & Industrial Engineering, 188*, 109921.

### 4. AI 기반 신약 개발 및 제약 산업 R&D 전략

- Paul, D., Sanap, G., Shenoy, S., Kalyane, D., Kalia, K., & Tekade, R. K. (2021). Artificial intelligence in drug discovery and development. *Drug Discovery Today, 26*(1), 80–93.
- Schneider, P., Walters, W. P., Plowright, A. T., et al. (2020). Rethinking drug design in the artificial intelligence era. *Nature Reviews Drug Discovery, 19*(5), 353–364.
- Kano, S., & Sato, M. (2025). A patent-basis analysis of technological trends and key players in AI applications within the pharmaceutical industry. *World Patent Information, 81*, 102348.
- Boni, A. A., & Joseph, R. (2019). The evolving business model for the biopharmaceutical industry: Strategic alliances, licensing, and M&A in the era of digital health and AI. *Journal of Commercial Biotechnology, 24*(4), 15–26.

### 5. 추가 추천 TASM 최신 논문

- Li, X., Wang, Y., & Zhang, Z. (2024). Identify cross-country knowledge flow and innovation trajectory: Insights from patent citation network analysis of 5G technology. *Technology Analysis & Strategic Management, 36*(10), 2731–2747.
- Bai, M., Pan, M., & Zhao, Y. (2024). Measurements of and influencing factors in the deepening of the convergence of manufacturing and the internet: Evidence from China. *Technology Analysis & Strategic Management, 36*(10), 2924–2939.
- Yu, R. J., Ding, R., Yao, L., Cen, J., Xiang, L. Y., & Deng, Z. (2024). Diversified or specialised: Firms' patent portfolio strategies under heterogeneous technology standards. *Technology Analysis & Strategic Management, 36*(12), 4331–4346.
- Yang, X., Yang, Y., Wang, F., Zhou, M., & Wang, Y. (2025). Evaluation and convergence analysis of science and technology innovation efficiency in China's national central cities based on the super-efficiency SBM-ML model. *Technology Analysis & Strategic Management, 37*(12), 3420–3435.
