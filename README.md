# 철강 제조 기술 키워드 유사도 분석

철강 제조 기술 관련 키워드들의 의미적 유사도를 분석하고 시각화하는 프로젝트입니다.

## 📋 프로젝트 개요

이 프로젝트는 다음과 같은 작업을 수행합니다:

1. **키워드 전처리**: 철강 제조 기술 키워드 정제 및 중복 제거
2. **의미 임베딩**: 한국어 특화 Sentence-BERT 모델을 사용한 벡터 임베딩
3. **유사도 계산**: 코사인 유사도를 통한 키워드 간 의미적 유사성 측정
4. **시각화**: 네트워크 그래프, 히트맵, 인터랙티브 그래프 생성

## 🔧 설치 방법

### 1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 시스템 요구사항

- Python 3.8 이상
- 메모리: 최소 4GB RAM (권장 8GB 이상)
- 디스크 공간: 약 2GB (모델 다운로드용)

## 🚀 사용 방법

### 단계 1: 키워드 유사도 분석 실행

```bash
python keyword_similarity_analysis.py
```

이 스크립트는 다음 파일들을 생성합니다:
- `keyword_similarity_matrix.csv`: 전체 유사도 행렬
- `top_similar_pairs.csv`: 상위 유사 키워드 쌍
- `keyword_embeddings.npy`: 임베딩 벡터
- `processed_keywords.txt`: 전처리된 키워드 리스트

### 단계 2: 시각화 생성

```bash
python visualize_similarity.py
```

이 스크립트는 다음 시각화 파일들을 생성합니다:
- `similarity_distribution.png`: 유사도 분포 히스토그램
- `similarity_network_XX.png`: 네트워크 그래프 (다양한 임계값)
- `similarity_network_interactive.html`: 인터랙티브 네트워크 그래프
- `similarity_heatmap.png`: 상위 키워드 히트맵
- `similarity_heatmap_clustered.png`: 클러스터링된 히트맵

## 📊 생성되는 시각화

### 1. 네트워크 그래프
키워드 간의 유사도를 네트워크로 표현합니다.
- 노드: 각 키워드
- 엣지: 유사도가 높은 키워드 간 연결
- 노드 크기: 다른 키워드와의 연결 수
- 엣지 두께: 유사도 정도

### 2. 인터랙티브 그래프
웹브라우저에서 확대/축소 및 드래그가 가능한 인터랙티브 그래프입니다.
```bash
# 브라우저에서 열기
firefox similarity_network_interactive.html
# 또는
google-chrome similarity_network_interactive.html
```

### 3. 히트맵
키워드 간 유사도를 색상으로 표현한 행렬입니다.
- 밝은 색: 높은 유사도
- 어두운 색: 낮은 유사도

### 4. 클러스터맵
계층적 클러스터링을 통해 유사한 키워드들을 그룹화한 히트맵입니다.

## 🔬 기술 스택

- **임베딩 모델**: `jhgan/ko-sroberta-multitask` (한국어 특화 Sentence-BERT)
- **유사도 측정**: Cosine Similarity
- **시각화**:
  - Matplotlib & Seaborn (정적 그래프)
  - NetworkX (네트워크 분석)
  - PyVis (인터랙티브 그래프)

## 📁 파일 구조

```
.
├── steel_keywords.py              # 키워드 데이터
├── keyword_similarity_analysis.py # 분석 메인 스크립트
├── visualize_similarity.py        # 시각화 스크립트
├── requirements.txt               # 패키지 의존성
├── README.md                      # 프로젝트 설명서
│
├── [생성되는 파일들]
├── keyword_similarity_matrix.csv  # 유사도 행렬
├── top_similar_pairs.csv          # 상위 유사 쌍
├── keyword_embeddings.npy         # 임베딩 벡터
├── processed_keywords.txt         # 전처리된 키워드
└── similarity_*.png/html          # 시각화 파일들
```

## ⚙️ 커스터마이징

### 임베딩 모델 변경

`keyword_similarity_analysis.py`에서 모델을 변경할 수 있습니다:

```python
# 기본값 (한국어 특화)
embedder = KeywordEmbedding('jhgan/ko-sroberta-multitask')

# 다국어 모델
embedder = KeywordEmbedding('paraphrase-multilingual-MiniLM-L12-v2')
```

### 시각화 임계값 조정

`visualize_similarity.py`에서 네트워크 그래프의 임계값을 조정할 수 있습니다:

```python
# 더 밀집된 네트워크를 원할 경우
visualizer.create_network_graph(threshold=0.6)

# 핵심 연결만 보고 싶을 경우
visualizer.create_network_graph(threshold=0.85)
```

### 키워드 추가/수정

`steel_keywords.py`의 `STEEL_MANUFACTURING_KEYWORDS` 리스트를 수정하세요.

## 📈 결과 해석

### 유사도 값의 의미
- **0.9 이상**: 매우 높은 유사도 (거의 동일한 의미)
- **0.7 ~ 0.9**: 높은 유사도 (관련 기술/개념)
- **0.5 ~ 0.7**: 중간 유사도 (같은 분야)
- **0.5 미만**: 낮은 유사도 (다른 분야)

### 네트워크 분석
- **중심성이 높은 노드**: 여러 기술과 관련된 핵심 키워드
- **클러스터 형성**: 유사한 기술 영역의 그룹
- **브릿지 노드**: 서로 다른 기술 영역을 연결하는 키워드

## 🐛 트러블슈팅

### 메모리 부족 오류
키워드 수가 많을 경우 메모리 부족이 발생할 수 있습니다:
```python
# 키워드 샘플링으로 해결
keywords_sample = keywords[:100]  # 상위 100개만 사용
```

### 한글 폰트 깨짐
한글이 깨질 경우 시스템에 한글 폰트를 설치하세요:
```bash
# Ubuntu/Debian
sudo apt-get install fonts-nanum

# 또는 사용자 폰트 디렉토리에 복사
mkdir -p ~/.fonts
cp /path/to/korean/font.ttf ~/.fonts/
fc-cache -f -v
```

### CUDA 오류 (GPU 관련)
GPU를 사용할 수 없는 환경에서는 자동으로 CPU를 사용합니다.
속도가 느릴 경우 키워드 수를 줄이거나 GPU 환경에서 실행하세요.

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 사용 가능합니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

## 📧 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.
