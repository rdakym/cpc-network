#!/bin/bash

# 철강 제조 기술 키워드 유사도 분석 실행 스크립트

echo "=========================================="
echo "철강 제조 기술 키워드 유사도 분석"
echo "=========================================="
echo ""

# 가상환경 확인
if [ ! -d "venv" ]; then
    echo "[1단계] 가상환경 생성 중..."
    python3 -m venv venv
    echo "✓ 가상환경 생성 완료"
    echo ""
fi

# 가상환경 활성화
echo "[2단계] 가상환경 활성화..."
source venv/bin/activate
echo "✓ 가상환경 활성화됨"
echo ""

# 패키지 설치
echo "[3단계] 필수 패키지 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ 패키지 설치 완료"
echo ""

# 분석 실행
echo "[4단계] 키워드 유사도 분석 실행 중..."
python keyword_similarity_analysis.py
if [ $? -eq 0 ]; then
    echo "✓ 분석 완료"
else
    echo "✗ 분석 실패"
    exit 1
fi
echo ""

# 시각화 실행
echo "[5단계] 시각화 생성 중..."
python visualize_similarity.py
if [ $? -eq 0 ]; then
    echo "✓ 시각화 완료"
else
    echo "✗ 시각화 실패"
    exit 1
fi
echo ""

echo "=========================================="
echo "모든 작업이 완료되었습니다!"
echo "=========================================="
echo ""
echo "생성된 파일들:"
echo "  분석 결과:"
echo "    - keyword_similarity_matrix.csv"
echo "    - top_similar_pairs.csv"
echo "    - keyword_embeddings.npy"
echo "    - processed_keywords.txt"
echo ""
echo "  시각화:"
echo "    - similarity_distribution.png"
echo "    - similarity_network_*.png"
echo "    - similarity_network_interactive.html"
echo "    - similarity_heatmap.png"
echo "    - similarity_heatmap_clustered.png"
echo ""
echo "인터랙티브 그래프 보기:"
echo "  firefox similarity_network_interactive.html"
echo ""
