"""
철강 제조 기술 키워드 유사도 분석 및 시각화
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from steel_keywords import STEEL_MANUFACTURING_KEYWORDS


class KeywordPreprocessor:
    """키워드 전처리 클래스"""

    @staticmethod
    def clean_keyword(keyword):
        """
        키워드 정리
        - 불필요한 공백 제거
        - 특수문자 정리
        """
        # 앞뒤 공백 제거
        keyword = keyword.strip()

        # 연속된 공백을 하나로
        keyword = re.sub(r'\s+', ' ', keyword)

        return keyword

    @staticmethod
    def preprocess_keywords(keywords):
        """
        키워드 리스트 전처리
        - 중복 제거
        - 정리
        """
        # 정리 및 중복 제거
        cleaned = [KeywordPreprocessor.clean_keyword(k) for k in keywords]
        unique_keywords = list(dict.fromkeys(cleaned))  # 순서 유지하며 중복 제거

        print(f"원본 키워드 수: {len(keywords)}")
        print(f"중복 제거 후: {len(unique_keywords)}")
        print(f"제거된 중복: {len(keywords) - len(unique_keywords)}")

        return unique_keywords


class KeywordEmbedding:
    """키워드 임베딩 클래스"""

    def __init__(self, model_name='jhgan/ko-sroberta-multitask'):
        """
        한국어 문장 임베딩 모델 초기화

        Args:
            model_name: 사용할 모델 (기본값: jhgan/ko-sroberta-multitask - 한국어 특화)
                      대안: 'paraphrase-multilingual-MiniLM-L12-v2' (다국어 지원)
        """
        print(f"임베딩 모델 로딩 중: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("모델 로딩 완료!")

    def encode_keywords(self, keywords):
        """
        키워드를 벡터로 인코딩

        Args:
            keywords: 키워드 리스트

        Returns:
            numpy array: 임베딩 벡터 (shape: [n_keywords, embedding_dim])
        """
        print(f"{len(keywords)}개 키워드 인코딩 중...")
        embeddings = self.model.encode(keywords, show_progress_bar=True)
        print(f"임베딩 완료! Shape: {embeddings.shape}")
        return embeddings


class SimilarityAnalyzer:
    """유사도 분석 클래스"""

    @staticmethod
    def calculate_similarity(embeddings):
        """
        코사인 유사도 계산

        Args:
            embeddings: 임베딩 벡터

        Returns:
            numpy array: 유사도 행렬
        """
        print("유사도 계산 중...")
        similarity_matrix = cosine_similarity(embeddings)
        print(f"유사도 행렬 Shape: {similarity_matrix.shape}")
        return similarity_matrix

    @staticmethod
    def get_top_similar_pairs(keywords, similarity_matrix, top_n=50, threshold=0.5):
        """
        가장 유사한 키워드 쌍 추출

        Args:
            keywords: 키워드 리스트
            similarity_matrix: 유사도 행렬
            top_n: 반환할 상위 n개 쌍
            threshold: 최소 유사도 임계값

        Returns:
            list: [(keyword1, keyword2, similarity), ...]
        """
        pairs = []
        n = len(keywords)

        for i in range(n):
            for j in range(i + 1, n):
                sim = similarity_matrix[i][j]
                if sim >= threshold:
                    pairs.append((keywords[i], keywords[j], sim))

        # 유사도 기준 내림차순 정렬
        pairs.sort(key=lambda x: x[2], reverse=True)

        return pairs[:top_n]

    @staticmethod
    def create_similarity_dataframe(keywords, similarity_matrix):
        """
        유사도 행렬을 데이터프레임으로 변환

        Args:
            keywords: 키워드 리스트
            similarity_matrix: 유사도 행렬

        Returns:
            pd.DataFrame: 유사도 데이터프레임
        """
        return pd.DataFrame(
            similarity_matrix,
            index=keywords,
            columns=keywords
        )


def main():
    """메인 실행 함수"""

    print("=" * 80)
    print("철강 제조 기술 키워드 유사도 분석")
    print("=" * 80)
    print()

    # 1. 키워드 전처리
    print("[1단계] 키워드 전처리")
    print("-" * 80)
    preprocessor = KeywordPreprocessor()
    keywords = preprocessor.preprocess_keywords(STEEL_MANUFACTURING_KEYWORDS)
    print()

    # 2. 임베딩 생성
    print("[2단계] 키워드 임베딩")
    print("-" * 80)
    embedder = KeywordEmbedding()
    embeddings = embedder.encode_keywords(keywords)
    print()

    # 3. 유사도 계산
    print("[3단계] 유사도 분석")
    print("-" * 80)
    analyzer = SimilarityAnalyzer()
    similarity_matrix = analyzer.calculate_similarity(embeddings)
    print()

    # 4. 결과 저장
    print("[4단계] 결과 저장")
    print("-" * 80)

    # 유사도 행렬 저장
    df_similarity = analyzer.create_similarity_dataframe(keywords, similarity_matrix)
    df_similarity.to_csv('keyword_similarity_matrix.csv', encoding='utf-8-sig')
    print("✓ 유사도 행렬 저장: keyword_similarity_matrix.csv")

    # 상위 유사 키워드 쌍 저장
    top_pairs = analyzer.get_top_similar_pairs(keywords, similarity_matrix, top_n=100, threshold=0.5)
    df_pairs = pd.DataFrame(top_pairs, columns=['Keyword1', 'Keyword2', 'Similarity'])
    df_pairs.to_csv('top_similar_pairs.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 상위 유사 키워드 쌍 저장: top_similar_pairs.csv ({len(top_pairs)}개)")

    # 임베딩 벡터 저장
    np.save('keyword_embeddings.npy', embeddings)
    print("✓ 임베딩 벡터 저장: keyword_embeddings.npy")

    # 키워드 리스트 저장
    with open('processed_keywords.txt', 'w', encoding='utf-8') as f:
        for i, keyword in enumerate(keywords):
            f.write(f"{i}\t{keyword}\n")
    print("✓ 처리된 키워드 리스트 저장: processed_keywords.txt")

    print()
    print("=" * 80)
    print("분석 완료!")
    print("=" * 80)
    print()

    # 상위 10개 유사 쌍 출력
    print("상위 10개 유사 키워드 쌍:")
    print("-" * 80)
    for i, (kw1, kw2, sim) in enumerate(top_pairs[:10], 1):
        print(f"{i:2d}. [{sim:.4f}] {kw1} ↔ {kw2}")
    print()

    return keywords, embeddings, similarity_matrix


if __name__ == "__main__":
    keywords, embeddings, similarity_matrix = main()
