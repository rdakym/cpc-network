"""
키워드 유사도 그래프 시각화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import networkx as nx
from pyvis.network import Network
import warnings

warnings.filterwarnings('ignore')


class SimilarityVisualizer:
    """유사도 시각화 클래스"""

    def __init__(self, keywords, similarity_matrix):
        """
        Args:
            keywords: 키워드 리스트
            similarity_matrix: 유사도 행렬
        """
        self.keywords = keywords
        self.similarity_matrix = similarity_matrix

        # 한글 폰트 설정
        self._setup_korean_font()

    def _setup_korean_font(self):
        """한글 폰트 설정"""
        try:
            # Linux에서 사용 가능한 한글 폰트 찾기
            font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
            korean_fonts = [
                f for f in font_list
                if 'Nanum' in f or 'Malgun' in f or 'Noto' in f or 'Gothic' in f
            ]

            if korean_fonts:
                font_path = korean_fonts[0]
                font_prop = fm.FontProperties(fname=font_path)
                plt.rc('font', family=font_prop.get_name())
            else:
                # 기본 설정
                plt.rc('font', family='DejaVu Sans')

            plt.rc('axes', unicode_minus=False)
        except Exception as e:
            print(f"폰트 설정 경고: {e}")
            plt.rc('font', family='DejaVu Sans')

    def create_network_graph(self, threshold=0.7, output_file='similarity_network.png', figsize=(20, 20)):
        """
        네트워크 그래프 생성

        Args:
            threshold: 엣지를 표시할 최소 유사도
            output_file: 출력 파일명
            figsize: 그림 크기
        """
        print(f"네트워크 그래프 생성 중... (임계값: {threshold})")

        # NetworkX 그래프 생성
        G = nx.Graph()

        # 노드 추가
        for keyword in self.keywords:
            G.add_node(keyword)

        # 엣지 추가 (임계값 이상의 유사도만)
        edge_count = 0
        for i in range(len(self.keywords)):
            for j in range(i + 1, len(self.keywords)):
                sim = self.similarity_matrix[i][j]
                if sim >= threshold:
                    G.add_edge(self.keywords[i], self.keywords[j], weight=sim)
                    edge_count += 1

        print(f"노드 수: {G.number_of_nodes()}, 엣지 수: {edge_count}")

        if edge_count == 0:
            print(f"경고: 임계값 {threshold} 이상의 유사도를 가진 쌍이 없습니다.")
            print("임계값을 낮춰보세요.")
            return

        # 레이아웃 계산
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # 그래프 그리기
        plt.figure(figsize=figsize)

        # 엣지 그리기 (가중치에 따라 두께 조절)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]

        nx.draw_networkx_edges(
            G, pos,
            width=[w * 3 for w in weights],
            alpha=0.3,
            edge_color='gray'
        )

        # 노드 그리기 (연결도에 따라 크기 조절)
        node_sizes = [G.degree(node) * 100 + 300 for node in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color='lightblue',
            alpha=0.7
        )

        # 라벨 그리기
        nx.draw_networkx_labels(
            G, pos,
            font_size=8,
            font_weight='bold'
        )

        plt.title(
            f'철강 제조 기술 키워드 유사도 네트워크\n(유사도 >= {threshold})',
            fontsize=16,
            pad=20
        )
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 네트워크 그래프 저장: {output_file}")
        plt.close()

    def create_interactive_network(self, threshold=0.7, output_file='similarity_network_interactive.html'):
        """
        인터랙티브 네트워크 그래프 생성 (PyVis)

        Args:
            threshold: 엣지를 표시할 최소 유사도
            output_file: 출력 HTML 파일명
        """
        print(f"인터랙티브 네트워크 그래프 생성 중... (임계값: {threshold})")

        # PyVis 네트워크 생성
        net = Network(
            height='900px',
            width='100%',
            bgcolor='#ffffff',
            font_color='black',
            notebook=False
        )

        # 물리 엔진 설정
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 200,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
          }
        }
        """)

        # 노드 추가
        for keyword in self.keywords:
            net.add_node(keyword, label=keyword, title=keyword)

        # 엣지 추가
        edge_count = 0
        for i in range(len(self.keywords)):
            for j in range(i + 1, len(self.keywords)):
                sim = self.similarity_matrix[i][j]
                if sim >= threshold:
                    net.add_edge(
                        self.keywords[i],
                        self.keywords[j],
                        value=float(sim),
                        title=f'유사도: {sim:.3f}'
                    )
                    edge_count += 1

        print(f"노드 수: {len(self.keywords)}, 엣지 수: {edge_count}")

        # HTML 파일로 저장
        net.save_graph(output_file)
        print(f"✓ 인터랙티브 그래프 저장: {output_file}")

    def create_heatmap(self, top_n=50, output_file='similarity_heatmap.png', figsize=(20, 16)):
        """
        유사도 히트맵 생성 (상위 N개 키워드)

        Args:
            top_n: 표시할 상위 키워드 개수
            output_file: 출력 파일명
            figsize: 그림 크기
        """
        print(f"히트맵 생성 중... (상위 {top_n}개 키워드)")

        # 각 키워드의 평균 유사도 계산
        avg_similarities = []
        for i in range(len(self.keywords)):
            # 자기 자신 제외
            similarities = [self.similarity_matrix[i][j] for j in range(len(self.keywords)) if i != j]
            avg_similarities.append(np.mean(similarities))

        # 상위 N개 키워드 선택
        top_indices = np.argsort(avg_similarities)[-top_n:]
        top_keywords = [self.keywords[i] for i in top_indices]
        top_matrix = self.similarity_matrix[np.ix_(top_indices, top_indices)]

        # 히트맵 생성
        plt.figure(figsize=figsize)
        sns.heatmap(
            top_matrix,
            xticklabels=top_keywords,
            yticklabels=top_keywords,
            cmap='YlOrRd',
            annot=False,
            fmt='.2f',
            square=True,
            cbar_kws={'label': '유사도'}
        )

        plt.title(f'철강 제조 기술 키워드 유사도 히트맵\n(상위 {top_n}개 키워드)', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 히트맵 저장: {output_file}")
        plt.close()

    def create_clustered_heatmap(self, output_file='similarity_heatmap_clustered.png', figsize=(24, 20)):
        """
        클러스터링된 히트맵 생성

        Args:
            output_file: 출력 파일명
            figsize: 그림 크기
        """
        print("클러스터링된 히트맵 생성 중...")

        # 클러스터맵 생성
        sns.clustermap(
            self.similarity_matrix,
            xticklabels=self.keywords,
            yticklabels=self.keywords,
            cmap='YlOrRd',
            figsize=figsize,
            cbar_kws={'label': '유사도'},
            dendrogram_ratio=0.1
        )

        plt.suptitle('철강 제조 기술 키워드 유사도 클러스터맵', y=0.995, fontsize=14)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 클러스터맵 저장: {output_file}")
        plt.close()

    def create_similarity_distribution(self, output_file='similarity_distribution.png'):
        """
        유사도 분포 히스토그램

        Args:
            output_file: 출력 파일명
        """
        print("유사도 분포 히스토그램 생성 중...")

        # 상삼각 행렬에서 유사도 추출 (중복 제거)
        similarities = []
        for i in range(len(self.keywords)):
            for j in range(i + 1, len(self.keywords)):
                similarities.append(self.similarity_matrix[i][j])

        # 히스토그램 생성
        plt.figure(figsize=(12, 6))
        plt.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('유사도', fontsize=12)
        plt.ylabel('빈도', fontsize=12)
        plt.title('키워드 쌍 유사도 분포', fontsize=14)
        plt.grid(True, alpha=0.3)

        # 통계 정보 추가
        mean_sim = np.mean(similarities)
        median_sim = np.median(similarities)
        plt.axvline(mean_sim, color='r', linestyle='--', label=f'평균: {mean_sim:.3f}')
        plt.axvline(median_sim, color='g', linestyle='--', label=f'중앙값: {median_sim:.3f}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 유사도 분포 그래프 저장: {output_file}")
        plt.close()


def main():
    """메인 실행 함수"""

    print("=" * 80)
    print("키워드 유사도 그래프 시각화")
    print("=" * 80)
    print()

    # 데이터 로딩
    print("[1단계] 데이터 로딩")
    print("-" * 80)

    try:
        # 키워드 로딩
        keywords = []
        with open('processed_keywords.txt', 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    keywords.append(parts[1])

        # 유사도 행렬 로딩
        df_similarity = pd.read_csv('keyword_similarity_matrix.csv', index_col=0, encoding='utf-8-sig')
        similarity_matrix = df_similarity.values

        print(f"✓ 키워드 수: {len(keywords)}")
        print(f"✓ 유사도 행렬 크기: {similarity_matrix.shape}")
    except FileNotFoundError as e:
        print(f"오류: 필요한 파일을 찾을 수 없습니다.")
        print("먼저 'keyword_similarity_analysis.py'를 실행해주세요.")
        return

    print()

    # 시각화 객체 생성
    print("[2단계] 시각화 생성")
    print("-" * 80)
    visualizer = SimilarityVisualizer(keywords, similarity_matrix)

    # 1. 유사도 분포
    visualizer.create_similarity_distribution()

    # 2. 네트워크 그래프 (여러 임계값)
    for threshold in [0.8, 0.7, 0.6]:
        visualizer.create_network_graph(
            threshold=threshold,
            output_file=f'similarity_network_{int(threshold*100)}.png'
        )

    # 3. 인터랙티브 네트워크
    visualizer.create_interactive_network(threshold=0.7)

    # 4. 히트맵
    visualizer.create_heatmap(top_n=50)

    # 5. 클러스터맵 (전체 키워드가 많지 않을 경우)
    if len(keywords) <= 100:
        visualizer.create_clustered_heatmap()
    else:
        print("키워드가 너무 많아 클러스터맵 생성을 건너뜁니다.")

    print()
    print("=" * 80)
    print("시각화 완료!")
    print("=" * 80)
    print()
    print("생성된 파일:")
    print("  - similarity_distribution.png: 유사도 분포")
    print("  - similarity_network_XX.png: 네트워크 그래프 (다양한 임계값)")
    print("  - similarity_network_interactive.html: 인터랙티브 그래프")
    print("  - similarity_heatmap.png: 히트맵")
    if len(keywords) <= 100:
        print("  - similarity_heatmap_clustered.png: 클러스터맵")
    print()


if __name__ == "__main__":
    main()
