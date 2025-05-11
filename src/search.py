import os
import pickle  # 추가
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import BASE_PATH

def search_files(query):
    metadata_path = os.path.join(BASE_PATH, 'metadata.pkl')
    index_path = os.path.join(BASE_PATH, 'index.faiss')

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # 리스트 → DataFrame 변환 [핵심 수정]
    if isinstance(metadata, list):
        metadata = pd.DataFrame(metadata)
    elif not isinstance(metadata, pd.DataFrame):
        raise ValueError("Invalid metadata format")

    # FAISS 인덱스 로드
    index = faiss.read_index(index_path)

    # 모델 로드 및 쿼리 임베딩 생성
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query]).astype("float32").reshape(1, -1)

    # 유사도 검색 (k=10 → 상위 10개 결과)
    num_docs = len(metadata)
    top_k = min(10, num_docs)  # 실제 문서 수보다 많이 요청하지 않음
    D, I = index.search(query_vec, k=top_k)

    # 거리를 유사도 점수로 변환 (1 - L2 거리)
    similarities = 1 - D[0]

    # 결과 정렬 (유사도 높은 순)
    sorted_indices = np.argsort(-similarities)
    sorted_similarities = similarities[sorted_indices]
    sorted_I = I[0][sorted_indices]

    # 메타데이터 매핑 및 결과 생성
    results = []
    for idx, sim in zip(sorted_I, sorted_similarities):

        if idx < 0 or idx >= len(metadata):  # 유사도 음수 제외
            continue

        doc_info = metadata.iloc[idx]
        results.append({
            'name': doc_info['name'],
            'dir': doc_info['dir'],
            'category_kor': doc_info['category_kor'],
            'similarity': float(sim)
        })

    return results
