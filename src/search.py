import os
import pickle
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import BASE_PATH

model = None

def search_files(query):
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
    metadata_path = os.path.join(BASE_PATH, 'metadata.pkl')
    index_path = os.path.join(BASE_PATH, 'index.faiss')

    # 1. 파일 존재 여부 확인
    if not (os.path.exists(metadata_path) and os.path.exists(index_path)):
        return []

    # 2. 메타데이터 로드 (예외 처리 추가)
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
        if isinstance(metadata, list):
            metadata = pd.DataFrame(metadata)
        elif not isinstance(metadata, pd.DataFrame):
            return []
            
        if metadata.empty:  # 빈 데이터 체크
            return []
            
    except Exception as e:
        print(f"Metadata loading error: {e}")
        return []

    # 3. 인덱스 로드 (예외 처리 추가)
    try:
        index = faiss.read_index(index_path)
        if index.ntotal == 0:  # 빈 인덱스 체크
            return []
    except Exception as e:
        print(f"Index loading error: {e}")
        return []

    # 4. 검색 실행
    try:
        query_vec = model.encode([query]).astype("float32").reshape(1, -1)
        num_docs = index.ntotal
        top_k = min(10, num_docs)
        
        if top_k <= 0:
            return []
            
        D, I = index.search(query_vec, k=top_k)
        similarities = -D[0]

        results = []
        for idx, sim in zip(I[0], similarities):
            # 유효한 인덱스 범위 확인 (0 <= idx < 문서 수)
            if 0 <= idx < len(metadata):
                doc_info = metadata.iloc[idx]
                results.append({
                    'name': doc_info.get('name', ''),
                    'dir': doc_info.get('dir', ''),
                    'category_kor': doc_info.get('category_kor', ''),
                    'similarity': float(sim)
                })
                
        return results
        
    except Exception as e:
        print(f"Search error: {e}")
        return []
