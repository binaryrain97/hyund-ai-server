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

    if not (os.path.exists(metadata_path) and os.path.exists(index_path)):
        return []

    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
        if isinstance(metadata, list):
            metadata = pd.DataFrame(metadata)
        elif not isinstance(metadata, pd.DataFrame):
            return []
            
        if metadata.empty:
            return []
            
    except Exception as e:
        print(f"Metadata loading error: {e}")
        return []

    try:
        index = faiss.read_index(index_path)
        if index.ntotal == 0:
            return []
    except Exception as e:
        print(f"Index loading error: {e}")
        return []

    try:
        query_vec = model.encode([query]).astype("float32").reshape(1, -1)
        num_docs = index.ntotal
        top_k = min(10, num_docs)
        
        if top_k <= 0:
            return []
            
        D, I = index.search(query_vec, k=top_k)
        similarities = -D[0]

        results = []
        seen_files = set()  # 추가된 부분: 중복 체크용 집합
        for idx, sim in zip(I[0], similarities):
            if 0 <= idx < len(metadata):
                doc_info = metadata.iloc[idx]
                file_name = doc_info.get('name', '')
                
                # 추가된 부분: 중복 파일 필터링
                if file_name not in seen_files:
                    seen_files.add(file_name)
                    results.append({
                        'name': file_name,
                        'dir': doc_info.get('dir', ''),
                        'category_kor': doc_info.get('category_kor', ''),
                        'similarity': float(sim)
                    })
                
        return results
        
    except Exception as e:
        print(f"Search error: {e}")
        return []