import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocess import preprocess_document
from config import BASE_PATH, CATEGORY_KOR_TO_ENG, CATEGORY_LABELS
from util import move_file

model = None

def classify_document(file_path):
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    # 문서 전처리
    text = preprocess_document(file_path)
    if not text.strip():
        raise ValueError("문서에 내용이 없습니다.")
    
    category = get_closest_category(text)
    embedding = model.encode([text])[0].astype('float32').reshape(1, -1)

    # 인덱스 저장 경로 설정
    dest_dir = f'{BASE_PATH}/{CATEGORY_KOR_TO_ENG[category]}'
    os.makedirs(dest_dir, exist_ok=True)
    index_path = os.path.join(BASE_PATH, 'index.faiss')
    metadata_path = os.path.join(BASE_PATH, 'metadata.pkl')

    # 기존 인덱스 불러오기
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(embedding.shape[1])
        metadata = []

    index.add(embedding)
    metadata.append({'name': os.path.basename(file_path), 'dir': dest_dir, 'category_kor': category})

    faiss.write_index(index, index_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    move_file(file_path, dest_dir)

    return {
        'status': 'success',
        'category': category,
        'path': dest_dir
    }
    

def get_closest_category(text):
    label_sentences = list(CATEGORY_LABELS.values())
    label_names = list(CATEGORY_LABELS.keys())

    doc_embedding = model.encode([text])[0]
    label_embeddings = model.encode(label_sentences)

    sims = np.dot(label_embeddings, doc_embedding) / (
        np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(doc_embedding)
    )

    best_index = np.argmax(sims)
    return label_names[best_index]