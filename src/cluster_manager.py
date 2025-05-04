import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocess import preprocess_document
from config import BASE_PATH, CATEGORY_KOR_TO_ENG, CATEGORY_LABELS
from util import move_file_to_category

model = SentenceTransformer('all-MiniLM-L6-v2')

def classify_document(text):
    label_sentences = list(CATEGORY_LABELS.values())
    label_names = list(CATEGORY_LABELS.keys())

    doc_embedding = model.encode([text])[0]
    label_embeddings = model.encode(label_sentences)

    sims = np.dot(label_embeddings, doc_embedding) / (
        np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(doc_embedding)
    )

    best_index = np.argmax(sims)
    return label_names[best_index]

def index_document_by_category(file_path):
    text = preprocess_document(file_path)
    if not text.strip():
        raise ValueError("문서에 내용이 없습니다.")

    category = classify_document(text)
    embedding = model.encode([text])[0].astype('float32').reshape(1, -1)

    # 인덱스 저장 경로 설정
    category_eng = CATEGORY_KOR_TO_ENG.get(category, "uncategorized")
    base_dir = f'{BASE_PATH}/{category_eng}'
    os.makedirs(base_dir, exist_ok=True)
    index_path = os.path.join(base_dir, 'vector.index')
    meta_path = os.path.join(base_dir, 'meta.pkl')

    # 기존 인덱스 불러오기
    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(embedding.shape[1])
        metadata = []

    index.add(embedding)
    metadata.append({'path': file_path})

    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)

    move_file_to_category(file_path, category)

    return {
        'status': 'success',
        'category': category,
        'path': file_path
    }
