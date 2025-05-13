import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocess import preprocess_document
from config import BASE_PATH, CATEGORY_KOR_TO_ENG, CATEGORY_LABELS, INDEX_PATH, METADATA_PATH
from util import move_file

model = None

def classify_document(file_path):
    global model
    try:
        if model is None:
            print("[DEBUG] 모델 초기화 시작")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[DEBUG] 모델 초기화 완료")

        print(f"[DEBUG] 파일 처리 시작: {file_path}")
        text = preprocess_document(file_path)
        print(f"[DEBUG] 전처리 후 텍스트: {text[:50]}...")  # 일부만 출력

        if not text.strip():
            raise ValueError("문서에 내용이 없습니다.")

        print("[DEBUG] 카테고리 분류 시작")
        closest_category = get_closest_category(text)
        print(f"[DEBUG] 예측 카테고리: {closest_category}")

        embeddings = model.encode([text]).astype('float32')  # shape (1, 384)
        print(f"[DEBUG] 임베딩 생성 완료: {embeddings.shape}")

        dest_dir = f'{BASE_PATH}/{CATEGORY_KOR_TO_ENG.get(closest_category)}'
        print(f"[DEBUG] 대상 디렉토리: {dest_dir}")

        # 기존 인덱스 불러오기
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, 'rb') as f:
                metadata = pickle.load(f)
        else:
            index = faiss.IndexFlatL2(embeddings.shape[1])
            metadata = []

        index.add(embeddings)
        metadata.append({'name': os.path.basename(file_path), 'dir': dest_dir, 'category_kor': closest_category})

        faiss.write_index(index, INDEX_PATH)
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata, f)

        move_file(file_path, dest_dir)

        return {
            'status': 'success',
            'category': closest_category,
            'path': dest_dir
        }
    except Exception as e:
        print(f"[최종 오류] {str(e)}")
        raise  # 상위로 전파

def get_closest_category(text):
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    label_sentences = list(CATEGORY_LABELS.values())
    label_names = list(CATEGORY_LABELS.keys())

    doc_embedding = model.encode([text])[0].astype('float32')
    label_embeddings = model.encode(label_sentences).astype('float32')

    sims = np.dot(label_embeddings, doc_embedding) / (
        np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(doc_embedding)
    )

    best_index = np.argmax(sims)
    return label_names[best_index]