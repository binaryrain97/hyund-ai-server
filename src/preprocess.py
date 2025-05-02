import os
import re
import nltk
import docx
import pandas as pd
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text as extract_pdf_text

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# --- 공통 텍스트 정제 함수 ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9가-힣 ]+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text.strip()

# --- 파일별 텍스트 추출 함수 ---
def extract_text_from_pdf(path):
    try:
        return extract_pdf_text(path)
    except Exception as e:
        print(f"[PDF 오류] {path}: {e}")
        return ""

def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"[DOCX 오류] {path}: {e}")
        return ""

def extract_text_from_txt(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"[TXT 오류] {path}: {e}")
        return ""

def extract_text_from_excel(path):
    try:
        df = pd.read_excel(path, engine='openpyxl')
        return "\n".join(df.astype(str).fillna('').apply(' '.join, axis=1))
    except Exception as e:
        print(f"[Excel 오류] {path}: {e}")
        return ""

# --- 파일 확장자에 따른 전처리 실행 ---
def preprocess_document(path):
    ext = os.path.splitext(path)[-1].lower()

    if ext == '.pdf':
        raw_text = extract_text_from_pdf(path)
    elif ext == '.docx':
        raw_text = extract_text_from_docx(path)
    elif ext == '.txt':
        raw_text = extract_text_from_txt(path)
    elif ext in ['.xls', '.xlsx']:
        raw_text = extract_text_from_excel(path)
    else:
        print(f"[지원되지 않는 형식] {path}")
        return ""

    return clean_text(raw_text)
