import os

BASE_PATH = r'result'
INDEX_PATH = r'result/index.faiss'
METADATA_PATH = r'result/metadata.pkl'

CATEGORY_KOR_TO_ENG = {
    "계약서": "contracts",
    "보고서": "reports",
    "회의록": "meeting_minutes",
    "인사문서": "hr_documents",
    "재무/회계": "finance_accounting",
    "기술 문서": "technical_documents",
    "홍보/마케팅": "marketing",
    "교육자료": "training_materials",
    "정책/규정": "policies",
    "일반 문서": "general_documents"
}

CATEGORY_LABELS = {
    "계약서": "법적 계약이나 합의에 대한 문서입니다.",
    "보고서": "업무 내용을 정리하거나 보고하기 위한 문서입니다.",
    "회의록": "회의의 내용과 결론을 요약한 문서입니다.",
    "인사문서": "직원의 이력, 평가, 채용 관련 문서입니다.",
    "재무/회계": "비용, 세금, 예산, 회계 관련 문서입니다.",
    "기술 문서": "제품 설명서, 기술 가이드, 사양서 등의 문서입니다.",
    "홍보/마케팅": "광고, 캠페인, 마케팅 전략 문서입니다.",
    "교육자료": "교육 목적의 문서, 강의안이나 학습자료입니다.",
    "정책/규정": "회사 정책, 규정, 지침 관련 문서입니다.",
    "일반 문서": "기타 또는 분류되지 않는 문서입니다."
}

def init_categories():
    for label in CATEGORY_LABELS.keys():
        label_eng = CATEGORY_KOR_TO_ENG[label]
        dir_path = os.path.join(BASE_PATH, label_eng)
        os.makedirs(dir_path, exist_ok=True)