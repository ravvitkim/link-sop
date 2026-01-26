# 🤖 RAG Chatbot v9.0

**LangGraph 기반 GMP/SOP 문서 처리 시스템**

PDF, DOCX, HTML 문서를 마크다운으로 변환하고, 계층적 섹션 구조를 추출하여 벡터 DB에 저장하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

---

## ✨ 주요 기능

### 🔄 LangGraph 상태 머신 파이프라인

```
┌─────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐
│  Load   │───▶│ Convert │───▶│ Validate │───▶│  Split  │───▶│ Optimize │───▶│ Finalize │
└─────────┘    └─────────┘    └──────────┘    └─────────┘    └──────────┘    └──────────┘
                    │              │
                    ▼              ▼
              ┌──────────┐  ┌──────────┐
              │ Fallback │  │  Repair  │
              └──────────┘  └──────────┘
```

| 노드 | 기능 | 분기 조건 |
|------|------|----------|
| **Load** | 파일 타입 감지 | - |
| **Convert** | 마크다운 변환 | 실패 시 → Fallback |
| **Fallback** | 대체 파서 시도 | PDF: pdfplumber → Docling → PyMuPDF |
| **Validate** | 품질 점수 계산 | 점수 < 50% → Repair |
| **Repair** | 헤더 추론, 테이블 복구 | - |
| **Split** | 헤더 기준 분할 | - |
| **Optimize** | 재분할 + 컨텍스트 프리픽스 | - |
| **Finalize** | 결과 정리 | - |

### 📄 지원 파일 형식

| 형식 | 파서 | 특징 |
|------|------|------|
| `.docx` | python-docx | Word 스타일 기반 헤더 감지 |
| `.pdf` | pdfplumber + 헤더 추론 | 다중 폴백 전략 |
| `.html` | BeautifulSoup | HTML 태그 → 마크다운 |
| `.md` | 직접 처리 | 패스스루 |
| `.txt` | 헤더 추론 | 패턴 기반 |

### 🔍 계층적 섹션 경로 (section_path)

```
📍 5 절차 Procedure
📍 5 절차 Procedure > 5.1 품질관리기준서의 구성 및 관리
📍 5 절차 Procedure > 5.1 품질관리기준서의 구성 및 관리 > 5.1.1 문서번호 체계
```

### 📊 품질 검증 (5가지 항목)

1. ✅ 텍스트 길이 (≥100자)
2. ✅ 헤더 개수 (≥3개)
3. ✅ 문단 구조 (≥5개)
4. ✅ 한글 비율 (≥10%)
5. ✅ 특수문자 오염 (<1%)

---

## 📁 프로젝트 구조

```
rag_chatbot/
├── main.py                      # FastAPI 서버 (v9.0)
├── requirements.txt             # 의존성
├── README.md
│
├── rag/                         # RAG 모듈
│   ├── __init__.py
│   ├── document_pipeline.py     # 🔥 LangGraph 파이프라인 (핵심)
│   ├── document_processor.py    # v8.0 선형 파이프라인 (폴백)
│   ├── document_loader.py       # 문서 로더 (그래프용)
│   ├── chunker.py               # 청킹 유틸리티
│   ├── prompt.py                # RAG 프롬프트 템플릿
│   ├── vector_store.py          # ChromaDB 벡터 스토어
│   ├── graph_store.py           # Neo4j 그래프 스토어
│   └── llm.py                   # LLM 연동 (Ollama)
│
├── frontend/                    # React 프론트엔드
│   ├── src/
│   │   ├── App.tsx
│   │   └── ...
│   └── package.json
│
└── chroma_db/                   # 벡터 DB 저장소
```

---

## 🚀 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 서버 실행

```bash
python main.py
```

서버가 `http://localhost:8000`에서 실행됩니다.

### 3. 프론트엔드 실행 (선택)

```bash
cd frontend
npm install
npm run dev
```

---

## 📡 API 엔드포인트

### 문서 업로드

```http
POST /rag/upload
```

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| file | File | 필수 | 업로드할 문서 |
| collection | string | "documents" | 컬렉션 이름 |
| chunk_size | int | 500 | 청크 크기 |
| chunk_method | string | "article" | 청킹 방법 |
| model | string | "multilingual-e5-small" | 임베딩 모델 |
| overlap | int | 50 | 청크 오버랩 |
| use_langgraph | bool | true | LangGraph 사용 여부 |

**응답:**
```json
{
  "success": true,
  "filename": "EQ-SOP-00010.docx",
  "sop_id": "EQ-SOP-00010",
  "chunks": 34,
  "pipeline_version": "v9.0-langgraph",
  "quality_score": 1.0,
  "conversion_method": "python-docx",
  "warnings": []
}
```

### 검색

```http
POST /rag/search
```

```json
{
  "query": "품질관리기준서의 목적은?",
  "collection": "documents",
  "top_k": 5
}
```

### 챗봇

```http
POST /rag/chat
```

```json
{
  "message": "품질관리기준서란 무엇인가요?",
  "collection": "documents",
  "model": "gemma3:4b"
}
```

---

## 📝 문서 형식별 처리

### DOCX (숫자형 헤더)

| 원본 | section_path |
|------|--------------|
| `5 절차 Procedure` | `5 절차 Procedure` |
| `5.1 품질관리기준서의 구성` | `5 절차 Procedure > 5.1 품질관리기준서의 구성` |

### DOCX (이름형 헤더)

| 원본 | section_path |
|------|--------------|
| `절차 Procedure` | `절차 Procedure` |
| `변경(개정) 관리 (Revision)` | `절차 Procedure > 변경(개정) 관리 (Revision)` |

### PDF (헤더 추론 적용)

| 추출된 텍스트 | 변환 후 |
|--------------|---------|
| `1 목적 Purpose` | `## 1 목적 Purpose` |
| `1.1 본 규정은...` | `### 1.1 본 규정은...` |

---

## ⚠️ 주의사항

### 문서 버전 차이

**같은 문서라도 파일 형식에 따라 내용이 다를 수 있습니다:**

| 파일 | 내용 |
|------|------|
| `EQ-SOP-00009.docx` | 숫자 없음 (`절차 Procedure`) |
| `EQ-SOP-00009.pdf` | 숫자 있음 (`5 절차 Procedure`) |

> 파싱 결과는 원본 문서의 내용을 그대로 반영합니다.

### PDF 파싱 제한

- ❌ 스캔된 이미지 PDF (OCR 필요)
- ⚠️ 복잡한 레이아웃 (텍스트 순서 섞임 가능)
- ⚠️ 헤더 추론 (패턴 기반, 100% 정확하지 않음)

---

## 🔄 버전 히스토리

| 버전 | 주요 변경 |
|------|----------|
| **v9.0** | LangGraph 상태 머신 파이프라인 |
| v8.1 | 컨텍스트 프리픽스, 테이블 보존 |
| v8.0 | 4단계 마크다운 파이프라인 |
| v7.0 | 소제목 패턴 확장, 확장자 버그 수정 |
| v6.3 | section_path 지원 |

---

## 📄 라이선스

MIT License