"""
LangGraph 기반 문서 처리 파이프라인 v9.0

🔥 상태 머신(State Machine) 기반 유연한 워크플로우:
- 문서 타입별 분기 처리
- 변환 실패 시 폴백 전략
- 품질 검증 및 보정 단계
- 조건부 재처리

노드 흐름:
┌─────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐
│  Load   │───▶│ Convert │───▶│ Validate │───▶│  Split  │
└─────────┘    └─────────┘    └──────────┘    └─────────┘
                    │              │                │
                    ▼              ▼                ▼
              ┌──────────┐  ┌──────────┐    ┌──────────┐
              │ Fallback │  │  Repair  │    │ Optimize │
              └──────────┘  └──────────┘    └──────────┘
                                                   │
                                                   ▼
                                            ┌──────────┐
                                            │ Finalize │
                                            └──────────┘
"""

from typing import TypedDict, List, Dict, Optional, Literal, Annotated
from dataclasses import dataclass, field
import re
from io import BytesIO
import operator


# ═══════════════════════════════════════════════════════════════════════════
# 상태 정의
# ═══════════════════════════════════════════════════════════════════════════

class PipelineState(TypedDict):
    """파이프라인 상태"""
    # 입력
    filename: str
    content: bytes
    
    # 설정
    chunk_size: int
    chunk_overlap: int
    
    # 중간 결과
    file_type: str
    markdown: str
    metadata: Dict
    sections: List[Dict]
    chunks: List[Dict]
    
    # 품질 지표
    quality_score: float
    conversion_method: str
    
    # 에러 처리
    errors: Annotated[List[str], operator.add]
    warnings: Annotated[List[str], operator.add]
    retry_count: int
    
    # 최종 결과
    success: bool


@dataclass
class Chunk:
    """청크 데이터"""
    text: str
    index: int = 0
    metadata: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# 노드 함수들
# ═══════════════════════════════════════════════════════════════════════════

def node_load(state: PipelineState) -> PipelineState:
    """
    1단계: 파일 로드 및 타입 감지
    """
    filename = state["filename"]
    content = state["content"]
    
    # 파일 타입 감지 (확장자 버그 수정 포함)
    filename_lower = filename.lower()
    
    if '.docx' in filename_lower:
        file_type = 'docx'
    elif '.doc' in filename_lower and '.docx' not in filename_lower:
        file_type = 'doc'
    elif '.pdf' in filename_lower:
        file_type = 'pdf'
    elif '.html' in filename_lower or '.htm' in filename_lower:
        file_type = 'html'
    elif '.md' in filename_lower:
        file_type = 'markdown'
    elif '.txt' in filename_lower:
        file_type = 'text'
    else:
        file_type = 'unknown'
    
    # 파일 크기 확인
    file_size = len(content)
    if file_size == 0:
        state["errors"] = ["파일이 비어있습니다."]
        state["success"] = False
        return state
    
    state["file_type"] = file_type
    state["metadata"] = {
        "file_name": filename,
        "file_type": file_type,
        "file_size": file_size,
    }
    
    return state


def node_convert(state: PipelineState) -> PipelineState:
    """
    2단계: 문서 → 마크다운 변환
    """
    file_type = state["file_type"]
    content = state["content"]
    filename = state["filename"]
    
    try:
        if file_type == 'docx':
            markdown, metadata = _convert_docx(filename, content)
            state["conversion_method"] = "python-docx"
            
        elif file_type == 'pdf':
            markdown, metadata, method = _convert_pdf_with_fallback(filename, content)
            state["conversion_method"] = method
            
        elif file_type == 'html':
            markdown, metadata = _convert_html(filename, content)
            state["conversion_method"] = "beautifulsoup"
            
        elif file_type == 'markdown':
            markdown = content.decode('utf-8', errors='ignore')
            metadata = {}
            state["conversion_method"] = "passthrough"
            
        elif file_type == 'text':
            markdown = _convert_text_to_markdown(content.decode('utf-8', errors='ignore'))
            metadata = {}
            state["conversion_method"] = "text-inference"
            
        else:
            # 알 수 없는 타입 → 텍스트로 시도
            markdown = content.decode('utf-8', errors='ignore')
            metadata = {}
            state["conversion_method"] = "fallback-text"
            state["warnings"] = [f"알 수 없는 파일 타입: {file_type}, 텍스트로 처리"]
        
        state["markdown"] = markdown
        state["metadata"].update(metadata)
        
    except Exception as e:
        state["errors"] = [f"변환 실패: {str(e)}"]
        state["markdown"] = ""
    
    return state


def node_convert_fallback(state: PipelineState) -> PipelineState:
    """
    2-1단계: 변환 실패 시 폴백 전략
    """
    content = state["content"]
    file_type = state["file_type"]
    
    state["warnings"] = [f"기본 변환 실패, 폴백 전략 시도 중..."]
    
    try:
        if file_type == 'pdf':
            # PDF 폴백: PyPDF2 → pdfplumber → 텍스트 추출
            markdown = _pdf_fallback_extract(content)
            state["conversion_method"] = "pdf-fallback"
            
        elif file_type == 'docx':
            # DOCX 폴백: XML 직접 파싱
            markdown = _docx_fallback_extract(content)
            state["conversion_method"] = "docx-fallback"
            
        else:
            # 최후의 수단: 바이너리에서 텍스트 추출 시도
            markdown = content.decode('utf-8', errors='ignore')
            state["conversion_method"] = "binary-text"
        
        state["markdown"] = markdown
        state["errors"] = []  # 에러 클리어
        
    except Exception as e:
        state["errors"] = [f"폴백 변환도 실패: {str(e)}"]
        state["success"] = False
    
    return state


def node_validate(state: PipelineState) -> PipelineState:
    """
    3단계: 마크다운 품질 검증
    """
    markdown = state.get("markdown", "")
    
    if not markdown:
        state["quality_score"] = 0.0
        state["errors"] = ["마크다운 변환 결과가 비어있습니다."]
        return state
    
    # 품질 점수 계산
    score = 0.0
    issues = []
    
    # 1. 길이 체크 (최소 100자)
    if len(markdown) >= 100:
        score += 0.2
    else:
        issues.append("텍스트가 너무 짧음")
    
    # 2. 헤더 존재 여부
    header_count = len(re.findall(r'^#{1,6}\s+', markdown, re.MULTILINE))
    if header_count >= 3:
        score += 0.3
    elif header_count >= 1:
        score += 0.15
        issues.append("헤더가 부족함")
    else:
        issues.append("헤더가 없음")
    
    # 3. 문단 구조 (빈 줄로 구분된 문단)
    paragraphs = [p for p in markdown.split('\n\n') if p.strip()]
    if len(paragraphs) >= 5:
        score += 0.2
    elif len(paragraphs) >= 2:
        score += 0.1
        issues.append("문단 구조가 부실함")
    
    # 4. 한글 비율 (SOP 문서 특성상 한글이 있어야 함)
    korean_chars = len(re.findall(r'[가-힣]', markdown))
    total_chars = len(markdown)
    korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
    
    if korean_ratio >= 0.1:
        score += 0.2
    else:
        issues.append("한글 비율이 낮음")
    
    # 5. 특수문자 오염 체크
    garbage_ratio = len(re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', markdown)) / len(markdown) if markdown else 0
    if garbage_ratio < 0.01:
        score += 0.1
    else:
        issues.append("특수문자 오염 감지")
    
    state["quality_score"] = min(score, 1.0)
    
    if issues:
        state["warnings"] = issues
    
    return state


def node_repair(state: PipelineState) -> PipelineState:
    """
    3-1단계: 마크다운 품질 보정
    """
    markdown = state.get("markdown", "")
    
    state["warnings"] = ["품질 보정 수행 중..."]
    
    # 1. 특수문자 제거
    markdown = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', markdown)
    
    # 2. 연속 빈 줄 정리
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    
    # 3. 헤더가 없으면 추론해서 추가
    if not re.search(r'^#{1,6}\s+', markdown, re.MULTILINE):
        markdown = _infer_headers(markdown)
    
    # 4. 깨진 테이블 복구 시도
    markdown = _repair_tables(markdown)
    
    state["markdown"] = markdown
    state["conversion_method"] += "+repaired"
    
    # 품질 재측정
    state = node_validate(state)
    
    return state


def node_split(state: PipelineState) -> PipelineState:
    """
    4단계: 헤더 기준 분할
    """
    markdown = state.get("markdown", "")
    
    if not markdown:
        state["sections"] = []
        return state
    
    lines = markdown.split('\n')
    sections = []
    
    current_headers = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
    current_content = []
    
    def flush_section():
        nonlocal current_content
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                header_path_parts = []
                headers_dict = {}
                for level in range(1, 7):
                    if current_headers[level]:
                        headers_dict[f"H{level}"] = current_headers[level]
                        if level >= 2:
                            header_path_parts.append(current_headers[level])
                
                sections.append({
                    "content": content,
                    "headers": headers_dict,
                    "header_path": " > ".join(header_path_parts) if header_path_parts else None
                })
        current_content = []
    
    for line in lines:
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        
        if header_match:
            flush_section()
            level = len(header_match.group(1))
            header_text = header_match.group(2).strip()
            
            current_headers[level] = header_text
            for l in range(level + 1, 7):
                current_headers[l] = None
            
            current_content.append(line)
        else:
            current_content.append(line)
    
    flush_section()
    
    state["sections"] = sections
    return state


def node_optimize(state: PipelineState) -> PipelineState:
    """
    5단계: 긴 섹션 재분할 + 컨텍스트 프리픽스
    """
    sections = state.get("sections", [])
    chunk_size = state.get("chunk_size", 500)
    chunk_overlap = state.get("chunk_overlap", 50)
    metadata = state.get("metadata", {})
    
    chunks = []
    idx = 0
    
    sop_id = metadata.get("sop_id")
    doc_name = metadata.get("file_name")
    
    for section in sections:
        content = section["content"]
        headers = section.get("headers", {})
        header_path = section.get("header_path")
        
        # 긴 섹션 재분할
        if len(content) > chunk_size:
            text_chunks = _split_recursive(content, chunk_size, chunk_overlap)
            is_split = len(text_chunks) > 1
        else:
            text_chunks = [content]
            is_split = False
        
        for i, text in enumerate(text_chunks):
            if not text.strip():
                continue
            
            # 재분할된 청크에 컨텍스트 프리픽스 추가
            if is_split and i > 0 and header_path:
                text = f"[Context: {header_path}]\n\n{text}"
            
            # 섹션 타입 결정
            section_type, section_num = _determine_section_type(headers)
            section_display = headers.get("H4") or headers.get("H3") or headers.get("H2") or headers.get("H1")
            
            chunks.append({
                "text": text.strip(),
                "index": idx,
                "metadata": {
                    "doc_name": doc_name,
                    "doc_title": sop_id or doc_name,
                    "sop_id": sop_id,
                    "article_num": section_num,
                    "article_type": section_type,
                    "section": section_display,
                    "section_path": header_path,
                    "section_path_readable": header_path,
                    "H1": headers.get("H1"),
                    "H2": headers.get("H2"),
                    "H3": headers.get("H3"),
                    "H4": headers.get("H4"),
                    "chunk_part": i + 1 if is_split else None,
                    "total_parts": len(text_chunks) if is_split else None,
                }
            })
            idx += 1
    
    state["chunks"] = chunks
    return state


def node_finalize(state: PipelineState) -> PipelineState:
    """
    6단계: 최종 결과 정리
    """
    chunks = state.get("chunks", [])
    
    if not chunks:
        state["success"] = False
        state["errors"] = ["청크 생성 실패: 결과가 비어있습니다."]
    else:
        state["success"] = True
    
    # 통계 추가
    state["metadata"]["total_chunks"] = len(chunks)
    state["metadata"]["quality_score"] = state.get("quality_score", 0)
    state["metadata"]["conversion_method"] = state.get("conversion_method", "unknown")
    
    return state


# ═══════════════════════════════════════════════════════════════════════════
# 조건부 라우팅 함수
# ═══════════════════════════════════════════════════════════════════════════

def should_fallback(state: PipelineState) -> Literal["fallback", "validate"]:
    """변환 실패 시 폴백으로 라우팅"""
    if state.get("errors") or not state.get("markdown"):
        return "fallback"
    return "validate"


def should_repair(state: PipelineState) -> Literal["repair", "split"]:
    """품질 점수가 낮으면 보정으로 라우팅"""
    quality_score = state.get("quality_score", 0)
    retry_count = state.get("retry_count", 0)
    
    # 품질이 낮고 재시도 횟수가 2회 미만이면 보정
    if quality_score < 0.5 and retry_count < 2:
        return "repair"
    return "split"


def is_failed(state: PipelineState) -> Literal["end", "continue"]:
    """실패 상태면 종료"""
    if state.get("errors") and state.get("success") == False:
        return "end"
    return "continue"


# ═══════════════════════════════════════════════════════════════════════════
# 헬퍼 함수들
# ═══════════════════════════════════════════════════════════════════════════

def _convert_docx(filename: str, content: bytes) -> tuple:
    """DOCX → Markdown"""
    from docx import Document
    
    doc = Document(BytesIO(content))
    md_lines = []
    metadata = {}
    
    sop_pattern = re.compile(r'((?:EQ-)?SOP[-_]?\d{4,5})', re.IGNORECASE)
    
    # 주요 섹션 키워드
    main_sections = ['목적', 'Purpose', '적용 범위', 'Scope', '정의', 'Definitions',
                     '책임', 'Responsibilities', '절차', 'Procedure', 
                     '참고문헌', 'Reference', '첨부', 'Attachments']
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            md_lines.append("")
            continue
        
        # SOP ID 추출
        sop_match = sop_pattern.search(text)
        if sop_match and "sop_id" not in metadata:
            sop_id = sop_match.group(1).upper().replace('_', '-')
            if not sop_id.startswith('EQ-'):
                sop_id = 'EQ-' + sop_id
            metadata["sop_id"] = sop_id
        
        # 헤더 레벨 결정
        header_level = None
        
        # Word 스타일 기반
        style_name = para.style.name.lower() if para.style else ""
        if 'heading 1' in style_name or 'title' in style_name:
            header_level = 1
        elif 'heading 2' in style_name:
            header_level = 2
        elif 'heading 3' in style_name:
            header_level = 3
        elif 'heading 4' in style_name:
            header_level = 4
        
        # 패턴 기반 감지
        if not header_level:
            for section in main_sections:
                if text.startswith(section) or re.match(rf'^\d+\s+{section}', text):
                    header_level = 2
                    break
            
            if not header_level:
                if re.match(r'^\d+\.\d+\.\d+\s+', text):
                    header_level = 4
                elif re.match(r'^\d+\.\d+\s+', text):
                    header_level = 3
                elif re.match(r'^\d+\.?\s+[가-힣A-Za-z]', text):
                    header_level = 2
                elif re.match(r'^[가-힣A-Z][가-힣\s\(\)/·\-]+\s*\([A-Za-z\s&/\-:]+\)\s*$', text):
                    header_level = 3
        
        if header_level:
            md_lines.append(f"{'#' * header_level} {text}")
        else:
            md_lines.append(text)
    
    # 테이블 처리
    for table in doc.tables:
        md_lines.append("")
        md_lines.append(_table_to_markdown(table))
    
    return '\n'.join(md_lines), metadata


def _convert_pdf_with_fallback(filename: str, content: bytes) -> tuple:
    """PDF 변환 (다중 폴백)"""
    
    # 1순위: Docling
    try:
        from docling.document_converter import DocumentConverter
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            converter = DocumentConverter()
            result = converter.convert(temp_path)
            markdown = result.document.export_to_markdown()
            return markdown, {"parser": "docling"}, "docling"
        finally:
            os.unlink(temp_path)
    except:
        pass
    
    # 2순위: PyMuPDF
    try:
        import fitz
        pdf = fitz.open(stream=content, filetype="pdf")
        md_lines = []
        for page_num, page in enumerate(pdf):
            text = page.get_text()
            if text.strip():
                md_lines.append(f"<!-- Page {page_num + 1} -->")
                md_lines.append(text)
        return '\n'.join(md_lines), {"parser": "pymupdf"}, "pymupdf"
    except:
        pass
    
    # 3순위: PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(BytesIO(content))
        md_lines = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ''
            if text.strip():
                md_lines.append(f"<!-- Page {i + 1} -->")
                md_lines.append(text)
        return '\n'.join(md_lines), {"parser": "pypdf2"}, "pypdf2"
    except:
        pass
    
    raise Exception("모든 PDF 파서 실패")


def _convert_html(filename: str, content: bytes) -> tuple:
    """HTML → Markdown"""
    from bs4 import BeautifulSoup
    
    html = content.decode('utf-8', errors='ignore')
    soup = BeautifulSoup(html, 'html.parser')
    
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()
    
    md_lines = []
    title = soup.title.string if soup.title else filename
    md_lines.append(f"# {title}")
    md_lines.append("")
    
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
        if tag.name.startswith('h'):
            level = int(tag.name[1])
            md_lines.append(f"{'#' * level} {tag.get_text(strip=True)}")
        elif tag.name == 'li':
            md_lines.append(f"- {tag.get_text(strip=True)}")
        else:
            text = tag.get_text(strip=True)
            if text:
                md_lines.append(text)
        md_lines.append("")
    
    return '\n'.join(md_lines), {"title": title}


def _convert_text_to_markdown(text: str) -> str:
    """텍스트 → 마크다운 (헤더 추론)"""
    lines = text.split('\n')
    md_lines = []
    
    main_sections = ['목적', '적용 범위', '정의', '책임', '절차', '참고문헌', '첨부']
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            md_lines.append("")
            continue
        
        is_header = False
        for section in main_sections:
            if stripped.startswith(section):
                md_lines.append(f"## {stripped}")
                is_header = True
                break
        
        if not is_header:
            if re.match(r'^\d+\.\d+\.\d+\s+', stripped):
                md_lines.append(f"#### {stripped}")
            elif re.match(r'^\d+\.\d+\s+', stripped):
                md_lines.append(f"### {stripped}")
            elif re.match(r'^\d+\.?\s+[가-힣A-Za-z]', stripped):
                md_lines.append(f"## {stripped}")
            else:
                md_lines.append(stripped)
    
    return '\n'.join(md_lines)


def _pdf_fallback_extract(content: bytes) -> str:
    """PDF 폴백 추출"""
    try:
        import pdfplumber
        with pdfplumber.open(BytesIO(content)) as pdf:
            texts = [page.extract_text() or '' for page in pdf.pages]
            return '\n\n'.join(texts)
    except:
        pass
    
    # 최후의 수단
    return content.decode('latin-1', errors='ignore')


def _docx_fallback_extract(content: bytes) -> str:
    """DOCX 폴백: XML 직접 파싱"""
    import zipfile
    from xml.etree import ElementTree
    
    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            xml_content = zf.read('word/document.xml')
            tree = ElementTree.fromstring(xml_content)
            
            # 모든 텍스트 추출
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            texts = []
            for t in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                if t.text:
                    texts.append(t.text)
            
            return '\n'.join(texts)
    except:
        return ""


def _table_to_markdown(table) -> str:
    """Word 테이블 → Markdown"""
    rows = []
    for row in table.rows:
        cells = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
        rows.append(cells)
    
    if not rows:
        return ""
    
    md_lines = []
    md_lines.append("| " + " | ".join(rows[0]) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
    for row in rows[1:]:
        while len(row) < len(rows[0]):
            row.append("")
        md_lines.append("| " + " | ".join(row[:len(rows[0])]) + " |")
    
    return '\n'.join(md_lines)


def _infer_headers(markdown: str) -> str:
    """헤더 추론 삽입"""
    lines = markdown.split('\n')
    result = []
    
    main_sections = ['목적', '적용 범위', '정의', '책임', '절차', '참고문헌', '첨부']
    
    for line in lines:
        stripped = line.strip()
        
        # 주요 섹션 키워드로 시작하면 H2
        matched = False
        for section in main_sections:
            if stripped.startswith(section):
                result.append(f"## {stripped}")
                matched = True
                break
        
        if not matched:
            # 숫자 패턴
            if re.match(r'^\d+\.\d+\.\d+\s+', stripped):
                result.append(f"#### {stripped}")
            elif re.match(r'^\d+\.\d+\s+', stripped):
                result.append(f"### {stripped}")
            elif re.match(r'^\d+\.?\s+[가-힣A-Za-z]', stripped):
                result.append(f"## {stripped}")
            else:
                result.append(line)
    
    return '\n'.join(result)


def _repair_tables(markdown: str) -> str:
    """깨진 테이블 복구"""
    lines = markdown.split('\n')
    result = []
    in_table = False
    table_cols = 0
    
    for line in lines:
        if line.strip().startswith('|') and line.strip().endswith('|'):
            cols = line.count('|') - 1
            
            if not in_table:
                in_table = True
                table_cols = cols
                result.append(line)
                # 구분선이 없으면 추가
                if len(result) >= 1:
                    next_idx = len(result)
            else:
                # 열 수 맞추기
                while line.count('|') - 1 < table_cols:
                    line = line.rstrip('|') + ' |'
                result.append(line)
        else:
            in_table = False
            result.append(line)
    
    return '\n'.join(result)


def _split_recursive(text: str, chunk_size: int, overlap: int) -> List[str]:
    """재귀적 텍스트 분할"""
    if len(text) <= chunk_size:
        return [text]
    
    separators = ["\n\n", "\n| ", "\n", ". ", "。", " ", ""]
    is_table = text.strip().startswith('|') or '\n|' in text
    effective_overlap = 0 if is_table else overlap
    
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            
            for part in parts:
                if len(current) + len(part) + len(sep) <= chunk_size:
                    current = current + sep + part if current else part
                else:
                    if current:
                        chunks.append(current)
                    if len(part) > chunk_size:
                        chunks.extend(_split_recursive(part, chunk_size, overlap))
                        current = ""
                    else:
                        current = part
            
            if current:
                chunks.append(current)
            
            return chunks
    
    # 강제 분할
    step = chunk_size - effective_overlap if effective_overlap > 0 else chunk_size
    return [text[i:i+chunk_size] for i in range(0, len(text), step)]


def _determine_section_type(headers: Dict) -> tuple:
    """섹션 타입 결정"""
    section_type = "text"
    section_num = None
    
    if headers.get("H4"):
        section_type = "subsubsection"
        match = re.match(r'^(\d+\.\d+\.\d+)', headers["H4"])
        if match:
            section_num = match.group(1)
    elif headers.get("H3"):
        section_type = "subsection"
        match = re.match(r'^(\d+\.\d+)', headers["H3"])
        if match:
            section_num = match.group(1)
    elif headers.get("H2"):
        section_type = "section"
        match = re.match(r'^(\d+)', headers["H2"])
        if match:
            section_num = match.group(1)
    
    return section_type, section_num


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph 파이프라인 빌더
# ═══════════════════════════════════════════════════════════════════════════

def build_pipeline():
    """
    LangGraph 파이프라인 구성
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        raise ImportError("langgraph 패키지가 필요합니다: pip install langgraph")
    
    # 그래프 생성
    workflow = StateGraph(PipelineState)
    
    # 노드 추가
    workflow.add_node("load", node_load)
    workflow.add_node("convert", node_convert)
    workflow.add_node("fallback", node_convert_fallback)
    workflow.add_node("validate", node_validate)
    workflow.add_node("repair", node_repair)
    workflow.add_node("split", node_split)
    workflow.add_node("optimize", node_optimize)
    workflow.add_node("finalize", node_finalize)
    
    # 엣지 정의 (흐름)
    workflow.set_entry_point("load")
    
    workflow.add_edge("load", "convert")
    
    workflow.add_conditional_edges(
        "convert",
        should_fallback,
        {
            "fallback": "fallback",
            "validate": "validate"
        }
    )
    
    workflow.add_edge("fallback", "validate")
    
    workflow.add_conditional_edges(
        "validate",
        should_repair,
        {
            "repair": "repair",
            "split": "split"
        }
    )
    
    workflow.add_edge("repair", "split")
    workflow.add_edge("split", "optimize")
    workflow.add_edge("optimize", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


# ═══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ═══════════════════════════════════════════════════════════════════════════

def process_document(
    filename: str,
    content: bytes,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    debug: bool = False
) -> dict:
    """
    문서 처리 메인 함수
    
    LangGraph 파이프라인 실행
    """
    # 초기 상태
    initial_state: PipelineState = {
        "filename": filename,
        "content": content,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "file_type": "",
        "markdown": "",
        "metadata": {},
        "sections": [],
        "chunks": [],
        "quality_score": 0.0,
        "conversion_method": "",
        "errors": [],
        "warnings": [],
        "retry_count": 0,
        "success": False,
    }
    
    try:
        # LangGraph 파이프라인 실행
        pipeline = build_pipeline()
        result = pipeline.invoke(initial_state)
        
        if debug:
            print(f"\n{'='*60}")
            print(f"📊 LangGraph 파이프라인 결과")
            print(f"{'='*60}")
            print(f"   파일: {filename}")
            print(f"   변환 방법: {result.get('conversion_method')}")
            print(f"   품질 점수: {result.get('quality_score', 0):.0%}")
            print(f"   총 청크: {len(result.get('chunks', []))}")
            if result.get('warnings'):
                print(f"   ⚠️ 경고: {result['warnings']}")
            if result.get('errors'):
                print(f"   ❌ 에러: {result['errors']}")
        
        return result
        
    except ImportError:
        # LangGraph 없으면 심플 파이프라인 사용
        if debug:
            print("⚠️ LangGraph 없음, 심플 파이프라인 사용")
        return _simple_pipeline(initial_state, debug)


def _simple_pipeline(state: PipelineState, debug: bool = False) -> dict:
    """
    LangGraph 없을 때 사용하는 심플 파이프라인
    """
    state = node_load(state)
    if state.get("errors"):
        return state
    
    state = node_convert(state)
    if not state.get("markdown"):
        state = node_convert_fallback(state)
    
    state = node_validate(state)
    
    if state.get("quality_score", 0) < 0.5:
        state = node_repair(state)
    
    state = node_split(state)
    state = node_optimize(state)
    state = node_finalize(state)
    
    if debug:
        print(f"\n📊 심플 파이프라인 결과: {len(state.get('chunks', []))} 청크")
    
    return state


# ═══════════════════════════════════════════════════════════════════════════
# 결과 변환 헬퍼
# ═══════════════════════════════════════════════════════════════════════════

def state_to_chunks(state: dict) -> List[Chunk]:
    """상태를 Chunk 객체 리스트로 변환"""
    return [
        Chunk(
            text=c["text"],
            index=c["index"],
            metadata=c["metadata"]
        )
        for c in state.get("chunks", [])
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 테스트
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 테스트 마크다운
    test_md = """# EQ-SOP-00010 품질관리기준서

## 목적 Purpose

본 기준서는 품질관리기준서의 작성, 검토, 승인에 관한 기준을 정한다.

## 적용 범위 Scope

본 기준서는 회사 내 품질관리 활동 전반에 적용된다.

## 절차 Procedure

품질관리기준서는 다음 항목을 포함한다.
"""
    
    result = process_document("test.md", test_md.encode(), chunk_size=300, debug=True)
    
    print(f"\n✅ 성공: {result.get('success')}")
    print(f"📊 청크 수: {len(result.get('chunks', []))}")
    
    for chunk in result.get('chunks', [])[:3]:
        print(f"\n📍 {chunk['metadata'].get('section_path_readable', 'N/A')}")
        print(f"   {chunk['text'][:60]}...")
