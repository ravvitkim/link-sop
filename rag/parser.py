"""
문서 파서
- SOP 문서를 구조화
- section / subsection 계층 유지
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────────────────────

@dataclass
class ContentBlock:
    text: str
    block_type: str                  # section / subsection / intro
    level: int = 0
    page: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    text: str
    blocks: List[ContentBlock]
    metadata: Dict


# ─────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────

def extract_title(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:5]:
        if line.lower().startswith(("title:", "제목:")):
            return line.split(":", 1)[1].strip()
        if len(line) > 5:
            return line[:100]
    return "제목 없음"


# ─────────────────────────────────────────────────────────────
# SOP 조항 패턴
# ─────────────────────────────────────────────────────────────

ARTICLE_PATTERNS = [
    (r'^(\d+\.\d+)\s+(.+)', 'subsection'),  # 4.1 시약 준비
    (r'^(\d+)\.\s+(.+)', 'section'),        # 1. 목적
]


# ─────────────────────────────────────────────────────────────
# SOP 파서
# ─────────────────────────────────────────────────────────────

def parse_articles(
    text: str,
    file_name: str,
    file_type: str
) -> ParsedDocument:

    lines = text.split("\n")
    blocks: List[ContentBlock] = []

    current_lines: List[str] = []
    current_meta = {
        "article_num": None,
        "article_type": "intro",
        "article_title": None,
        "level": 0,
        "parent": None,
    }

    # SOP 메타데이터
    sop_id_match = re.search(r'SOP[-_]([A-Z]+)[-_](\d+)', text, re.IGNORECASE)
    version_match = re.search(r'(?:Version|Ver|버전)[\s:]*(\d+\.?\d*)', text, re.IGNORECASE)

    sop_id = sop_id_match.group(0) if sop_id_match else "SOP-UNKNOWN"
    version = version_match.group(1) if version_match else "1.0"
    doc_title = extract_title(text)

    def enrich_meta(meta: Dict):
        num = meta["article_num"]
        if meta["article_type"] == "section":
            meta["level"] = 1
            meta["parent"] = None
        elif meta["article_type"] == "subsection":
            meta["level"] = 2
            meta["parent"] = num.split(".")[0]

    def flush():
        if not current_lines:
            return

        block_text = "\n".join(current_lines).strip()
        if not block_text:
            return

        blocks.append(ContentBlock(
            text=block_text,
            block_type=current_meta["article_type"],
            level=current_meta.get("level", 0),
            section=current_meta.get("article_num"),
            metadata={
                "article_num": current_meta.get("article_num"),
                "article_type": current_meta.get("article_type"),
                "title": current_meta.get("article_title"),
                "parent": current_meta.get("parent"),
            }
        ))

    for line in lines:
        line_strip = line.strip()
        matched = False

        for pattern, a_type in ARTICLE_PATTERNS:
            m = re.match(pattern, line_strip)
            if m:
                flush()
                current_lines.clear()

                current_meta = {
                    "article_num": m.group(1),          # 1 / 4.1
                    "article_type": a_type,              # section / subsection
                    "article_title": m.group(2).strip(), # 목적 / 시약 준비
                }
                enrich_meta(current_meta)
                matched = True
                break

        current_lines.append(line)

    flush()

    return ParsedDocument(
        text=text,
        blocks=blocks,
        metadata={
            "file_name": file_name,
            "file_type": file_type,
            "title": doc_title,
            "doc_type": "SOP",
            "sop_id": sop_id,
            "version": version,
            "status": "Effective",
            "is_gxp": True,
        }
    )
