"""
Neo4j 그래프 저장소 모듈

SOP 문서의 지식 그래프를 Neo4j Aura에 저장하고 조회합니다.

노드 타입:
- Document: SOP 문서 (sop_id, title, version)
- Section: 섹션 (name, type, content)
- Term: 정의된 용어 (name, definition)
- Role: 책임 역할 (name, responsibilities)

관계 타입:
- HAS_SECTION: Document -> Section
- PARENT_OF: Section -> Section (계층 구조)
- DEFINES: Document -> Term
- ASSIGNS: Document -> Role
- REFERENCES: Document -> Document (상호 참조)
- RELATED_TO: Term -> Term
"""

from neo4j import GraphDatabase
from typing import List, Dict, Optional, Any
import re


class Neo4jGraphStore:
    """Neo4j 그래프 저장소 클래스"""
    
    def __init__(
        self,
        uri: str = "neo4j+s://d00efa60.databases.neo4j.io",
        user: str = "neo4j",
        password: str = "4Qs45al1Coz_NwZDSMcFV9JIFjU7zXPjdKyptQloS6c",
        database: str = "neo4j"
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
    
    def connect(self):
        """Neo4j 연결"""
        if not self.driver:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
        return self
    
    def close(self):
        """연결 종료"""
        if self.driver:
            self.driver.close()
            self.driver = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 'Connected!' AS message")
                record = result.single()
                print(f"✅ Neo4j 연결 성공: {record['message']}")
                return True
        except Exception as e:
            print(f"❌ Neo4j 연결 실패: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 스키마 초기화
    # ═══════════════════════════════════════════════════════════════════════════
    
    def init_schema(self):
        """인덱스 및 제약조건 생성"""
        constraints = [
            "CREATE CONSTRAINT doc_sop_id IF NOT EXISTS FOR (d:Document) REQUIRE d.sop_id IS UNIQUE",
            "CREATE CONSTRAINT term_name IF NOT EXISTS FOR (t:Term) REQUIRE t.name IS UNIQUE",
            "CREATE INDEX doc_title IF NOT EXISTS FOR (d:Document) ON (d.title)",
            "CREATE INDEX section_name IF NOT EXISTS FOR (s:Section) ON (s.name)",
            "CREATE INDEX section_path IF NOT EXISTS FOR (s:Section) ON (s.section_path)",
        ]
        
        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"⚠️ 제약조건 생성 스킵: {e}")
        
        print("✅ 스키마 초기화 완료")
    
    def clear_all(self):
        """모든 노드와 관계 삭제"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("🗑️ 모든 데이터 삭제 완료")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 문서 노드 생성
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_document(
        self,
        sop_id: str,
        title: str,
        version: str = "1.0",
        doc_type: str = "SOP",
        level: int = 2,
        metadata: Dict = None
    ) -> Dict:
        """Document 노드 생성"""
        query = """
        MERGE (d:Document {sop_id: $sop_id})
        SET d.title = $title,
            d.version = $version,
            d.doc_type = $doc_type,
            d.level = $level,
            d.metadata = $metadata,
            d.updated_at = datetime()
        RETURN d
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                sop_id=sop_id,
                title=title,
                version=version,
                doc_type=doc_type,
                level=level,
                metadata=str(metadata or {})
            )
            record = result.single()
            return dict(record["d"]) if record else None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 섹션 노드 생성
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_section(
        self,
        sop_id: str,
        section_id: str,  # "목적", "5.1", "5.1.1" 등
        name: str,
        section_type: str,  # "named_section", "section", "subsection", "level"
        content: str = "",
        section_path: str = None,
        section_path_readable: str = None
    ) -> Dict:
        """Section 노드 생성 및 Document와 연결"""
        query = """
        MATCH (d:Document {sop_id: $sop_id})
        MERGE (s:Section {doc_sop_id: $sop_id, section_id: $section_id})
        SET s.name = $name,
            s.section_type = $section_type,
            s.content = $content,
            s.section_path = $section_path,
            s.section_path_readable = $section_path_readable
        MERGE (d)-[:HAS_SECTION]->(s)
        RETURN s
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                sop_id=sop_id,
                section_id=section_id,
                name=name,
                section_type=section_type,
                content=content[:5000] if content else "",  # 최대 5000자
                section_path=section_path,
                section_path_readable=section_path_readable
            )
            record = result.single()
            return dict(record["s"]) if record else None
    
    def create_section_hierarchy(
        self,
        sop_id: str,
        parent_section_id: str,
        child_section_id: str
    ):
        """섹션 간 계층 관계 생성"""
        query = """
        MATCH (parent:Section {doc_sop_id: $sop_id, section_id: $parent_id})
        MATCH (child:Section {doc_sop_id: $sop_id, section_id: $child_id})
        MERGE (parent)-[:PARENT_OF]->(child)
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                sop_id=sop_id,
                parent_id=parent_section_id,
                child_id=child_section_id
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 용어 노드 생성
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_term(
        self,
        name: str,
        definition: str,
        english_name: str = None,
        sop_id: str = None
    ) -> Dict:
        """Term 노드 생성"""
        query = """
        MERGE (t:Term {name: $name})
        SET t.definition = $definition,
            t.english_name = $english_name
        WITH t
        OPTIONAL MATCH (d:Document {sop_id: $sop_id})
        FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END |
            MERGE (d)-[:DEFINES]->(t)
        )
        RETURN t
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                name=name,
                definition=definition[:2000] if definition else "",
                english_name=english_name,
                sop_id=sop_id
            )
            record = result.single()
            return dict(record["t"]) if record else None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 역할 노드 생성
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_role(
        self,
        name: str,
        responsibilities: str,
        sop_id: str = None
    ) -> Dict:
        """Role 노드 생성"""
        query = """
        MERGE (r:Role {name: $name})
        SET r.responsibilities = $responsibilities
        WITH r
        OPTIONAL MATCH (d:Document {sop_id: $sop_id})
        FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END |
            MERGE (d)-[:ASSIGNS]->(r)
        )
        RETURN r
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                name=name,
                responsibilities=responsibilities[:2000] if responsibilities else "",
                sop_id=sop_id
            )
            record = result.single()
            return dict(record["r"]) if record else None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 문서 간 참조 관계
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_reference(
        self,
        from_sop_id: str,
        to_sop_id: str,
        reference_type: str = "REFERENCES"
    ):
        """문서 간 참조 관계 생성"""
        query = """
        MATCH (from:Document {sop_id: $from_sop_id})
        MATCH (to:Document {sop_id: $to_sop_id})
        MERGE (from)-[r:REFERENCES]->(to)
        SET r.type = $ref_type
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                from_sop_id=from_sop_id,
                to_sop_id=to_sop_id,
                ref_type=reference_type
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 조회 함수
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_document(self, sop_id: str) -> Optional[Dict]:
        """문서 조회"""
        query = """
        MATCH (d:Document {sop_id: $sop_id})
        OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
        RETURN d, collect(s) as sections
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, sop_id=sop_id)
            record = result.single()
            if record:
                return {
                    "document": dict(record["d"]),
                    "sections": [dict(s) for s in record["sections"]]
                }
            return None
    
    def get_all_documents(self) -> List[Dict]:
        """모든 문서 목록"""
        query = """
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
        RETURN d, count(s) as section_count
        ORDER BY d.sop_id
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return [
                {**dict(record["d"]), "section_count": record["section_count"]}
                for record in result
            ]
    
    def search_by_term(self, term: str) -> List[Dict]:
        """용어로 검색"""
        query = """
        MATCH (t:Term)
        WHERE t.name CONTAINS $term OR t.definition CONTAINS $term
        OPTIONAL MATCH (d:Document)-[:DEFINES]->(t)
        RETURN t, collect(d.sop_id) as documents
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, term=term)
            return [
                {
                    "term": dict(record["t"]),
                    "documents": list(record["documents"])
                }
                for record in result
            ]
    
    def search_sections(self, keyword: str, sop_id: str = None) -> List[Dict]:
        """섹션 내용 검색"""
        if sop_id:
            query = """
            MATCH (d:Document {sop_id: $sop_id})-[:HAS_SECTION]->(s:Section)
            WHERE s.name CONTAINS $keyword OR s.content CONTAINS $keyword
            RETURN s, d.sop_id as sop_id
            """
            params = {"keyword": keyword, "sop_id": sop_id}
        else:
            query = """
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
            WHERE s.name CONTAINS $keyword OR s.content CONTAINS $keyword
            RETURN s, d.sop_id as sop_id
            LIMIT 20
            """
            params = {"keyword": keyword}
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return [
                {
                    "section": dict(record["s"]),
                    "sop_id": record["sop_id"]
                }
                for record in result
            ]
    
    def get_document_references(self, sop_id: str) -> Dict:
        """문서 참조 관계 조회"""
        query = """
        MATCH (d:Document {sop_id: $sop_id})
        OPTIONAL MATCH (d)-[:REFERENCES]->(ref:Document)
        OPTIONAL MATCH (cited:Document)-[:REFERENCES]->(d)
        RETURN d, collect(DISTINCT ref.sop_id) as references, collect(DISTINCT cited.sop_id) as cited_by
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, sop_id=sop_id)
            record = result.single()
            if record:
                return {
                    "document": dict(record["d"]),
                    "references": list(record["references"]),
                    "cited_by": list(record["cited_by"])
                }
            return None
    
    def get_section_hierarchy(self, sop_id: str) -> List[Dict]:
        """문서의 섹션 계층 구조"""
        query = """
        MATCH (d:Document {sop_id: $sop_id})-[:HAS_SECTION]->(s:Section)
        OPTIONAL MATCH (s)-[:PARENT_OF]->(child:Section)
        RETURN s, collect(child.section_id) as children
        ORDER BY s.section_path
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, sop_id=sop_id)
            return [
                {
                    "section": dict(record["s"]),
                    "children": list(record["children"])
                }
                for record in result
            ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 그래프 분석
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_graph_stats(self) -> Dict:
        """그래프 통계"""
        query = """
        MATCH (d:Document) WITH count(d) as docs
        MATCH (s:Section) WITH docs, count(s) as sections
        MATCH (t:Term) WITH docs, sections, count(t) as terms
        MATCH (r:Role) WITH docs, sections, terms, count(r) as roles
        MATCH ()-[rel]->() WITH docs, sections, terms, roles, count(rel) as rels
        RETURN docs, sections, terms, roles, rels
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            record = result.single()
            if record:
                return {
                    "documents": record["docs"],
                    "sections": record["sections"],
                    "terms": record["terms"],
                    "roles": record["roles"],
                    "relationships": record["rels"]
                }
            return {}


# ═══════════════════════════════════════════════════════════════════════════
# 문서 파싱 → 그래프 변환
# ═══════════════════════════════════════════════════════════════════════════

def extract_terms_from_text(text: str) -> List[Dict]:
    """정의 섹션에서 용어 추출"""
    terms = []
    
    # 패턴: "용어(English Term): 정의..." 또는 "용어: 정의..."
    patterns = [
        r'^([가-힣A-Za-z\s]+)\s*\(([A-Za-z\s]+)\)\s*[:：]\s*(.+)',  # 용어(English): 정의
        r'^([가-힣]+)\s*[:：]\s*(.+)',  # 용어: 정의 (한글만)
    ]
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        for i, pattern in enumerate(patterns):
            match = re.match(pattern, line)
            if match:
                if i == 0:  # 영문명 포함
                    terms.append({
                        "name": match.group(1).strip(),
                        "english_name": match.group(2).strip(),
                        "definition": match.group(3).strip()
                    })
                else:  # 한글만
                    terms.append({
                        "name": match.group(1).strip(),
                        "english_name": None,
                        "definition": match.group(2).strip()
                    })
                break
    
    return terms


def extract_references_from_text(text: str) -> List[str]:
    """텍스트에서 SOP 참조 추출"""
    # EQ-SOP-00004, SOP-00001 등의 패턴
    pattern = r'[A-Z]{2,}-?SOP-?\d{4,5}'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # 정규화
    normalized = set()
    for m in matches:
        m = m.upper().replace('_', '-')
        if not m.startswith('EQ-'):
            m = 'EQ-' + m
        normalized.add(m)
    
    return list(normalized)


def document_to_graph(
    graph_store: Neo4jGraphStore,
    parsed_doc,  # ParsedDocument
    sop_id: str = None
):
    """ParsedDocument를 Neo4j 그래프로 변환"""
    
    # SOP ID 추출
    sop_id = sop_id or parsed_doc.metadata.get("sop_id") or "UNKNOWN"
    title = parsed_doc.metadata.get("title") or parsed_doc.metadata.get("file_name") or "문서"
    version = parsed_doc.metadata.get("version") or "1.0"
    
    print(f"\n📄 문서 그래프 생성: {sop_id} - {title}")
    
    # 1. Document 노드 생성
    graph_store.create_document(
        sop_id=sop_id,
        title=title,
        version=version,
        doc_type="SOP",
        metadata=parsed_doc.metadata
    )
    
    # 2. 블록 → Section 노드 변환
    section_stack = {}  # 계층 추적용
    
    for block in parsed_doc.blocks:
        meta = block.metadata
        section_id = meta.get("article_num") or meta.get("title") or "intro"
        section_type = meta.get("article_type", "intro")
        section_name = meta.get("title", "")
        section_path = meta.get("section_path")
        section_path_readable = meta.get("section_path_readable")
        
        # Section 노드 생성
        graph_store.create_section(
            sop_id=sop_id,
            section_id=str(section_id),
            name=section_name,
            section_type=section_type,
            content=block.text,
            section_path=section_path,
            section_path_readable=section_path_readable
        )
        
        # 계층 관계 설정
        if section_type == "subsection" and section_stack.get("section"):
            graph_store.create_section_hierarchy(
                sop_id=sop_id,
                parent_section_id=section_stack["section"],
                child_section_id=str(section_id)
            )
        elif section_type == "subsubsection" and section_stack.get("subsection"):
            graph_store.create_section_hierarchy(
                sop_id=sop_id,
                parent_section_id=section_stack["subsection"],
                child_section_id=str(section_id)
            )
        elif section_type == "level" and section_stack.get("named_section"):
            graph_store.create_section_hierarchy(
                sop_id=sop_id,
                parent_section_id=section_stack["named_section"],
                child_section_id=str(section_id)
            )
        
        # 스택 업데이트
        if section_type in ["section", "named_section"]:
            section_stack["section"] = str(section_id)
            section_stack["named_section"] = str(section_id)
            section_stack["subsection"] = None
            section_stack["subsubsection"] = None
        elif section_type in ["subsection", "level"]:
            section_stack["subsection"] = str(section_id)
            section_stack["subsubsection"] = None
        elif section_type == "subsubsection":
            section_stack["subsubsection"] = str(section_id)
        
        # 3. 정의 섹션에서 용어 추출
        if section_name and ("정의" in section_name or "Definition" in section_name):
            terms = extract_terms_from_text(block.text)
            for term in terms:
                graph_store.create_term(
                    name=term["name"],
                    definition=term["definition"],
                    english_name=term.get("english_name"),
                    sop_id=sop_id
                )
            print(f"   📖 용어 {len(terms)}개 추출")
    
    # 4. 문서 내 참조 추출
    all_refs = extract_references_from_text(parsed_doc.text)
    for ref_sop_id in all_refs:
        if ref_sop_id != sop_id:  # 자기 참조 제외
            graph_store.create_reference(sop_id, ref_sop_id)
    
    if all_refs:
        print(f"   🔗 참조 문서: {all_refs}")
    
    print(f"   ✅ 섹션 {len(parsed_doc.blocks)}개 생성 완료")


# ═══════════════════════════════════════════════════════════════════════════
# CLI 테스트
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 연결 테스트
    with Neo4jGraphStore() as graph:
        graph.test_connection()
        
        # 스키마 초기화
        graph.init_schema()
        
        # 통계
        stats = graph.get_graph_stats()
        print(f"\n📊 그래프 통계: {stats}")