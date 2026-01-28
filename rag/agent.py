"""
GMP/SOP 에이전트 모듈 v1.0

🤖 ReAct 에이전트 + LangSmith 추적
- 도구: ChromaDB 검색, Neo4j 그래프 검색
- LLM이 상황에 맞는 도구를 선택해서 실행
- LangSmith로 실행 과정 모니터링
"""

import os
from typing import List, Dict, Optional, Any, Annotated, TypedDict
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# LangSmith 설정 (맨 위에서 설정해야 함)
# ═══════════════════════════════════════════════════════════════════════════

# 환경변수로 설정하거나 직접 입력
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "gmp-sop-agent")

if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    print(f"✅ LangSmith 연동 활성화: {LANGSMITH_PROJECT}")
else:
    print("⚠️ LangSmith API 키 없음 - 로컬 모드로 실행")


# ═══════════════════════════════════════════════════════════════════════════
# 의존성 임포트
# ═══════════════════════════════════════════════════════════════════════════

try:
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatResult, ChatGeneration
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LangChain 패키지 필요: pip install langchain langchain-core")
    LANGCHAIN_AVAILABLE = False

try:
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LangGraph 에이전트 패키지 필요: pip install langgraph")
    LANGGRAPH_AGENT_AVAILABLE = False

# Z.AI SDK
try:
    from zai import ZaiClient
    ZAI_AVAILABLE = True
except ImportError:
    ZAI_AVAILABLE = False
    print(f"⚠️ Z.AI SDK 필요: pip install zai-sdk")


# ═══════════════════════════════════════════════════════════════════════════
# 도구 정의 (Tools)
# ═══════════════════════════════════════════════════════════════════════════

# 전역 변수 (초기화 시 설정)
_vector_store = None
_graph_store = None


def init_agent_tools(vector_store_module, graph_store_instance):
    """에이전트 도구 초기화"""
    global _vector_store, _graph_store
    _vector_store = vector_store_module
    _graph_store = graph_store_instance
    print("✅ 에이전트 도구 초기화 완료")


@tool
def search_sop_documents(query: str) -> str:
    """
    SOP 문서에서 관련 내용을 의미 기반으로 검색합니다.
    일반적인 질문, 절차 확인, 정의 검색 등에 사용하세요.
    
    Args:
        query: 검색할 내용 (예: "품질관리책임자의 역할", "문서 변경 절차")
    
    Returns:
        검색된 문서 내용과 출처
    """
    if not _vector_store:
        return "❌ 벡터 스토어가 초기화되지 않았습니다."
    
    try:
        results = _vector_store.search(
            query=query,
            collection_name="documents",
            model_name="intfloat/multilingual-e5-small",
            n_results=10,
            similarity_threshold=0.3
        )
        
        if not results:
            return f"'{query}'에 대한 검색 결과가 없습니다."
        
        output = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            sop_id = meta.get("sop_id", "N/A")
            section = meta.get("section_path", meta.get("section", ""))
            page = meta.get("page", "")
            similarity = r.get("similarity", 0)
            text = r.get("text", "")[:500]
            
            source = f"[{sop_id}]"
            if section:
                source += f" > {section}"
            if page:
                source += f" (p.{page})"
            
            output.append(f"📄 {source} (유사도: {similarity:.0%})\n{text}...")
        
        return "\n\n---\n\n".join(output)
    
    except Exception as e:
        return f"❌ 검색 오류: {str(e)}"


@tool
def get_document_references(sop_id: str) -> str:
    """
    특정 SOP 문서가 참조하는 다른 문서 목록을 조회합니다.
    문서 간 관계나 연관 규정을 찾을 때 사용하세요.
    
    Args:
        sop_id: SOP 문서 ID (예: "EQ-SOP-00001")
    
    Returns:
        참조하는 문서 목록과 참조받는 문서 목록
    """
    if not _graph_store:
        return "❌ 그래프 스토어가 초기화되지 않았습니다."
    
    try:
        # sop_id 정규화
        sop_id = sop_id.upper().strip()
        if not sop_id.startswith("EQ-"):
            sop_id = "EQ-" + sop_id
        
        refs = _graph_store.get_document_references(sop_id)
        
        if not refs:
            return f"'{sop_id}' 문서를 찾을 수 없거나 참조 관계가 없습니다."
        
        doc = refs.get("document", {})
        references = refs.get("references", [])
        cited_by = refs.get("cited_by", [])
        
        output = [f"📄 문서: {doc.get('sop_id', sop_id)} - {doc.get('title', '')}"]
        
        if references:
            output.append(f"\n🔗 참조하는 문서 ({len(references)}개):")
            for ref in references:
                output.append(f"  → {ref}")
        else:
            output.append("\n🔗 참조하는 문서: 없음")
        
        if cited_by:
            output.append(f"\n📥 참조받는 문서 ({len(cited_by)}개):")
            for ref in cited_by:
                output.append(f"  ← {ref}")
        else:
            output.append("\n📥 참조받는 문서: 없음")
        
        return "\n".join(output)
    
    except Exception as e:
        return f"❌ 참조 조회 오류: {str(e)}"


@tool
def search_sections_by_keyword(keyword: str, sop_id: str = None) -> str:
    """
    키워드로 문서 섹션을 검색합니다.
    특정 용어나 개념이 어느 섹션에 정의되어 있는지 찾을 때 사용하세요.
    
    Args:
        keyword: 검색할 키워드 (예: "책임", "절차", "기록")
        sop_id: 특정 문서로 한정할 경우 SOP ID (선택사항)
    
    Returns:
        키워드가 포함된 섹션 목록
    """
    if not _graph_store:
        return "❌ 그래프 스토어가 초기화되지 않았습니다."
    
    try:
        results = _graph_store.search_sections(keyword, sop_id=sop_id, limit=5)
        
        if not results:
            scope = f" ({sop_id} 내)" if sop_id else ""
            return f"'{keyword}' 키워드가 포함된 섹션을 찾을 수 없습니다{scope}."
        
        output = [f"🔍 '{keyword}' 검색 결과 ({len(results)}개):"]
        
        for sec in results:
            sop = sec.get("doc_sop_id", "N/A")
            name = sec.get("name", "")
            path = sec.get("section_path", "")
            page = sec.get("page", "")
            
            location = f"[{sop}] {name}"
            if path:
                location += f"\n   📍 {path}"
            if page:
                location += f" (p.{page})"
            
            output.append(f"\n📄 {location}")
        
        return "\n".join(output)
    
    except Exception as e:
        return f"❌ 섹션 검색 오류: {str(e)}"


@tool
def get_document_structure(sop_id: str) -> str:
    """
    특정 SOP 문서의 섹션 계층 구조를 조회합니다.
    문서의 목차나 구성을 파악할 때 사용하세요.
    
    Args:
        sop_id: SOP 문서 ID (예: "EQ-SOP-00001")
    
    Returns:
        문서의 섹션 계층 구조
    """
    if not _graph_store:
        return "❌ 그래프 스토어가 초기화되지 않았습니다."
    
    try:
        sop_id = sop_id.upper().strip()
        if not sop_id.startswith("EQ-"):
            sop_id = "EQ-" + sop_id
        
        hierarchy = _graph_store.get_section_hierarchy(sop_id)
        
        if not hierarchy:
            return f"'{sop_id}' 문서의 구조를 찾을 수 없습니다."
        
        output = [f"📋 {sop_id} 문서 구조:"]
        
        for item in hierarchy[:15]:  # 상위 15개만
            sec = item.get("section", {})
            name = sec.get("name", "")
            sec_type = sec.get("section_type", "")
            children = item.get("children", [])
            
            # 들여쓰기
            indent = ""
            if sec_type == "subsection":
                indent = "  "
            elif sec_type == "subsubsection":
                indent = "    "
            
            child_info = f" ({len(children)}개 하위)" if children else ""
            output.append(f"{indent}• {name}{child_info}")
        
        if len(hierarchy) > 15:
            output.append(f"  ... 외 {len(hierarchy) - 15}개 섹션")
        
        return "\n".join(output)
    
    except Exception as e:
        return f"❌ 구조 조회 오류: {str(e)}"


@tool
def list_all_documents() -> str:
    """
    시스템에 등록된 모든 SOP 문서 목록을 조회합니다.
    어떤 문서가 있는지 파악할 때 사용하세요.
    
    Returns:
        등록된 SOP 문서 목록
    """
    if not _graph_store:
        return "❌ 그래프 스토어가 초기화되지 않았습니다."
    
    try:
        docs = _graph_store.get_all_documents()
        
        if not docs:
            return "등록된 문서가 없습니다."
        
        output = [f"📚 등록된 SOP 문서 ({len(docs)}개):"]
        
        for doc in docs:
            sop_id = doc.get("sop_id", "N/A")
            title = doc.get("title", "")
            sections = doc.get("section_count", 0)
            output.append(f"  • {sop_id}: {title} ({sections}개 섹션)")
        
        return "\n".join(output)
    
    except Exception as e:
        return f"❌ 문서 목록 조회 오류: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
# 에이전트 생성
# ═══════════════════════════════════════════════════════════════════════════

# 도구 리스트
AGENT_TOOLS = [
    search_sop_documents,
    get_document_references,
    search_sections_by_keyword,
    get_document_structure,
    list_all_documents,
]


# 시스템 프롬프트
AGENT_SYSTEM_PROMPT = """당신은 GMP(의약품 제조 및 품질관리) 규정 전문가 AI 에이전트입니다.

## 역할
사용자의 질문에 대해 SOP(표준작업절차서) 문서를 검색하고 정확한 답변을 제공합니다.

## 사용 가능한 도구
1. **search_sop_documents**: SOP 문서 내용 검색 (의미 기반)
2. **get_document_references**: 문서 간 참조 관계 조회
3. **search_sections_by_keyword**: 키워드로 섹션 검색
4. **get_document_structure**: 문서 목차/구조 조회
5. **list_all_documents**: 전체 문서 목록 조회

## 답변 원칙
1. 반드시 도구를 사용해서 검색한 후 답변하세요.
2. 출처(SOP ID, 섹션)를 명확히 밝히세요.
3. 검색 결과가 없으면 솔직히 말하세요.
4. 추측하지 말고 문서 내용만 기반으로 답변하세요.

## 도구 선택 가이드
- 일반적인 질문 → search_sop_documents
- "참조하는 문서", "관련 규정" → get_document_references
- 특정 용어/키워드 위치 → search_sections_by_keyword
- 문서 구성/목차 → get_document_structure
- 어떤 문서가 있는지 → list_all_documents
"""


class AgentState(TypedDict):
    """에이전트 상태"""
    messages: List[Any]
    

# 메모리 (대화 히스토리 유지)
_memory_saver = None
_agent = None


def create_agent(model_name: str = "glm-4.7-flash"):
    """에이전트 생성 (Z.AI 기반)"""
    global _agent, _memory_saver
    
    if not ZAI_AVAILABLE:
        raise ImportError("Z.AI SDK가 필요합니다: pip install zai-sdk")
    
    # Z.AI 클라이언트 초기화
    api_key = os.getenv("ZAI_API_KEY", "")
    if not api_key:
        raise ValueError("ZAI_API_KEY 환경변수가 설정되지 않았습니다")
    
    _agent = {
        "model": model_name,
        "api_key": api_key,
        "client": ZaiClient(api_key=api_key)
    }
    
    _memory_saver = MemorySaver() if LANGGRAPH_AGENT_AVAILABLE else {}
    
    print(f"✅ Z.AI 에이전트 생성 완료: {model_name}")
    return _agent


def run_agent(
    query: str,
    session_id: str = "default",
    model_name: str = "glm-4.7-flash"
) -> Dict[str, Any]:
    """
    Z.AI 에이전트 실행 (ReAct 스타일)
    
    Args:
        query: 사용자 질문
        session_id: 세션 ID
        model_name: Z.AI 모델명
    
    Returns:
        {
            "answer": "답변 텍스트",
            "tool_calls": [...],
            "session_id": "세션ID"
        }
    """
    global _agent
    
    # 에이전트가 없으면 생성
    if _agent is None:
        create_agent(model_name)
    
    client = _agent["client"]
    tool_calls = []
    context_parts = []
    
    print(f"🔄 Z.AI 에이전트 실행 중... ({model_name})")
    
    try:
        # 1. 먼저 LLM에게 도구 선택을 요청
        tool_selection_prompt = f"""{AGENT_SYSTEM_PROMPT}

사용자 질문: {query}

위 질문에 답하려면 어떤 도구를 사용해야 하는지 판단하세요.
도구 이름만 짧게 응답하세요:
- search_sop_documents: 일반적인 내용 검색
- get_document_references: 문서 간 참조 관계
- search_sections_by_keyword: 키워드로 섹션 검색
- get_document_structure: 문서 목차/구조
- list_all_documents: 전체 문서 목록

응답 형식: [도구이름]
예: search_sop_documents"""

        tool_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": tool_selection_prompt}],
            max_tokens=100,
            temperature=0.1,
        )
        
        tool_msg = tool_response.choices[0].message
        selected_tool = getattr(tool_msg, 'content', "") or ""
        
        # '생각' 모드로 인해 content가 비어있을 경우 대응
        if not selected_tool:
            reasoning = getattr(tool_msg, 'reasoning_content', "").lower()
            selected_tool = "search_sop_documents" if "search" in reasoning or "검색" in reasoning else "search_sop_documents"
        
        selected_tool = selected_tool.strip().lower()
        print(f"🔧 선택된 도구: {selected_tool}")
        
        # 2. 도구 실행
        tool_result = ""
        
        if "search_sop_documents" in selected_tool or "검색" in selected_tool:
            tool_result = search_sop_documents.invoke(query)
            tool_calls.append({"tool": "search_sop_documents", "input": query, "output": tool_result[:300]})
            
        elif "references" in selected_tool or "참조" in selected_tool:
            import re
            sop_match = re.search(r'(EQ-?SOP-?\d+)', query, re.IGNORECASE)
            if sop_match:
                sop_id = sop_match.group(1).upper()
                tool_result = get_document_references.invoke(sop_id)
                tool_calls.append({"tool": "get_document_references", "input": sop_id, "output": tool_result[:300]})
            else:
                tool_result = search_sop_documents.invoke(query)
                tool_calls.append({"tool": "search_sop_documents", "input": query, "output": tool_result[:300]})
                
        elif "keyword" in selected_tool or "키워드" in selected_tool:
            # 주요 키워드 추출
            keywords = query.replace("?", "").replace("은", "").replace("는", "").split()[-1]
            tool_result = search_sections_by_keyword.invoke(keywords)
            tool_calls.append({"tool": "search_sections_by_keyword", "input": keywords, "output": tool_result[:300]})
            
        elif "structure" in selected_tool or "구조" in selected_tool or "목차" in selected_tool:
            import re
            sop_match = re.search(r'(EQ-?SOP-?\d+)', query, re.IGNORECASE)
            if sop_match:
                sop_id = sop_match.group(1).upper()
                tool_result = get_document_structure.invoke(sop_id)
                tool_calls.append({"tool": "get_document_structure", "input": sop_id, "output": tool_result[:300]})
            else:
                tool_result = list_all_documents.invoke({})
                tool_calls.append({"tool": "list_all_documents", "input": "", "output": tool_result[:300]})
                
        elif "list" in selected_tool or "목록" in selected_tool:
            tool_result = list_all_documents.invoke({})
            tool_calls.append({"tool": "list_all_documents", "input": "", "output": tool_result[:300]})
            
        else:
            # 기본: 검색
            tool_result = search_sop_documents.invoke(query)
            tool_calls.append({"tool": "search_sop_documents", "input": query, "output": tool_result[:300]})
        
        print(f"📄 도구 결과 길이: {len(tool_result)} 글자")
        
        # 3. 최종 답변 생성
        final_prompt = f"""{AGENT_SYSTEM_PROMPT}

[검색 결과]
{tool_result}

[사용자 질문]
{query}

위 검색 결과를 바탕으로 질문에 정확히 답변하세요.
반드시 출처(SOP ID, 섹션)를 명시하세요."""

        final_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=2048,  # 토큰 상향
            temperature=0.7,
        )
        
        msg_obj = final_response.choices[0].message
        answer = getattr(msg_obj, 'content', "") or ""
        reasoning = getattr(msg_obj, 'reasoning_content', "") or ""
        
        if not answer and reasoning:
            answer = f"[분석 내용]\n{reasoning}\n\n⚠️ 답변 생성 중 토큰 제한으로 중단되었습니다."
            
        print(f"✅ 최종 답변 길이: {len(answer)} 글자")
        
        return {
            "answer": answer,
            "tool_calls": tool_calls,
            "session_id": session_id,
            "success": True
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "answer": f"Z.AI 에이전트 실행 오류: {str(e)}",
            "tool_calls": [],
            "session_id": session_id,
            "success": False
        }


# ═══════════════════════════════════════════════════════════════════════════
# 간단한 폴백 에이전트 (LangGraph 없이)
# ═══════════════════════════════════════════════════════════════════════════

def run_simple_agent(
    query: str,
    vector_store_module,
    graph_store_instance,
    llm_model: str = "qwen2.5:3b"
) -> Dict[str, Any]:
    """
    간단한 규칙 기반 에이전트 (LangGraph 없이 동작)
    
    키워드 기반으로 도구 선택
    """
    from rag.llm import get_llm_response
    
    tool_calls = []
    context_parts = []
    
    query_lower = query.lower()
    
    # 1. 문서 목록 질문
    if any(kw in query_lower for kw in ["문서 목록", "어떤 문서", "등록된 문서", "sop 목록"]):
        try:
            docs = graph_store_instance.get_all_documents()
            result = "\n".join([f"• {d['sop_id']}: {d.get('title', '')}" for d in docs])
            context_parts.append(f"📚 등록된 문서:\n{result}")
            tool_calls.append({"tool": "list_all_documents", "result": result[:200]})
        except Exception as e:
            context_parts.append(f"문서 목록 조회 실패: {e}")
    
    # 2. 참조 관계 질문
    elif any(kw in query_lower for kw in ["참조", "관련 문서", "연관", "관계"]):
        import re
        sop_match = re.search(r'(EQ-?SOP-?\d+)', query, re.IGNORECASE)
        if sop_match:
            sop_id = sop_match.group(1).upper().replace("SOP", "-SOP-").replace("--", "-")
            try:
                refs = graph_store_instance.get_document_references(sop_id)
                if refs:
                    context_parts.append(f"📄 {sop_id} 참조 관계:\n- 참조: {refs.get('references', [])}\n- 피참조: {refs.get('cited_by', [])}")
                    tool_calls.append({"tool": "get_document_references", "input": sop_id})
            except:
                pass
    
    # 3. 문서 구조 질문
    elif any(kw in query_lower for kw in ["목차", "구조", "구성", "섹션"]):
        import re
        sop_match = re.search(r'(EQ-?SOP-?\d+)', query, re.IGNORECASE)
        if sop_match:
            sop_id = sop_match.group(1).upper().replace("SOP", "-SOP-").replace("--", "-")
            try:
                hierarchy = graph_store_instance.get_section_hierarchy(sop_id)
                if hierarchy:
                    sections = [h['section']['name'] for h in hierarchy[:10]]
                    context_parts.append(f"📋 {sop_id} 구조:\n" + "\n".join([f"• {s}" for s in sections]))
                    tool_calls.append({"tool": "get_document_structure", "input": sop_id})
            except:
                pass
    
    # 4. 기본: 벡터 검색
    if not context_parts:
        try:
            results = vector_store_module.search(
                query=query,
                collection_name="documents",
                model_name="intfloat/multilingual-e5-small",
                n_results=3
            )
            for r in results:
                meta = r.get("metadata", {})
                sop = meta.get("sop_id", "")
                path = meta.get("section_path", "")
                text = r.get("text", "")[:400]
                context_parts.append(f"[{sop}] {path}\n{text}")
            tool_calls.append({"tool": "search_sop_documents", "input": query})
        except Exception as e:
            context_parts.append(f"검색 실패: {e}")
    
    # LLM 답변 생성
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""당신은 GMP/SOP 규정 전문가입니다. 아래 검색 결과를 바탕으로 질문에 답변하세요.

[검색 결과]
{context}

[질문]
{query}

[답변] (출처를 명시하세요):"""
    
    answer = get_llm_response(prompt, llm_model=llm_model, max_tokens=512)
    
    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "context": context[:500],
        "success": True
    }


# ═══════════════════════════════════════════════════════════════════════════
# 테스트
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 GMP/SOP 에이전트 테스트")
    print("="*60)
    
    # 테스트 (도구 초기화 없이 구조만 확인)
    print("\n📋 등록된 도구:")
    for tool in AGENT_TOOLS:
        print(f"  • {tool.name}: {tool.description[:50]}...")
    
    print("\n✅ 에이전트 모듈 로드 완료!")
    print("   사용법: from agent import run_agent, init_agent_tools")