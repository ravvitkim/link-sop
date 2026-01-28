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
import operator

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
반드시 아래의 **구조화된 답변 형식**을 준수하여 한국어로 답변하세요.

## 답변 형식 (필수)
1. **검증 의견**: 질문에 대한 결론과 전문적인 분석 내용을 상세히 서술합니다.
2. **검증 근거 및 출처**: 
   - 각 근거별로 숫자를 매겨 제목(`**1. 제목**`)을 작성합니다.
   - 해당 근거에 대한 상세 설명을 불렛 포인트로 작성합니다.
   - 마지막에는 반드시 `**[출처]** [SOP ID] > [장/절 제목] > [상세 문구 인용] (p.페이지)` 형식을 지킵니다.

## 핵심 규칙
1. **상세성**: 단순히 짧게 대답하지 말고, 규정의 맥락을 충분히 설명하세요.
2. **근거 중심**: 모든 주장은 반드시 검색된 SOP의 구체적인 조항에 기반해야 합니다.
3. **객관성**: 추측을 배제하고 문서에 명시된 사실만을 전달하세요.
"""


class AgentState(TypedDict):
    """멀티 에이전트 공유 상태"""
    messages: Annotated[List[Any], operator.add]
    query: str
    next_node: str
    search_results: List[Dict]
    verification: str
    answer: str
    tool_calls: List[Dict]
    session_id: str
    model_name: str
    

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


# ═══════════════════════════════════════════════════════════════════════════
# 멀티 에이전트 노드 구현
# ═══════════════════════════════════════════════════════════════════════════

def orchestrator_node(state: AgentState):
    """의도 분석 및 작업 분배 노드"""
    query = state["query"]
    model_name = state["model_name"]
    client = _agent["client"]
    
    print(f"🎯 [Orchestrator] 의도 분석 중: {query}")
    
    prompt = f"""당신은 GMP 규정 시스템의 오케스트레이터입니다. 사용자의 질문을 분석하여 다음 단계(next_node)를 결정하세요.
    - search_agent: 규정 검색이 필요한 경우
    - verifier_agent: 특정 상황이나 행위가 규정에 맞는지 검착/검증이 필요한 경우 (검색 결과가 이미 있다면)
    - list_agent: 문서 목록 조회가 필요한 경우
    - writer_agent: 이미 충분한 정보가 있어 답변을 생성하면 되는 경우

    현재 질문: {query}
    응답 형식: [노드이름]
    예: search_agent"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.1
    )
    
    next_node = response.choices[0].message.content.strip().lower()
    if "search" in next_node: next_node = "search_agent"
    elif "verify" in next_node or "검증" in next_node: next_node = "verifier_agent"
    elif "list" in next_node: next_node = "list_agent"
    else: next_node = "search_agent" # 기본값

    return {"next_node": next_node}


def search_agent_node(state: AgentState):
    """검색 전문 에이전트 노드"""
    query = state["query"]
    print(f"🔍 [SearchAgent] 규정 검색 시도: {query}")
    
    # 벡터 검색 및 그래프 검색 통합 사용
    results = search_sop_documents.invoke(query)
    
    # 질문에 '맞는지', '적절한지' 등의 키워드가 있으면 검증 노드로 유도
    should_verify = any(kw in query for kw in ["맞는지", "적절한지", "가능한지", "위반", "검증", "의견"])
    
    return {
        "search_results": [{"content": results}],
        "next_node": "verifier_agent" if should_verify else "writer_agent"
    }


def verifier_agent_node(state: AgentState):
    """규정 검증 및 최종 답변 생성 노드 (통합)"""
    query = state["query"]
    search_results = state.get("search_results", [])
    context = "\n".join([r["content"] for r in search_results]) if search_results else "참조할 규정 검색 결과가 없습니다."
    
    model_name = state["model_name"]
    client = _agent["client"]
    
    print(f"⚖️ [VerifierAgent] 규정 검증 및 답변 생성 중")
    
    prompt = f"""{AGENT_SYSTEM_PROMPT}

[규정 컨텍스트]
{context}

[사용자 질문]
{query}

위 규정을 면밀히 분석하여, 이미지와 같이 **상세하고 전문적인 검증 결과**를 작성하세요. 
규정 조항의 문구를 직접 인용하며 설득력 있는 답변을 제공해야 합니다.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500,
        temperature=0.3
    )
    
    msg_obj = response.choices[0].message
    content = getattr(msg_obj, 'content', "") or ""
    reasoning = getattr(msg_obj, 'reasoning_content', "") or ""
    
    final_answer = content if content else (f"[분석 내용]\n{reasoning}" if reasoning else "[오류] 답변을 생성하지 못했습니다.")
    
    return {"answer": final_answer, "reasoning": reasoning, "next_node": "end"}


def list_agent_node(state: AgentState):
    """문서 목록 조회 전문 에이전트 노드"""
    print(f"📚 [ListAgent] 전체 문서 목록 조회 중")
    docs_info = list_all_documents.invoke({})
    return {"search_results": [{"content": docs_info}], "next_node": "writer_agent"}


def writer_agent_node(state: AgentState):
    """일반 답변 생성 노드 (통합)"""
    query = state["query"]
    search_results = state.get("search_results", [])
    context = "\n".join([r["content"] for r in search_results]) if search_results else "검색 결과 없음"
    model_name = state["model_name"]
    client = _agent["client"]
    
    print(f"✍️ [WriterAgent] 일반 답변 작성 중")
    
    prompt = f"""{AGENT_SYSTEM_PROMPT}

[참고 규정]
{context}

[사용자 질문]
{query}

위 규정을 바탕으로 질문에 대해 **충분한 설명과 구체적인 출처**를 포함하여 답변을 작성하세요.
이미지의 형식(`**검증 의견**`, `**검증 근거 및 출처**`)을 엄격히 준수하세요.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500,
        temperature=0.7
    )
    
    msg_obj = response.choices[0].message
    content = getattr(msg_obj, 'content', "") or ""
    reasoning = getattr(msg_obj, 'reasoning_content', "") or ""
    
    final_answer = content if content else (f"[분석 내용]\n{reasoning}" if reasoning else "[오류] 답변을 생성하지 못했습니다.")

    return {"answer": final_answer, "reasoning": reasoning, "next_node": "end"}


def run_agent(
    query: str,
    session_id: str = "default",
    model_name: str = "glm-4.7-flash"
) -> Dict[str, Any]:
    """멀티 에이전트 워크플로우 실행 (노드 기반 시뮬레이션)"""
    global _agent
    
    # 에이전트가 없으면 생성
    if _agent is None:
        create_agent(model_name)
    
    # 초기 상태
    state: AgentState = {
        "messages": [],
        "query": query,
        "next_node": "orchestrator",
        "search_results": [],
        "verification": "",
        "answer": "",
        "tool_calls": [],
        "session_id": session_id,
        "model_name": model_name
    }
    
    try:
        # 1. Orchestrator
        res = orchestrator_node(state)
        state.update(res)
        
        # 2. Search (필요 시)
        if state["next_node"] == "search_agent":
            res = search_agent_node(state)
            state.update(res)
            state["tool_calls"].append({"tool": "search_sop_documents", "input": query})
        
        # 3. List (필요 시)
        elif state["next_node"] == "list_agent":
            res = list_agent_node(state)
            state.update(res)
            state["tool_calls"].append({"tool": "list_all_documents", "input": ""})
            # ListAgent에서 바로 결과가 나오므로 Writer 호출 불필요 (현재 구현상)
            state["answer"] = res.get("search_results", [{}])[0].get("content", "")
            state["next_node"] = "end"

        # 4. Verifier / Writer (분기하여 하나만 실행)
        if state["next_node"] == "verifier_agent":
            res = verifier_agent_node(state)
            state.update(res)
            state["tool_calls"].append({"tool": "verifier_agent", "input": "compliance_check"})
        elif state["next_node"] == "writer_agent":
            res = writer_agent_node(state)
            state.update(res)
        
        return {
            "answer": state["answer"],
            "tool_calls": state["tool_calls"],
            "session_id": session_id,
            "success": True,
            "reasoning": state.get("reasoning", "")
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "answer": f"❌ 멀티 에이전트 실행 중 오류: {str(e)}",
            "tool_calls": state.get("tool_calls", []),
            "session_id": session_id,
            "success": False
        }


# ═══════════════════════════════════════════════════════════════════════════
# 테스트 및 사용법
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 멀티 에이전트 시스템 테스트 (v11.0)")
    print("="*60)
    
    # 에이전트 모듈 및 노드 로드 상태 확인
    print("\n✅ 에이전트 모듈 및 노드 로드 완료!")
    print(f"   - 오케스트레이터: {orchestrator_node.__name__}")
    print(f"   - 검색 에이전트: {search_agent_node.__name__}")
    print(f"   - 검증 에이전트: {verifier_agent_node.__name__}")
    print(f"   - 라이터 에이전트: {writer_agent_node.__name__}")
    
    # 사용법 안내
    print("\n💡 사용법:")
    print("   from rag.agent import run_agent, init_agent_tools")
    print("   init_agent_tools(vector_store, graph_store)")
    print("   # 예시 호출:")
    print("   # result = run_agent(\"품질관리책임자의 역할이 뭐야? 규정에 맞는지 검증해줘.\")")
    print("   # print(result['answer'])")
    print("="*60)
