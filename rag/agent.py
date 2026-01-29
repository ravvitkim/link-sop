"""
SOP 에이전트 모듈 v12.8 (Hybrid Deep Search)

🤖 하이브리드 ReAct 멀티 에이전트
- 하이브리드 검색: Vector(의미) + SQL(키워드) + Graph(참조) 결합
- 검색 누락 방지: 벡터 검색 임계값 최적화(0.20) 및 SQL 기반 전역 폴백 검색
- 전문 답변 보장: 내부 규정 기반 상세 검증 보고서 레이아웃 고정
"""

import os
from typing import List, Dict, Optional, Any, Annotated, TypedDict
from datetime import datetime
import operator
import re

# ═══════════════════════════════════════════════════════════════════════════
# 임포트 및 설정
# ═══════════════════════════════════════════════════════════════════════════

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "gmp-sop-agent")

try:
    from zai import ZaiClient
    ZAI_AVAILABLE = True
except ImportError:
    ZAI_AVAILABLE = False

try:
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AGENT_AVAILABLE = True
except ImportError:
    LANGGRAPH_AGENT_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# 하이브리드 도구 정의
# ═══════════════════════════════════════════════════════════════════════════

_vector_store = None
_graph_store = None
_sql_store = None

def init_agent_tools(vector_store_module, graph_store_instance, sql_store_instance=None):
    global _vector_store, _graph_store, _sql_store
    _vector_store = vector_store_module
    _graph_store = graph_store_instance
    _sql_store = sql_store_instance

@tool
def hybrid_search_sop(query: str, embedding_model: str = "jhgan/ko-sroberta-multitask") -> str:
    """SOP 문서 검색 (Vector + SQL 하이브리드)"""
    if not _vector_store: return "❌ 벡터 스토어 미설정"
    
    combined_results = []
    seen_ids = set()
    
    # 1. 벡터 검색 (의미 중심, 임계값 하향 조정)
    try:
        results = _vector_store.search(
            query=query, 
            collection_name="documents", 
            n_results=10,
            model_name=embedding_model, # 모델명 명시적 전달
            similarity_threshold=0.15 # 누락 방지를 위해 더 하향 (0.20 -> 0.15)
        )
        for r in results:
            meta = r.get("metadata", {})
            text = r.get("text", "")
            doc_id = meta.get('sop_id', 'N/A')
            
            # 출처 정보 가공
            source = f"[{doc_id}]"
            if meta.get('section_path'): source += f" > {meta.get('section_path')}"
            source += f" (p.{meta.get('page', 'N/A')})"
            
            content = f"📄 출처: {source} (신뢰도: {r.get('confidence', 'N/A')})\n{text}"
            combined_results.append(content)
            seen_ids.add(doc_id)
    except Exception: pass

    # 2. SQL 키워드 폴백 검색 (결과가 적거나 신뢰도가 낮을 때)
    if len(combined_results) < 5 and _sql_store:
        try:
            # 질문에서 의미 있는 키워드 추출 (간이)
            raw_keywords = re.findall(r'[가-힣A-Z0-9]{2,}', query)
            unique_keywords = list(set([k for k in raw_keywords if len(k) > 1]))[:5]
            
            for kw in unique_keywords:
                docs = _sql_store.list_documents()
                for doc in docs:
                    sop_id = doc.get("sop_id", "")
                    title = doc.get("title", "")
                    if kw.upper() in sop_id.upper() or kw in title:
                        if sop_id not in seen_ids:
                            full_doc = _sql_store.get_document_by_id(sop_id)
                            if full_doc:
                                text = full_doc.get("markdown_content", "")[:4000]
                                combined_results.append(f"📄 [SQL 전역 검색] 출처: {sop_id} (키워드: {kw})\n{text}")
                                seen_ids.add(sop_id)
        except Exception: pass
        
    return "\n\n".join(combined_results)

@tool
def get_document_references(sop_id: str) -> str:
    """참조 문서 조회 (Graph)"""
    if not _graph_store: return ""
    try:
        refs = _graph_store.get_document_references(sop_id.upper())
        if not refs: return ""
        doc = refs.get("document", {})
        return f"📄 {doc.get('sop_id')} 참조문서: {', '.join(refs.get('references', []))}"
    except Exception: return ""

AGENT_TOOLS = [hybrid_search_sop, get_document_references]

# ═══════════════════════════════════════════════════════════════════════════
# 에이전트 시스템 프롬프트
# ═══════════════════════════════════════════════════════════════════════════

AGENT_SYSTEM_PROMPT = """당신은 회사 내부 GMP 규정(SOP) 전문가입니다.
제공된 규정 데이터를 기반으로 사용자의 상황을 진단하고 전문적인 '검증 보고서'를 작성하세요.

## 🎯 핵심 원칙: 능동적 추론 (Active Reasoning)
1. **명시적 규정 우선**: 문서에 직설적으로 "금지" 또는 "허용"이 명시된 경우 이를 최우선으로 합니다.
2. **논리적 추론 (Deduction)**: 구체적인 허용 여부가 없더라도, 상위 규정(예: "모든 OOS는 조사가 선행되어야 한다")을 바탕으로 하위 상황(예: "따라서 즉시 재시험은 불가하다")을 논리적으로 추론하여 결론을 도출하세요. "규정이 없어서 모른다"는 답변은 지양하고, "규정의 취지상 ~해야 한다"는 방향을 제시하세요.
3. **증거 기반 (Evidence-based)**: 추론의 근거는 반드시 제공된 SOP 텍스트의 특정 조항이어야 합니다.

## 📋 답변 구조 (필수)
### **1. 검증 의견**
- [**핵심 결론**]: 결론을 한 문장으로 명확히 제시 (예: 재시험 불가/조건부 허용 등)
- [**상세 분석**]: 규정의 취지와 사용자 상황을 대조하여 논리적으로 설명

### **2. 검증 근거 및 출처**
- 각 근거별 번호와 제목
- 정확한 출처 표기 필수: `**[출처]** [SOP ID] > [제목] > [상세 문구 인용] (p.페이지)`

### **3. 조치 권고 및 제언**
- 발견된 규정의 공백을 메우기 위해 사용자가 즉시 취해야 할 행동 가이드
- 관련하여 추가로 확인해야 할 하위 지침서(SOP) 명칭 제안
"""

class AgentState(TypedDict):
    query: str
    model_name: str
    embedding_model: str # 추가
    search_results: List[Dict]
    answer: str
    reasoning: str
    queries: List[str]

_agent = None

def create_agent(model_name: str = "glm-4.7-flash"):
    global _agent
    api_key = os.getenv("ZAI_API_KEY")
    if not api_key: raise ValueError("ZAI_API_KEY 설정 필요")
    _agent = {"model": model_name, "client": ZaiClient(api_key=api_key)}
    return _agent

def query_expansion_node(state: AgentState):
    """멀티 쿼리 확장"""
    client = _agent["client"]
    print(f"🧠 [Agent] 검색어 확장 및 하이브리드 검색 준비 중...")
    prompt = f"질문에서 핵심 기술 용어 및 규정 명칭 3개를 추출하세요. (쉼표 구분)\n질문: {state['query']}"
    try:
        res = client.chat.completions.create(model=state["model_name"], messages=[{"role": "user", "content": prompt}], max_tokens=100)
        expanded = [q.strip() for q in res.choices[0].message.content.split(',') if q.strip()]
    except Exception: expanded = []
    
    q_list = [state["query"]] + expanded
    return {"queries": q_list[:4]}

def verifier_agent_node(state: AgentState):
    print(f"⚖️ [VerifierAgent] 하이브리드 데이터 취합 및 보고서 생성 중")
    
    all_context = []
    seen_content = set()
    
    # 멀티 쿼리별 하이브리드 검색 수행
    for q in state.get("queries", [state["query"]]):
        print(f"🔍 [HybridSearch] '{q}' 실행 중 (Model: {state.get('embedding_model')})")
        res = hybrid_search_sop.invoke({"query": q, "embedding_model": state.get("embedding_model")})
        if res and "❌" not in res:
            for snippet in res.split("\n\n"):
                if snippet and snippet not in seen_content:
                    all_context.append(snippet)
                    seen_content.add(snippet)
                    
    if not all_context:
        return {
            "answer": "❌ 모든 데이터베이스(Vector, SQL)를 검색했으나 관련 규정을 찾지 못했습니다. SOP 제목이나 핵심 키워드(예: OOS, 재시험 등)를 포함하여 다시 질문해 주세요.",
            "reasoning": "Zero results from 3-tier hybrid search."
        }

    context = "\n\n".join(all_context[:12])
    
    # 능동적 추론을 돕기 위해 질문의 의도를 다시 한 번 강조
    prompt = f"""{AGENT_SYSTEM_PROMPT}

[검색된 내부 SOP 데이터]
{context}

[사용자 상황 및 의도]
"{state['query']}"에 대해 단순히 규정 유무만 따지지 말고, 
검색된 규정의 '취지'와 '책임' 조항을 근거로 시험자가 즉시 취해야 할 행동의 적절성을 판정하세요.
특히 "OO 절차에 따른다"는 문구가 있다면, 해당 절차 없이 독단적으로 행동하는 것이 규정 위반임을 강조하세요.
"""
    
    try:
        res = _agent["client"].chat.completions.create(
            model=state["model_name"], 
            messages=[{"role": "user", "content": prompt}], 
            max_tokens=4000,
            temperature=0.1
        )
        msg = res.choices[0].message
        return {
            "answer": getattr(msg, 'content', "") or getattr(msg, 'reasoning_content', ""),
            "reasoning": getattr(msg, 'reasoning_content', "")
        }
    except Exception as e:
        return {"answer": f"❌ 오류 발생: {e}", "reasoning": str(e)}

def run_agent(query: str, session_id: str = "default", model_name: str = "glm-4.7-flash", embedding_model: str = "jhgan/ko-sroberta-multitask"):
    if not _agent: create_agent(model_name)
    state = {"query": query, "model_name": model_name, "embedding_model": embedding_model}
    
    # 실행 파이프라인
    expanded = query_expansion_node(state)
    state.update(expanded)
    
    final = verifier_agent_node(state)
    
    return {
        "answer": final["answer"], 
        "reasoning": final.get("reasoning", ""),
        "success": True,
        "tool_calls": [{"tool": "hybrid_search", "queries": state.get("queries")}]
    }
