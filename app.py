import streamlit as st
import pandas as pd
import os
import time

# --- Streamlit 설정 ---
st.set_page_config(page_title="Biz-Insight 3.0", page_icon="✨", layout="wide")

DEMO_THESIS = (
    "대형주는 이미 해석된 정보가 많습니다. Biz-Insight는 공시, 뉴스, 실적, "
    "산업 데이터, 직원 리뷰가 흩어져 있는 코스닥 성장기업을 연결해 투자자가 "
    "성장 논리와 리스크를 빠르게 이해하도록 돕습니다."
)


def _configured_data_dir() -> str:
    return os.path.abspath(os.getenv("BIZINSIGHT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data")))


def _secret_value(name: str) -> str | None:
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def configure_from_secrets() -> None:
    data_dir = _secret_value("BIZINSIGHT_DATA_DIR")
    if data_dir and not os.getenv("BIZINSIGHT_DATA_DIR"):
        os.environ["BIZINSIGHT_DATA_DIR"] = str(data_dir)


def require_password() -> None:
    expected = os.getenv("BIZINSIGHT_APP_PASSWORD") or _secret_value("APP_PASSWORD")
    if not expected:
        return
    if st.session_state.get("authenticated"):
        return

    st.title("Biz-Insight Demo")
    password = st.text_input("비밀번호", type="password")
    if st.button("입장"):
        if password == expected:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")
    st.stop()


def display_value(row: pd.Series, key: str, default: str = "N/A") -> str:
    value = row.get(key, default)
    if pd.isna(value) or value == "":
        return default
    return str(value)


def render_execution_trace(trace: dict) -> None:
    agents = trace.get("agents") or []
    tool_calls = trace.get("tool_calls") or []
    sources = trace.get("data_sources") or []

    st.markdown("### 실행 흐름")

    graph_lines = [
        "digraph BizInsight {",
        "rankdir=LR;",
        'node [shape=box, style="rounded,filled", color="#D8DEE9", fillcolor="#F8FAFC", fontname="Arial"];',
        '"User Request" -> "Supervisor";',
    ]
    agent_order = ["supervisor", "researcher", "analyst", "reviewer", "synthesis"]
    active = [agent for agent in agent_order if agent in agents]
    for left, right in zip(active, active[1:]):
        graph_lines.append(f'"{left.title()}" -> "{right.title()}";')
    for call in tool_calls[:12]:
        tool = call.get("tool", "tool")
        graph_lines.append(f'"Researcher" -> "{tool}";')
    graph_lines.append("}")
    st.graphviz_chart("\n".join(graph_lines), width="stretch")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("에이전트", len(agents))
    col_b.metric("도구 호출", len(tool_calls))
    col_c.metric("데이터 출처", len(sources))

    if trace.get("requested_domains"):
        st.markdown("**분석 도메인**")
        st.write(", ".join(trace["requested_domains"]))

    if tool_calls:
        st.markdown("**호출된 도구**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "tool": call.get("tool"),
                        "sources": ", ".join(call.get("sources") or []),
                    }
                    for call in tool_calls
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    if sources:
        st.markdown("**데이터 출처 marker**")
        st.write(", ".join(sources))

    if trace.get("errors"):
        st.markdown("**실행 이슈**")
        st.warning("; ".join(str(item) for item in trace["errors"]))


configure_from_secrets()
require_password()

# --- 커스텀 CSS (프리미엄 디자인) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Pretendard:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Pretendard', 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #0e1117;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(167, 119, 227, 0.4);
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #a777e3, #6e8efb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 데이터 로드 함수 ---
@st.cache_data
def load_data():
    data_dir = _configured_data_dir()
    company_info = pd.read_csv(os.path.join(data_dir, "company_info.csv"))
    # 향후 다른 데이터들도 필요시 로드
    return company_info

try:
    df_company = load_data()
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

# --- 사이드바 ---
with st.sidebar:
    st.markdown("<h2 class='gradient-text'>Biz-Insight AI</h2>", unsafe_allow_html=True)
    st.markdown("KOSPI 기준점과 KOSDAQ 성장기업을 같은 분석 흐름으로 비교합니다.")
    st.divider()
    
    search_query = st.text_input("🔍 기업 검색", placeholder="예: 케이아이엔엑스, 덕산네오룩스")
    
    if search_query:
        st.write("---")
        st.write(f"검색어: **{search_query}**")

# --- 메인 화면 ---
if not search_query:
    st.title("Biz-Insight")
    st.markdown(f"### {DEMO_THESIS}")
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=2070", width="stretch")
    if {"시장", "데모역할", "데모포인트"}.issubset(df_company.columns):
        st.markdown("### Demo Universe")
        st.dataframe(
            df_company[["회사명", "종목코드", "시장", "데모역할", "데모포인트"]],
            width="stretch",
            hide_index=True,
        )
    
else:
    # 1. 기업 기본 정보 표시
    company_data = df_company[df_company['회사명'].str.contains(search_query, na=False)]
    
    if company_data.empty:
        st.warning(f"'{search_query}'에 대한 기업 정보를 찾을 수 없습니다.")
    else:
        # 정확히 일치하는 기업이 있으면 그걸 우선 선택, 아니면 첫 번째 기업 선택
        exact_match = company_data[company_data['회사명'] == search_query]
        if not exact_match.empty:
            target_company = exact_match.iloc[0]
        else:
            target_company = company_data.iloc[0]
            
        corp_name = target_company['회사명']
        
        st.markdown(f"<h1><span class='gradient-text'>{corp_name}</span> 기업 분석 대시보드</h1>", unsafe_allow_html=True)
        
        # 글래스모피즘 카드 형태로 기본 정보 렌더링
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='glass-card'>
                <h4>📌 종목코드</h4>
                <p style='font-size: 24px; font-weight: bold;'>{display_value(target_company, '종목코드')}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='glass-card'>
                <h4>🏛️ 시장</h4>
                <p style='font-size: 20px; font-weight: 600;'>{display_value(target_company, '시장')}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='glass-card'>
                <h4>🎯 데모 역할</h4>
                <p style='font-size: 20px; font-weight: 600;'>{display_value(target_company, '데모역할')}</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class='glass-card'>
            <h4>📦 주요 제품 / 분석 포인트</h4>
            <p><strong>{display_value(target_company, '업종')}</strong> · {display_value(target_company, '주요제품')}</p>
            <p>{display_value(target_company, '데모포인트')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()

        # 2. LangGraph AI 리포트 생성 섹션
        st.markdown("### 🤖 LangGraph AI 멀티 에이전트 분석")
        st.markdown("Supervisor가 자연어 요청을 해석해 필요한 데이터 수집, 분석, 검토 흐름을 구성합니다.")
        
        # 분석 유형 선택
        query_type = st.selectbox(
            "분석 유형 선택",
            options=["full_report", "financial", "credit", "overview", "investment", "review", "risk"],
            format_func=lambda x: {
                "full_report": "📊 전체 종합 리포트",
                "financial": "💰 재무분석",
                "credit": "🏦 신용분석",
                "overview": "📋 기업개요",
                "investment": "📈 투자/상세지표",
                "review": "👥 직원리뷰",
                "risk": "⚠️ 리스크 분석",
            }.get(x, x),
        )

        default_requests = {
            "full_report": f"{corp_name}의 공시/실적, 산업 맥락, 주가 흐름, 직원 리뷰를 연결해서 성장 논리와 핵심 리스크를 분석해줘.",
            "financial": f"{corp_name}의 실적과 재무 안정성을 산업 맥락과 비교해서 분석해줘.",
            "credit": f"{corp_name}의 신용등급, 부채 상환능력, 재무 건전성 관점에서 리스크를 분석해줘.",
            "overview": f"{corp_name}의 핵심 사업과 기업 기본 정보를 요약해줘.",
            "investment": f"{corp_name}의 성장 논리, 시장 위치, 주가 흐름을 바탕으로 투자 관점의 체크포인트를 분석해줘.",
            "review": f"{corp_name}의 직원 리뷰를 바탕으로 조직문화, 복지, 워라밸 리스크를 분석해줘.",
            "risk": f"{corp_name}의 재무, 신용, 주가, 조직문화 데이터를 종합해서 주요 리스크와 대응 포인트를 분석해줘.",
        }
        user_request = st.text_area(
            "분석 요청",
            value=default_requests.get(query_type, default_requests["full_report"]),
            height=120,
            help="Supervisor가 이 자연어 요청을 바탕으로 필요한 도메인과 도구를 선택합니다.",
        )
        
        if st.button("AI 리포트 생성 시작 ✨"):
            with st.spinner(f"'{corp_name}' 데이터를 분석 중입니다. 약 1~2분 정도 소요될 수 있습니다..."):
                start_time = time.time()
                try:
                    from src.graph import generate_ai_report_result

                    # LangGraph 호출
                    result = generate_ai_report_result(corp_name, query_type, user_request)
                    report = result["report"]
                    trace = result["trace"]
                    st.success(f"🎉 분석 완료! (소요 시간: {time.time() - start_time:.1f}초)")
                    
                    st.markdown("""
                    <div class='glass-card'>
                    """, unsafe_allow_html=True)
                    st.markdown(report)
                    st.markdown("""
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("리포트 작성 과정 보기", expanded=False):
                        render_execution_trace(trace)
                    
                except Exception as e:
                    st.error(f"리포트 생성 중 오류가 발생했습니다: {e}")
