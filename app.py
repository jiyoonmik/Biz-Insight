import streamlit as st
import pandas as pd
import os
import time
from html import escape

# --- Streamlit 설정 ---
st.set_page_config(page_title="Biz-Insight Research", page_icon="📈", layout="wide")

DEMO_THESIS = (
    "Biz-Insight는 공시, 재무, 신용, 시장 가격, 직원 리뷰를 하나의 분석 문맥으로 "
    "결합해 성장기업의 투자 가설과 리스크를 빠르게 검증하는 리서치 워크벤치입니다."
)


def _default_data_dir() -> str:
    project_dir = os.path.dirname(__file__)
    demo_dir = os.path.join(project_dir, "data", "demo")
    if os.path.isdir(demo_dir):
        return demo_dir
    return os.path.join(project_dir, "data")


def _configured_data_dir() -> str:
    return os.path.abspath(os.getenv("BIZINSIGHT_DATA_DIR", _default_data_dir()))


def _secret_value(name: str) -> str | None:
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def configure_from_secrets() -> None:
    data_dir = _secret_value("BIZINSIGHT_DATA_DIR")
    if not os.getenv("BIZINSIGHT_DATA_DIR"):
        os.environ["BIZINSIGHT_DATA_DIR"] = str(data_dir or _default_data_dir())

    for name in (
        "GOOGLE_API_KEY",
        "DART_API_KEY",
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_TRACING",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGCHAIN_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_ENDPOINT",
    ):
        value = _secret_value(name)
        if value and not os.getenv(name):
            os.environ[name] = str(value)

    try:
        from src.config import configure_langsmith_environment

        configure_langsmith_environment()
    except Exception:
        pass


def require_password() -> None:
    expected = (
        os.getenv("BIZINSIGHT_APP_PASSWORD")
        or _secret_value("BIZINSIGHT_APP_PASSWORD")
        or _secret_value("APP_PASSWORD")
    )
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


def html_value(row: pd.Series, key: str, default: str = "N/A") -> str:
    return escape(display_value(row, key, default))


def format_count(value: int) -> str:
    return f"{value:,}"


def render_execution_trace(trace: dict) -> None:
    agents = trace.get("agents") or []
    tool_calls = trace.get("tool_calls") or []
    sources = trace.get("data_sources") or []

    st.markdown("### 분석 실행 로그")

    graph_lines = [
        "digraph BizInsight {",
        "rankdir=LR;",
        'node [shape=box, style="rounded,filled", color="#D8DEE9", fillcolor="#F8FAFC", fontname="Arial"];',
        '"Analyst Query" -> "Supervisor";',
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
        st.markdown("**분석 범위**")
        st.write(", ".join(trace["requested_domains"]))

    if tool_calls:
        st.markdown("**사용 도구**")
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
        st.markdown("**데이터 출처**")
        st.write(", ".join(sources))

    if trace.get("errors"):
        st.markdown("**확인 필요 항목**")
        st.warning("; ".join(str(item) for item in trace["errors"]))


configure_from_secrets()
require_password()

# --- 커스텀 CSS (리서치 터미널 디자인) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Pretendard:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Pretendard', 'Inter', sans-serif;
        letter-spacing: 0;
    }
    
    .stApp {
        background: #0b0f14;
        color: #d7dee8;
    }
    
    .main .block-container {
        max-width: 1380px;
        padding-top: 2.25rem;
        padding-bottom: 4rem;
    }

    header[data-testid="stHeader"] {
        background: #0b0f14;
        border-bottom: 1px solid #222c38;
    }

    div[data-testid="stToolbar"] {
        color: #c9d4e2;
    }

    section[data-testid="stSidebar"] {
        background: #10161d;
        border-right: 1px solid #222c38;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #c9d4e2;
    }

    h1, h2, h3, h4 {
        color: #f1f5f9;
        letter-spacing: 0;
    }

    p, li, label, .stMarkdown {
        color: #c9d4e2;
    }

    .stButton>button {
        background: #f1f5f9;
        color: #0b0f14;
        border: 1px solid #f1f5f9;
        border-radius: 6px;
        padding: 0.55rem 0.95rem;
        font-weight: 700;
        transition: border-color 0.16s ease, background 0.16s ease;
    }
    
    .stButton>button:hover {
        background: #dbe4ef;
        border-color: #dbe4ef;
        color: #0b0f14;
    }

    .stButton>button:focus,
    .stButton>button:focus:not(:active) {
        border-color: #58a6ff;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.22);
        color: #0b0f14;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    textarea {
        background: #0d131a !important;
        border-color: #2a3644 !important;
        border-radius: 6px !important;
        color: #edf2f7 !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #222c38;
        border-radius: 6px;
        overflow: hidden;
    }

    .research-hero {
        border: 1px solid #243140;
        border-radius: 8px;
        background: #10161d;
        padding: 1.65rem 1.8rem;
        margin-bottom: 1.25rem;
    }

    .research-hero h1 {
        margin: 0.2rem 0 0.45rem 0;
        font-size: 2.15rem;
        line-height: 1.15;
    }

    .research-hero p {
        max-width: 920px;
        margin: 0;
        color: #aebacc;
        font-size: 1rem;
        line-height: 1.65;
    }

    .terminal-card, .research-card {
        background: #10161d;
        border: 1px solid #263241;
        border-radius: 6px;
        padding: 1.05rem 1.15rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        min-height: 118px;
        background: #10161d;
        border: 1px solid #263241;
        border-left: 3px solid #58a6ff;
        border-radius: 6px;
        padding: 1rem 1.05rem;
        margin-bottom: 1rem;
    }
    
    .metric-card.accent-amber {
        border-left-color: #d29922;
    }

    .metric-card.accent-green {
        border-left-color: #3fb950;
    }

    .metric-card h4 {
        margin: 0 0 0.6rem 0;
        color: #8b98a8;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .metric-card .value {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 700;
        line-height: 1.25;
        word-break: keep-all;
    }

    .metric-card .sub {
        margin-top: 0.45rem;
        color: #94a3b8;
        font-size: 0.86rem;
        line-height: 1.45;
    }

    .section-label {
        color: #58a6ff;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }

    .muted {
        color: #8b98a8;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        border: 1px solid #2a3644;
        border-radius: 999px;
        color: #c9d4e2;
        background: #0d131a;
        padding: 0.25rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .company-heading {
        margin: 0.15rem 0 0.25rem 0;
        font-size: 2rem;
        line-height: 1.2;
    }

    .divider-line {
        border-top: 1px solid #222c38;
        margin: 1.1rem 0 1.35rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- 데이터 로드 함수 ---
@st.cache_data
def load_data():
    data_dir = _configured_data_dir()
    company_info = pd.read_csv(os.path.join(data_dir, "company_info.csv"), dtype={"종목코드": str})
    # 향후 다른 데이터들도 필요시 로드
    return company_info

try:
    df_company = load_data()
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

# --- 사이드바 ---
with st.sidebar:
    st.markdown("<h2>Biz-Insight Research</h2>", unsafe_allow_html=True)
    st.markdown("공시, 재무, 신용, 시장 가격, 조직 신호를 연결해 기업별 투자 논리를 점검합니다.")
    st.divider()
    
    with st.form("company_search"):
        search_input = st.text_input("기업명 검색", placeholder="예: 케이아이엔엑스, 덕산네오룩스")
        submitted = st.form_submit_button("기업 조회", use_container_width=True)
        if submitted:
            st.session_state["active_search_query"] = search_input.strip()

    search_query = st.session_state.get("active_search_query", "")
    
    if search_query:
        st.write("---")
        st.write(f"조회 기업: **{search_query}**")

# --- 메인 화면 ---
if not search_query:
    kospi_count = int((df_company.get("시장", pd.Series(dtype=str)) == "KOSPI").sum())
    kosdaq_count = int((df_company.get("시장", pd.Series(dtype=str)) == "KOSDAQ").sum())
    st.markdown(f"""
    <div class='research-hero'>
        <div class='section-label'>Equity Research Terminal</div>
        <h1>Biz-Insight</h1>
        <p>{escape(DEMO_THESIS)}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Coverage</h4>
            <div class='value'>{format_count(len(df_company))} companies</div>
            <div class='sub'>KOSPI {format_count(kospi_count)} · KOSDAQ {format_count(kosdaq_count)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card accent-amber'>
            <h4>Analysis Stack</h4>
            <div class='value'>Canonical Context + Dynamic Signals</div>
            <div class='sub'>재무제표, 신용, 주가, 리뷰 데이터를 동일한 질의 흐름으로 조회합니다.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card accent-green'>
            <h4>Workflow</h4>
            <div class='value'>Research · Analysis · Review</div>
            <div class='sub'>에이전트 실행 로그와 데이터 출처를 리포트와 함께 확인합니다.</div>
        </div>
        """, unsafe_allow_html=True)

    if {"시장", "데모역할", "데모포인트"}.issubset(df_company.columns):
        st.markdown("### SAMPLE 분석 대상 기업")
        st.dataframe(
            df_company[["회사명", "종목코드", "시장", "데모역할", "데모포인트"]],
            width="stretch",
            hide_index=True,
        )
    
else:
    # 1. 기업 기본 정보 표시
    company_data = df_company[df_company['회사명'].str.contains(search_query, na=False)]
    
    if company_data.empty:
        st.warning(f"'{search_query}'와 일치하는 기업을 찾지 못했습니다. 기업명을 다시 확인해 주세요.")
    else:
        # 정확히 일치하는 기업이 있으면 그걸 우선 선택, 아니면 첫 번째 기업 선택
        exact_match = company_data[company_data['회사명'] == search_query]
        if not exact_match.empty:
            target_company = exact_match.iloc[0]
        else:
            target_company = company_data.iloc[0]
            
        corp_name = target_company['회사명']
        
        st.markdown(f"""
        <div class='research-hero'>
            <span class='status-pill'>{html_value(target_company, '시장')}</span>
            <h1 class='company-heading'>{escape(corp_name)}</h1>
            <p>{html_value(target_company, '업종')} · {html_value(target_company, '주요제품')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 리서치 터미널 카드 형태로 기본 정보 렌더링
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Ticker</h4>
                <div class='value'>{html_value(target_company, '종목코드')}</div>
                <div class='sub'>상장 종목 식별 코드</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card accent-amber'>
                <h4>Market</h4>
                <div class='value'>{html_value(target_company, '시장')}</div>
                <div class='sub'>비교 기준 시장</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card accent-green'>
                <h4>Research Role</h4>
                <div class='value'>{html_value(target_company, '데모역할')}</div>
                <div class='sub'>분석 포지션</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class='research-card'>
            <div class='section-label'>Investment Context</div>
            <h4>핵심 사업 및 점검 포인트</h4>
            <p><strong>{html_value(target_company, '업종')}</strong> · {html_value(target_company, '주요제품')}</p>
            <p class='muted'>{html_value(target_company, '데모포인트')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='divider-line'></div>", unsafe_allow_html=True)

        # 2. LangGraph AI 리포트 생성 섹션
        st.markdown("### 기업 리서치 실행")
        st.markdown("분석 요청을 입력하면 에이전트가 필요한 데이터 범위와 도구를 선택하고, 리포트와 실행 로그를 함께 반환합니다.")
        
        # 분석 유형 선택
        query_type = st.selectbox(
            "리서치 범위",
            options=["full_report", "financial", "credit", "overview", "investment", "review", "risk"],
            format_func=lambda x: {
                "full_report": "종합 리서치 리포트",
                "financial": "재무 성과 분석",
                "credit": "신용 및 상환능력 분석",
                "overview": "기업 개요",
                "investment": "투자 체크포인트",
                "review": "조직 및 직원 리뷰",
                "risk": "리스크 분석",
            }.get(x, x),
        )

        default_requests = {
            "full_report": f"{corp_name}의 공시, 실적, 산업 맥락, 주가 흐름, 직원 리뷰를 연결해 투자 가설과 핵심 리스크를 분석해줘.",
            "financial": f"{corp_name}의 매출 성장성, 수익성, 현금흐름, 재무 안정성을 산업 맥락과 비교해 분석해줘.",
            "credit": f"{corp_name}의 신용등급, 부채 상환능력, 유동성, 재무 건전성 관점의 위험 요인을 분석해줘.",
            "overview": f"{corp_name}의 사업 구조, 주요 제품, 시장 포지션, 최근 관찰 포인트를 요약해줘.",
            "investment": f"{corp_name}의 성장 논리, 시장 위치, 주가 흐름을 바탕으로 투자 검토 체크포인트를 정리해줘.",
            "review": f"{corp_name}의 직원 리뷰를 바탕으로 조직문화, 복지, 근무환경 관련 리스크를 분석해줘.",
            "risk": f"{corp_name}의 재무, 신용, 시장 가격, 조직 신호를 종합해 주요 리스크와 모니터링 포인트를 정리해줘.",
        }
        user_request = st.text_area(
            "분석 요청",
            value=default_requests.get(query_type, default_requests["full_report"]),
            height=120,
            help="요청 문장에 따라 분석 범위와 도구 호출이 달라집니다.",
        )
        
        if st.button("기업 분석 실행"):
            with st.spinner(f"'{corp_name}' 리서치 데이터를 분석 중입니다. 일반적으로 1~2분 정도 소요됩니다."):
                start_time = time.time()
                try:
                    from src.graph import generate_ai_report_result

                    # LangGraph 호출
                    result = generate_ai_report_result(corp_name, query_type, user_request)
                    report = result["report"]
                    trace = result["trace"]
                    st.success(f"분석이 완료되었습니다. 소요 시간: {time.time() - start_time:.1f}초")
                    
                    st.markdown("""
                    <div class='research-card'>
                    """, unsafe_allow_html=True)
                    st.markdown(report)
                    st.markdown("""
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("실행 로그 및 데이터 출처 보기", expanded=False):
                        render_execution_trace(trace)
                    
                except Exception as e:
                    st.error(f"리포트 생성 중 오류가 발생했습니다: {e}")
