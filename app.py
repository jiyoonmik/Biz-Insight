import streamlit as st
import pandas as pd
import os
import time

# --- LangGraph 멀티 에이전트 연동 ---
from src.graph import generate_ai_report

# --- Streamlit 설정 ---
st.set_page_config(page_title="Biz-Insight 3.0", page_icon="✨", layout="wide")

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
    data_dir = os.path.join(os.path.dirname(__file__), "data")
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
    st.markdown("기업 데이터를 바탕으로 AI가 즉각적으로 리포트를 생성해주는 서비스입니다.")
    st.divider()
    
    search_query = st.text_input("🔍 기업 검색", placeholder="예: 삼성전자, 카카오")
    
    if search_query:
        st.write("---")
        st.write(f"검색어: **{search_query}**")

# --- 메인 화면 ---
if not search_query:
    st.title("✨ 환영합니다! Biz-Insight 3.0")
    st.markdown("좌측 메뉴에서 분석하고자 하는 **기업명을 검색**해주세요.")
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=2070", use_container_width=True)
    
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
                <p style='font-size: 24px; font-weight: bold;'>{target_company.get('종목코드', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='glass-card'>
                <h4>🏭 주요 업종</h4>
                <p style='font-size: 20px; font-weight: 600;'>{target_company.get('업종', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='glass-card'>
                <h4>👤 대표자</h4>
                <p style='font-size: 20px; font-weight: 600;'>{target_company.get('대표자명', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class='glass-card'>
            <h4>📦 주요 제품</h4>
            <p>{target_company.get('주요제품', 'N/A')}</p>
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
            "full_report": f"{corp_name}의 기업 개요, 재무 안정성, 신용 리스크, 투자 매력도, 직원 리뷰를 종합해서 분석해줘.",
            "financial": f"{corp_name}의 재무 성과와 안정성을 산업 평균과 비교해서 분석해줘.",
            "credit": f"{corp_name}의 신용등급, 부채 상환능력, 재무 건전성 관점에서 리스크를 분석해줘.",
            "overview": f"{corp_name}의 핵심 사업과 기업 기본 정보를 요약해줘.",
            "investment": f"{corp_name}의 투자 지표와 주가 흐름을 바탕으로 투자 매력도를 분석해줘.",
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
                    # LangGraph 호출
                    report = generate_ai_report(corp_name, query_type, user_request)
                    st.success(f"🎉 분석 완료! (소요 시간: {time.time() - start_time:.1f}초)")
                    
                    st.markdown("""
                    <div class='glass-card'>
                    """, unsafe_allow_html=True)
                    st.markdown(report)
                    st.markdown("""
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"리포트 생성 중 오류가 발생했습니다: {e}")
