import streamlit as st


PAGE_OPTIONS = {
    "overview": "🏠 개요",
    "data_intro": "📁 데이터 소개 및 한계",
    "preprocessing": "🧹 데이터 전처리",
    "damage_insights": "📊 피해 현황 및 특성",
    "call_insights": "📞 통화 패턴 및 핵심 인사이트",
    "ml_validation": "🤖 ML 위험 탐지 및 검증",
    "demo_v2": "🛡️ 서비스 데모 V2",
}


def render_sidebar() -> str:
    """대시보드 이름과 페이지 메뉴를 표시하고 선택값을 반환합니다."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <span class="sidebar-brand-icon">🛡️</span>
                <span>보이스피싱 현황 및 위험 분석</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # radio가 선택 상태를 관리하므로 별도의 Session State가 필요하지 않습니다.
        return st.radio(
            "페이지 선택",
            options=tuple(PAGE_OPTIONS),
            format_func=PAGE_OPTIONS.get,
            key="selected_page",
            label_visibility="collapsed",
        )
