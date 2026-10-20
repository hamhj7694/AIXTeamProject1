import streamlit as st

# 페이지 설정은 다른 Streamlit 명령보다 먼저 실행해야 합니다.
st.set_page_config(
    page_title="보이스피싱 현황 및 위험 분석",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="locked",
)

from components.overview_clean import render_overview
from components.pages import (
    render_call_insights,
    render_damage_insights,
    render_data_intro,
    render_demo_v2,
    render_ml_validation,
    render_preprocessing,
)
from components.sidebar import render_sidebar
from styles.style import apply_styles


PAGE_RENDERERS = {
    "overview": render_overview,
    "data_intro": render_data_intro,
    "preprocessing": render_preprocessing,
    "damage_insights": render_damage_insights,
    "call_insights": render_call_insights,
    "ml_validation": render_ml_validation,
    "demo_v2": render_demo_v2,
}


def main() -> None:
    """대시보드 공통 레이아웃과 현재 페이지를 표시합니다."""
    apply_styles()
    selected_page = render_sidebar()

    # 사이드바에서 선택한 하나의 값으로 메인 페이지를 결정합니다.
    PAGE_RENDERERS[selected_page]()


if __name__ == "__main__":
    main()
