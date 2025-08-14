import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="시니어 연금 계산기",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 스타일링 - KB 스타일 적용
st.markdown("""
<style>
    /* 전체 배경 */
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* KB 로고 및 헤더 스타일 */
    .kb-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    
    .kb-logo {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFCC00;
        margin-bottom: 0.5rem;
    }
    
    .kb-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #333;
        margin: 1rem 0;
    }
    
    /* 메인 버튼 스타일 */
    .main-buttons {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .pension-button {
        background: linear-gradient(135deg, #fff2cc 0%, #ffeaa7 100%);
        border: none;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .pension-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .pension-button.receiving {
        background: linear-gradient(135deg, #a8d8ff 0%, #74b9ff 100%);
    }
    
    .pension-text {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2d3436;
        margin: 0;
    }
    
    /* 하단 버튼들 */
    .bottom-buttons {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 2rem;
    }
    
    .bottom-button {
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
    }
    
    .info-button {
        background: linear-gradient(135deg, #98fb98 0%, #90ee90 100%);
        color: #2d3436;
    }
    
    .contact-button {
        background: linear-gradient(135deg, #ffb3ba 0%, #ff9999 100%);
        color: #2d3436;
    }
    
    .bottom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* 입력 폼 스타일 */
    .input-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .input-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2d3436;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Streamlit 기본 요소 스타일 조정 */
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 10px;
    }
    
    .stNumberInput > div > div {
        background-color: white;
        border-radius: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        width: 100%;
        margin: 1rem 0;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0984e3 0%, #74b9ff 100%);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'pension_status' not in st.session_state:
    st.session_state.pension_status = None

# 메인 페이지
def main_page():
    # KB 헤더
    st.markdown("""
    <div class="kb-header">
        <div class="kb-logo">🏦 KB</div>
        <div class="kb-title">시니어 연금 계산기</div>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
            <span style="background: #e17055; color: white; padding: 0.25rem 0.75rem; border-radius: 15px; font-size: 0.9rem;">현재 연금</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 메인 선택 버튼들
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("현재 연금\n미수령 중", key="not_receiving", help="아직 연금을 받지 않는 경우"):
            st.session_state.pension_status = "not_receiving"
            st.session_state.page = "input_form"
            st.rerun()
    
    with col2:
        if st.button("현재 연금\n수령 중", key="receiving", help="이미 연금을 받고 있는 경우"):
            st.session_state.pension_status = "receiving"
            st.session_state.page = "input_form"
            st.rerun()
    
    # 하단 버튼들
    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("상품\n정보", key="product_info"):
            st.info("📋 연금 상품 정보를 확인할 수 있습니다.")
    
    with col4:
        if st.button("전화\n상담", key="phone_consult"):
            st.info("📞 전화 상담: 1588-9999")

# 입력 폼 페이지
def input_form_page():
    st.markdown(f"""
    <div class="kb-header">
        <div class="kb-logo">🏦 KB</div>
        <div style="font-size: 1.5rem; color: #666;">
            {'연금 수령 중' if st.session_state.pension_status == 'receiving' else '연금 미수령 중'} 계산
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("← 메인으로 돌아가기"):
        st.session_state.page = 'main'
        st.rerun()
    
    # 단계별 입력 폼
    with st.form("pension_form"):
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # 1. 평균 월소득
        st.markdown('<div class="input-title">1. 평균 월소득을 입력해주세요.</div>', unsafe_allow_html=True)
        monthly_income = st.number_input(
            "월소득 (만원)", 
            min_value=0, 
            max_value=10000, 
            value=300, 
            step=10,
            help="최근 3년간의 평균 월소득을 입력하세요"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 2. 국민연금 가입기간
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<div class="input-title">2. 국민연금 가입기간을 입력해주세요.</div>', unsafe_allow_html=True)
        pension_years = st.number_input(
            "가입기간 (년)", 
            min_value=0, 
            max_value=50, 
            value=20, 
            step=1,
            help="국민연금에 가입한 총 기간을 입력하세요"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 3. 나이
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<div class="input-title">3. 나이를 입력해주세요.</div>', unsafe_allow_html=True)
        age = st.number_input(
            "현재 나이", 
            min_value=20, 
            max_value=100, 
            value=60, 
            step=1
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 4. 성별
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<div class="input-title">4. 성별을 입력해주세요.</div>', unsafe_allow_html=True)
        gender = st.selectbox("성별", ["남성", "여성"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 5. 가구원 수
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<div class="input-title">5. 가구원 수를 입력해주세요.</div>', unsafe_allow_html=True)
        household_size = st.number_input(
            "가구원 수 (명)", 
            min_value=1, 
            max_value=10, 
            value=2, 
            step=1
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 6. 피부양자 여부 (수령 중인 경우만)
        if st.session_state.pension_status == "receiving":
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.markdown('<div class="input-title">6. 피부양자가 있나요?</div>', unsafe_allow_html=True)
            has_dependents = st.selectbox("피부양자 여부", ["없음", "있음"])
            st.markdown("</div>", unsafe_allow_html=True)
        
        # 7. 현재 보유 금융자산
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<div class="input-title">7. 현재 보유한 금융자산을 입력해주세요.</div>', unsafe_allow_html=True)
        financial_assets = st.number_input(
            "금융자산 (만원)", 
            min_value=0, 
            max_value=100000, 
            value=5000, 
            step=100
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 8. 월 수령 연금 (수령 중인 경우만)
        if st.session_state.pension_status == "receiving":
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.markdown('<div class="input-title">8. 월 수령하는 연금 금액을 입력해주세요.</div>', unsafe_allow_html=True)
            current_pension = st.number_input(
                "월 연금액 (만원)", 
                min_value=0, 
                max_value=500, 
                value=100, 
                step=5
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        # 9. 월 평균 지출
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<div class="input-title">9. 월 평균 지출비를 입력해주세요.</div>', unsafe_allow_html=True)
        monthly_expense = st.number_input(
            "월 지출 (만원)", 
            min_value=0, 
            max_value=1000, 
            value=250, 
            step=10
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 10. 투자 성향
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<div class="input-title">10. 투자 성향을 선택해주세요.</div>', unsafe_allow_html=True)
        investment_style = st.selectbox(
            "투자 성향", 
            ["안정형", "안정추구형", "위험중립형", "적극투자형", "공격투자형"]
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 계산 버튼
        submitted = st.form_submit_button("연금 계산하기")
        
        if submitted:
            st.session_state.page = 'results'
            st.session_state.form_data = {
                'monthly_income': monthly_income,
                'pension_years': pension_years,
                'age': age,
                'gender': gender,
                'household_size': household_size,
                'financial_assets': financial_assets,
                'monthly_expense': monthly_expense,
                'investment_style': investment_style,
                'current_pension': current_pension if st.session_state.pension_status == "receiving" else 0
            }
            st.rerun()

# 결과 페이지
def results_page():
    data = st.session_state.form_data
    
    st.markdown("""
    <div class="kb-header">
        <div class="kb-logo">🏦 KB</div>
        <div class="kb-title">연금 계산 결과</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("← 다시 계산하기"):
        st.session_state.page = 'input_form'
        st.rerun()
    
    # 간단한 연금 계산 로직
    base_pension = data['monthly_income'] * 0.4 * (data['pension_years'] / 40)
    
    # 투자 수익률 설정
    investment_rates = {
        "안정형": 0.02,
        "안정추구형": 0.035,
        "위험중립형": 0.05,
        "적극투자형": 0.07,
        "공격투자형": 0.09
    }
    
    rate = investment_rates[data['investment_style']]
    
    # 결과 표시
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "예상 월 연금액",
            f"{base_pension:,.0f}만원",
            delta=f"현재 대비 {((base_pension / data['monthly_expense']) * 100):,.0f}%"
        )
    
    with col2:
        st.metric(
            "생활비 충족률",
            f"{(base_pension / data['monthly_expense'] * 100):,.0f}%",
            delta="목표 70%" if base_pension / data['monthly_expense'] < 0.7 else "충분"
        )
    
    # 투자 포트폴리오 추천
    st.subheader("📊 추천 투자 포트폴리오")
    
    if data['investment_style'] == "안정형":
        portfolio = {"예금/적금": 60, "국채": 30, "회사채": 10}
    elif data['investment_style'] == "안정추구형":
        portfolio = {"예금/적금": 40, "국채": 35, "회사채": 15, "주식": 10}
    elif data['investment_style'] == "위험중립형":
        portfolio = {"예금/적금": 30, "채권": 30, "주식": 25, "리츠": 15}
    elif data['investment_style'] == "적극투자형":
        portfolio = {"예금/적금": 20, "채권": 20, "주식": 40, "리츠": 20}
    else:  # 공격투자형
        portfolio = {"예금/적금": 10, "채권": 15, "주식": 50, "리츠": 25}
    
    # 포트폴리오 차트
    fig = px.pie(
        values=list(portfolio.values()),
        names=list(portfolio.keys()),
        title="투자 포트폴리오 구성",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # 연령별 예상 자산 증가
    st.subheader("📈 연령별 예상 자산 증가")
    
    years = list(range(data['age'], min(data['age'] + 20, 90)))
    assets = []
    current_assets = data['financial_assets']
    
    for year in years:
        if year >= 65:  # 연금 수령 시작
            current_assets = current_assets * (1 + rate) + base_pension * 12 - data['monthly_expense'] * 12
        else:
            current_assets = current_assets * (1 + rate) + data['monthly_income'] * 12 * 0.1 - data['monthly_expense'] * 12
        assets.append(max(0, current_assets))
    
    df = pd.DataFrame({
        '나이': years,
        '예상 자산 (만원)': assets
    })
    
    fig = px.line(
        df, 
        x='나이', 
        y='예상 자산 (만원)',
        title="연령별 예상 자산 변화",
        color_discrete_sequence=['#74b9ff']
    )
    fig.update_layout(
        xaxis_title="나이",
        yaxis_title="자산 (만원)",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 권장사항
    st.subheader("💡 맞춤 권장사항")
    
    if base_pension / data['monthly_expense'] < 0.7:
        st.warning("⚠️ 연금액이 생활비의 70%에 못 미칩니다. 추가적인 노후 준비가 필요합니다.")
        st.info("📌 개인연금 가입이나 투자 확대를 고려해보세요.")
    else:
        st.success("✅ 연금액이 생활비를 충분히 커버할 수 있습니다!")
    
    if data['age'] < 50:
        st.info("📌 아직 젊으시므로 적극적인 투자로 자산을 늘려나가세요.")
    elif data['age'] < 60:
        st.info("📌 은퇴가 가까워지고 있습니다. 안정적인 포트폴리오로 전환을 고려하세요.")
    else:
        st.info("📌 안정적인 수익에 집중하여 자산을 보전하세요.")

# 페이지 라우팅
if st.session_state.page == 'main':
    main_page()
elif st.session_state.page == 'input_form':
    input_form_page()
elif st.session_state.page == 'results':
    results_page()
