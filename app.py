import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# 페이지 설정
st.set_page_config(
    page_title="SECOM 공정 수율 분석 시스템",
    page_icon="💻",
    layout="wide"
)

# 데이터 로드
@st.cache_data
def load_data():
    data_path = "secom.data"
    label_path = "secom_labels.data"
    
    if not os.path.exists(data_path) or not os.path.exists(label_path):
        return None, None

    try:
        # 데이터 파일 로딩
        df_data = pd.read_csv(data_path, sep=r'\s+', header=None)
        df_data.columns = [f"Feature_{i}" for i in range(1, df_data.shape[1] + 1)]
        
        # 라벨 로딩
        df_labels = pd.read_csv(label_path, sep=r'\s+', header=None)
        labels = df_labels[0]
        
        # 간단한 결측치 전처리: 중앙값으로 채우기
        df_data = df_data.fillna(df_data.median())
        return df_data, labels
    except Exception as e:
        st.error(f"데이터 로딩 중 에러 발생: {e}")
        return None, None

raw_data, labels = load_data()

# 사이드바 및 UI
st.title("💻 SECOM 공정 수율 분석 시스템")

if raw_data is None:
    st.error("데이터 파일을 찾을 수 없습니다. (secom.data, secom_labels.data 확인 필요)")
    st.stop()

# 요약 메트릭
col1, col2, col3 = st.columns(3)
col1.metric("전체 샘플 수", f"{len(raw_data):,}")
col2.metric("전체 센서(Feature) 수", f"{raw_data.shape[1]:,}")
col3.metric("불량(Fail) 건수", f"{(labels == 1).sum():,}")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["네트워크 상관관계", "이상치(Outlier) 분포", "센서 불안정성(변동성) 분석"])

with tab1:
    st.subheader("주요 센서 상관관계 (Top 20)")
    variances = raw_data.var()
    top_cols = variances.nlargest(20).index
    corr = raw_data[top_cols].corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=top_cols, y=top_cols, colorscale='RdBu'))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("이상치 분포 확인")
    cols = raw_data.columns[:50].tolist()
    x_ax = st.selectbox("X축 센서", cols, index=0)
    y_ax = st.selectbox("Y축 센서", cols, index=1)
    
    fig_scat = px.scatter(
        x=raw_data[x_ax], y=raw_data[y_ax], 
        color=labels.astype(str), 
        labels={'color': 'Status (1:Fail, -1:Pass)'},
        title=f"{x_ax} vs {y_ax} 관계도"
    )
    st.plotly_chart(fig_scat, use_container_width=True)

with tab3:
    st.subheader("센서 변동성(Volatility) 히트맵")
    st.write("각 센서들의 표준편차를 계산하여, 데이터 변동이 가장 심한 센서들을 식별합니다.")
    
    # 변동성 계산 (표준편차)
    std_devs = raw_data.std().sort_values(ascending=False).head(50)
    
    fig_vol = go.Figure(go.Bar(
        x=std_devs.index,
        y=std_devs.values,
        marker_color='orange'
    ))
    fig_vol.update_layout(title="Top 50 불안정한 센서 (표준편차 기준)", xaxis_tickangle=-45)
    st.plotly_chart(fig_vol, use_container_width=True)
    
    st.info("💡 인사이트: 변동성이 매우 높은 센서들은 공정 중에 갑자기 튀는 신호를 보낼 확률이 높습니다. 이 센서들을 먼저 집중적으로 조사하면 공정 불량의 실마리를 찾을 수 있습니다.")
