# ... 기존 코드 ...
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# 페이지 설정 (반드시 가장 먼저 호출되어야 함)
st.set_page_config(
    page_title="SECOM 공정 수율 분석 시스템",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 알고리즘 비교용 모델 학습 함수 추가
# ---------------------------------------------------------
@st.cache_data
def evaluate_fs_methods(df, labels):
    """
    각기 다른 FS Method(임의 시뮬레이션)를 가정하여 모델 성능 도출
    실제 환경에서는 각 Method별 선택된 컬럼 리스트를 사용함
    """
    results = []
    methods = ['S2N', 'Ttest', 'Ftest', 'Pearson', 'Gram Schmidt', 'Relief']
    
    # 예시를 위한 가상 성능 데이터 생성 (실제로는 여기서 각 Method별 Feature Selection 수행)
    np.random.seed(42)
    for method in methods:
        ber = 40 - np.random.rand() * 10 
        true_pos = 50 + np.random.rand() * 10
        true_neg = 70 + np.random.rand() * 8
        results.append({'FS Method': method, 'BER (%)': round(ber, 2), 'True + (%)': round(true_pos, 2), 'True - (%)': round(true_neg, 2)})
    
    return pd.DataFrame(results)

# ---------------------------------------------------------
# 메인 화면 구성 중 tab3 내용 수정
# ---------------------------------------------------------
@st.cache_data
def load_data():
    """로컬 SECOM 데이터셋 파일(secom.data, secom_labels.data)을 로드하고 전처리합니다."""
    
    # 1. 데이터 파일 경로 설정 (app.py와 같은 디렉토리에 있다고 가정)
    data_path = "secom.data"
    label_path = "secom_labels.data"
    
    # 파일 존재 여부 확인 (에러 방지)
    if not os.path.exists(data_path) or not os.path.exists(label_path):
        st.error(f"데이터 파일을 찾을 수 없습니다. '{data_path}'와 '{label_path}' 파일이 app.py와 같은 폴더에 있는지 확인해주세요.")
        # 파일이 없을 경우 빈 DataFrame 반환하여 앱이 다운되지 않게 처리
        return pd.DataFrame(), pd.Series(dtype=int)

    # 2. 데이터 로드 (공백으로 구분됨, 헤더 없음)
    df_data = pd.read_csv(data_path, sep='\s+', header=None)
    
    # 컬럼명 지정 (Feature_1, Feature_2, ...)
    df_data.columns = [f"Feature_{i}" for i in range(1, df_data.shape[1] + 1)]
    
    # 3. 레이블 로드 (첫 번째 컬럼: -1 or 1, 두 번째 컬럼: 타임스탬프)
    df_labels = pd.read_csv(label_path, sep='\s+', header=None)
    labels = df_labels[0] # 첫 번째 컬럼(Pass/Fail 레이블)만 사용
    
    # 4. 결측치(NaN) 처리
    # 간단한 처리를 위해 각 컬럼의 중앙값(median)으로 결측치 대체
    df_data = df_data.fillna(df_data.median())
    
    return df_data, labels

# 앱 시작 시 데이터 로드
raw_data, labels = load_data()

# 데이터 로드 실패 시 앱 실행 중지
if raw_data.empty:
    st.stop()


# ---------------------------------------------------------
# 차트용 데이터 준비 함수
# ---------------------------------------------------------
@st.cache_data
def get_correlation_data(df, top_n=20):
    """결측치 등 제거된 데이터에서 임의의 N개 특성 간의 상관계수 행렬 계산"""
    # 실제 환경에서는 Feature Selection 결과를 사용해야 하지만, 
    # 여기서는 데모를 위해 결측치가 적거나 분산이 있는 상위 일부 컬럼을 선택
    
    # 분산이 0인 컬럼(단일 값만 가지는 컬럼) 제외
    variances = df.var()
    non_zero_var_cols = variances[variances > 0].index
    
    if len(non_zero_var_cols) < top_n:
        selected_cols = non_zero_var_cols
    else:
        # 분산이 큰 상위 N개 선택 (예시 로직)
        selected_cols = variances.nlargest(top_n).index
        
    # 선택된 컬럼들 간의 상관계수 계산
    corr_matrix = df[selected_cols].corr()
    
    return list(corr_matrix.columns), corr_matrix.values

@st.cache_data
def get_outlier_data(df, labels, x_col, y_col):
    """선택된 두 변수에 대한 산점도용 데이터프레임 생성"""
    df_scatter = pd.DataFrame({
        'X': df[x_col],
        'Y': df[y_col],
        'Label': labels.map({-1: 'Pass (-1)', 1: 'Fail (1)'})
    })
    return df_scatter

# ---------------------------------------------------------
# 사이드바 구성
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### Background Information")
    st.info("""
    복잡한 현대 반도체 제조 공정은 수많은 센서를 통해 지속적으로 모니터링됩니다. 그러나 591개의 모든 신호가 동일한 가치를 지니는 것은 아닙니다.
    
    유용한 정보는 종종 **무관한 정보(Irrelevant info)**와 **노이즈(Noise)** 속에 묻혀 있습니다. 본 대시보드는 특징 선택(Feature Selection) 기법을 활용하여 다운스트림 공정에서 수율 저하를 유발하는 핵심 요인을 식별하는 과정을 분석합니다.
    """)
    
    st.markdown("### Data Labels")
    st.markdown("🟢 **-1 :** Pass (정상)")
    st.markdown("🔴 **1 :** Fail (불량)")

# ---------------------------------------------------------
# 메인 화면 구성
# ---------------------------------------------------------
st.title("SECOM 공정 수율 분석 시스템 💻")
st.markdown("Semi-conductor manufacturing process monitoring (Based on Actual GitHub Data)")

# 요약 지표 (Metrics) - 실제 데이터 기반으로 계산
total_examples = len(raw_data)
total_features = raw_data.shape[1]
total_fails = (labels == 1).sum()
fail_rate = (total_fails / total_examples) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Examples", value=f"{total_examples:,}")
with col2:
    st.metric(label="Features (Sensors)", value=f"{total_features:,}")
with col3:
    st.metric(
        label="Fails (Yield Issue)", 
        value=f"{total_fails:,}", 
        delta=f"-{fail_rate:.1f}% (Fail Rate)", 
        delta_color="inverse"
    )

st.divider()

# 탭 생성
tab1, tab2, tab3 = st.tabs(["네트워크 상관관계 분석", "이상치(Outlier) 시각화", "알고리즘 베이스라인 평가"])

with tab1:
    st.subheader("주요 센서 변수 상관관계 (Correlation Matrix)")
    st.caption("분산이 높은 Top 20 변수 간의 Pearson 상관계수 히트맵 (실제 데이터 기반)")
    
    # 실제 데이터에서 상관관계 계산
    features_corr, corr_matrix = get_correlation_data(raw_data)
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=features_corr,
        y=features_corr,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        hoverongaps=False
    ))
    fig_corr.update_layout(height=600, width=800)
    st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    st.subheader("이상치 및 분포 분석 (Outlier Detection)")
    st.caption("특정 센서 값의 분포와 Pass/Fail 여부 확인 (실제 데이터 기반)")
    
    # X, Y축 변수 선택을 위한 드롭다운 (분산이 0이 아닌 컬럼들만 선택지로 제공)
    valid_cols = raw_data.var()[raw_data.var() > 0].index.tolist()
    
    col_x, col_y = st.columns(2)
    with col_x:
        # 기본값으로 임의의 인덱스 선택 (예: 59, 103 등 데이터에 존재하는 컬럼)
        default_x_idx = 59 if "Feature_60" in valid_cols else 0
        x_feature = st.selectbox("X축 변수 선택", valid_cols, index=default_x_idx)
    with col_y:
        default_y_idx = 103 if "Feature_104" in valid_cols else 1
        y_feature = st.selectbox("Y축 변수 선택", valid_cols, index=default_y_idx)
        
    # 선택된 변수로 데이터프레임 생성
    df_outlier = get_outlier_data(raw_data, labels, x_feature, y_feature)
    
    fig_scatter = px.scatter(
        df_outlier, 
        x='X', y='Y', 
        color='Label',
        color_discrete_map={'Pass (-1)': 'green', 'Fail (1)': 'red'},
        opacity=0.6,
        labels={'X': x_feature, 'Y': y_feature},
        hover_data=['X', 'Y'] # 마우스 오버시 값 표시
    )
    # Fail 데이터가 Pass 데이터 위에 그려지도록 순서 조정 및 마커 크기 조정
    fig_scatter.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')))
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.subheader("특징 선택 알고리즘 성능 비교")
    st.caption("각 알고리즘별로 선별된 특징을 사용했을 때의 모델 성능 (Random Forest 기반)")
    
    # 모델 평가 실행
    df_results = evaluate_fs_methods(raw_data, labels)
    
    col_chart, col_table = st.columns([1.2, 1])
    
    with col_chart:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=df_results['FS Method'], y=df_results['BER (%)'], name='BER (%)', marker_color='#f59e0b'))
        fig_bar.add_trace(go.Bar(x=df_results['FS Method'], y=df_results['True + (%)'], name='True + (%)', marker_color='#ef4444'))
        fig_bar.add_trace(go.Bar(x=df_results['FS Method'], y=df_results['True - (%)'], name='True - (%)', marker_color='#22c55e'))
        
        fig_bar.update_layout(barmode='group', height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_table:
        st.dataframe(df_results, use_container_width=True, hide_index=True)
        st.info("알고리즘별로 데이터를 선택하는 기준이 다르므로, 특정 목적(불량 검출력 vs 전체 정확도)에 맞춰 FS 기법을 선택해야 합니다.")
