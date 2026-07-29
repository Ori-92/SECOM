import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# 안전한 ML 라이브러리 로드
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

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
        return pd.DataFrame(), pd.Series(dtype=int)

    df_data = pd.read_csv(data_path, sep=r'\s+', header=None)
    df_data.columns = [f"Feature_{i}" for i in range(1, df_data.shape[1] + 1)]
    df_labels = pd.read_csv(label_path, sep=r'\s+', header=None)
    labels = df_labels[0]
    df_data = df_data.fillna(df_data.median())
    return df_data, labels

raw_data, labels = load_data()

if raw_data.empty:
    st.error("데이터 파일을 찾을 수 없습니다. (secom.data, secom_labels.data 확인 필요)")
    st.stop()

# 메인 UI
st.title("SECOM 공정 수율 분석 시스템 💻")

col1, col2, col3 = st.columns(3)
col1.metric("Total Examples", f"{len(raw_data):,}")
col2.metric("Features (Sensors)", f"{raw_data.shape[1]:,}")
col3.metric("Fails (Yield Issue)", f"{(labels == 1).sum():,}")

tab1, tab2, tab3 = st.tabs(["네트워크 상관관계 분석", "이상치(Outlier) 시각화", "알고리즘 비교"])

with tab1:
    st.subheader("상관관계 히트맵 (Top 20)")
    variances = raw_data.var()
    top_cols = variances.nlargest(20).index
    corr = raw_data[top_cols].corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=top_cols, y=top_cols, colorscale='RdBu'))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("이상치 분석")
    cols = raw_data.columns[:10].tolist()
    x_ax = st.selectbox("X축 선택", cols, index=0)
    y_ax = st.selectbox("Y축 선택", cols, index=1)
    fig_scat = px.scatter(x=raw_data[x_ax], y=raw_data[y_ax], color=labels.astype(str))
    st.plotly_chart(fig_scat, use_container_width=True)

with tab3:
    st.subheader("알고리즘 비교 분석")
    if ML_AVAILABLE:
        st.success("Scikit-learn이 로드되었습니다. 모델 학습을 수행합니다.")
        # 간략한 모델 성능 측정 예시
        X_train, X_test, y_train, y_test = train_test_split(raw_data, labels, test_size=0.2)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        score = balanced_accuracy_score(y_test, model.predict(X_test))
        st.metric("Random Forest Balanced Accuracy", f"{score:.2%}")
    else:
        st.error("Scikit-learn 라이브러리가 설치되지 않았습니다. requirements.txt를 확인하세요.")
