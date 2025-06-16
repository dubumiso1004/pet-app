import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 데이터
data = {
    "SVF": [0.238, 0.168, 0.080, 0.126, 0.005, 0.074, 0.078, 0.037, 0.037, 0.074, 0.122, 0.154],
    "GVI": [0.205, 0.353, 0.393, 0.402, 0.534, 0.815, 0.808, 0.663, 0.518, 0.662, 0.800, 0.542],
    "BVI": [0.139, 0.181, 0.192, 0.374, 0.119, 0.000, 0.000, 0.083, 0.263, 0.220, 0.099, 0.022],
    "PET": [29.2, 28.9, 27.0, 25.9, 27.4, 24.1, 26.2, 26.2, 25.9, 26.4, 26.6, 29.2]
}
df = pd.DataFrame(data)

# 모델 학습
x = df[["SVF", "GVI", "BVI"]]
y = df["PET"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x, y)

# Streamlit 앱 시작
st.title("보행자 열쾌적성 예측기 (PET Prediction App)")
st.write("SVF, GVI, BVI를 입력하면 PET를 예측합니다.")

# 입력 슬라이더
svf = st.slider("SVF (Sky View Factor)", 0.0, 1.0, 0.3, step=0.01)
gvi = st.slider("GVI (Green View Index)", 0.0, 1.0, 0.3, step=0.01)
bvi = st.slider("BVI (Building View Index)", 0.0, 1.0, 0.3, step=0.01)

# 예측
input_data = pd.DataFrame({"SVF": [svf], "GVI": [gvi], "BVI": [bvi]})
predicted_pet = model.predict(input_data)[0]

# 결과 출력
st.header("예측 결과")
st.metric("예측된 PET (℃)", f"{predicted_pet:.2f}")

