import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# 웹페이지 기본 설정
st.set_page_config(
    page_title="손글씨 숫자 인식기",
    page_icon="✍️",
    layout="centered",
)

st.title("✍️ 손글씨 숫자 인식기")
st.markdown("업로드한 이미지에서 숫자를 인식합니다. (0~9)")

st.markdown("---")

# 모델 로딩 함수
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.h5")

model = load_model()

# 이미지 전처리 함수
def preprocess_image(image):
    image = image.convert("L")  # 흑백 변환
    image = ImageOps.invert(image)  # 색 반전 (흰배경->검정글씨)
    image = image.resize((28, 28))  # 모델 입력 크기
    image = np.array(image).astype("float32") / 255.0  # 정규화
    image = image.reshape(1, 784)  # 입력 형태에 맞게 reshape
    return image

# 이미지 업로드 받기
uploaded_file = st.file_uploader("🖼️ 손글씨 숫자 이미지 업로드", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 이미지 표시
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드한 이미지", use_column_width=False, width=200)

    # 전처리 후 예측
    input_data = preprocess_image(image)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)

    st.success(f"🔍 예측된 숫자: **{predicted_class}**")
    st.bar_chart(prediction[0])

    st.markdown("---")

    st.markdown("✅ 숫자가 정확하게 인식되지 않는다면 다음을 확인하세요:")
    st.markdown("- 이미지 배경이 **흰색**, 숫자는 **진하게**")
    st.markdown("- 숫자 하나만 포함되어 있어야 해요")
