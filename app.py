import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

st.title("✍️ 손글씨 숫자 인식기")
st.write("손글씨 숫자 이미지를 업로드하면 인식 결과를 보여줍니다.")

# 모델 로드
model = load_model("my_model.h5")

# 이미지 업로드
uploaded_file = st.file_uploader("숫자 이미지 파일을 업로드하세요", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # 흑백
    st.image(image, caption="업로드한 이미지", use_column_width=False)

    image = ImageOps.invert(image)                 # 색 반전
    image = image.resize((28, 28))                 # 28x28 크기로 조정
    img_array = np.array(image) / 255.0            # 정규화
    img_array = img_array.reshape(1, 784)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.subheader(f"✅ 예측 결과: {predicted_digit}")
