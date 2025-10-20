import streamlit as st
from app.model import Predictor

@st.cache_resource
def load_model():
    return Predictor(trained_dir="./trained_results_tf2_XXXX")  # replace folder name

predictor = load_model()

st.set_page_config(page_title="Text Classification", page_icon="🧠")

st.title("🧠 Multi-Class Text Classification App")
st.markdown("Enter a sentence and see its predicted class!")

text_input = st.text_area("Enter text:", height=150)

if st.button("Predict"):
    if text_input.strip():
        label = predictor.predict([text_input])[0]
        st.success(f"**Predicted label:** {label}")
    else:
        st.warning("Please enter some text.")
