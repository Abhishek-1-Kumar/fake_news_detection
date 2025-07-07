import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os
HF_TOKEN = os.getenv("HF_TOKEN")


MODEL_NAME = "shi13u/fake_news_detection_bert"  

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,token=HF_TOKEN)

model.eval()

# Streamlit page settings
st.set_page_config(
    page_title="📰 Fake News Detector",
    layout="centered"
)

# Title
st.markdown("<h1 style='text-align: center;'>🤖 Fake News Detection Chatbot</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Paste a news snippet below to check if it's <strong>Real ✅</strong> or <strong>Fake ❌</strong> using BERT!</p>",
    unsafe_allow_html=True
)

# Input area
user_input = st.text_area("🗞️ Enter News Article or Headline:", height=200)

# On click
if st.button("🔍 Check Now"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
            pred = np.argmax(probs)
            confidence = float(np.max(probs) * 100)

        # Label with emoji
        if pred == 1:
            label = "✅ **This looks like REAL news!**"
            emoji = "🟢"
        else:
            label = "❌ **This might be FAKE news!**"
            emoji = "🔴"

        # Display results
        st.markdown(f"### {emoji} {label}")
        st.markdown(f"📊 **Confidence:** `{confidence:.2f}%`")
