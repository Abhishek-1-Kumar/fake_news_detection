import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import numpy as np

# Load model and tokenizer from Google Drive (after downloading or syncing)
model_path = "bert_fake_news_model"  # or full Drive path if running locally
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
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
