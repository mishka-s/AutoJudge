import streamlit as st
import joblib
import numpy as np
import re
from scipy.sparse import hstack

# --------------------------------
# Load models and preprocessors
# --------------------------------
svm_clf = joblib.load("svm_classifier.pkl")
rf_reg = joblib.load("rf_regressor.pkl")

tfidf_cls = joblib.load("tfidf_classifier.pkl")
tfidf_reg = joblib.load("tfidf_regressor.pkl")

scaler = joblib.load("feature_scaler.pkl")

# --------------------------------
# Feature extraction
# --------------------------------
def extract_extra_features(text):
    text = text.lower()

    # log text length
    text_len_log = np.log1p(len(text))

    # keyword count
    keywords = [
        "dp", "dynamic programming", "graph", "tree",
        "dfs", "bfs", "recursion", "binary search", "greedy"
    ]
    keyword_count = sum(text.count(k) for k in keywords)

    # math symbol density
    math_density = (
        len(re.findall(r"[=<>+\-*/%]", text)) / len(text)
        if len(text) > 0 else 0
    )

    return np.array([[text_len_log, keyword_count, math_density]])

# --------------------------------
# Streamlit UI
# --------------------------------
st.set_page_config(page_title="Problem Difficulty Predictor")

st.title("🧠 Problem Difficulty Predictor")

desc = st.text_area("📘 Problem Description", height=200)
inp = st.text_area("📥 Input Description", height=120)
out = st.text_area("📤 Output Description", height=120)

if st.button("🔮 Predict Difficulty"):
    if desc.strip() == "":
        st.warning("Please enter the problem description.")
    else:
        full_text = desc + " " + inp + " " + out

        # ---------------- CLASSIFICATION ----------------
        X_text_cls = tfidf_cls.transform([full_text])
        X_extra = extract_extra_features(full_text)
        X_extra_scaled = scaler.transform(X_extra)

        X_cls_final = hstack([X_text_cls, X_extra_scaled])
        pred_class = svm_clf.predict(X_cls_final)[0]

        # ---------------- REGRESSION ----------------
        X_text_reg = tfidf_reg.transform([full_text])
        X_reg_final = hstack([X_text_reg, X_extra_scaled])
        pred_score = rf_reg.predict(X_reg_final.toarray())[0]

        # ---------------- OUTPUT ----------------
        st.success(f"📊 Predicted Difficulty Class: **{pred_class}**")
        st.info(f"📈 Predicted Difficulty Score: **{pred_score:.2f}**")

