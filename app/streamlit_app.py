import sys
import os
import streamlit as st
import PyPDF2

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.contract_service import process_contract, analyze

st.set_page_config(page_title="ClauseAI", layout="wide")

st.title("📄 ClauseAI - Contract Analyzer")

# =========================
# Session State
# =========================
if "contract_processed" not in st.session_state:
    st.session_state.contract_processed = False


# =========================
# Upload Contract
# =========================
uploaded_file = st.file_uploader("Upload Contract (PDF)", type=["pdf"])

if uploaded_file is not None and not st.session_state.contract_processed:

    with st.spinner("Processing contract..."):

        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

        if not text.strip():
            st.error("❌ Could not extract text from PDF. It may be scanned.")
        else:
            result = process_contract(text)
            st.session_state.contract_processed = True
            st.success(result)


# =========================
# Ask Questions
# =========================
if st.session_state.contract_processed:

    st.divider()
    st.subheader("🔎 Ask About This Contract")

    query = st.text_area("Enter your legal question:")

    if st.button("Analyze"):

        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing contract..."):
                response = analyze(query)

            st.markdown("### 📑 Legal Analysis")
            st.write(response)
