import streamlit as st
from src.services.contract_service import process_contract, analyze

st.set_page_config(page_title="ClauseAI - Milestone 1")

st.title("📄 ClauseAI - Contract Analyzer")

uploaded_file = st.file_uploader("Upload Contract PDF", type=["pdf"])

agent_option = st.selectbox(
    "Select Agent",
    ["Compliance", "Finance", "Legal", "Operations"]
)

if uploaded_file and st.button("Process Contract"):
    st.info("Processing contract...")
    process_contract(uploaded_file)
    st.success("Contract processed successfully!")

if st.button("Run Analysis"):
    st.info("Analyzing...")
    result = analyze(agent_option)
    st.write(result)

