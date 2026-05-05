import os
import streamlit as st
import pandas as pd
from modules.ingestion import process_pdf
from modules.rag_pipeline import RAGPipeline
from modules.workflow import create_workflow
from modules.react_agent import run_react_agent
from config import APP_TITLE, DATA_DIR

# --- UI Configuration ---
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🧠")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { border-radius: 20px; background-color: #4CAF50; color: white; }
    .stTextInput>div>div>input { border-radius: 10px; }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7d32, #1b5e20); color: white; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title(APP_TITLE)
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.markdown("---")
    
    st.subheader("📁 Document Management")
    uploaded_file = st.file_uploader("Upload Knowledge PDF", type="pdf")
    
    if uploaded_file:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Process & Index Document"):
            with st.spinner("Processing PDF..."):
                chunks = process_pdf(file_path)
                if chunks:
                    rag = RAGPipeline()
                    if rag.build_index(chunks):
                        st.success("Document indexed successfully!")
                    else:
                        st.error("Indexing failed.")
                else:
                    st.error("Could not extract text from PDF.")

    st.markdown("---")
    mode = st.radio("Select Mode", ["Intelligent Workflow", "Direct RAG", "ReAct Reasoning", "General Chat"])

# --- Main Interface ---
st.header(f"Mode: {mode}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        try:
            if mode == "Intelligent Workflow":
                with st.status("Analyzing & Routing...", expanded=True) as status:
                    app = create_workflow()
                    config = {"configurable": {"thread_id": "1"}}
                    result = app.invoke({"query": prompt}, config)
                    status.update(label="Response Generated!", state="complete", expanded=False)
                response = result["response"]
                
            elif mode == "Direct RAG":
                rag = RAGPipeline()
                res = rag.answer_question(prompt)
                response = res["result"] if isinstance(res, dict) else str(res)
                
            elif mode == "ReAct Reasoning":
                with st.spinner("Reasoning..."):
                    response = run_react_agent(prompt)
                    
            else: # General Chat
                from modules.general_llm import get_general_answer
                res = get_general_answer(prompt)
                response = res.content if hasattr(res, 'content') else str(res)

            response_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("GenAI Research & Decision Assistant | Built with LangChain, LangGraph & Streamlit")
