#!/usr/bin/env python
# coding: utf-8

################### Imports ###################
from constants import *
from pdfqa import PdfQA

import time
import shutil
import os
from dotenv import load_dotenv
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.mention import mention
################### Imports ###################

################### Helper Functions ###################
@st.cache_resource
def load_emb(emb): # will cache for multiple sessions
    if emb == EMB_INSTRUCTOR_XL:
        return PdfQA.create_instructor_xl()
    elif emb == EMB_BGE_SMALL_EN:
        return PdfQA.create_bge_small()
    else:
        raise ValueError("Invalid embedding setting")
        
@st.cache_resource
def load_llm(llm, load_in_8bit): # will cache for multiple sessions
    if llm == LLM_FLAN_T5_LARGE:
        return PdfQA.create_flan_t5_large(load_in_8bit)
    elif llm == LLM_FALCON_SMALL:
        return PdfQA.create_falcon_instruct_small(load_in_8bit)
    else:
        raise ValueError("Invalid LLM setting")
################### Helper Functions ###################


def main():
    load_dotenv()
    HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    
    if "pdf_qa_model" not in st.session_state:
        st.session_state["pdf_qa_model"] = PdfQA()
            
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")
                
    ################### Side Navigation ###################
    with st.sidebar:
        st.subheader(":notebook: Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here :arrow_down:", accept_multiple_files=False, type="pdf")
        
        if pdf_docs and st.button("Upload"):
            with st.spinner("Processing"):
                with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    shutil.copyfileobj(pdf_docs, tmp)
                    tmp_path = Path(tmp.name)
                    st.session_state["pdf_qa_model"].config = {
                        "pdf_path": tmp_path,
                        "embedding": EMB_TO_USE,
                        "llm": LLM_TO_USE,
                        "load_in_8bit": LOAD_IN_8BIT
                    }
                    st.session_state["pdf_qa_model"].embedding = load_emb(EMB_TO_USE)
                    st.session_state["pdf_qa_model"].llm = load_llm(LLM_TO_USE, LOAD_IN_8BIT)        
                    st.session_state["pdf_qa_model"].init_embeddings()
                    st.session_state["pdf_qa_model"].init_llm()
                    st.session_state["pdf_qa_model"].vector_db_pdf()
            st.success('Upload complete! Please proceed to ask your questions', icon="✅")

        st.sidebar.header((":pushpin: Resources:"))
        st.sidebar.markdown(("""
        - Text Embedding Model - [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en)
        - Open Source LLM - [google/flan-t5-large](https://huggingface.co/google/flan-t5-large)
        """
        ))
        
        add_vertical_space(26)   # Push it to the bottom
        
        mention(
            label="PDFGPT",
            icon="github",
            url="https://github.com/nav727/PDFGPT",
        )
    ################### Side Navigation ###################    
    
    
    ################### Main Page ###################
    user_question = st.text_input("Ask a question about your documents", placeholder="type your question here!", max_chars=1000)
    
    if user_question and len(user_question.split()) < 3: # removing invalid questions
        st.error('Invalid question!', icon="🚨")
        return
    
    if user_question:
        if pdf_docs is None: # can't answer until doc not uploaded
            st.error('Please upload a PDF document!', icon="🚨")
        else:
            try:
                st.session_state["pdf_qa_model"].retreival_qa_chain()
                answer = st.session_state["pdf_qa_model"].answer_query(user_question)
                st.info(f"{answer}")
            except Exception as e:
                st.error(f"Error answering the question")                
#                 st.error(f"Error answering the question: {str(e)}")
    ################### Main Page ###################
    
    
# References:                
# https://github.com/streamlit/streamlit/wiki/Installing-in-a-virtual-environment
# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
# https://extras.streamlit.app/
# https://github.com/streamlit/streamlit/issues/652
# https://medium.com/analytics-vidhya/installing-cuda-and-cudnn-on-windows-d44b8e9876b5
# https://medium.com/@jjlovesstudying/python-cuda-set-up-on-windows-10-for-gpu-support-78126284b085
# https://gist.github.com/chizhang529/3c414428cd1c82e38e7dde0be70e2955
# https://huggingface.co/blog/hf-bitsandbytes-integration#:~:text=Hardware%20requirements,A40%2DA100%2C%20T4%2B).
# https://heidloff.net/article/running-llm-flan-t5-locally/
# https://github.com/xlang-ai/instructor-embedding/blob/main/requirements.txt

if __name__ == '__main__':
    main()