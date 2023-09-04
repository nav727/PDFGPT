#!/usr/bin/env python
# coding: utf-8

################### Imports ###################
import os
from dotenv import load_dotenv

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.mention import mention

from pypdf import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings

from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
################### Imports ###################

################### Helper Functions ###################
def get_pdf_text(pdf_docs):
    """
    Extract text from list of pdfs files
    """
    combined_text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)        
        for page in pdf_reader.pages:
            combined_text += page.extract_text()
    return combined_text

def get_text_chunks(text):
    """
    Breakup text into chunks before 
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings()
    
    # https://huggingface.co/spaces/mteb/leaderboard
#     embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-large-en")

    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en",
    encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI()
    # https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":2048})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_question(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.info(response["answer"])
################### Helper Functions ###################



def main():
    load_dotenv()
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    ################### Main Page ###################
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")
    
    user_question = st.text_input("Ask a question about your documents", placeholder="write your question here!", max_chars=1000)
    if user_question:
        handle_user_question(user_question)
    ################### Main Page ###################

    ################### Side Navigation ###################
    with st.sidebar:
        st.subheader(":notebook: Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here :arrow_down:", accept_multiple_files=True)
        
        if pdf_docs and st.button("Upload"):
            with st.spinner("Processing"):
                # get pdf text
                combined_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(combined_text)
                
                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
        
                st.success('Upload complete! Please proceed to ask your questions', icon="✅")

        st.sidebar.header((":pushpin: Resources:"))
        st.sidebar.markdown(("""
        - Text Embedding Model - [Instructor-xl](https://huggingface.co/hkunlp/instructor-xl)
        - Open Source LLM - [flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl)
        """
        ))
        
        # Push it to the bottom
        add_vertical_space(26)
        
        mention(
            label="PDFGPT",
            icon="github",
            url="https://github.com/nav727/PDFGPT",
        )
    ################### Side Navigation ###################

    
# References:                
# https://github.com/streamlit/streamlit/wiki/Installing-in-a-virtual-environment
# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
# https://extras.streamlit.app/
if __name__ == '__main__':
    main()