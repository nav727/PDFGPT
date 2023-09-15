from constants import *
import os

import streamlit as st

from pypdf import PdfReader
import torch

from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

from transformers import AutoTokenizer
from transformers import pipeline

class PdfQA:
    def __init__(self, config:dict = {}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None
    
    
    ################# Embeddings Options #################
    @classmethod
    def create_instructor_xl(cls):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceInstructEmbeddings(model_name=EMB_INSTRUCTOR_XL, model_kwargs={"device": device})
    
    @classmethod
    def create_bge_small(cls):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceBgeEmbeddings(model_name=EMB_BGE_SMALL_EN, model_kwargs={"device": device}, 
                                        encode_kwargs= {'normalize_embeddings': True})
    ################# Embeddings Options #################

    
    ################# LLM Options #################
    @classmethod
    def create_flan_t5_large(cls, load_in_8bit=False):
        return pipeline(
            task="text2text-generation",
            model=LLM_FLAN_T5_LARGE,
            max_new_tokens=512,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.2}
        )
    
    @classmethod
    def create_falcon_instruct_small(cls, load_in_8bit=False):
        tokenizer = AutoTokenizer.from_pretrained(model)
        hf_pipeline = pipeline(
                task="text-generation",
                model = LLM_FALCON_SMALL,
                tokenizer = tokenizer,
                trust_remote_code = True,
                max_new_tokens=100,
                model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.01,
                              "torch_dtype":torch.bfloat16}
            )
        
        return hf_pipeline
    ################# LLM Options #################   
    
    def init_embeddings(self) -> None:
        """
        Initialize embedding models based on config.
        """
        
        if self.config["embedding"] == EMB_INSTRUCTOR_XL:
            if self.embedding is None:
                self.embedding = PdfQA.create_instructor_xl()
                
        elif self.config["embedding"] == EMB_BGE_SMALL_EN:
            if self.embedding is None:
                self.embedding = PdfQA.create_bge_small()
        else:
            self.embedding = None
            raise ValueError("Invalid config")
            
            
    def init_llm(self) -> None:
        """ 
        Initialize LLM models based on config.
        """
        
        load_in_8bit = self.config.get("load_in_8bit", False)
        if self.config["llm"] == LLM_FLAN_T5_LARGE:
            if self.llm is None:
                self.llm = PdfQA.create_flan_t5_large(load_in_8bit=load_in_8bit)
        
        elif self.config["llm"] == LLM_FALCON_SMALL:
            if self.llm is None:
                self.llm = PdfQA.create_falcon_instruct_small(load_in_8bit=load_in_8bit)
        else:
            raise ValueError("Invalid config")       
        
    
    def vector_db_pdf(self) -> None:
        """
        creates vector db for the embeddings and persists them 
        OR loads a vector db from the persist directory.
        """
        
        pdf_path = self.config.get("pdf_path", None)
        persist_directory = self.config.get("persist_directory", None)
        
        if persist_directory and os.path.exists(persist_directory):
            self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
        
        elif pdf_path and os.path.exists(pdf_path):
            loader = PyPDFLoader(fr"{pdf_path}")
            pages = loader.load_and_split(text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        ))            
            self.vectordb = Chroma.from_documents(documents=pages, embedding=self.embedding, persist_directory=persist_directory)
            
        else:
            raise ValueError("No PDF file found")

            
    def retreival_qa_chain(self):
        """
        Creates retrieval qa chain using vectordb as retrivar and LLM to complete the prompt
        """
        
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k":3})
        hf_llm = HuggingFacePipeline(pipeline=self.llm, model_id=self.config["llm"])

        self.qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff", retriever=self.retriever)
        self.qa.combine_documents_chain.verbose = True
        self.qa.return_source_documents = True
            
            
    def answer_query(self, question:str) -> str:
        """
        Respond to the question
        """

        answer_dict = self.qa({"query": question})
#         st.write(answer_dict)
        answer = answer_dict["result"]
        return answer
    