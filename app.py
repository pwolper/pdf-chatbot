#!/usr/bin/env python3

import os, sys, time, re
import streamlit as st

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import faiss


### Functions ###
def get_api_key():
    if "openai_api_key" not in st.session_state:
        if not os.getenv("OPENAI_API_KEY"):
            openai_input_field=st.empty()
            openai_input_field.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...",
                                          key="openai_api_key_input", type="password",
                                          help="Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)")
            if st.session_state.openai_api_key_input != "":
                st.session_state.openai_api_key=st.session_state.openai_api_key_input
                openai_input_field.success("API key saved...")
                time.sleep(.5)
                openai_input_field.empty()
        else:
            st.session_state.openai_api_key=os.getenv("OPENAI_API_KEY")
            return



def process_pdfs():
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load_and_split()
    st.write(str(str(len(docs))+ " documents extracted"))
    return(docs)

def chunk_text(docs):
    documents = []
    for p in range(0, len(docs)):
        text = docs[p].page_content
        single_sentence_list = re.split(r"(?<=[.?!])\s+", text)
        print(f"{len(single_sentence_list)} sentences were found in {docs[p].metadata}")
        for i in range(0, len(single_sentence_list)-3, 3):
            chunk = single_sentence_list[i:i+4]
            chunk = " ".join(chunk)
            doc = Document(page_content=chunk, metadata=docs[p].metadata)
            documents.append(doc)
    return(documents)


#@st.cache_data(show_spinner="Generating vectorstore from embeddings and text...")
def create_vectorstore(chunks):
        embeddings = OpenAIEmbeddings()
        vectorstore = faiss.FAISS.from_documents(chunks, embeddings)
        return(vectorstore)

#@st.cache_data(show_spinner="Summarizing the document...")
def summary_chain(chunks, openai_api_key):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.5, openai_api_key=openai_api_key)
    chain = load_summarize_chain(llm=llm, chain_type="stuff")
    summarization = chain.invoke(chunks)
    return(summarization)




### Streamlit page starts here ###

st.set_page_config(page_title="PDF Chatbot", page_icon=":books:", initial_sidebar_state="collapsed")
st.header("PDF Chatbot")

get_api_key()

for paper in os.listdir("data/"):
    st.write(paper)

docs = process_pdfs()
chunks = chunk_text(docs)

    # chunks.append


# chunks = semantic_chunking(docs)
