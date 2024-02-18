#!/usr/bin/env python3

import os, sys, time, re
import base64
import streamlit as st

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_experimental.text_splitter import SemanticChunker
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
    #st.write(str(str(len(docs))+ " documents extracted"))
    return(docs)

def chunk_text(docs):
    documents = []
    for p in range(0, len(docs)):
        text = docs[p].page_content
        single_sentence_list = re.split(r"(?<=[.?!])\s+", text)
        #print(f"{len(single_sentence_list)} sentences were found in {docs[p].metadata['source']} on page {docs[p].metadata['page']}")
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
        vectorstore.save_local("vector_db", index_name="pdf")
        return(vectorstore)

def get_context_retriever_chain(vectorstore):
    llm = ChatOpenAI()
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return(retriever_chain)

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question in a brief manner based on the context below:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}")
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return(create_retrieval_chain(retriever_chain, stuff_documents_chain))

def get_response(user_query):
    retriever_chain = get_context_retriever_chain(vector_db)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return(response['answer'])


def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display =  f"""<embed
    class="pdfobject"
    type="application/pdf"
    title="Embedded PDF"
    src="data:application/pdf;base64,{base64_pdf}"
    style="overflow: auto; width: 100%; height: 100%;">"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


### Streamlit page starts here ###

st.set_page_config(page_title="PDF Chatbot", page_icon=":books:", initial_sidebar_state="collapsed")
st.header("PDF Chatbot")

get_api_key()

# for paper in os.listdir("data/"):
#     st.write(paper)

## Processing pdfs
docs = process_pdfs()
chunks = chunk_text(docs)
#st.write(f"Number of resulting text chunks: {len(chunks)}")

## Creating vector store
#create_vectorstore(chunks)
vector_db = faiss.FAISS.load_local("vector_db", embeddings = OpenAIEmbeddings(), index_name="pdf")

displayPDF("data/johri-2022-recom-improv.pdf")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a helpful AI expert. How can I help you?"),]


user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)


with st.sidebar:
    st.write(st.session_state.chat_history)
