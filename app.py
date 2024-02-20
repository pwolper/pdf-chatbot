#!/usr/bin/env python3

import os, sys, time, re
import base64
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from streamlit_float import *
import bibtexparser

# Langchain imports
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
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

def process_dir():
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load_and_split()
    #st.write(str(str(len(docs))+ " documents extracted"))
    return(docs)

def process_file(file):
    loader = PyPDFLoader(file)
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

#@st.cache_data()
def create_vectorstore(chunks, file):
        embeddings = OpenAIEmbeddings()
        vectorstore = faiss.FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("vector_db", index_name=pdf)
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
    pdf_viewer(file, width=800)
    # # Opening file from file path
    # with open(file, "rb") as f:
    #     base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # # Embedding PDF in HTML
    # pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

    # # Displaying File
    # st.markdown(pdf_display, unsafe_allow_html=True)
    return

def parse_bibtex(file="articles.bib"):
    with open(file, 'r') as bib:
        library = bibtexparser.parse_string(bib.read())

        articles = []
        for entry in library.entries:
            info = {}
            info['title'] = entry['title']
            info['authors'] = entry['author']
            info['pdf'] = str(entry.key + ".pdf")
            articles.append(info)
    return(articles)



### Streamlit page starts here ###

st.set_page_config(page_title="PDF Chatbot", page_icon=":books:", initial_sidebar_state="collapsed", layout="wide")
st.title("pdf-chatbot: Question AI models about papers, while reading them")

float_init()

get_api_key()

articles = parse_bibtex()

a, b = st.columns([1,1])
with a:
    article = st.selectbox("Choose an article",
                           [a['title'] for a in articles],
                           index = None,
                           label_visibility="collapsed")

with b:
    st.button("Click for PDF", type="primary")

if article is None:
    st.stop()

dir = "data/"
pdf = str([a['pdf'] for a in articles if a['title'] == article][0])
file = dir + pdf

## Processing pdfs
docs = process_file(file)
chunks = chunk_text(docs)

#st.write(f"Number of resulting text chunks: {len(chunks)}")


## Creating vector store
if str(pdf + ".faiss") not in os.listdir("vector_db"):
    st.toast(":page_facing_up: Generating embeddings...")
    vector_db = create_vectorstore(chunks,file)
else:
    # st.toast(f"Embeddings found for the file {pdf} :smile:")
    vector_db = faiss.FAISS.load_local("vector_db", embeddings = OpenAIEmbeddings(), index_name=pdf)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a helpful AI expert. How can I help you?"),]

if "retrieved" not in st.session_state:
    st.session_state.retrieved = None

col1, col2 = st.columns([1.4,1], gap="small")

with col1:
    st.markdown("")
    displayPDF(file)

with col2:
    container = st.container()
    with container:
        st.markdown(str("**" + article + "**"))
        chat = st.container(height=410)
        user_query = st.chat_input("Type your message here...")

        if user_query is not None and user_query != "":
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

        # Print chat_history to chat
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with chat.chat_message("AI"):
                    st.write(message.content)
            if isinstance(message, HumanMessage):
                with chat.chat_message("Human"):
                    st.write(message.content)
    container.float()
    # container.button("Start", type="primary")


with st.sidebar:
    with st.expander("chat history"):
        st.write(st.session_state.chat_history)
    with st.expander("Retrieved chunks for last prompt"):
        st.write(st.session_state.retrieved)
