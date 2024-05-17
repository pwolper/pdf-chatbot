#!/usr/bin/env python3

import os, sys, time, re
import base64
import streamlit as st
import streamlit.components.v1 as components
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from streamlit_float import *

from langchain_core.messages import AIMessage, HumanMessage
from functions import parse_bibtex
from functions import process_dir, process_file, chunk_text
from functions import create_vectorstore, load_vectorstore
from functions import get_context_retriever_chain, get_conversational_rag_chain
from functions import get_response, llm_network_call, json_parsing, pyvis_graph


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

### Streamlit page starts here ###

st.set_page_config(page_title="PDF Chatbot", page_icon=":books:", initial_sidebar_state="expanded", layout="wide")

float_init()

get_api_key()

articles = parse_bibtex()


with st.sidebar:
    st.sidebar.title("PDF chatbot")
    st.markdown("**Question AI models about articles and generate knowledge graphs to enhance text understanding. Powered by OpenAI API and LangChain.**")


# a, b, c = st.columns([2,0.1,0.5])
# with a:
    article = st.selectbox("Choose an article",
                           [a['title'] for a in articles],
                           index = None,
                           label_visibility="collapsed")

    st.markdown("**About**")
    st.markdown("Created by Philip Wolper ([phi.wolper@gmail.com](philip.wolper@gmail.com)). Code is available at [https://github.com/pwolper/scimapai.git](https://github.com/pwolper/pdf-chatbot) here. Feeback is very welcome.")

if article is not None:
    dir = "data/"
    pdf = str([a['pdf'] for a in articles if a['title'] == article][0])
    file = dir + pdf

######## with b:
########     #st.button("Display PDF", type="primary")
########     st.write('or')

######## with c:
########     uploader = st.button("Upload a file")
########     if uploader:
########         upload = st.file_uploader(label="Upload your pdf",
########                                 label_visibility="collapsed")
########         if upload is not None:
########             # To read file as bytes:
########             bytes = upload.getvalue()
########             new_file = open('data/tmp/new.pdf', 'wb')
########             new_file.write(bytes)
########             article = 'User uploaded file'
########             file = 'data/tmp/new.pdf'
########             pdf = 'new.pdf'
########             st.write(article)


if article is None:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a helpful AI expert. How can I help you?"),]
    source_code = None
    st.stop()


## Processing pdfs
# docs = process_file(file)
# chunks = chunk_text(docs)

#st.write(f"Number of resulting text chunks: {len(chunks)}")


## Creating vector store
if str(pdf + ".faiss") not in os.listdir("vector_db"):
    st.toast(":page_facing_up: Generating embeddings...")
    docs = process_file(file)
    chunks = chunk_text(docs)
    vector_db = create_vectorstore(chunks,file)

else:
    #st.toast(f"Embeddings found for the file {pdf} :smile:")
    vector_db = load_vectorstore(pdf)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a helpful AI expert. How can I help you?"),]

if "retrieved" not in st.session_state:
    st.session_state.retrieved = None

col1, col2 = st.columns([1.3,1], gap="small")

with col2:
    tab1, tab2 = st.tabs(["GPT-4", "Knowledge Graph"])
    with tab1:
        container = st.container()
        with container:
            st.write("")
            #st.write("")
            #header = str([a['title'] + ' (' + a['authors'] + ', ' + a['year'] + ')' for a in articles if a['title'] == article][0])
            header = article
            st.markdown(str('**' + header + '**'))
            chat = st.container(height=400)
            user_query = st.chat_input("Type your message here...")

        # Print chat_history to chat
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with chat.chat_message("AI"):
                    st.write(message.content)
            if isinstance(message, HumanMessage):
                with chat.chat_message("Human"):
                    st.write(message.content)

        if user_query is not None and user_query != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))

            with chat.chat_message("Human"):
                st.markdown(user_query)

            with chat.chat_message("AI"):
                response = get_response(user_query, st.session_state.chat_history, vector_db)
                streamed_response = st.write_stream(response)

            st.session_state.chat_history.append(AIMessage(content=streamed_response))
        container.float()

    with tab2:
        if st.session_state.chat_history != [AIMessage(content="Hello, I am a helpful AI expert. How can I help you?"),]:
            mapping_output = llm_network_call(st.session_state.chat_history)
            nodes, edges = json_parsing(mapping_output)
            source_code=pyvis_graph(nodes, edges)
            st.markdown("**Knowledge Graph:**")
            components.html(source_code, height=500,width=600)
            download=st.download_button("Download HTML", data=source_code, file_name="knowledge_graph.html")

            with st.sidebar.expander("Debug"):
                st.write(mapping_output)
                st.write(nodes)
                st.write(edges)
   # container.button("Start", type="primary")

with col1:
    display_container = st.container()
    with display_container:
        # displayPDF(file)
        pdf_viewer(file, width=700, height=1000)

    display_container.float()

