#!/usr/bin/env python3

import os, sys, time, re
import base64
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv, find_dotenv
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
from functions import get_response, llm_network_call, json_parsing, split_edge_labels, pyvis_graph

load_dotenv(find_dotenv())

def get_api_key():
    if "openai_api_key" not in st.session_state:
        if not os.getenv("OPENAI_API_KEY"):
            st.write("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
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
    else:
        st.stop()
        return


### Streamlit page starts here ###

st.set_page_config(page_title="PDF Chatbot", page_icon=":books:", initial_sidebar_state="expanded", layout="wide")

margins_css = """
    <style>
        .main > div {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
    </style>
"""

st.markdown(margins_css, unsafe_allow_html=True)

float_init()

get_api_key()

articles = parse_bibtex()

if 'uploads' not in st.session_state:
    st.session_state['uploads'] = None

if 'article' not in st.session_state:
     st.session_state['article'] = None

if 'vector_db' not in st.session_state:
     st.session_state['vector_db'] = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a helpful AI expert. How can I help you?"),]

if "retrieved" not in st.session_state:
    st.session_state.retrieved = None

st.sidebar.title("PDF chatbot")
st.sidebar.markdown("**Question AI models about articles and generate knowledge graphs to enhance text understanding.** Powered by OpenAI API and LangChain.")

uploaded = st.sidebar.file_uploader("Upload an article",
                                    type="pdf", accept_multiple_files=True)

st.session_state.article = st.sidebar.selectbox("Browse library",
                                        [a['title'] for a in articles],
                                        index = None)

st.sidebar.markdown("**_Tips_**")
st.sidebar.markdown("Ask specific questions using keywords found in the paper!")
st.sidebar.markdown("Be patient and try different prompts.")

st.sidebar.markdown("**_About_**")
st.sidebar.markdown("Created by Philip Wolper ([phi.wolper@gmail.com](philip.wolper@gmail.com)). Code is available at [https://github.com/pwolper/pdf-chatbot.git](https://github.com/pwolper/pdf-chatbot) here. Feeback is very welcome.")

if uploaded:
    st.session_state.uploads = {}
    for file in uploaded:
        binary = file.getvalue()
        st.session_state.uploads[file.name] = binary
        with NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(binary)
            st.session_state['tmp_file_path'] = tmp_file.name
            st.session_state.article = "Upload"

            if str(file.name + ".faiss") not in os.listdir("vector_db"):
                st.toast(f":page_facing_up: Reading file \"{file.name}\" and generating embeddings...")
                st.write("")
                with st.spinner(f"Reading file \"{file.name}\" and generating embeddings"):
                    docs = process_file(st.session_state.tmp_file_path)
                    chunks = chunk_text(docs)
                    st.session_state.vector_db = create_vectorstore(chunks,file.name)

    if len(st.session_state.uploads.keys()) > 1:
        st.session_state.vector_db = {}
        for i, upload in enumerate(list(st.session_state.uploads.keys())):
            if i == 0:
                st.session_state.vector_db = load_vectorstore(upload)
                print(f"Created vectorstore for {upload}")
            else:
                db = load_vectorstore(upload)
                st.session_state.vector_db.merge_from(db)
                print(f"Merged {upload} into existing vectorstore")
    else:
        st.session_state.vector_db = load_vectorstore(list(st.session_state.uploads.keys())[0])
        print(f"Created vectorstore for {list(st.session_state.uploads.keys())[0]}")

else: #article from library list
    st.session_state.uploads = {}
    if st.session_state.article is not None:
        dir = "articles/"
        pdf = str([a['pdf'] for a in articles if a['title'] == st.session_state.article][0])
        file = dir + pdf

        if str(pdf + ".faiss") not in os.listdir("vector_db"):
            st.toast(":page_facing_up: Generating embeddings...")
            docs = process_file(file)
            chunks = chunk_text(docs)
            st.session_state.vector_db = create_vectorstore(chunks,file)
        else:
            st.session_state.vector_db = load_vectorstore(pdf)


if st.session_state.article is None:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a helpful AI expert. How can I help you?"),]
    source_code = None
    st.stop()


col1, col2 = st.columns([1,1], gap="small")

with col1:
    display_container = st.container()
    with display_container:
        if st.session_state['uploads']:
            if len(st.session_state.uploads.keys()) > 1:
                tab_labels = st.session_state.uploads.keys()
                tabs = st.tabs(tab_labels)
                for label, tab in zip(tab_labels, tabs):
                    with tab:
                        pdf_viewer(st.session_state.uploads[label], width=800, height=1000)
            else:
                pdf_viewer(list(st.session_state.uploads.values())[0], width=800, height=1000)
        else:
            pdf_viewer(file, width=800, height=1000)
    display_container.float()

with col2:
    tab1, tab2 = st.tabs(["GPT-4", "Knowledge Graph"])
    with tab1:
        container = st.container()
        with container:
            st.write("")
            #st.write("")
            #header = str([a['title'] + ' (' + a['authors'] + ', ' + a['year'] + ')' for a in articles if a['title'] == article][0])
            header = st.session_state.article
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
                response = get_response(user_query, st.session_state.chat_history, st.session_state.vector_db)
                streamed_response = st.write_stream(response)

            st.session_state.chat_history.append(AIMessage(content=streamed_response))
        container.float()

    with tab2:
        if st.session_state.chat_history != [AIMessage(content="Hello, I am a helpful AI expert. How can I help you?"),]:
            st.toast(":spider_web: Generating knowledge graph...")
            with (st.spinner('Generating knowledge graph...')):
                mapping_output = llm_network_call(st.session_state.chat_history)
                nodes, edges = json_parsing(mapping_output)
                split_edge_labels(edges)
                source_code=pyvis_graph(nodes, edges)
            st.markdown("**Knowledge Graph:**")
            components.html(source_code, height=500,width=800)
            download=st.download_button("Download HTML", data=source_code, file_name="knowledge_graph.html")

            with st.sidebar.expander("Debug"):
                st.write(mapping_output)
                st.write(nodes)
                st.write(edges)
   # container.button("Start", type="primary")
