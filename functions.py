#!/usr/bin/env python3

import re, time, os
from typing import List
import json
from pyvis.network import Network
from pyvis import network as net
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

def process_dir():
    loader = PyPDFDirectoryLoader("articles/")
    docs = loader.load_and_split()
    #st.write(str(str(len(docs))+ " documents extracted"))
    return(docs)

def process_file(file):
    loader = PyPDFLoader(file)
    docs = loader.load_and_split()
    #st.write(str(str(len(docs))+ " documents extracted"))
    #print('file processed...')
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
def create_vectorstore(chunks, pdf):
    if not os.path.exists("vector_db"):
        os.makedirs("vector_db")

    embeddings = OpenAIEmbeddings()
    vectorstore = faiss.FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vector_db", index_name=pdf)
    return(vectorstore)

def load_vectorstore(index_name):
    vectorstore = faiss.FAISS.load_local("vector_db",
                                         embeddings = OpenAIEmbeddings(),
                                         index_name = index_name,
                                         allow_dangerous_deserialization=True)
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
    llm = ChatOpenAI(model='gpt-4')
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question in a brief manner based on the context below:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}")
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return(create_retrieval_chain(retriever_chain, stuff_documents_chain))

def get_response(user_query, chat_history, vector_db):
    retriever_chain = get_context_retriever_chain(vector_db)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)


    response = conversation_rag_chain.stream({"chat_history": chat_history,
                                              "input": user_query})
    for chunk in response:
        content = chunk.get("answer", "")
        yield content

def llm_network_call(chat_history):
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)

    # mapping_template = """
    # You are an scientific assistant, tasked with extracting the main concepts and key words from articles and abstracts. I want to create a concept map to enhance understanding of the text.
    # Breaking down the text hierarchically and semantically, I want you to choose high-level concepts as nodes, avoiding overlap between concepts. Keep the number of nodes to a minimum.
    # Concepts should be as atomistic as possible, but chosen so that there is high connectivity in the graph.
    # Return nodes and edges for the concept map and come up with a sentence explaining each edge as well.
    # Text: {text}

    # Strictly return a list of json objects, with the following fields:
    # "node_1", "node_2" and "edge". Where "node_1" and "node_2", represent two nodes and "edge" is a string containing a sentence describing the relationship between the nodes.
    # Do not wrap the output in ```json ```.
    # """

    mapping_template = """
    You are an assistant, tasked with extracting the main concepts and topics from conversation. I want to create a concept map to enhance understanding of the complex topics.
    Breaking down the conversation hierarchically and semantically, I want you to choose high-level concepts as nodes, avoiding overlap between concepts. Keep the number of nodes to a minimum.

    Concepts should be as atomistic as possible, but chosen so that there is high connectivity in the graph.
    Return nodes and edges for the concept map and come up with a very short sentence explaining each edge as well.
    Text: {text}

    Strictly return a list of json objects, with the following fields:
    "node_1", "node_2" and "edge". Where "node_1" and "node_2", represent two nodes and "edge" is a string containing a sentence describing the relationship between the nodes.
    Do not wrap the output in ```json ```.
    """
    mapping_prompt = ChatPromptTemplate.from_messages(
        [("system", mapping_template),
         ("human", "{text}"),
         ])

    messages = mapping_prompt.format_messages(text=chat_history)
    answer = llm(messages)
    output = answer.content
    return(output)

def json_parsing(mapping_output):
    output_dict = json.loads(mapping_output)

    nodes = []
    edges = []

    for dict in output_dict:
        node_1 = dict['node_1']
        node_2 = dict['node_2']
        edge = dict['edge']

        nodes.append(node_1)
        nodes.append(node_2)

        edges.append((node_1, node_2, edge))
    return(nodes, edges)

def split_edge_labels(edges):
    labels = []
    for edge in edges:
        labels.append(edge[2])
        label = edge[2]
        if len(label.split(' ')) > 11:
            new_label = ' '.join(edge[2].split(' ', 10)[:10]) + '\n' + edge[2].split(' ', 10)[10]
            print(new_label)
    return(labels)


def pyvis_graph(nodes, edges):
    nt = Network(directed=False,
                 notebook=True,height="490px",width="799px",
                #height="480px",
                #width="620px",
                #width="940px",
                heading='')

    for n in nodes:
        nt.add_node(n,
                    title=n,
                    size=15)

    for source, target, label in edges:
        nt.add_edge(source,
                    target,
                    title=label)
    # nt.barnes_hut()
    nt.show('pyvis_knowledge_graph.html')
    html_file = open('./pyvis_knowledge_graph.html', 'r', encoding='utf-8')
    source_code = html_file.read()
    return(source_code)

def parse_bibtex(file="articles/articles.bib"):
    with open(file, 'r') as bib:
        library = bibtexparser.load(bib)

        articles = []
        for entry in library.entries:
            info = {}
            info['title'] = entry['title']
            info['authors'] = entry['author']
            info['pdf'] = str(entry['ID'] + ".pdf")
            info['year'] = entry['year']
            info['url'] = entry['url']
            articles.append(info)
    return(articles)
