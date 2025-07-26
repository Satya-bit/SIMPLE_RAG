import streamlit as st
import time
from langchain_openai import OpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()




st.title("RAG App Demo")
st.sidebar.markdown("This is a demo application for Retrieval-Augmented Generation (RAG) using LangChain and OpenAI.")

urls = st.text_input("Enter the URL")
submit_button = st.button("Submit")

# Use session state to persist retriever and llm
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None
if 'llm' not in st.session_state:
    st.session_state['llm'] = None

if submit_button:
    if urls:
        try:
            url_list = [urls]
            loader = UnstructuredURLLoader(urls=url_list)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
            docs = text_splitter.split_documents(data)
            all_splits = docs
            vectorstore = FAISS.from_documents(all_splits, OpenAIEmbeddings())
            st.session_state['retriever'] = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            st.session_state['llm'] = OpenAI(temperature=0.4, max_tokens=500)
            st.success("URL processed. You can now ask questions.")
        except Exception as e:
            st.error("Please enter a correct URL or check your connection.")
    else:
        st.write("Please enter a URL to proceed.")

# Only show chat input if retriever and llm are available
if st.session_state['retriever'] and st.session_state['llm']:
    query = st.chat_input("Ask me anything related to provided URL: ")
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    if query:
        question_answer_chain = create_stuff_documents_chain(st.session_state['llm'], prompt)
        rag_chain = create_retrieval_chain(st.session_state['retriever'], question_answer_chain)
        response = rag_chain.invoke({"input": query})
        print(response["answer"])
        st.write(response["answer"])

    