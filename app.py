import json
import os
import sys

import streamlit as st
import boto3, base64, time, tempfile

# Using Titan Embeddings Model to generate Embedding
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat


# Data Ingestion
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# Vector Embedding and Vector Store
from langchain_community.vectorstores import FAISS


## LLM Models
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA


# AWS Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)



# Data Ingestion, Vector Embeding and Vector Store
def create_vector_store_from_pdfs(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    docs = []
    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(file_path)
        pdf_docs = loader.load()
        docs.extend(pdf_docs)

    # Splitting documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    split_docs = splitter.split_documents(docs)

    # Creating FAISS index
    vectorstore = FAISS.from_documents(split_docs, embedding=bedrock_embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore


# LLM Model selection
def get_llm(model_choice):
    if model_choice == "Claude 3 Haiku":
        return BedrockChat(model_id="anthropic.claude-3-haiku-20240307-v1:0", client=bedrock,
                           model_kwargs={'max_tokens': 512})
    elif model_choice == "Llama3 70B":
        return BedrockChat(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock,
                           model_kwargs={'max_gen_len': 512})
    elif model_choice == "Titan Text G1":
        return BedrockChat(model_id="amazon.titan-text-premier-v1:0", client=bedrock,
                           model_kwargs={'maxTokenCount': 512})
    else:
        raise ValueError("Invalid model choice")


prompt_template = """

Human: Use the following context from the PDF files to answer the user's question in a 
well-structured, detailed, and concise manner ( approximately 250 words). If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# RAG Response
def get_response_llm(llm,vectorstore,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer["result"],answer["source_documents"]



# Streamlit Application
def main():
    st.set_page_config(page_title="Chat with PDF using AWS Bedrock", layout="wide")

    # ---- Custom CSS ----
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 10% 20%, #0f2027, #203a43, #2c5364);
            color: white;
        }
        [data-testid="stSidebar"] {
            background: #111418;
            color: white;
        }
        .stButton>button {
            background-color: #00c3ff;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center;color:#00c3ff;'>Chat with Your Own PDFs using AWS Bedrock 💬📘</h1>",
                unsafe_allow_html=True)

    # ---- Sidebar ----
    with st.sidebar:
        st.header("📂 Upload PDF(s)")
        uploaded_files = st.file_uploader("Upload one or multiple PDFs", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            if st.button("📚 Create Vector Store"):
                with st.spinner("Processing PDFs and generating embeddings..."):
                    create_vector_store_from_pdfs(uploaded_files)
                    st.success("✅ Vector Store Created Successfully!")

        st.divider()
        st.subheader("🤖 Choose Model")
        model_choice = st.selectbox("", ["Claude 3 Haiku", "Llama3 70B", "Titan Text G1"])

        st.divider()
        st.caption("Made using LangChain + AWS Bedrock")

    # ---- Main Chat Section ----
    query = st.text_input("Ask a question from your uploaded PDFs:")
    if st.button("💬 Get Answer"):
        if not query.strip():
            st.warning("Please enter a question first.")
            return

        # Load FAISS (must exist)
        try:
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,
                                           allow_dangerous_deserialization=True)
        except:
            st.error("⚠️ Please upload PDFs and build vector store first.")
            return

        with st.spinner("Retrieving relevant context and generating answer..."):
            llm = get_llm(model_choice)
            answer, sources = get_response_llm(llm, faiss_index, query)

            # Typing effect
            placeholder = st.empty()
            typed = ""
            for ch in answer:
                typed += ch
                placeholder.markdown(f"<div style='color:white;font-size:18px;'>{typed}</div>",
                                     unsafe_allow_html=True)
                time.sleep(0.01)

            st.success("✅ Answer Generated!")

            # ---- Show Sources ----
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}:** {os.path.basename(doc.metadata.get('source','Unknown'))}")
                    st.markdown(f"Page: {doc.metadata.get('page', 'N/A')}")
                    st.markdown(f"Excerpt:** {doc.page_content[:300]}...**")
                    st.divider()


if __name__ == "__main__":
    main()