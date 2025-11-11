# 💬 Chat with Your Own PDFs using AWS Bedrock

> “Turn static PDFs into interactive conversations — powered by AWS Bedrock, LangChain, and Streamlit.”

---

## App Interface
![Demo](demo/main_page.png)

## 🚀 Overview
**Chat with Your Own PDFs using AWS Bedrock** is a **Retrieval-Augmented Generation (RAG)** application that allows users to **upload any PDF**, automatically generate embeddings using **Amazon Titan**, and query the content conversationally using **Claude 3 Haiku**, **Llama3 70B**, or **Titan Text G1** — all served via **AWS Bedrock**.

This project bridges cloud AI models with document intelligence and a clean, modern Streamlit interface — enabling real-time, explainable responses grounded in your data.

---

## 🧠 Core Features

| Category | Description |
|-----------|--------------|
| **📂 Real-time PDF Uploads** | Upload multiple PDFs directly through the UI and instantly build a FAISS vector store. |
| **🧮 Titan Text Embeddings** | Each document chunk is vectorized using `amazon.titan-embed-text-v1` for high-precision retrieval. |
| **🤖 Multi-Model Orchestration** | Choose between **Claude 3 Haiku**, **Llama3 70B**, or **Titan Text G1** at runtime. |
| **🔍 RAG Pipeline** | Combines contextual retrieval + prompt generation for accurate answers. |
| **🎨 Interactive UI** | Streamlit-powered with gradient backgrounds, dynamic progress spinners, and typing animations. |
| **📚 Source Transparency** | Displays the original page number, file name, and excerpt for every generated response. |
| **🧱 Modular Design** | Fully modular functions for ingestion, embeddings, retrieval, and model execution. |

---

## 🏗️ Tech Stack

| Layer | Technology |
|--------|-------------|
| **Frontend/UI** | Streamlit |
| **Backend** | Python 3.10 + Boto3 |
| **Cloud AI Models** | AWS Bedrock (Claude 3 Haiku, Llama3 70B, Titan Text G1, Titan Embeddings) |
| **Vector Store** | FAISS |
| **Frameworks** | LangChain, LangChain-Community |
| **Environment** | MacBook M3 + Visual Studio Code |

---

## 🧩 Architecture

```mermaid
flowchart TD
    A["Upload PDFs"] --> B["Text Splitter (RecursiveCharacterTextSplitter)"]
    B --> C["Titan Embeddings via Bedrock"]
    C --> D["FAISS Vector Store"]
    D --> E["Bedrock LLM (Claude / Llama3 / Titan)"]
    E --> F["Streamlit UI Response"]
    D --> G["Source Documents Display"]


