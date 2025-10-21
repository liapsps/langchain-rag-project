# 📖 My First LangChain Project: A RAG System for Document Q&A

This repository documents my first project as an AI Engineer Trainee, exploring the **LangChain** framework to build a Question & Answer (Q&A) system based on the seminal paper "Attention Is All You Need."

The goal is to create an application that allows a user to "chat" with a PDF document. Instead of reading the entire paper, we can ask questions in natural language and receive concise answers based exclusively on the document's content.

This project implements the **Retrieval-Augmented Generation (RAG)** technique, which enhances a Large Language Model's (LLM) capabilities by providing it with relevant information from an external knowledge base (in this case, the PDF).

## 🚀 How It Works

The project's workflow is divided into two main phases, orchestrated by the `rag_application.py` script:

### 1. Indexing Phase (Preparing the Knowledge Base)

Before we can ask questions, we need to process the document and make it "searchable" for our system.

1.  **PDF Loading**: The text from the `attention_is_all_you_need.pdf` file is loaded page by page.
2.  **Chunking**: The text is broken down into smaller pieces (chunks) of 1000 characters, with a 150-character overlap to ensure context is not lost between chunks.
3.  **Embedding Creation**: Each text chunk is converted into a numerical vector (embedding) using the `sentence-transformers/all-MiniLM-L6-v2` model. These vectors capture the semantic meaning of the text.
4.  **Vector Storage**: The embeddings are stored and indexed in a **FAISS** (from Meta AI) vector database. This data structure is optimized for high-speed similarity searches, allowing us to find the most relevant text chunks for any given query.

### 2. Generation Phase (Answering Questions)

Once the knowledge base is indexed, the system is ready to answer questions.

1.  **User Question**: The system receives a question in natural language (e.g., "What is a Transformer?").
2.  **Retrieval**: The question is also converted into an embedding vector. FAISS uses this vector to search its database for the most semantically similar text chunks (the most relevant ones).
3.  **Augmentation**: The retrieved chunks are combined with the original question into a structured prompt. This prompt instructs the LLM to use **only** the provided context to formulate its answer.
4.  **Generation**: The complete prompt is sent to the LLM (`google/flan-t5-base`). The model then generates a cohesive and contextualized response based on the provided document excerpts.

This cycle ensures that the answers are factual and grounded in the source document, minimizing the risk of LLM "hallucinations."

## 🛠️ Tech Stack

- **Framework**: LangChain
- **LLM (Language Model)**: `google/flan-t5-base` via Hugging Face Hub
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Document Loader**: `PyPDFLoader`

## ⚙️ How to Run the Project

Follow the steps below to run the application on your local machine.

### Prerequisites

- Python 3.8 or higher
- A Hugging Face API key (token) from [Hugging Face](https://huggingface.co/settings/tokens)

### Steps

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create a virtual environment and install dependencies:**

    ```bash
    # Create the environment
    python -m venv .venv

    # Activate the environment (Windows)
    .venv\Scripts\activate
    # Activate the environment (Linux/macOS)
    source .venv/bin/activate

    # Install the libraries
    pip install langchain langchain-community langchain-huggingface faiss-cpu pypdf sentence-transformers transformers torch
    ```

3.  **Download the PDF document:**
    Download the paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) and save it in the project's root directory with the filename `attention_is_all_you_need.pdf`.

4.  **Set up your API key:**
    Open the `rag_application.py` file and replace the placeholder with your Hugging Face API key:

    ```python
    api_token = "YOUR_HF_API_KEY_HERE"
    ```

5.  **Run the script:**
    ```bash
    python rag_application.py
    ```
    The script will execute the entire process: it will load the PDF, create the vector database, and finally, ask a few sample questions, displaying the generated answer and the source document chunks used to create it.

## 📂 Project Structure

```
/
├── rag_application.py          # Main script containing all the RAG logic
└── attention_is_all_you_need.pdf # The knowledge base for our system
```
