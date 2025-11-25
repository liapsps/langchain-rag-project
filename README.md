# 📖 My First LangChain Project: A RAG System for Document Q\&A

This repository documents my first project as an AI Engineer Trainee, exploring the LangChain framework to build a Question & Answer (Q\&A) system based on the seminal paper "Attention Is All You Need."

The goal is to create an application that allows a user to "chat" with a PDF document. Instead of reading the entire paper, we can ask questions in natural language and receive concise answers based exclusively on the document's content.

This project implements the **Retrieval-Augmented Generation (RAG)** technique, which enhances a Large Language Model's (LLM) capabilities by providing it with relevant information from an external knowledge base (in this case, the PDF).

## 🚀 How It Works

The project's workflow is divided into two main phases, orchestrated by the `app.py` script:

### 1\. Indexing Phase (Preparing the Knowledge Base)

Before we can ask questions, we need to process the document and make it "searchable" for our system.

  * **PDF Loading:** The text from the `attention_is_all_you_need.pdf` file is loaded page by page.
  * **Chunking:** The text is broken down into smaller pieces (chunks). This was a critical tuning point:
      * **Initial:** 1000 characters (Broke the LLM).
      * **Final:** **500 characters** with a 50-character overlap. This size is required to fit within the small context window of our chosen LLM.
  * **Embedding Creation:** Each text chunk is converted into a numerical vector (embedding) using the `sentence-transformers/all-MiniLM-L6-v2` model.
  * **Vector Storage:** The embeddings are stored and indexed in a **FAISS** (from Meta AI) vector database for high-speed similarity searches.

### 2\. Generation Phase (Answering Questions)

Once the knowledge base is indexed, the system is ready to answer questions.

  * **User Question:** The system receives a question in natural language (e.g., "What is a Transformer?").
  * **Retrieval:** The question is embedded, and FAISS searches the database for the **top 3 most similar text chunks** (`k=3`).
  * **Augmentation:** The 3 retrieved chunks are combined with the original question into a structured prompt (`PromptTemplate`). This prompt includes **strong guardrails**, instructing the LLM to use *only* the provided context and how to behave if the answer is not found.
  * **Generation:** The complete prompt is sent to the LLM (**`google/flan-t5-base`**). The model then generates a cohesive response based on the provided document excerpts.

This cycle ensures that the answers are factual and grounded in the source document, minimizing the risk of LLM "hallucinations."

-----

## 🔬 Key Engineering Challenge: The 512-Token Bottleneck

A major part of this sprint was diagnosing why the RAG pipeline produced nonsensical answers.

1.  **The Problem:** The chosen LLM, `google/flan-t5-base`, has a **512-token context window**.
2.  **Failure 1:** Using `chunk_size=1000` caused the input (`chunk` + `prompt`) to exceed 512 tokens, leading to input truncation and failed responses.
3.  **Failure 2:** Using `chunk_size=500` with `k=1` (only 1 chunk) fixed the token limit but suffered from poor retrieval, where the single retrieved chunk was often irrelevant.
4.  **Current State:** Using `chunk_size=500` and `k=3` (3 chunks) provides richer context. However, this *also* pushes the token limit, and the LLM struggles to synthesize the information, often mixing correct answers with prompt instructions.

**Conclusion:** The `flan-t5-base` model is the system's primary bottleneck. A future step is to replace it with a model with a larger context window (e.g., `Mistral-7B`).

-----

## 🛠️ Tech Stack

  * **Framework:** LangChain
  * **LLM (Language Model):** `google/flan-t5-base` (via Hugging Face Pipeline)
  * **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
  * **Vector Database:** FAISS (Facebook AI Similarity Search)
  * **Document Loader:** `PyPDFLoader`
  * **Orchestration:** `dotenv`, `transformers`

## ⚙️ How to Run the Project

Follow the steps below to run the application on your local machine.

### Prerequisites

  * Python 3.8 or higher
  * A Hugging Face API key (token). You can get one from [Hugging Face](https://huggingface.co/settings/tokens).

### Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/projeto-rag.git
    cd projeto-rag
    ```

2.  **Create a virtual environment and install dependencies:**

    ```bash
    # Create the environment
    python -m venv venv

    # Activate the environment (Windows)
    venv\Scripts\activate

    # Activate the environment (Linux/macOS)
    source venv/bin/activate

    # Install all required libraries from the file
    pip install -r requirements.txt
    ```

3.  **Download the PDF document:**
    Download the paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) and save it in the project's root directory with the filename `attention_is_all_you_need.pdf`.

4.  **Set up your API key:**
    Create a file named `.env` in the root of the project directory. Add your Hugging Face API key to it:

    ```
    HUGGINGFACEHUB_API_TOKEN="YOUR_HF_API_KEY_HERE"
    ```

5.  **Run the script:**

    ```bash
    python app.py
    ```

The script will execute the entire process: it will load the PDF, create the vector database, and finally, ask the 10 benchmark questions, displaying the (often flawed) answer and the source document chunks used to create it.

## 📂 Project Structure

```
/
├── app.py                               # Main script containing all the RAG logic
├── requirements.txt                     # All project dependencies
├── attention_is_all_you_need.pdf        # The knowledge base for our system
├── .env                                 # API keys configuration (not in git)
├── .gitignore                           # Git ignore rules
├── venv/                                # Virtual environment directory
|
├── gabarito_notebooklm.md               # The "golden set" (benchmark) of 10 Q&A pairs
├── comparacao_RAG_vs_NotebookLM.md      # Side-by-side analysis of RAG vs. baseline
└── sprint1_review.md                    # Research notes (Vector DBs, FAISS, Latency)
```