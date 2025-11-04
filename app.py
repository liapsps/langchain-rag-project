# --- Passo 0: Importando as Ferramentas Essenciais ---
# Aqui eu importo tudo que vou precisar. É como separar os ingredientes antes de cozinhar.
# Tenho ferramentas do LangChain para carregar o PDF, dividir o texto, criar os embeddings,
# e orquestrar todo o fluxo de Pergunta e Resposta (Q&A).

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# --- Modularidade e Compatibilidade do LangChain ---
# A biblioteca LangChain mudou bastante, foi dividida em pacotes menores.
# Importamos a classe do HuggingFace
# from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

load_dotenv()

def main():

    # --- Configuração Inicial: Chave da API ---
    # Para usar os modelos do Hugging Face, preciso me autenticar.
    # A função load_dotenv() já carregou a chave do arquivo .env para o ambiente.
    # O LangChain vai encontrar a variável de ambiente HUGGINGFACEHUB_API_TOKEN automaticamente.
    # Apenas adicionamos uma verificação para garantir que ela existe.
    if os.getenv("HUGGINGFACEHUB_API_TOKEN") is None:
        print("ERRO: A chave HUGGINGFACEHUB_API_TOKEN não foi encontrada.")
        print("Certifique-se de que você tem um arquivo .env com a sua chave.")
        # return # Você pode descomentar esta linha para parar o script se a chave não for encontrada

    # O documento que será a base de conhecimento do nosso sistema RAG
    pdf_path = "attention_is_all_you_need.pdf"
    
    # Uma verificação simples para garantir que o arquivo PDF está no lugar certo.
    if not os.path.exists(pdf_path):
        print(f"ERRO: O arquivo '{pdf_path}' não foi encontrado.")
        print("Por favor, baixe o artigo 'Attention Is All You Need' em https://arxiv.org/pdf/1706.03762.pdf e salve-o na mesma pasta.")
        return

    # --- FASE DE INDEXAÇÃO: Preparando a Base de Conhecimento ---
    # O objetivo desta fase é transformar nosso PDF em algo que a máquina possa buscar e entender.

    # --- Passo 1: Carregar o Documento ---
    # A primeira coisa a fazer é extrair o texto do PDF.
    # O PyPDFLoader faz isso, criando um objeto de "Documento" para cada página.
    print("Carregando o documento...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Documento carregado com {len(documents)} páginas.")

    # --- Passo 2: Dividir o Documento em Chunks ---
    # Os LLMs têm um limite de quantos tokens conseguem processar de uma vez (janela de contexto).
    # Por isso, preciso quebrar o texto em pedaços menores, os "chunks".
    # Escolhi o RecursiveCharacterTextSplitter porque ele tenta manter parágrafos e frases juntos.
    # 'chunk_size=1000' define o tamanho de cada pedaço e 'chunk_overlap=150' cria uma sobreposição
    # para não perdermos o contexto entre os chunks.
    print("Dividindo o documento em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"Documento dividido em {len(docs)} chunks.")

    # --- Passo 3: Criar os Embeddings ---
    # Agora, preciso converter esses chunks de texto em vetores numéricos (embeddings).
    # É assim que a máquina entende o "significado" do texto. Chunks com significados parecidos
    # terão vetores próximos no espaço vetorial.
    # Escolhi o 'all-MiniLM-L6-v2' porque é um modelo leve, rápido e muito bom para essa tarefa.
    print("Inicializando o modelo de embedding...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # --- Passo 4: Armazenar os Vetores no FAISS ---
    # Com os vetores criados, eu os armazeno em um banco de dados vetorial.
    # O FAISS (da Meta) é ótimo para isso, pois permite buscar vetores similares de forma muito rápida.
    # Basicamente, estou criando um índice pesquisável do meu documento. Essa é a nossa "base de conhecimento".
    print("Criando o banco de dados vetorial com FAISS...")
    # Esta etapa pode demorar alguns minutos na primeira vez, pois baixará o modelo.
    db = FAISS.from_documents(docs, embeddings)
    print("Banco de dados vetorial criado com sucesso!")

    # --- FASE DE GERAÇÃO: Respondendo às Perguntas ---
    # Agora que a base está pronta, o sistema pode receber perguntas e gerar respostas.

    # --- Passo 5: Escolher o LLM ---
    # Este é o cérebro da operação. O LLM vai receber a pergunta e o contexto
    # e gerar a resposta em linguagem natural.
    # Estou usando o 'flan-t5-xxl' do Google, que é um bom modelo para tarefas de instrução.
    # A 'temperature=0.1' torna a resposta mais precisa e menos "criativa", o que é ideal para Q&A.
    print("Inicializando o LLM do Hugging Face Hub...")
    pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=1000,
    temperature=0.1
)
    llm = HuggingFacePipeline(pipeline=pipe)
    # llm = HuggingFaceEndpoint(
    #     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    #     task="text2text-generation",
    #     temperature=0.1,
    #     max_new_tokens=512,
    #     model_kwargs={
    #         "temperature": 0.1,
    #         "max_new_tokens": 512
    #     }
    # )

    # --- Passo 6: Criar a Cadeia de RAG (Retrieval-Augmented Generation) ---
    # Esta é a parte que conecta tudo. O RAG funciona em dois tempos:
    # 1. Retrieval (Recuperação): Encontra os chunks relevantes no FAISS.
    # 2. Generation (Geração): Envia esses chunks junto com a pergunta para o LLM gerar a resposta.
    print("Criando a cadeia de RAG...")
    # Criei um template para instruir o LLM sobre como ele deve se comportar.
    # Ele recebe o contexto ({context}) que veio do FAISS e a pergunta do usuário ({question}).
    # Isso ajuda a evitar que o modelo "alucine" e a garantir respostas baseadas nos fatos do documento.
    prompt_template = """
    Você é um assistente de IA especialista, focado em responder perguntas sobre o artigo "Attention Is All You Need".
    Sua resposta deve ser amigável e direta.

    Use **apenas** os trechos de contexto fornecidos abaixo para formular sua resposta.
    - Não invente informações que não estejam no texto.
    - Se a resposta não puder ser encontrada no contexto, diga educadamente: "Desculpe, mas não encontrei essa informação específica no documento."

    Limites da Resposta:
    - Mantenha a resposta o mais concisa possível.
    - Use no máximo três frases.

    Contexto Fornecido:
    {context}

    Pergunta do Usuário:
    {question}

    Resposta Amigável:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    # O 'retriever' é o componente responsável por buscar os documentos no FAISS.
    # 'search_kwargs={"k": 3}' significa que ele vai buscar os 3 chunks mais relevantes para a pergunta.
    retriever = db.as_retriever(search_kwargs={"k": 1}) #parâmetros do FAISS (olhar documentação): score de similaridade, tipos de busca
    
    # Finalmente, monto a cadeia 'RetrievalQA'.
    # Ela junta o LLM, o retriever e o prompt.
    # 'chain_type="stuff"' significa que ele vai "enfiar" (stuff) todos os chunks recuperados no prompt.
    # 'return_source_documents=True' é muito útil para podermos ver quais partes do texto
    # foram usadas para gerar a resposta, o que garante a transparência do processo.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("Cadeia de RAG pronta!")

    # --- Passo 7: Testar o Sistema ---
    # Para validar, criei uma lista de perguntas sobre o artigo e iterei sobre elas,
    # mostrando a pergunta, a resposta do modelo e os trechos que ele usou como fonte.
    perguntas = [
        "What is the main topic of the document?",
        "What is a Transformer?",
        "How does multi-head attention work?"
    ]
    
    for pergunta in perguntas:
        print(f"\n{'='*30}\n--> Pergunta: {pergunta}")
        # Chamo a cadeia com a pergunta. A cadeia faz todo o trabalho de RAG por baixo dos panos.
        resultado = qa_chain.invoke({'query':pergunta})
        print("\n--> Resposta:")
        print(resultado["result"])
        print("\n--> Documentos Fonte Utilizados:")
        # Isso aqui é a prova de que o RAG funcionou: mostro os trechos que fundamentaram a resposta.
        for doc in resultado["source_documents"]:
            print(f"  - Página: {doc.metadata.get('page', 'N/A')}, Trecho: '{doc.page_content[:150].strip()}'...")

# Padrão Python para garantir que a função 'main' só rode quando o script for executado diretamente.
if __name__ == '__main__':
    main()