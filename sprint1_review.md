## 🚀 Resumo da Sprint: RAG, Pesquisa e Debugging

Aqui está o progresso feito no projeto RAG e nas tasks de pesquisa. O foco principal se tornou um diagnóstico de engenharia para identificar um gargalo de arquitetura no nosso RAG.

### 1\. 💻 Progresso no Projeto (Implementação)

  * **Task 1: Dependências do Projeto (Concluído)**

      * Travei as versões de todas as bibliotecas (`langchain`, `faiss-cpu`, etc.) no arquivo `requirements.txt` para garantir a reprodutibilidade do ambiente e evitar quebras por atualizações.

    <!-- end list -->

    ```bash
    pip freeze > requirements.txt
    ```

  * **Task 4: Refinamento do Prompt (Concluído)**

      * "Amarrei" o *prompt template* do RAG, adicionando "guardrails" e instruções claras (ex: "Use **apenas** o contexto fornecido", "Seja amigável", "Não invente informações") para controlar melhor a saída do LLM.

### 2\. 📚 Pesquisa e Arquitetura (Estudo)

  * **Task 2: Análise de Bancos Vetoriais (Concluído)**

      * Pesquisei as principais soluções de mercado para armazenamento vetorial, comparando opções *open-source* (locais) vs. *gerenciadas* (SaaS).

    | Banco | Tipo | Prós (Resumo) | Contras (Resumo) |
    | :--- | :--- | :--- | :--- |
    | **FAISS** | Aberta | Leve, rápido, roda local (ótimo p/ protótipo) | Não é um "serviço" (sem API, não escala) |
    | **ChromaDB** | Aberta | "API-first", fácil de usar | Performance menor que FAISS em larga escala |
    | **Qdrant** | Aberta | Rápido, filtros de metadados avançados | Mais complexo de configurar |
    | **Pinecone** | Fechada | Escala "infinita", gerenciado, fácil | Custo, "vendor lock-in" |
    | **Azure AI Search**| Fechada | Busca híbrida (keyword + semântica), ecossistema | Custo, complexo, "vendor lock-in" (Azure) |
    | **CosmosDB** | Fechada | DB NoSQL completo (multi-modelo) | Não é otimizado *apenas* para vetores |

  * **Task 3: Parâmetros de Busca (FAISS) (Concluído)**

      * Analisei os parâmetros do `retriever` do FAISS. A configuração-chave é o `search_kwargs`:

    <!-- end list -->

    ```python
    # k: Define quantos chunks o retriever deve buscar.
    # score_threshold (opcional): Define uma nota de corte para a similaridade.
    retriever = db.as_retriever(search_kwargs={"k": 3})
    ```

      * Também pesquisei sobre tipos de busca, identificando que usamos a **Semântica** (vetores) e que o "padrão-ouro" da indústria é a **Híbrida** (Semântica + Keyword).

  * **Task 3: Criação de "Gabarito" para Benchmark (Concluído)**

      * Usei o NotebookLM para fazer upload do PDF e criei um documento `gabarito_notebooklm.md` com 10 perguntas-chave e suas respostas "perfeitas", baseadas nos trechos corretos. O objetivo era usar isso para *avaliar* a qualidade do nosso RAG.

  * **Task 5: Latência vs. Assertividade (Concluído)**

      * Pesquisei o trade-off: Nosso RAG usa o `flan-t5-base` (pequeno), priorizando **Baixa Latência** (rapidez) em troca de **Baixa Assertividade** (respostas mais simples/confusas). Modelos maiores (GPT-4) fariam o oposto.

### 3\. 🔬 Diagnóstico de Engenharia (A Descoberta Principal)

Ao tentar rodar o benchmark (Task 3) contra o nosso RAG (Task 4), identifiquei o principal gargalo técnico do projeto:

**O `context window` (limite de 512 tokens) do `google/flan-t5-base` é incompatível com a nossa arquitetura de RAG.**

Minha investigação de debugging mostrou:

1.  **Teste 1 (`chunk_size=1000`, `k=1`):** O input (`chunk` + `prompt`) estourou o limite de 512 tokens (Erro `655 > 512`), truncando o prompt e gerando respostas confusas.
2.  **Teste 2 (`chunk_size=500`, `k=1`):** O limite de tokens foi respeitado. **Porém**, o retriever (com `k=1`) errou a busca, trazendo um chunk irrelevante (sobre os autores do paper) e fazendo o LLM (obedientemente) gerar uma resposta errada.
3.  **Teste 3 (`chunk_size=500`, `k=3`):** O retriever acertou (trouxe os chunks corretos, `k=3`). **Porém**, o input total (`3 chunks` + `prompt`) estourou o limite de 512 tokens novamente, fazendo o LLM se confundir e misturar trechos de resposta com instruções do prompt.

**Conclusão:** O `chain_type="stuff"` (que "enfia" todos os chunks no prompt) não funciona com o `flan-t5-base`.

**Sugestões para Próximos Passos:**

  * **Opção A (Melhor Qualidade):** Trocar o LLM por um com *context window* maior (ex: `Mistral-7B`, que tem 8k de tokens).
  * **Opção B (Manter o LLM):** Trocar o `chain_type` para `"map_reduce"`, que processa os chunks um de cada vez (mais lento, mas respeita o limite de 512).