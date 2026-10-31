# Travel Assistant Agent

Projeto experimental que reúne fluxos para um assistente de viagens: recomendação de destinos, conversa com histórico, encaminhamento para especialistas por tipo de destino e respostas baseadas em documentos sobre benefícios e coberturas de viagem. Os exemplos estão organizados em scripts e notebooks independentes.

O desenvolvimento foi feito nos notebooks, que funcionam como um diário de desenvolvimento e registram a evolução de cada fluxo. Os scripts Python apresentam o resultado final de cada implementação em versões mais diretas e resumidas.

## Estrutura do repositório

```text
travel-assistant-agent/
├── .env.example
├── .gitignore
├── 1_langchain.py
├── 2_chat.py
├── 3_langgraph.py
├── 4_rag.py
├── data/
│   └── faiss_pdf/
│       ├── index.faiss
│       └── index.pkl
├── documents/
├── notebooks/
│   ├── 1_langchain.ipynb
│   ├── 2_chat.ipynb
│   ├── 3_langgraph.ipynb
│   └── 4_rag.ipynb
└── requirements.txt
```

## Fluxos implementados

### Recomendações com LangChain

`1_langchain.py` encadeia etapas para sugerir uma cidade de acordo com o interesse informado, recomendar restaurantes e listar atividades culturais. A resposta é organizada em um relatório; parsers estruturam os dados de destino e restaurantes.

### Conversa com histórico

`2_chat.py` configura um guia de viagens especializado em destinos brasileiros e mantém o histórico da conversa em memória por sessão. O histórico é temporário e não é persistido entre execuções.

### Roteamento com LangGraph

`3_langgraph.py` usa um grafo com um nó roteador e dois fluxos especializados: praias ou montanhas. A classificação da consulta determina qual fluxo responde, com estado compartilhado e execução assíncrona.

### Respostas com RAG

`4_rag.py` demonstra recuperação de trechos relevantes de documentos PDF e geração de respostas fundamentadas no contexto recuperado. O notebook `4_rag.ipynb` também demonstra a leitura do guia de benefícios Mastercard Gold em `documents/GTB_gold_Nov23.txt`, a divisão do conteúdo em trechos, a criação de embeddings e a busca semântica. O índice FAISS persistido fica em `data/faiss_pdf/`.

## API de IA generativa e tecnologias

- **OpenAI API:** geração de respostas via `ChatOpenAI`, da integração `langchain-openai`. O notebook e os fluxos principais obtêm o nome do modelo da variável de ambiente `OPENAI_MODEL_VERSION`.
- **OpenAI Embeddings:** vetoriza os trechos dos documentos para a busca semântica do RAG.
- **LangChain:** compõe prompts e cadeias, interpreta saídas, gerencia o histórico da conversa e integra carregamento, divisão e recuperação de documentos.
- **LangGraph:** modela o roteamento condicional entre os fluxos especializados de praia e montanha.
- **FAISS:** armazena os vetores e recupera trechos semanticamente próximos à consulta.
- **Python e Pydantic:** implementam os fluxos, os estados e a estruturação de respostas.

## Configuração

Instale as dependências:

```bash
pip install -r requirements.txt
```

Configure as variáveis de ambiente indicadas em `.env.example`:

```env
OPENAI_API_KEY=sua_chave
OPENAI_MODEL_VERSION=nome_do_modelo
```

Não compartilhe nem versione uma chave real da API. Para executar `4_rag.py`, disponibilize os arquivos PDF na pasta `documents/`; o notebook também contém o fluxo de leitura do arquivo `.txt` incluído no repositório.
