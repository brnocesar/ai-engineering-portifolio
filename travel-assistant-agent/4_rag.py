import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatOpenAI(model="gpt-6-luna")


# carrega documentos PDF
path_documents = "documents"
pdf_files      = [os.path.join(path_documents, file) for file in os.listdir(path_documents) if file.endswith(".pdf")]
documents      = sum([PyPDFLoader(pdf).load() for pdf in pdf_files], [])

# quebra conteudo dos documentos em chunks
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(documents)

# vetoriza chunks
embeddings = OpenAIEmbeddings() # nao eh necessario passar a chave da API, o langchain faz uma busca por uma variavel de ambiente chamada OPENAI_API_KEY
retriever  = FAISS.from_documents(chunks, embeddings).as_retriever(search_kwargs={"k": 3})

# define template do prompt para pergunta sobre conteudo dos documentos
prompt_insurance = ChatPromptTemplate.from_messages([
    ("system", "Responda usando exclusivamente o conteúdo fornecido como contexto"),
    ("human", "{query}\n\nContexto: \n{context}\n\nResposta:")
])

# define cadeia para pergunta com resposta baseada nos documentos
chain = prompt_insurance | llm | StrOutputParser()

# funcao para buscar pelos chunks mais similares e disparar execucao da cadeia para responder usuario
def response(query: str):
    selected_chunks = retriever.invoke(query)
    context         = "\n\n".join(t.page_content for t in selected_chunks)
    return chain.invoke({"query": query, "context": context})

print(response(query="Quais são os cartões que possuem algum benefício em relação a roubo?"))
