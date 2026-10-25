import os
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
import asyncio

load_dotenv()
api_key       = os.getenv("OPENAI_API_KEY")
model_version = os.getenv("OPENAI_MODEL_VERSION")


llm = ChatOpenAI(
    model=model_version,
    api_key=api_key
)


# define template dos prompts: especializados e roteador
prompt_beach = ChatPromptTemplate.from_messages([
    ("system",      "Apresente-se como Sra Praia. Você é uma especialista em viagens com destinos para praia."),
    ("placeholder", "{conversation_historic}"),
    ("human",       "{query}")
])

prompt_mountain = ChatPromptTemplate.from_messages([
    ("system",      "Apresente-se como Sr Montanha. Você é uma especialista em viagens com destinos para montanhas e atividades radicais."),
    ("placeholder", "{conversation_historic}"),
    ("human",       "{query}")
])

prompt_router = ChatPromptTemplate.from_messages([
    ("system", "Responda apenas com 'praia' ou 'montanha'"),
    ("human", "{query}")
])


# define estrutura da resposta esperada para o prompt roteador
class Route(TypedDict):
    destination: Literal["praia", "montanha"]

# define o "estado" do grafo: schema do contexto global da execucao
class State(TypedDict):
    query:       str
    destination: Route # Route retorna um dict com chave destination, entao eh ["destination"]["destination"] para acessar o valor
    response:    str


# define cadeias especializadas
chain_beach    = prompt_beach    | llm | StrOutputParser()
chain_mountain = prompt_mountain | llm | StrOutputParser()
chain_router   = prompt_router   | llm.with_structured_output(Route)


# define funcao para nó roteador, acessa a pergunta do usuario do estado (query)
async def node_router(state: State, config: RunnableConfig):
    return {"destination": await chain_router.ainvoke({"query": state["query"]}, config)}

# define funcoes para nós das cadeias de resposta especifica
async def node_beach(state: State, config=RunnableConfig):
    return {"response": await chain_beach.ainvoke({"query": state["query"]}, config)}

async def node_mountain(state: State, config=RunnableConfig):
    return {"response": await chain_mountain.ainvoke({"query": state["query"]}, config)}

# define funcao que escolhe proximo nó de acordo com estado
def chose_node(state: State) -> Literal["praia", "montanha"]:
    return "praia" if state["destination"]["destination"] == "praia" else "montanha"


# inicializa grafo e define estrutura do estado que deve ser usado
graph = StateGraph(State)

# adiciona os nós do grafo: define o que sera executado em cada nó
graph.add_node(node="routing",  action=node_router)
graph.add_node(node="praia",    action=node_beach)
graph.add_node(node="montanha", action=node_mountain)

# adiciona arestas: regras para navegar entre os nós
graph.add_edge(START, "routing")
graph.add_conditional_edges("routing", chose_node)
graph.add_edge("praia", END)
graph.add_edge("montanha", END)


# compila o grafo definido e gera objeto executavel
app = graph.compile()

async def main():
    response = await app.ainvoke({"query": "Quero visitar um lugar no Brasil famoso por praias e cultura"})
    # print(response)
    print(response["response"])

asyncio.run(main()) 