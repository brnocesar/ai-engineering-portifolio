import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
api_key       = os.getenv("OPENAI_API_KEY")
model_version = os.getenv("OPENAI_MODEL_VERSION")

llm = ChatOpenAI(
    model=model_version,
    api_key=api_key
)

# cria prompt template para chat
prompt_destination = ChatPromptTemplate.from_messages([
    ("system",      "Você é um guia de viagem especializado em destinos brasileiros. Apresente-se como Sr. Passeios"),
    ("placeholder", "{conversation_historic}"),
    ("human",       "{query}")
])

# define a cadeia base
chain      = prompt_destination | llm | StrOutputParser()

# dicionario para armazenar os enderecos de memoria com os historicos de conversa
memory     = {}
# identificador da conversa desse script
session_id = "travel_plan_"

# funcao que permite recuperar historico da conversa
def conversarion_per_session(session_id: str):
    if session_id not in memory:
        memory[session_id] = InMemoryChatMessageHistory()
    
    return memory[session_id]

# define cadeia com memoria
chain_with_memory = RunnableWithMessageHistory(
    runnable=chain,                                 # runnable base para o qual sera gerenciado o historico de mensagens
    get_session_history=conversarion_per_session,   # funcao que retorna historico de mensagens a partir de um session_id
    input_messages_key="query",                     # inputs definidos no runnable base
    history_messages_key="conversation_historic"    # placeholder definido no runnable base
)

messages = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

for query in messages:
    response = chain_with_memory.invoke(
        {"query": query},
        {"configurable": {"session_id": session_id}}
    )
    
    print("- User:", query)
    print("- IA:", response, "\n")
