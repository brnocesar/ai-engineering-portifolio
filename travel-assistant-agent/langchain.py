import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.globals import set_debug
from pydantic import BaseModel, Field


# set_debug(True)

load_dotenv()
api_key       = os.getenv("OPENAI_API_KEY")
model_version = os.getenv("OPENAI_MODEL_VERSION")
llm           = ChatOpenAI(model=model_version, api_key=api_key, cache=False)


# define classes dos formatos esperados para respostas
class Destination(BaseModel):
    city:   str = Field("A cidade recomendada para visitar") # esse campo sera passado como input para a proxima cadeia
    reason: str = Field("motivo que pelo qual é interessante visitar essa cidade")

class Restaurants(BaseModel):
    city:        str = Field("A cidade recomendada para visitar")
    restaurants: str = Field("Restaurantes recomendandos na cidade")


# define parsers para tranformar repostas de acordo com formato esperado
parser_destination = JsonOutputParser(pydantic_object=Destination)
parser_restaurants = JsonOutputParser(pydantic_object=Restaurants)
parser_culture     = StrOutputParser()


# define prompt template com o formato de saida esperado
prompt_destination = PromptTemplate(
    template="""
        Sugira uma cidade para minha viagem, dado meu interesse por {interesse}.
        {formato_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_saida": parser_destination.get_format_instructions()}
)

prompt_restaurants = PromptTemplate(
    template="""
        Sugira restaurantes populares entre locais em {city}.
        {formato_saida}
    """,
    partial_variables={"formato_saida": parser_restaurants.get_format_instructions()}
)

prompt_culture = PromptTemplate(
    template="sugira atividades e locais culturais em {city}."
)


# define cada uma das cadeias
chain_1 = prompt_destination | llm | parser_destination
chain_2 = prompt_restaurants | llm | parser_restaurants
chain_3 = prompt_culture     | llm | parser_culture

# define o fluxo composto de cadeias, adicionando etapa no final para montar relatorio final
chain = (
    RunnablePassthrough
    .assign(destination=chain_1)
    .assign(
        restaurants=RunnableLambda(
            lambda state: chain_2.invoke(
                {"city": state["destination"]["city"]}
            )
        )
    )
    .assign(
        culture=RunnableLambda(
            lambda state: chain_3.invoke(
                {"city": state["destination"]["city"]}
            )
        )
    )
    .assign(
        report=RunnableLambda(
            lambda state: (
                f"Cidade sugerida: {state['destination']['city']}\n"
                f"Motivo para visitar: {state['destination']['reason']}\n"
                f"Restaurantes: {state['restaurants']['restaurants']}\n"
                f"Atividades e locais culturais: {state['culture']}"
            )
        )
    )
)

# dispara execucao da sequencia de cadeias
response = chain.invoke({"interesse": "praias"})
print(response["report"])