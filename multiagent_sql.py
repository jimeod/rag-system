
# 1. IMPORTS

import os
import re
import time
import getpass
from typing import Literal, Annotated

from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.types import Command



# 2. CONFIGURACION: API KEYS Y MODELO

if not os.environ.get("groq api key"):
    os.environ["groq api key"] = getpass.getpass("Ingresa tu GROQ_API_KEY: ")
if not os.environ.get("tvly api key"):
    os.environ["tvly api key"] = getpass.getpass("Ingresa tu TAVILY_API_KEY: ")

llm = ChatGroq(
    api_key=os.environ["groq api key"],
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_retries=0,
    timeout=30,
)

search_tool = TavilySearchResults(
    max_results=5,
    tavily_api_key=os.environ["tvly api key"],
)



# 3. CONEXION A LA BASE DE DATOS

db = SQLDatabase.from_uri("sqlite:///Chinook.db", sample_rows_in_table_info=0)

print("Dialecto:", db.dialect)
print("Tablas disponibles:", db.get_usable_table_names())



# 4. SISTEMA DE MEMORIA POR SESION

session_memories = {}


class SessionMemory:

    def __init__(self):
        self.history = InMemoryChatMessageHistory()

    def load_memory_variables(self, _inputs):
        return {"chat_history": self.history.messages}

    def save_context(self, inputs: dict, outputs: dict):
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        if user_input:
            self.history.add_message(HumanMessage(content=user_input))
        if ai_output:
            self.history.add_message(AIMessage(content=ai_output))


def get_memory_for_session(session_id: str) -> SessionMemory:
    if session_id not in session_memories:
        session_memories[session_id] = SessionMemory()
    return session_memories[session_id]



# 5. ESTADO DEL GRAFO

class State(MessagesState):
    next: str
    session_id: str = "default"
    context: dict = {}
    sql_history: list = []
    last_agent: str = ""


class Router(TypedDict):
    next: Literal["researcher", "sql_agent", "FINISH"]



# 6. NODO SUPERVISOR (decide a que agente enrutar cada pregunta)

MEMBERS = ["researcher", "sql_agent"]
OPTIONS = MEMBERS + ["FINISH"]

SUPERVISOR_PROMPT = (
    "You are a supervisor tasked with managing a conversation between the "
    f" following workers: {MEMBERS}. Given the following user request, "
    " choose the most appropriate worker to handle it. "
    " The sql_agent handles ANY question about the music store database: "
    " genres, tracks, albums, artists, customers, invoices, sales, employees, playlists, "
    " revenue, top-selling anything, counts, totals, or rankings from the data. "
    " The researcher handles ONLY general knowledge or current events questions "
    " that are NOT about the music store database (e.g. capitals, history, world news). "
    " When the conversation is complete, respond with FINISH. "
    " Consider the context of previous interactions when making your decisions."
    ' Respond ONLY with a JSON object of the form {"next": "<worker_or_FINISH>"},'
    f" where <worker_or_FINISH> is exactly one of: {OPTIONS}."
)


def supervisor_node(state: State) -> Command[Literal["researcher", "sql_agent", "__end__"]]:
    session_id = state["session_id"]
    memory = get_memory_for_session(session_id)

    messages = [{"role": "system", "content": SUPERVISOR_PROMPT}] + state["messages"]

    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    if chat_history:
        messages[0]["content"] += f"\nChat History: {chat_history}"

    if state["sql_history"]:
        sql_context = "\nPrevious SQL interactions: " + "; ".join(state["sql_history"][-3:])
        messages[0]["content"] += sql_context

 
    response = llm.with_structured_output(Router, method="json_mode").invoke(messages)
    goto = response["next"]

    if goto != END:
        last_message = state["messages"][-1].content if state["messages"] else ""
        memory.save_context({"input": last_message}, {"output": goto})
        state["last_agent"] = goto

    if goto == "FINISH":
        goto = END

    return Command(
        goto=goto,
        update={"next": goto, "last_agent": state["last_agent"]},
    )



# 7. NODO INVESTIGADOR (busqueda web + respuesta)

def research_node(state: State) -> Command[Literal["supervisor"]]:
    session_id = state["session_id"]
    memory = get_memory_for_session(session_id)

    question = state["messages"][-1].content
    chat_history = memory.load_memory_variables({}).get("chat_history", [])

    search_results = search_tool.invoke(question)

    context_str = f"\nContexto previo: {chat_history}" if chat_history else ""
    prompt = f"""Eres un experto investigador. Responde la pregunta del usuario en espanol,
usando la siguiente informacion encontrada en la web.{context_str}

Pregunta: {question}

Resultados de busqueda: {search_results}

Responde de forma clara y concisa:"""

    response = llm.invoke(prompt)

    return Command(
        update={"messages": [HumanMessage(content=response.content, name="researcher")]},
        goto="supervisor",
    )



# 8. AGENTE SQL: funciones auxiliares (generar / ejecutar / responder)

def sql_write_query(question: str) -> str:
    """Genera una consulta SQL a partir de una pregunta en lenguaje natural,
    usando el esquema real de la base de datos."""
    prompt = f"""Eres un experto en SQL. Dado el siguiente esquema de base de datos:

{db.get_table_info()}

Escribe UNA consulta SQL valida para SQLite que responda la siguiente pregunta.

Reglas importantes:
- Si la pregunta pide identificar una persona, empleado, cliente, artista, etc., asegurate de
  hacer los JOIN necesarios para incluir su NOMBRE (no solo su ID) en el resultado.
- En la tabla Employee, usa FirstName y LastName para el nombre.
- En Chinook, el empleado responsable de una venta suele relacionarse via
  Invoice -> Customer.SupportRepId -> Employee.EmployeeId.
- Responde UNICAMENTE con el SQL, sin explicaciones, sin comentarios, sin bloques de markdown.

Pregunta: {question}
"""
    response = llm.invoke(prompt)
    query = response.content.strip()
    query = query.replace("```sql", "").replace("```", "").strip()
    return query


def sql_execute_query(query: str) -> str:
    """Ejecuta la consulta SQL contra la base de datos Chinook."""
    execute_query_tool = QuerySQLDataBaseTool(db=db)
    return execute_query_tool.invoke(query)


def sql_generate_answer(question: str, query: str, result: str) -> str:

    prompt = f"""Dada la pregunta del usuario, la consulta SQL utilizada y el resultado obtenido,
responde la pregunta en lenguaje natural (en espanol), de forma clara y concisa.

Pregunta: {question}
Consulta SQL: {query}
Resultado: {result}

Respuesta:"""
    response = llm.invoke(prompt)
    return response.content



# 9. NODO SQL AGENT (orquesta las 3 funciones de arriba)

def sql_agent_node(state: State) -> Command[Literal["supervisor"]]:
    session_id = state["session_id"]
    memory = get_memory_for_session(session_id)
    question = state["messages"][-1].content

    query = sql_write_query(question)
    result = sql_execute_query(query)
    answer = sql_generate_answer(question, query, result)

    state["sql_history"].append(f"Q: {question} | SQL: {query}")
    if len(state["sql_history"]) > 10:
        state["sql_history"] = state["sql_history"][-10:]

    memory.save_context({"input": question}, {"output": answer})

    return Command(
        update={
            "messages": [HumanMessage(content=answer, name="sql_agent")],
            "sql_history": state["sql_history"],
        },
        goto="supervisor",
    )



# 10. CONSTRUCCION DEL GRAFO

workflow = StateGraph(State)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", research_node)
workflow.add_node("sql_agent", sql_agent_node)
workflow.add_edge(START, "supervisor")
graph = workflow.compile()



# 11. FUNCION PRINCIPAL DE ENTRADA 

def process_message(message: str, session_id: str = "default", max_retries: int = 2):
    for attempt in range(max_retries + 1):
        try:
            config = {"configurable": {"session_id": session_id}, "recursion_limit": 15}

            initial_state = {
                "messages": [HumanMessage(content=message)],
                "session_id": session_id,
                "context": {},
                "sql_history": [],
                "last_agent": "",
                "next": "supervisor",
            }

            result = graph.invoke(initial_state, config=config)

            return {
                "response": result["messages"][-1].content,
                "last_agent": result.get("last_agent", ""),
                "session_id": session_id,
            }

        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate_limit_exceeded" in error_str

            if is_rate_limit and attempt < max_retries:
                match = re.search(r"try again in ([\d.]+)(m?s)", error_str)
                if match:
                    wait_time = float(match.group(1))
                    if match.group(2) == "m":
                        wait_time *= 60
                    elif "ms" in match.group(0):
                        wait_time /= 1000
                else:
                    wait_time = 3

                wait_time = min(wait_time + 0.5, 60)
                print(f"Limite de Groq alcanzado, esperando {wait_time:.1f}s "
                      f"antes de reintentar (intento {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue

            return {"error": error_str, "session_id": session_id}



# 12. PRUEBAS

if __name__ == "__main__":
    sql_result = process_message("Que agente genero mas dinero en el año 2022?")
    print("SQL Query Result:", sql_result)

    research_result = process_message("¿Qué es LangGraph?")
    print("\nResearch Result:", research_result)

    followup_result = process_message("¿Cuál es el género musical más vendido?")
    print("\nFollow-up Result:", followup_result)