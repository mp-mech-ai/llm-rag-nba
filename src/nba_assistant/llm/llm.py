import logging
from mistralai import Mistral
from nba_assistant.llm.vector_store_management import VectorStoreManager
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from nba_assistant.llm.vector_store_tool import vector_store_research
from langchain.agents import create_tool_calling_agent
import ast
from langchain_mistralai import ChatMistralAI
from nba_assistant.config.config import (
    MODEL_NAME, DATABASE_URL, DATABASE_TABLES_FOR_LLM, SYSTEM_PROMPT, MISTRAL_API_KEY
)
from langchain.agents import AgentExecutor

class ClientCreationError(Exception):
    pass

class LLMError(Exception):
    pass

class RAGError(Exception):
    """Base exception for RAG-related failures."""
    pass

class MessageHistory(ChatPromptTemplate):
    def __init__(self, system_prompt, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

def create_agent() -> AgentExecutor:
    try:
        llm = ChatMistralAI(model=MODEL_NAME)
    except Exception as e:
        raise ClientCreationError(f"Une erreur est survenue lors de la connexion à l'API Mistral: {e}")

    # Connect to your SQLite database
    db = SQLDatabase.from_uri(
        database_uri=DATABASE_URL + '?mode=ro',
        include_tables=DATABASE_TABLES_FOR_LLM
        )
    
    stat_meaning = db.run("SELECT * FROM Stats")
    stat_meaning = "\n".join([f"{code}: {meaning}" for (code, meaning) in ast.literal_eval(stat_meaning)])

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    tools = toolkit.get_tools() + [vector_store_research]

    system_prompt = SYSTEM_PROMPT.format(
        dialect=db.dialect,
        stat_meaning=stat_meaning
    )

    history = MessageHistory(system_prompt)
    prompt = history.prompt

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )


def generer_reponse_stream():
    pass