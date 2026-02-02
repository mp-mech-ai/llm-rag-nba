import logging
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from nba_assistant.llm.vector_store_tool import vector_store_research
from langchain.agents import create_tool_calling_agent
import ast
from langchain_mistralai import ChatMistralAI
from nba_assistant.config.config import (
    MODEL_NAME, DATABASE_URL, DATABASE_TABLES_FOR_LLM, SYSTEM_PROMPT
)
from langchain.agents import AgentExecutor
from nba_assistant.llm.sql_tool import get_database, get_sql_toolkit
import logfire

class AgentCreationError(Exception):
    pass

class LLMError(Exception):
    pass

class RAGError(Exception):
    """Base exception for RAG-related failures."""
    pass

class RAGAgent:
    def __init__(self):
        try:
            self.llm = ChatMistralAI(
                model=MODEL_NAME,
                model_kwargs={
                "parallel_tool_calls": False
            })
        except Exception as e:
            raise AgentCreationError(f"Error creating Mistral client: {e}")
        
        # Connect to database
        self.db = get_database(
            db_uri=DATABASE_URL + '?mode=ro',
            include_tables=DATABASE_TABLES_FOR_LLM
        )
        
        stat_meaning = self.db.run("SELECT * FROM Stats")
        stat_meaning = "\n".join([f"{code}: {meaning}" 
                                  for (code, meaning) in ast.literal_eval(stat_meaning)])
        
        # Setup tools
        self.toolkit = get_sql_toolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools() + [vector_store_research]
        
        # Setup prompt
        system_prompt = SYSTEM_PROMPT.format(
            dialect=self.db.dialect,
            stat_meaning=stat_meaning
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent (once)
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create executor (once)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            early_stopping_method="generate",
            handle_parsing_errors=True,  # Gracefully handle errors
            return_intermediate_steps=True,  # Keep track of steps
        )
        
        # Store chat history
        self.chat_history = []
    
    def invoke(self, message: str):
        """Send a message and get a response."""
        with logfire.span("llm_invoke"):
            response = self.agent_executor.invoke({
                "input": message,
                "chat_history": self.chat_history
            })
            
            # Update history
            self.chat_history.append(("human", message))
            self.chat_history.append(("ai", response["output"]))
            
            return response["output"]
    
    def stream(self, message: str):
        """Stream tool calls and output chunks."""
        with logfire.span("llm_stream"):
            full_response = ""
            
            for chunk in self.agent_executor.stream({
                "input": message,
                "chat_history": self.chat_history
            }):
                # Yield tool calls (actions)
                if "actions" in chunk:
                    for action in chunk["actions"]:
                        yield {
                            "type": "tool_call",
                            "tool": action.tool,
                            "tool_input": action.tool_input,
                            "log": action.log
                        }
                
                # Yield tool results
                elif "steps" in chunk:
                    for step in chunk["steps"]:
                        yield {
                            "type": "tool_result",
                            "tool": step.action.tool,
                            "result": step.observation
                        }
                
                # Yield output chunks (final response)
                elif "output" in chunk:
                    new_text = chunk["output"][len(full_response):]
                    full_response = chunk["output"]
                    if new_text:
                        yield {
                            "type": "output",
                            "content": new_text
                        }
            
            # Update history after streaming completes
            self.chat_history.append(("human", message))
            self.chat_history.append(("ai", full_response))

if __name__ == "__main__":
    from nba_assistant.utils.logging_handler import setup_logging
    setup_logging()
    
    agent = RAGAgent()
    print(f"list tables: {agent.tools[2]._run()}")
    print(f"schema: {agent.tools[1]._run("Teams")}")
    # question = "Quelle équipe contient le plus de joueurs dans le top 15 des meilleurs marqueurs de la saison ? Tu me donneras le nom de l'équipe ainsi que le nombre de joueurs de l'équipe qui sont dans le top 15. Tu me diras aussi ce que pense les fans de cette équipe"
    # # question = "Quel est le pourcentage de tir réussi du top 3 des meilleurs marqueurs ?"
    # question = "Call successively every tool that you can use"
    # for chunk in agent.stream(question):
    #     if chunk["type"] == "tool_call":
    #         logging.info(f"Tool call: {chunk['tool']} with input: {chunk['tool_input']}")
    #     elif chunk["type"] == "tool_result":
    #         logging.info(f"Tool result: {chunk['tool']} with output: {chunk['result']}")
    #     elif chunk["type"] == "output":
    #         logging.info(f"Output: {chunk['content']}")