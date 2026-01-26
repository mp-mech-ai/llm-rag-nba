from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_mistralai import ChatMistralAI
from nba_assistant.config.config import MODEL_NAME, DATABASE_URL, DATABASE_TABLES_FOR_LLM, SYSTEM_PROMPT
import ast

# Initialize the Mistral model
llm = ChatMistralAI(model=MODEL_NAME)


# Connect to your SQLite database
db = SQLDatabase.from_uri(
    database_uri=DATABASE_URL + '?mode=ro',
    include_tables=DATABASE_TABLES_FOR_LLM
    )

stat_meaning = db.run("SELECT * FROM Stats")
stat_meaning = "\n".join([f"{code}: {meaning}" for (code, meaning) in ast.literal_eval(stat_meaning)])

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

system_prompt = SYSTEM_PROMPT.format(
    dialect=db.dialect,
    stat_meaning=stat_meaning
)


if __name__ == "__main__":
    from langchain.agents import create_tool_calling_agent
    from langchain.agents import AgentExecutor
    from nba_assistant.llm.vector_store_tool import vector_store_research
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from nba_assistant.utils.logging_handler import setup_logging

    setup_logging()

    question = "Donne moi le top 15 des joueurs ayant marqués le plus de points"
    print(question)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(
        llm=llm,
        tools=[*tools, vector_store_research],
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=[*tools, vector_store_research],
        verbose=True,
    )

    # print(agent_executor.invoke({"input": question}))
    for step in agent_executor.stream({"input": question}):
        
        # 'actions' contains tool calls
        if "actions" in step:
            for action in step["actions"]:
                print(f"🔧 Tool: {action.tool}")
                print(f"📥 Input: {action.tool_input}")
                print("---")
        
        # 'steps' contains tool results
        if "steps" in step:
            for agent_step in step["steps"]:
                print(f"✅ Tool result: {agent_step.observation}")
                print("---")
        
        # 'output' contains the final response
        if "output" in step:
            print(f"🎯 Final Answer: {step['output']}")

        
