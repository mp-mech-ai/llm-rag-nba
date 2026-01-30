from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import logfire

def get_database(db_uri: str, include_tables: list):
    try:
        with logfire.span("sql_db_init"):
            return SQLDatabase.from_uri(
                database_uri=db_uri + '?mode=ro',
                include_tables=include_tables
            )
    except Exception as e:
        logfire.error(f"Error connecting to database: {e}")
        raise


def get_sql_toolkit(db, llm):
    with logfire.span("sql_toolkit_init"):
        return SQLDatabaseToolkit(db=db, llm=llm)