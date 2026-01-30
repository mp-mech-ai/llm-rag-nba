import logging
import logfire
from dotenv import load_dotenv
import os

def setup_logging():
    load_dotenv()
    if os.environ.get("ON_STREAMLIT", False) == "True": 
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        logfire.configure()
