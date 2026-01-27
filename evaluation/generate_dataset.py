import pandas as pd
from nba_assistant.llm.llm import RAGAgent
from nba_assistant.config.config import EVALUATION_DIR, SYSTEM_PROMPT
import os
from tqdm import tqdm
import time
from nba_assistant.utils.logging_handler import setup_logging
import logging
import httpx

setup_logging()

questions = pd.read_csv(os.path.join(EVALUATION_DIR, "questions.csv"), comment="#", header=None, names=["question"])

answers = []
contexts = []

for prompt in tqdm(questions["question"]):
    c = ""
    a = ""
    double_output = False
    agent = RAGAgent()
    
    try:
        for chunk in agent.stream(prompt):
            if chunk["type"] == "tool_call":
                c += f"Tool call: {chunk['tool']} with input: {chunk['tool_input']}"
            elif chunk["type"] == "tool_result":
                c +=  f"Tool result: {chunk['tool']} with output: {chunk['result']}"
            elif chunk["type"] == "output":
                if double_output:
                    print(f"Duplicated answer: \n{a}\n\n\n{chunk['content']}")
                double_output = True
                a += chunk['content']
    except httpx.HTTPStatusError as e:
        print(e)
        logging.warning("Sleeping for 60 seconds before retrying...")
        time.sleep(60)
        for chunk in agent.stream(prompt):
            if chunk["type"] == "tool_call":
                c += f"Tool call: {chunk['tool']} with input: {chunk['tool_input']}"
            elif chunk["type"] == "tool_result":
                c +=  f"Tool result: {chunk['tool']} with output: {chunk['result']}"
            elif chunk["type"] == "output":
                if double_output:
                    print(f"Duplicated answer: \n{a}\n\n\n{chunk['content']}")
                double_output = True
                a += chunk['content']
    
    contexts.append(c)
    answers.append(a)

questions["answer"] = answers
questions["context"] = contexts
questions.to_csv(os.path.join(EVALUATION_DIR, "questions_answers.csv"), index=False)