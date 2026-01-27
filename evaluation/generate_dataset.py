import pandas as pd
from nba_assistant.llm.llm import RAGAgent
from nba_assistant.llm.vector_store_management import create_vector_store_manager
from nba_assistant.config.config import SEARCH_K, EVALUATION_DIR
import os
from tqdm import tqdm

vector_store_manager = create_vector_store_manager()

questions = pd.read_csv(os.path.join(EVALUATION_DIR, "questions.csv"), comment="#", header=None, names=["question"])

answers = []
contexts = []
for prompt in tqdm(questions["question"]):
    c = ""
    a = ""
    double_output = False
    agent = RAGAgent()

    for chunk in agent.stream(prompt):
        if chunk["type"] == "tool_call":
            c += f"Tool call: {chunk['tool']} with input: {chunk['tool_input']}"
        elif chunk["type"] == "tool_result":
            c +=  f"Tool result: {chunk['tool']} with output: {chunk['result']}"
        elif chunk["type"] == "output":
            if double_output:
                print(f"Duplicated answer: \n{a}\n{chunk['content']}")

            double_output = True
            a += chunk['content']
    
    contexts.append(c)
    answers.append(a)

questions["answer"] = answers
questions["context"] = contexts
questions.to_csv(os.path.join(EVALUATION_DIR, "questions_answers.csv"), index=False)