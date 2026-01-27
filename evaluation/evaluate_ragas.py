from pydantic import BaseModel, Field
from enum import Enum
from mistralai import Mistral
import os
import pandas as pd
from tqdm import tqdm
import logging
from nba_assistant.utils.logging_handler import setup_logging
from nba_assistant.config.config import (
    MISTRAL_API_KEY,
    MODEL_NAME,
    EVALUATION_DIR
)

setup_logging()

file_path = os.path.join(EVALUATION_DIR, "questions_answers_evaluations.csv")

if not os.path.isfile(file_path):
    logging.info(f"File {file_path} does not exist. Starting evaluation from scratch...")

    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    client = Mistral(api_key=MISTRAL_API_KEY)
    logging.info("Client initialised")

    # Define Enum for scores
    class Score(str, Enum):
        no_relevance = "0"
        low_relevance = "1"
        medium_relevance = "2"
        high_relevance = "3"


    # Define a constant for the score description
    SCORE_DESCRIPTION = (
        "Score as a string between '0' and '3'. "
        "0: No relevance/Not grounded/Irrelevant - The context/answer is completely unrelated or not based on the context. "
        "1: Low relevance/Low groundedness/Somewhat relevant - The context/answer has minimal relevance or grounding. "
        "2: Medium relevance/Medium groundedness/Mostly relevant - The context/answer is somewhat relevant or grounded. "
        "3: High relevance/High groundedness/Fully relevant - The context/answer is highly relevant or grounded."
    )

    # Define separate classes for each criterion with detailed descriptions
    class ContextRelevance(BaseModel):
        explanation: str = Field(..., description=("Step-by-step reasoning explaining how the retrieved context aligns with the user's query. "
                        "Consider the relevance of the information to the query's intent and the appropriateness of the context "
                        "in providing a coherent and useful response."))
        score: Score = Field(..., description=SCORE_DESCRIPTION)

    class AnswerRelevance(BaseModel):
        explanation: str = Field(..., description=("Step-by-step reasoning explaining how well the generated answer addresses the user's original query. "
                        "Consider the helpfulness and on-point nature of the answer, aligning with the user's intent and providing valuable insights."))
        score: Score = Field(..., description=SCORE_DESCRIPTION)

    class Groundedness(BaseModel):
        explanation: str = Field(..., description=("Step-by-step reasoning explaining how faithful the generated answer is to the retrieved context. "
                        "Consider the factual accuracy and reliability of the answer, ensuring it is grounded in the retrieved information."))
        score: Score = Field(..., description=SCORE_DESCRIPTION)

    class RAGEvaluation(BaseModel):
        context_relevance: ContextRelevance = Field(..., description="Evaluation of the context relevance to the query, considering how well the retrieved context aligns with the user's intent." )
        answer_relevance: AnswerRelevance = Field(..., description="Evaluation of the answer relevance to the query, assessing how well the generated answer addresses the user's original query." )
        groundedness: Groundedness = Field(..., description="Evaluation of the groundedness of the generated answer, ensuring it is faithful to the retrieved context." )


    # Function to evaluate RAG metrics
    def evaluate_rag(query: str, retrieved_context: str, generated_answer: str):
        chat_response = client.chat.parse(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a judge for evaluating a Retrieval-Augmented Generation (RAG) system. "
                        "Evaluate the context relevance, answer relevance, and groundedness based on the following criteria: "
                        "Provide a reasoning and a score as a string between '0' and '3' for each criterion. "
                        "Context Relevance: How relevant is the retrieved context to the query? "
                        "Answer Relevance: How relevant is the generated answer to the query? "
                        "Groundedness: How faithful is the generated answer to the retrieved context?"
                    )
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\nRetrieved Context: {retrieved_context}\nGenerated Answer: {generated_answer}"
                },
            ],
            response_format=RAGEvaluation,
            temperature=0
        )
        return chat_response.choices[0].message.parsed

    logging.info("Evaluating RAGs...")
    dataset = pd.read_csv(os.path.join(EVALUATION_DIR, "questions_answers.csv"))
    questions = dataset["question"]
    answers = dataset["answer"]
    context = dataset["context"]
    context_relevance_score = []
    answer_relevance_score = []
    groundedness_score = []

    context_relevance_explanation = []
    answer_relevance_explanation = []
    groundedness_explanation = []


    for question, answer, context in tqdm(zip(questions, answers, context)):
        evaluation = evaluate_rag(question, context, answer)
        logging.info(f"Question: {question}")
        logging.info(f"Answer: {answer}")
        logging.info(f"Context: {context}")
        logging.info(f"Evaluation: {evaluation}")
        context_relevance_score.append(evaluation.context_relevance.score)
        answer_relevance_score.append(evaluation.answer_relevance.score)
        groundedness_score.append(evaluation.groundedness.score)

        context_relevance_explanation.append(evaluation.context_relevance.explanation)
        answer_relevance_explanation.append(evaluation.answer_relevance.explanation)
        groundedness_explanation.append(evaluation.groundedness.explanation)

    dataset["context_relevance_score"] = context_relevance_score
    dataset["answer_relevance_score"] = answer_relevance_score
    dataset["groundedness_score"] = groundedness_score

    dataset["context_relevance_explanation"] = context_relevance_explanation
    dataset["answer_relevance_explanation"] = answer_relevance_explanation
    dataset["groundedness_explanation"] = groundedness_explanation

    dataset.to_csv(file_path, index=False)
    logging.info(f"Evaluations saved to {file_path}")
else:
    logging.info(f"File {file_path} already exists. Skipping evaluation.")


evaluations = pd.read_csv(file_path)


logging.info(f"{"Mean context relevance score:":40s} {evaluations['context_relevance_score'].mean():.2f} / 3")
logging.info(f"{"Mean answer relevance score:":40s} {evaluations['answer_relevance_score'].mean():.2f} / 3")
logging.info(f"{"Mean groundedness score:":40s} {evaluations['groundedness_score'].mean():.2f} / 3")


"""
OUTPUT 1st run:
2026-01-25 11:42:37,675 - INFO - evaluate_ragas - File evaluation/questions_answers_evaluations.csv already exists. Skipping evaluation.
2026-01-25 11:42:37,682 - INFO - evaluate_ragas - Mean context relevance score:            2.06 / 3
2026-01-25 11:42:37,682 - INFO - evaluate_ragas - Mean answer relevance score:             2.42 / 3
2026-01-25 11:42:37,682 - INFO - evaluate_ragas - Mean groundedness score:                 2.17 / 3
"""

"""
OUTPUT 2nd run:
2026-01-27 17:06:07,672 - INFO - root - Evaluations saved to evaluation/questions_answers_evaluations.csv
2026-01-27 17:06:07,686 - INFO - root - Mean context relevance score:            2.89 / 3
2026-01-27 17:06:07,687 - INFO - root - Mean answer relevance score:             2.92 / 3
2026-01-27 17:06:07,687 - INFO - root - Mean groundedness score:                 2.92 / 3
"""