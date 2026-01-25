from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_recall
import pandas as pd
from utils.config import MODEL_NAME, MISTRAL_API_KEY
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from mistralai import Mistral
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas import evaluate

# 1. Initialize the Mistral Client
# This client is compatible with the "Instructor" logic Ragas now uses
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# 2. Use llm_factory with the Mistral client
# We use 'openai' as the base because Mistral's SDK is compatible 
# with the structured output patterns Ragas expects.
evaluator_llm = llm_factory(
    model=MODEL_NAME, 
    client=mistral_client
)

# 3. Initialize Embeddings using the dedicated class
# This avoids the deprecated embedding_factory
evaluator_embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=MISTRAL_API_KEY
)

# 4. Initialize Metrics with the new LLM
# Note: Ragas metrics now prefer being passed directly to evaluate()
metrics = [
    faithfulness.Faithfulness(evaluator_llm, evaluator_embeddings),
    answer_relevancy.AnswerRelevancy(evaluator_llm, evaluator_embeddings),
    context_recall.ContextRecall(evaluator_llm, evaluator_embeddings)
]

dataset = pd.read_csv("evaluate/questions_answers.csv", comment="#", header=0)

# 5. Run Evaluation
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy]
)
print(results)

with open("evaluate/results.txt", "w") as f:
    f.write(str(results))