import pandas as pd
from llm.llm import generer_reponse
from llm.vector_store_management import create_vector_store_manager
import logging
from config.config import SEARCH_K, EVALUATION_DIR
import os

vector_store_manager = create_vector_store_manager()

SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:"""

questions = pd.read_csv(os.path.join(EVALUATION_DIR, "questions.csv"), comment="#", header=None, names=["question"])

answers = []
context = []
for prompt in questions["question"]: 
    if vector_store_manager is None:
        logging.error("VectorStoreManager non disponible pour la recherche.")
        break

    # 3. Rechercher le contexte dans le Vector Store
    try:
        logging.info(f"Recherche de contexte pour la question: '{prompt}' avec k={SEARCH_K}")
        search_results = vector_store_manager.search(prompt, k=SEARCH_K)
        logging.info(f"{len(search_results)} chunks trouvés dans le Vector Store.")
    except Exception:
        logging.exception(f"Erreur pendant vector_store_manager.search pour la query: {prompt}")
        search_results = [] # On continue sans contexte si la recherche échoue

    # 4. Formater le contexte pour le prompt LLM
    context_str = "\n\n---\n\n".join([
        f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
        for res in search_results
    ])
    context.append(context_str)

    if not search_results:
        context_str = "Aucune information pertinente trouvée dans la base de connaissances pour cette question."
        logging.warning(f"Aucun contexte trouvé pour la query: {prompt}")

    # 5. Construire le prompt final pour l'API Mistral en utilisant le System Prompt RAG
    final_prompt_for_llm = SYSTEM_PROMPT.format(context_str=context_str, question=prompt)

    # Créer la liste de messages pour l'API (juste le prompt système/utilisateur combiné)
    messages_for_api = [
        {"role": "user", "content": final_prompt_for_llm}
    ]
    answers.append(generer_reponse(messages_for_api))


questions["answer"] = answers
questions["context"] = context
questions.to_csv(os.path.join(EVALUATION_DIR, "questions_answers.csv"), index=False)