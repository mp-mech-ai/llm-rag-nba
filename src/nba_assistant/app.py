import streamlit as st
import logging
import time

# --- Importations depuis vos modules ---
try:
    from nba_assistant.config.config import (
        APP_TITLE, NAME, MODEL_NAME, 
        MISTRAL_API_KEY, SEARCH_K, SYSTEM_PROMPT
    )
    from nba_assistant.utils.logging_handler import setup_logging
    from nba_assistant.llm.llm import RAGError, RAGAgent
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()

# --- Configuration du Logging ---
setup_logging()

@st.cache_resource()
def get_agent():
    return RAGAgent()

# --- Creation du client Mistral ---
try:
    agent = get_agent()
except RAGError as e:
    st.error(f"Une erreur est survenue lors de la création de l'agent: {e}")
    st.stop()

# --- Initialisation de l'historique de conversation ---
if "messages" not in st.session_state:
    # Message d'accueil initial
    st.session_state.messages = [{"role": "assistant", "content": f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions sur les équipes, les joueurs ou les statistiques, et je vous répondrai en me basant sur les données les plus récentes."}]

# --- Interface Utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle: {MODEL_NAME}")

# Affichage des messages de l'historique (pour l'UI)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    # 1. Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Afficher indicateur + Générer la réponse de l'assistant via LLM
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...") # Indicateur simple

        # Génération de la réponse de l'assistant en utilisant la RAG
        logging.info(f"Génération de la réponse de l'assistant pour la question: {prompt}")
        try:
            for chunk in agent.stream(prompt):
                if chunk["type"] == "tool_call":
                    logging.info(f"Tool call: {chunk['tool']} with input: {chunk['tool_input']}")
                    message_placeholder.write(chunk['tool_input'])
                elif chunk["type"] == "tool_result":
                    logging.info(f"Tool result: {chunk['tool']} with output: {chunk['result']}")
                    message_placeholder.write(chunk['result'])
                elif chunk["type"] == "output":
                    logging.info(f"Output: {chunk['content']}")
                    message_placeholder.write(chunk['content'] + "▮")
                    time.sleep(0.5)

        except RAGError as e:
            st.error(f"Une erreur est survenue lors de la generation de la réponse de l'assistant: {e}")


    # # 3. Ajouter la réponse de l'assistant à l'historique (pour affichage UI)
    # st.session_state.messages.append({"role": "assistant", "content": answer})

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")