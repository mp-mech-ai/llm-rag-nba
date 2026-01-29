import streamlit as st
import logging
from nba_assistant.utils.st_helper import render_chat_history
from dotenv import load_dotenv

load_dotenv()

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

def trim_output(output):
    max_length = 1000
    if len(output) > max_length:
        return output[:max_length // 2] + "\n[...]\n" + output[-max_length // 2:]
    return output

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
render_chat_history(st.session_state.messages)

# Zone de saisie utilisateur
if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    # 1. Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Afficher indicateur + Générer la réponse de l'assistant via LLM
    with st.chat_message("assistant"):
        tool_placeholder = st.empty()
        message_placeholder = st.empty()
        message_placeholder.text("...") # Indicateur simple

        tool = ""
        answer = ""
        # Génération de la réponse de l'assistant en utilisant la RAG
        logging.info(f"Génération de la réponse de l'assistant pour la question: {prompt}")
        try:
            for chunk in agent.stream(prompt):
                if chunk["type"] == "tool_call":
                    logging.info(f"Tool call: {chunk['tool']} with input: {chunk['tool_input']}")

                    tool += f"Tool call: `{chunk['tool']}` with input: `{chunk['tool_input']}`"
                    tool_placeholder.expander("Tool call", expanded=False).markdown(tool)

                elif chunk["type"] == "tool_result":
                    logging.info(f"Tool result: {chunk['tool']} with output: {chunk['result']}")

                    tool += f"Tool result: `{chunk['tool']}` with output: `{trim_output(chunk['result'])}`"
                    tool_placeholder.expander("Tool call", expanded=False).write(tool)
                elif chunk["type"] == "output":
                    logging.info(f"Output: {chunk['content']}")
                    message_placeholder.write(chunk['content'])
                    answer += chunk['content'] + "\n"

        except RAGError as e:
            st.error(f"Une erreur est survenue lors de la generation de la réponse de l'assistant: {e}")
            answer = "Une erreur est survenue lors de la generation de la réponse de l'assistant."


    # 3. Ajouter la réponse de l'assistant à l'historique (pour affichage UI)
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "tools": tool
    })

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")