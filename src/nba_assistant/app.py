import streamlit as st
import logging

# --- Importations depuis vos modules ---
try:
    from nba_assistant.config.config import (
        APP_TITLE, NAME, MODEL_NAME, 
        MISTRAL_API_KEY, SEARCH_K, SYSTEM_PROMPT
    )
    from nba_assistant.llm.vector_store_management import create_vector_store_manager, VectorStoreInitializationError
    from nba_assistant.utils.logging_handler import setup_logging
    from nba_assistant.llm.llm import create_client, generer_reponse,ClientCreationError, RAGError
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()

# --- Configuration du Logging ---
setup_logging()

# --- Creation du client Mistral ---
try:
    client = create_client(MISTRAL_API_KEY)
except ClientCreationError as e:
    st.error(f"Une erreur est survenue lors de la connexion à l'API Mistral: {e}")
    st.stop()

# --- Chargement du Vector Store (mis en cache) ---
@st.cache_resource # Garde le manager chargé en mémoire pour la session
def get_vector_store_manager():
    logging.info("Chargement du Vector Store...")
    return create_vector_store_manager()

try:
    vector_store_manager = get_vector_store_manager()
except VectorStoreInitializationError:
    st.error(
        "La base de connaissances n'est pas disponible. "
        "Veuillez vérifier l'index."
    )
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
            answer = generer_reponse(client, prompt, vector_store_manager, SEARCH_K, SYSTEM_PROMPT)
        except RAGError as e:
            st.error(str(e))
            answer = "Une erreur technique est survenue."
        except ValueError as e:
            st.warning(str(e))
            answer = "Veuillez poser une question valide."

        # Affichage de la réponse complète
        message_placeholder.write(answer)

    # 3. Ajouter la réponse de l'assistant à l'historique (pour affichage UI)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")