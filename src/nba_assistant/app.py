import streamlit as st
import logging
from nba_assistant.utils.st_helper import render_chat_history, trim_output, render_tool_output, render_tool_input
from dotenv import load_dotenv
import time

load_dotenv()

# --- Importations depuis vos modules ---
try:
    from nba_assistant.config.config import APP_TITLE, NAME, MODEL_NAME
    from nba_assistant.utils.logging_handler import setup_logging
    from nba_assistant.llm.llm import RAGError, RAGAgent
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()

st.markdown("""
<style>
    .typing-indicator {
        display: inline-block;
        font-size: 20px;
        font-weight: bold;
        color: #555;
    }
    .dot {
        animation: blink 1.4s infinite both;
        font-size: 25px;
    }
    .dot:nth-child(2) { animation-delay: .2s; }
    .dot:nth-child(3) { animation-delay: .4s; }

    @keyframes blink {
        0% { opacity: .2; }
        20% { opacity: 1; }
        100% { opacity: .2; }
    }
</style>
""", unsafe_allow_html=True)

typing_dots = '<div class="typing-indicator"><span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>'

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
render_chat_history(st.session_state.messages)

# Zone de saisie utilisateur
if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    # 1. Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Afficher indicateur + Générer la réponse de l'assistant via LLM
    with st.chat_message("assistant"):
        tools_container = st.container()

        message_placeholder = st.empty()
        message_placeholder.markdown(typing_dots, unsafe_allow_html=True)

        answer = ""
        tool_steps = []
        current_status = None

        # Génération de la réponse de l'assistant en utilisant la RAG
        logging.info(f"Génération de la réponse de l'assistant pour la question: {prompt}")
        try:
            for chunk in agent.stream(prompt):
                # Appel d'un nouvel outil
                if chunk["type"] == "tool_call":
                    logging.info(f"Tool call: {chunk['tool']} with input: {chunk['tool_input']}")
                    t0 = time.time()
                    tool_name = chunk['tool']
                    tool_input = chunk['tool_input']
                    current_status = tools_container.status(f"🛠️ Executing: {tool_name}...", expanded=False)
                    with current_status:
                        render_tool_input(tool_name, tool_input)

                    tool_steps.append({
                        "name": tool_name,
                        "input": tool_input,
                        "output": None
                    })

                # Résultat d'un outil
                elif chunk["type"] == "tool_result":
                    t1 = time.time()
                    logging.info(f"Tool result: {chunk['tool']} with output: {chunk['result']} in {(t1-t0)*1000:.2f} ms")
                    tool_name = chunk['tool']
                    result_content = chunk['result']

                    if current_status:
                        with current_status:
                            render_tool_output(tool_name, result_content)
                        current_status.update(
                            label=f"{tool_name} in {(t1-t0)*1000:.2f} ms", 
                            state="complete", 
                            expanded=False
                        )
                        current_status = None

                    if tool_steps:
                        tool_steps[-1]["output"] = result_content
                
                # Réponse finale du LLM
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
        "tool_steps": tool_steps 
    })

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")