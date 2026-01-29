import streamlit as st

def render_chat_history(messages):
    """
    Renders the chat history from st.session_state.messages, 
    including optional tool call logs in an expander.
    """
    for message in messages:
        with st.chat_message(message["role"]):
            # Check if there are tool logs stored for this assistant message
            if "tools" in message and message["tools"]:
                with st.expander("Tool call", expanded=False):
                    st.markdown(message["tools"])
            
            # Display the actual text response
            st.write(message["content"])