import streamlit as st
import ast
import pandas as pd

def trim_output(output):
    max_length = 1000
    if len(output) > max_length:
        return output[:max_length // 2] + "\n[...]\n" + output[-max_length // 2:]
    return output

def render_tool_output(tool_name, result_str):
    """
    Renders the tool output nicely inside the current Streamlit container.
    """
    st.markdown("**Result:**")
    if tool_name == "sql_db_query":
        try:
            # Parse string representation of list -> Python List
            data = ast.literal_eval(result_str)
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("Query returned no results.")
        except:
            st.code(result_str) # Fallback

    elif tool_name == "sql_db_schema":
        st.code(result_str, language="sql")

    elif tool_name == "vector_store_research":
        st.markdown(trim_output(result_str))

    elif tool_name == "sql_db_list_tables":
        # Split "A, B, C" into badges
        tables = result_str.split(',')
        st.markdown(" ".join([f"`{t.strip()}`" for t in tables]))
        
    else:
        st.write(result_str)

def render_tool_input(tool_name, tool_input):
    if tool_name == "sql_db_query":
        st.markdown(f"**Input:**")
        st.code(tool_input['query'], language="sql")
    elif tool_name == "vector_store_research":
        st.markdown(f"**Input:**")
        st.json(tool_input)
    elif tool_name == "sql_db_list_tables":
        pass
    elif tool_name == "sql_db_schema":
        st.markdown(f"**Input:**")
        st.json(tool_input)
    else:
        st.write(tool_input)

def render_chat_history(messages):
    """
    Renders chat history. 
    Reconstructs rich tool outputs from the saved 'tool_steps' list.
    """
    for message in messages:
        with st.chat_message(message["role"]):
            
            # 1. Re-render Tool Calls (if any exist)
            if "tool_steps" in message and message["tool_steps"]:
                
                # Iterate through the list of tool executions saved for this message
                for step in message["tool_steps"]:
                    tool_name = step["name"]
                    tool_input = step["input"]
                    tool_output = step["output"]

                    # Re-create the UI element in 'complete' state
                    with st.status(f"✅ {tool_name} (Complete)", state="complete", expanded=False):
                        render_tool_input(tool_name, tool_input)
                        st.markdown("**Result:**")
                        # Use the same helper function to get the nice tables/code
                        render_tool_output(tool_name, tool_output)

            # 2. Render Final Answer
            st.markdown(message["content"])