import logging
import logfire
from langchain.callbacks.base import BaseCallbackHandler

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    logfire.configure()

def log_retry(retry_state):
    exc = retry_state.outcome.exception()
    wait = retry_state.next_action.sleep
    logging.warning(
        "Retrying LLM call (%s/%s) after %.1fs due to %s",
        retry_state.attempt_number,
        retry_state.stop.max_attempt_number,
        wait,
        exc,
    )

class LogfireCallback(BaseCallbackHandler):
    """Custom callback to send LangChain events to Logfire."""
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        logfire.info(f"Tool call: {tool_name}", input=input_str)
    
    def on_tool_end(self, output, **kwargs):
        logfire.info("Tool result", output=output)
    
    def on_tool_error(self, error, **kwargs):
        logfire.error("Tool error", error=str(error))
    
    def on_agent_action(self, action, **kwargs):
        logfire.info(f"Agent action: {action.tool}", input=action.tool_input)
    
    def on_agent_finish(self, finish, **kwargs):
        logfire.info("Agent finished", output=finish.return_values)
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        logfire.info("LLM call started")
    
    def on_llm_end(self, response, **kwargs):
        logfire.info("LLM call completed")