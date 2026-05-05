from langchain_openai import ChatOpenAI
from utils.logger import get_logger

logger = get_logger(__name__)

def get_general_answer(prompt):
    """Get a response from the LLM without any context or tools."""
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        return llm.invoke(prompt)
    except Exception as e:
        logger.error(f"General LLM error: {str(e)}")
        return f"Error: {str(e)}"
