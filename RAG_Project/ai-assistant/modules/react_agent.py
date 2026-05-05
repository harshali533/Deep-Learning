from langchain_classic.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from tools.calculator import calculate
from utils.logger import get_logger

logger = get_logger(__name__)

def run_react_agent(query):
    """Run a ReAct agent to demonstrate explicit reasoning."""
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        tools = [calculate]
        
        # Using ZERO_SHOT_REACT_DESCRIPTION which is the classic ReAct agent
        agent = initialize_agent(
            tools, 
            llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            verbose=True,
            handle_parsing_errors=True
        )
        
        logger.info(f"Running ReAct agent for query: {query}")
        response = agent.invoke({"input": query})
        return response["output"]
    except Exception as e:
        logger.error(f"ReAct Agent error: {str(e)}")
        return f"Error: {str(e)}"
