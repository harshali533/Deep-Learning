from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_classic import hub
from tools.calculator import calculate
from langchain_community.tools.tavily_search import TavilySearchResults
from utils.logger import get_logger

logger = get_logger(__name__)

class GeneralAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Initialize search tool
        try:
            self.search = TavilySearchResults()
            self.tools = [calculate, self.search]
        except Exception:
            logger.warning("Tavily API key not found. Search tool disabled.")
            self.tools = [calculate]

        # Get prompt from hub or define manually
        self.prompt = hub.pull("hwchase17/openai-functions-agent")
        
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def run(self, query: str):
        """Run the agent on a query."""
        try:
            response = self.agent_executor.invoke({"input": query})
            return response["output"]
        except Exception as e:
            logger.error(f"Agent error: {str(e)}")
            return f"An error occurred while processing your request: {str(e)}"
