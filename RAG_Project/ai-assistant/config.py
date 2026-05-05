import os
from dotenv import load_dotenv

load_dotenv()

# App Settings
APP_TITLE = "🧠 Intelligent GenAI Assistant"
DATA_DIR = os.path.join(os.getcwd(), "data")
VECTOR_STORE_DIR = os.path.join(os.getcwd(), "vectorstore")
FAISS_INDEX_NAME = "index"

# API Keys
OPENAI_API_KEY = os.getenv("college") or os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Ensure OPENAI_API_KEY is in environment for LangChain
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
