print("--- Checking Chains ---")
try:
    from langchain_classic.chains.retrieval import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    print("SUCCESS: Langchain Classic Chains imported")
except ImportError as e:
    print(f"FAILED: Langchain Classic Chains: {e}")

print("\n--- Checking Agents ---")
try:
    from langchain_classic.agents import initialize_agent, AgentType, AgentExecutor, create_openai_functions_agent
    print("SUCCESS: Langchain Classic Agents imported")
except ImportError as e:
    print(f"FAILED: Langchain Classic Agents: {e}")

print("\n--- Checking Hub ---")
try:
    from langchain_classic import hub
    print("SUCCESS: Langchain Classic Hub imported")
except ImportError as e:
    print(f"FAILED: Langchain Classic Hub: {e}")

print("\n--- Checking LangChain Core/Community ---")
try:
    import langchain_core
    import langchain_community
    import langchain_openai
    print("SUCCESS: Core/Community/OpenAI packages imported")
except ImportError as e:
    print(f"FAILED: Core/Community/OpenAI: {e}")
