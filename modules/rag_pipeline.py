import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from config import VECTOR_STORE_DIR, FAISS_INDEX_NAME
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
        self.vector_store = None

    def build_index(self, chunks):
        """Build and save FAISS index from document chunks."""
        try:
            logger.info("Building FAISS index...")

            self.vector_store = FAISS.from_documents(
                chunks,
                self.embeddings
            )

            self.vector_store.save_local(
                VECTOR_STORE_DIR,
                index_name=FAISS_INDEX_NAME
            )

            logger.info("FAISS index saved successfully.")
            return True

        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            return False

    def load_index(self):
        """Load existing FAISS index."""
        try:
            index_path = os.path.join(
                VECTOR_STORE_DIR,
                f"{FAISS_INDEX_NAME}.faiss"
            )

            if os.path.exists(index_path):
                self.vector_store = FAISS.load_local(
                    VECTOR_STORE_DIR,
                    self.embeddings,
                    index_name=FAISS_INDEX_NAME,
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS index loaded successfully.")
                return True

            logger.warning("FAISS index not found.")
            return False

        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False

    def answer_question(self, question):
        """Retrieve context and answer the question using modern retrieval chain."""
        try:
            # Ensure vector store is loaded
            if not self.vector_store:
                if not self.load_index():
                    return {
                        "result": "Knowledge base is empty. Please upload documents first.",
                        "source_documents": []
                    }

            # Prompt
            system_prompt = (
                "Use the given context to answer the question. "
                "If you don't know the answer, say you don't know.\n\n"
                "Context:\n{context}"
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])

            # Create chains
            document_chain = create_stuff_documents_chain(
                self.llm,
                prompt
            )

            retrieval_chain = create_retrieval_chain(
                self.vector_store.as_retriever(search_kwargs={"k": 3}),
                document_chain
            )

            # Run
            response = retrieval_chain.invoke({
                "input": question
            })

            return {
                "result": response.get("answer", ""),
                "source_documents": response.get("context", [])
            }

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "result": "An error occurred while processing your question.",
                "source_documents": []
            }