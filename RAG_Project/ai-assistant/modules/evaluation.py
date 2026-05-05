from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision
from datasets import Dataset
from utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_response(query, response, context):
    """Evaluate a single RAG response using RAGAS metrics."""
    try:
        data = {
            "question": [query],
            "answer": [response],
            "contexts": [[context]]
        }
        dataset = Dataset.from_dict(data)
        
        # Evaluate
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevance, context_precision]
        )
        return result.to_pandas().to_dict(orient='records')[0]
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return {"error": str(e)}
