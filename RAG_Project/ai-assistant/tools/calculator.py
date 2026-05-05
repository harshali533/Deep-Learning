import numexpr
from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    """Useful for when you need to answer questions about math. 
    Input should be a mathematical expression, for example: 2*2+3"""
    try:
        # Using numexpr for safer and more powerful math evaluation
        result = numexpr.evaluate(expression).item()
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}. Please provide a valid mathematical expression."
