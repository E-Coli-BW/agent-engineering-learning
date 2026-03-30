"""
MCP Server - Black Box Optimization Game
The function implementation is hidden in black_box.py
"""

from mcp.server.fastmcp import FastMCP
from black_box import evaluate, get_optimal
from datetime import datetime

mcp = FastMCP("BlackBox Optimizer")

# In-memory history
_history: list[dict] = []
_best_score = float('-inf')
_best_x = None
_best_y = None

@mcp.tool()
def query(x: int, y: int) -> dict:
    """
    Query the black-box function with given x and y values.
    
    Args:
        x: An integer input
        y: An integer input
    
    Returns:
        A dict with the score and current best record, or an error if out of bounds.
    """
    global _best_score, _best_x, _best_y

    if not isinstance(x, int) or not isinstance(y, int):
        return {
            "error": "x and y must be integers",
            "valid": False
        }

    if not (0 <= x <= 20) or not (0 <= y <= 20):
        return {
            "error": "x and/or y is out of the valid range",
            "valid": False
        }

    score = evaluate(x, y)
    is_new_best = score > _best_score

    if is_new_best:
        _best_score = score
        _best_x = x
        _best_y = y

    record = {
        "x": x,
        "y": y,
        "score": score,
        "is_new_best": is_new_best,
        "current_best": {
            "x": _best_x,
            "y": _best_y,
            "score": _best_score
        },
        "total_queries": len(_history) + 1
    }
    _history.append(record)
    return record


@mcp.tool()
def get_history() -> dict:
    """
    Get the full query history and current best result so far.
    
    Returns:
        All past queries and the current best (x, y, score).
    """
    return {
        "total_queries": len(_history),
        "best_so_far": {
            "x": _best_x,
            "y": _best_y,
            "score": _best_score
        } if _best_x is not None else None,
        "history": _history
    }


@mcp.tool()
def reset() -> dict:
    """
    Reset all query history and best record. Start a new game session.
    """
    global _history, _best_score, _best_x, _best_y
    _history = []
    _best_score = float('-inf')
    _best_x = None
    _best_y = None
    return {"message": "Game reset. Good luck!"}


@mcp.tool()
def judge(x: int, y: int) -> dict:
    """
    Judge your final answer: submit (x, y) as your best guess for the optimal point.
    This will tell you how close you are to the true optimum.

    Args:
        x: Your guessed optimal x (integer)
        y: Your guessed optimal y (integer)

    Returns:
        Whether you found the optimum, your score, and how far you are from the true best.
    """
    optimal_x, optimal_y, optimal_score = get_optimal()
    your_score = evaluate(x, y)
    gap = round(optimal_score - your_score, 4)
    found = (x == optimal_x and y == optimal_y)

    result = {
        "your_answer": {"x": x, "y": y, "score": your_score},
        "found_optimal": found,
        "score_gap": gap,
        "total_queries_used": len(_history),
    }

    if found:
        result["message"] = "🎉 Perfect! You found the global optimum!"
    elif gap < 1.0:
        result["message"] = f"🔥 Very close! Gap is only {gap}. Almost there!"
    elif gap < 5.0:
        result["message"] = f"😊 Getting warm! Gap is {gap}."
    else:
        result["message"] = f"❄️  Still far off. Gap is {gap}. Keep exploring!"

    return result


if __name__ == "__main__":
    import uvicorn
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="127.0.0.1", port=8339)
