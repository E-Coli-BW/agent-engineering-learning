"""
Black Box 函数实现 — 面试题的隐藏函数
======================================

⚠️ 这个文件是黑盒函数的实现，游戏期间不要打开看！
⚠️ DO NOT READ THIS FILE - it's the secret function!
"""

import math

def evaluate(x: int, y: int) -> float:
    """
    Black-box function. Takes discrete x and y, returns a score.
    x: integer in range [0, 20]
    y: integer in range [0, 20]
    Returns: float score
    """
    # Secret formula - hidden from the player
    score = (
        - (x - 7) ** 2
        - 2 * (y - 13) ** 2
        + 3 * math.sin(x * 0.8) * math.cos(y * 0.5)
        + x * 0.5
        + y * 0.3
        - 0.1 * (x - y) ** 2
    )
    return round(score, 4)

def get_optimal():
    """Returns the true optimal for validation."""
    best_score = float('-inf')
    best_x, best_y = 0, 0
    for x in range(21):
        for y in range(21):
            s = evaluate(x, y)
            if s > best_score:
                best_score = s
                best_x, best_y = x, y
    return best_x, best_y, best_score
