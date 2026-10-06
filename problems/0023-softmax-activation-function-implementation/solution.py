import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    m = max(scores)

    exps = []
    for x in scores:
        exps.append(math.exp(x - m))

    total = sum(exps)

    result = []
    for x in exps:
        result.append(x / total)

    return result