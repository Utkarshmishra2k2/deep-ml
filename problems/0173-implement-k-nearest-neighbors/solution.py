import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    if k <= 0:
        return []

    qx, qy = query_point
    ranked = []

    for point in points:
        x = point[0]
        y = point[1]
        distance = (x - qx) ** 2 + (y - qy) ** 2
        ranked.append((distance, point))

    ranked.sort(key=lambda item: item[0])
    return [item[1] for item in ranked[:k]]