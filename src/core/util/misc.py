def yield_n(iter, n):
    """
    Returns the next n elements from an iterator.

    Args:
        iter (iterator): Iterator.
        n (int): Number of elements.

    """
    return [next(iter) for _ in range(n)]
