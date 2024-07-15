def alternating_indices(window_size):
    """Generate alternating indices from -window_size//2 to window_size//2

    Example: window_size=7 -> [0, -1, 1, -2, 2, -3, 3]
    """
    yield 0
    for i in range(1, window_size // 2 + 1):
        yield -i
        yield i
