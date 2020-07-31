import math

def combinations(n, k):
    if n < k:
        raise ValueError('n cannot be smaller than k in combination(n, k)')
    return int(math.factorial(n) / (math.factorial(n-k) * math.factorial(k)))