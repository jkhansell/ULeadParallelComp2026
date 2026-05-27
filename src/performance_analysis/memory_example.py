
import numpy as np

def allocate_memory():
    x = np.random.rand(10_000_000)
    y = np.random.rand(10_000_000)

    return x + y
