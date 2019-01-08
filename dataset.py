import numpy as np
import math
def dateMaker():
    i=0
    while True:
        result = np.deg2rad(np.arange(i*10, (1+i)*10))
        yield np.cos(result) if i%40==0 else np.sin(result)
        i += 1