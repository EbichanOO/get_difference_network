import numpy as np
import math
def dateMaker():
    i=1
    send = np.array([])
    while True:
        result = np.deg2rad(np.arange(i*10, (1+i)*10))
        send = np.append(send, np.cos(result, dtype=np.float32) if i%40==0 else np.sin(result, dtype=np.float32))
        if i%10==0:
            yield np.array(send.reshape(10, 10), dtype=np.float32)
            send = np.array([])
        i += 1