import threading
from queue import Queue
import time
import numpy as np
import tqdm

np.random.seed(42)

raw_data = np.random.normal(size=(20, 3))
BZ = 2
N = raw_data.shape[0]

buffer = Queue()
ans = []
def data_loader(buffer, ):
    for i in range(0, N, BZ):
        time.sleep(0.1)
        x_batch = raw_data[i:i+BZ]
        buffer.put(x_batch)

def predictor(buffer, ):
    for _ in tqdm.trange(0, N, BZ):
        time.sleep(0.05)
        ans.append(buffer.get())

q_size = 2
q = Queue(maxsize=q_size)

t1 = threading.Thread(target=data_loader, name='DataLoader', args=(q,))
t2 = threading.Thread(target=predictor, name='Predictor', args=(q,))

t1.start()
t2.start()

t1.join()
t2.join()
