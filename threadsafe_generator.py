import threading
import numpy as np


class ThreadSafeIterator:

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*args, **kwargs):
        return ThreadSafeIterator(f(*args, **kwargs))

    return g


@threadsafe_generator
def train_generator(dt, batch_size, shuffle=False):
	num_data = dt.shape[0]
	num_batch = num_data // batch_size
	num_left = num_data % batch_size
	if shuffle:
		idx = np.random.permutation(num_data)
	else:
		idx = np.arange(num_data)
	while True:
		for batch in range(num_batch):
			idx_start = batch * batch_size
			idx_end = idx_start + batch_size
			idx_batch = idx[idx_start:idx_end]
			yield dt[idx_batch]
		if num_left:
			idx_batch = idx[idx_end:]
			yield dt[idx_batch]


if __name__ == '__main__':
    raw_data = np.random.normal(size=(20, 3))
    BZ = 2
    gen = train_generator(raw_data, BZ)

    i = 0
    for dt in gen:
        print(dt)
        i += 1
        if i > 20:
            break
