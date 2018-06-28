from collections import deque
import random
import numpy as np


class RecencyBuffer(object):

    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        # Right side of deque contains newest experience
        self.buffer = deque(maxlen = self.buffer_size)

    def add(self, s, a, sp, r, gamma):
        self.buffer.append([s, a, sp, r, gamma])

    def getSize(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def clear(self):
        self.buffer.clear()