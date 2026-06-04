import math
import random
import numpy as np

class OUE:

    def __init__(self, d, epsilon):
        self.epsilon = epsilon
        self. d = d
        self.p = 0.5
        self.q = 1/ (np.exp(self.epsilon) + 1)

    def perturb(self, index):
        oh_vec = np.random.choice([1, 0], size=self.d, p=[self.q, 1 - self.q])  # If entry is 0, flip with prob q
        if random.random() < self.p:
            oh_vec[index] = 1
        else:
            oh_vec[index] = 0
        return oh_vec