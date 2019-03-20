import numpy as np
import random
import pickle
import os

class SoundscapeNoise(object):
    def __init__(self, noise_dir, scaling=0.2):
        self.noise_dir = noise_dir
        self.files = os.listdir(noise_dir)
        self.scaling = scaling
        
    def __call__(self, img):
        file = random.choice(self.files)
        noise = self._unpickle(os.path.join(self.noise_dir, file))
        noise -= noise.min()
        noise /= noise.max()
        return img + self.scaling * noise
        
    def _unpickle(self, path):
        with open(path, 'rb') as f:
            slice_ = pickle.load(f)
        return slice_

    def __repr__(self):
        return self.__class__.__name__ + 'parameters'

class VerticalRoll(object):
    def __init__(self, amount=10):
        self.amount = amount

    def __call__(self, img):
        roll = np.random.randint(-self.amount, self.amount)
        return np.roll(img, roll, axis=0)

    def __repr__(self):
        return self.__class__.__name__ + f' Amount={self.amount}'

class GaussianNoise(object):
    def __init__(self, scaling=0.05):
        self.scaling = scaling
    
    def __call__(self, img):
        noise = np.random.normal(size=img.shape) * np.random.uniform(low = -self.scaling, high = self.scaling)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + f' Scaling={self.scaling}'
