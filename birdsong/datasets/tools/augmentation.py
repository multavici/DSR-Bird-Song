import numpy as np

class SoundscapeNoise(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return

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
