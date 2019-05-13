import numpy as np
import random
import pickle
import os
from PIL import Image
from torchvision import transforms

class SoundscapeNoise(object):
    def __init__(self, noise_dir, scaling=0.2):
        self.scaling = scaling
        self.noise_bank = self._load_noise(noise_dir)
        
    def __call__(self, img):
        noise = random.choice(self.noise_bank)
        noise -= noise.min()
        noise /= noise.max()
        noise *= img.max()
        return img + self.scaling * noise
    
    def _load_noise(self, noise_dir):
        print('Loading noise bank into RAM.')
        noise = []
        for file in os.listdir(noise_dir):
            if file.endswith('.pkl'):
                path = os.path.join(noise_dir, file)
                noise_slice = self._load(path)
                if noise_slice.max() != 0:
                    noise.append(noise_slice)
        return noise
    
    def _load(self, path):
        with open(path, 'rb') as f:
            slice_ = pickle.load(f)
        return slice_

    def __repr__(self):
        return self.__class__.__name__ + f' Scaling: {self.scaling}, Total Noise: {len(self.noise_bank)}'



class ImageSoundscapeNoise(SoundscapeNoise):
    def __init__(self, noise_dir, scaling=0.2):
        super(ImageSoundscapeNoise, self).__init__(noise_dir, scaling)
    
    def __call__(self, img):
        noise = random.choice(self.noise_bank)
        return img + self.scaling * noise
    
    def _load_noise(self, noise_dir):
        print('Loading noise bank into RAM.')
        noise = []
        for file in os.listdir(noise_dir):
            path = os.path.join(noise_dir, file)
            noise_slice = self._load(path)
            if noise_slice.max() != 0:
                noise.append(noise_slice)
        return noise
    
    def _load(self, path):
        noise = Image.open(path)
        return transforms.ToTensor()(noise)


class VerticalRoll(object):
    def __init__(self, amount=10):
        self.amount = amount

    def __call__(self, img):
        roll_axis = np.random.choice(10, 2)
        roll_freq = np.random.randint(-self.amount, self.amount)
        roll_time = np.random.randint(-self.amount, self.amount)
        if roll_axis[0] >= 5:
            return np.roll(np.roll(img, roll_freq, axis=0), roll_time, axis= 1)
        elif roll_axis[1] >= 5:
            return np.roll(img, roll_freq, axis=0)
        else:
            return np.roll(img, roll_time, axis=1)


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
