import pickle
import os
from tqdm import tqdm
from PIL import Image

class Reformatter:
    def __init__(self, storage_dir, target_dir):
        """ A df with filename, rec_id, label """
        self.storage_dir = storage_dir
        self.target_dir = target_dir
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
    
    def unpack(self, path):
        with open(path, 'rb') as f:
            slice_ = pickle.load(f)
        return slice_
    
    def transform(self, slice_):
        img = Image.fromarray(slice_).convert('L')
        return img
    
    def store(self, img, image_path):
        img.save(image_path)
        
    def __call__(self):
        for file in tqdm(os.listdir(self.storage_dir)):
            name = file.replace('.pkl', '.png')
            image_path = os.path.join(self.target_dir, name)
            if not os.path.isfile(image_path):
                path = os.path.join(self.storage_dir, file)
                try:
                    slice_ = self.unpack(path)
                    slice_ -= slice_.min()
                    slice_ /= slice_.max()
                    slice_ *= 255
                    img = self.transform(slice_)
                    self.store(img, image_path)
                except:
                    print('Problem with', file)
            

ref = Reformatter('storage/noise_slices', 'storage/noise_images')
ref()
