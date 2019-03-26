class Exponent:
    def __init__(self, exp=0.17):
        self.exp = exp
        
    def __call__(self, img):
        return img ** 0.17
        
    def __repr__(self):
        return self.__class__.__name__ + f' Factor: {self.exp}'
