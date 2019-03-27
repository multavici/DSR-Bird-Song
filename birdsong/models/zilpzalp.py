import torch
import torch.nn as nn
import torch.nn.functional as F


class Zilpzalp(nn.Module):
    def __init__(self, freq_axis=701, time_axis=80,  no_classes=10):
        super(Zilpzalp, self).__init__()

        self.time_axis = time_axis
        self.freq_axis = freq_axis
        self.__name__='Zilpzalp'

        # Frequency block
        self.frequency = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=(9, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride =3),

            nn.Conv2d(32,64, kernel_size=(9, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride =3),

            nn.Conv2d(64,128, kernel_size=(9, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=3, stride =3),
            )

        # Time block
        self.time = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=(3, 9), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride =3),

            nn.Conv2d(32,64, kernel_size=(3, 9), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride =3),

            nn.Conv2d(64,128, kernel_size=(3, 9), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=3, stride =3),
            )

        # Summary block
        self.summary = nn.Sequential(
            nn.Conv2d(128,64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride =2),

            nn.Conv2d(64,no_classes, kernel_size=1, stride=1),
            )

    def forward(self, x):
        batch_size = x.shape[0]
        freq = self.frequency(x).view(batch_size, 128, -1, 1)
        time = self.time(x).view(batch_size, 128, 1, -1)
        
        comb = torch.matmul(freq, time)
        summ = self.summary(comb)
        out = F.max_pool2d(summ, kernel_size=summ.size()[2:]).view(batch_size, summ.size()[1])

        return out

def test():
    image = torch.randn(64, 1, 256, 216)
    cnn = Zilpzalp(256, 216, 10)
    output = cnn(image)
    print("input shape:")
    print(image.shape)
    print("output shape:")
    print(output.shape)

if __name__ == '__main__':
    test()