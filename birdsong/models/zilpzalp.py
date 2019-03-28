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
            
            nn.Conv2d(128,256, kernel_size=(3, 5), stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=3, stride =3)
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
            
            nn.Conv2d(128,256, kernel_size=(6, 2), stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=3, stride =3)
            )

        # Summary block
        self.summary = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, no_classes),
            nn.ReLU(),
            
            )

    def forward(self, x):
        freq = self.frequency(x)
        time = self.time(x)
        comb = torch.cat((freq, time), 1).squeeze()
        out = self.summary(comb)
        return out

def test():
    cnn = Zilpzalp (256, 216, 100)
    summary(cnn, (1, 256, 216))

if __name__=="__main__":
    from torchsummary import summary
    test()
