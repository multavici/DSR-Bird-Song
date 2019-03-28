
import torch.nn as nn

class Pigeon(nn.Module):
    def __init__(self, freq_axis=256, time_axis=216,  no_classes=100):
        super(Pigeon, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.summ1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(30, 1), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
            )
        
        self.summ2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
            )
        
        self.summ3 = nn.Sequential(
            nn.Conv2d(64, no_classes, kernel_size=(1, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3))
            )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.summ1(out)
        out = self.summ2(out)
        out = self.summ3(out)
        return out.squeeze()



def test():
    cnn = Pigeon(256, 216, 100)
    summary(cnn, (1, 256, 216))

if __name__=="__main__":
    from torchsummary import summary
    test()