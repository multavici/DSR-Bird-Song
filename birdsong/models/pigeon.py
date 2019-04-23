import torch
import torch.nn as nn

class Pigeon(nn.Module):
    def __init__(self, freq_axis=256, time_axis=216,  no_classes=100):
        super(Pigeon, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.Dropout(0.3),
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.Dropout(0.3),
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3)
            )
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3)
            )
        
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            )
        
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
            )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, no_classes),
            nn.ReLU()
            )
        
    def pad_wrap1d(self, x, pad):

        x = torch.cat([x, x[:,:,:, 0:pad]], dim=3)
        x = torch.cat([x[:,:,:, -2 * pad:-pad], x], dim=3)

        return x

    def forward(self, x):
        x = self.pad_wrap1d(x, 20)
        out = self.layer1(x)
        print('1.', out.shape)
        out = self.layer2(out)
        print('2.', out.shape)
        out = self.layer3(out)
        print('3.', out.shape)
        out = self.layer4(out)
        print('4.', out.shape)
        out = self.layer5(out)
        print('5.', out.shape)
        out = self.layer6(out)
        print('6.', out.shape)
        out = self.layer7(out)
        out = out.view(out.shape[0], 512)
        print('7.', out.shape)
        out = self.fc1(out)
        print('8.', out.shape)
        out = self.fc2(out)
        print('9.', out.shape)
        return out

def test():
    cnn = Pigeon(256, 216, 100)
    summary(cnn, (1, 256, 216))

if __name__=="__main__":
    from torchsummary import summary
    test()