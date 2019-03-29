import torch
from torch import nn


class LstmModel(nn.Module):

    def __init__(self, freq_axis, time_axis, no_classes):
        super(LstmModel, self).__init__()
        
        self.freq_axis = freq_axis #input_dim
        self.time_axis = time_axis
        self.no_classes = no_classes
        
        self.input_features = 1622 #input_dim
        self.seq_length = 66 
        
        # Hyper parameters
        # Hidden dimensions and number of hidden layers
        self.hidden_dim = 200 #500
        self.layer_dim = 3 #7
        
        
        self.timbral_features = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=(9,3), stride=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    #nn.Dropout(0.3),

                    nn.Conv2d(16, 32, kernel_size=(5,3), stride=1),
                    #nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.MaxPool2d(kernel_size=(3,3), stride=(3,3)),


                    nn.Conv2d(32, 64, kernel_size=5, stride=1),
                    #nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.MaxPool2d(kernel_size=(3,1), stride=(3,1)),
                    )

          
        
        self.rhythm = nn.Sequential(
                    nn.AvgPool2d(kernel_size=(150,19), stride=(5,3))
                    )

        
        # batch_first=True shapes Tensors : batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(self.input_features, self.hidden_dim, self.layer_dim, dropout=0.5, batch_first=True)
        
        
        self.fc = nn.Linear(self.hidden_dim, self.no_classes)
        
    def forward(self, x):   
        rhythm_out = self.rhythm(x)
        #print(rhythm_out.shape)

        rhythm_out = rhythm_out.view(rhythm_out.shape[0], -1 , 66)
        
        timbral_out = self.timbral_features(x)
        #print(timbral_out.shape)

        timbral_out = timbral_out.view(timbral_out.shape[0], -1 , 66)
        stack = torch.cat((rhythm_out, timbral_out), 1).permute(0,2,1)  

       
        #LSTM input Shape: batch_dim, sequence_dim, feature_dim
        #print(stack.shape)
        output_seq, hidden_state = self.lstm(stack)
        last_output = output_seq[:, -1]
        out = self.fc(last_output)
        
        return out
