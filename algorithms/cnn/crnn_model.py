import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=32, hidden_size=256, in_channels=1):
        super().__init__()
        assert img_height % 16 == 0, "CRNN requires the input height to be a multiple of 16."

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                               

            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                  

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),                         

            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),                      

            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
           
        )

        self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True,
                            batch_first=False, dropout=0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes + 1)  

    def forward(self, x):
        feat = self.cnn(x)                    
        assert feat.size(2) == 1, f"The height should be reduced to 1, but it is actually{feat.size(2)}, Check if the input height is 32"
        feat = feat.squeeze(2)                 
        feat = feat.permute(2, 0, 1)           

        rnn_out, _ = self.rnn(feat)            
        out = self.fc(rnn_out)                 
        return out.log_softmax(2)              