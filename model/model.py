import torch.nn as nn
import brevitas as qnn


class QuadbitMnistModel(nn.Module):
    def __init__(self, n_pixel):
        super(QuadbitMnistModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                      padding_mode="replicate"),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        n_pixel = n_pixel//2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                      padding_mode="replicate"),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        n_pixel = n_pixel//2
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                      padding_mode="replicate"),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        n_pixel = n_pixel//2
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=128*(n_pixel ** 2), out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)

        return x
