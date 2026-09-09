import torch
import torch.nn as nn

class VGG16(nn.Module):
    '''
    VGG16 model (same with ATS and RTS)
    '''
    def __init__(self):
        super(VGG16, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=(1, 1), padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=(1, 1), padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=(1, 1), padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=(1, 1), padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=(1, 1), padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=(1, 1), padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=(1, 1), padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Block 5
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=(1, 1), padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=(1, 1), padding=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=(1, 1), padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        # Block 1
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)

        # Block 2
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)

        # Block 3
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.pool3(x)

        # Block 4
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.pool4(x)

        # Block 5
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.pool5(x)

        # x = x.view(batch_size, -1)
        x = x.permute(0, 2, 3, 1)  # keep same flatten operate with keras model
        x = x.reshape(batch_size, -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))

        return x
