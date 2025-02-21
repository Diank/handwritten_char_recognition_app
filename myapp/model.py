import os
import torch
import torch.nn as nn


class EMNIST_3CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(in_features=1152, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_classes)
        )

    def forward(self, x):
        return self.model(x)


class Model:
    def __init__(self):
        self.model = EMNIST_3CNN(n_classes=47)
        model_path = os.path.join('myapp', 'best_model3CNN.ckpt')
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        self.mapping = {}
        with open('emnist-balanced-mapping.txt', 'r') as file:
            for line in file:
                label, code = line.split()
                self.mapping[int(label)] = chr(int(code))

    def predict(self, x):
        self.model.eval()

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = self.model(x)

        label_prediction = output.argmax(dim=1).item()
        char_prediction = self.mapping.get(label_prediction, '?')

        return char_prediction