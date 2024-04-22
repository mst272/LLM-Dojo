import time
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

# --------data process------------------------------------------------

train_dataset = datasets.MNIST(root='data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='data',
                              train=False,
                              transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

# ----------------Hyperparameters-------------------------------------
random_seed = 123
learning_rate = 0.005
num_epochs = 2

# ----------------Architecture-----------------------------------------
num_features = 784
num_hidden_1 = 128
num_hidden_2 = 256
num_classes = 10

torch.manual_seed(random_seed)


# ---------------Model-----------------------------------------------
class TestMLP(nn.Module):
    def __init__(self, num_features, num_hidden1, num_hidden2, num_class):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, num_hidden1),
            nn.ReLU(),
            nn.Linear(num_hidden1, num_hidden2),
            nn.ReLU(),

            nn.Linear(num_hidden2, num_class)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


model = TestMLP(
    num_features=num_features, num_hidden1=num_hidden_1, num_hidden2=num_hidden_2, num_class=num_classes
)

model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# ---------------------Eval---------------------------------------------

def computer_metrics(model, data_loader, device):
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.view(-1, 28*28).to(device)
            labels = labels.to(device)
            logits = model(images)











