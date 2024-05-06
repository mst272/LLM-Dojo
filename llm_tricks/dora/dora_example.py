import time
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy

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
num_epochs = 1

# ----------------Architecture-----------------------------------------
num_features = 784
num_hidden_1 = 32
num_hidden_2 = 64
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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# ---------------------Eval---------------------------------------------

def computer_metrics(model, data_loader, device):
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            # Image batch dimensions: torch.Size([64, 1, 28, 28])
            # Image label dimensions: torch.Size([64])

            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)
            logits = model(images)
            _, predicted_labels = torch.max(logits, 1)
            num_examples = labels.size(0)
            correct_pred += (predicted_labels == labels).sum()
            return correct_pred.float() / num_examples * 100


# ---------------------Train---------------------------------------------


def train(epochs, model, optimizer, train_loader, device):
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # forward and back
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            if not batch_idx % 400:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.4f'
                      % (epoch + 1, epochs, batch_idx,
                         len(train_loader), loss))
        with torch.set_grad_enabled(False):
            print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
                epoch + 1, epochs,
                computer_metrics(model, train_loader, device)))
        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))


# ---------------------Lora Model---------------------------------------------
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.rand(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


# ---------------------DoRA Model---------------------------------------------
class LinearWithDoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
        self.m = nn.Parameter(torch.ones(1, linear.out_features))

    def forward(self, x):
        linear_out = self.linear(x)
        lora_out = self.lora(x)
        lora_out_norm = lora_out / (lora_out.norm(p=2, dim=1, keepdim=True) + 1e-9)
        dora_modification = self.m * lora_out_norm
        return linear_out + dora_modification


# 冻结模型的线性层，即可达到lora的只训练额外的lora层
def freeze_linear_layers(model):
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively freeze linear layers in children modules
            freeze_linear_layers(child)


# 将模型中的linear层替换为 LinearWithLoRA
def convert_lora_layers(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank=4, alpha=8))
        else:
            convert_lora_layers(module)


# 将模型中的linear层替换为 LinearWithDoRA
def convert_dora_layers(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithDoRA(module, rank=4, alpha=8))
        else:
            convert_lora_layers(module)


if __name__ == '__main__':
    train(num_epochs, model, optimizer, train_loader, DEVICE)
    print(f'Test accuracy: {computer_metrics(model, test_loader, DEVICE):.2f}%')

    # 复制两份模型，以供lora 和 dora分别实验
    model_lora = copy.deepcopy(model)
    model_dora = copy.deepcopy(model)

    # lora_qlora 训练
    convert_lora_layers(model_lora)
    freeze_linear_layers(model_lora)
    model_lora.to(DEVICE)
    optimizer_lora = torch.optim.Adam(model_lora.parameters(), lr=learning_rate)
    train(2, model_lora, optimizer_lora, train_loader, DEVICE)
    print(f'Test accuracy LoRA finetune: {computer_metrics(model_lora, test_loader, DEVICE):.2f}%')

    # dora 训练
    convert_dora_layers(model_dora)
    freeze_linear_layers(model_dora)
    model_dora.to(DEVICE)
    optimizer_dora = torch.optim.Adam(model_dora.parameters(), lr=learning_rate)
    train(2, model_dora, optimizer_dora, train_loader, DEVICE)
    print(f'Test accuracy DoRA finetune: {computer_metrics(model_dora, test_loader, DEVICE):.2f}%')
