# Experiment 5: Convolutional Neural Network (CNN) for Image Recognition (PyTorch)
# Student Name: Rahul Bharathwaj
# Student Roll Number: CH.SC.U4AIE24058

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

print("=" * 60)
print("Student Name: Rahul Bharathwaj")
print("Student Roll Number: CH.SC.U4AIE24058")
print("Experiment 5: PyTorch CNN on SVHN Dataset")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Roll No: CH.SC.U4AIE24058] Using compute device: {device}")

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load SVHN Dataset
train_dataset = datasets.SVHN(root="./data", split="train", download=True, transform=transform)
test_dataset = datasets.SVHN(root="./data", split="test", download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# CNN Architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        # Input size: 32x32 -> Pool1: 16x16 -> Pool2: 8x8 -> 64 * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ConvNet().to(device)

# Display tensor shapes after each layer on a sample batch
sample_batch, _ = next(iter(train_loader))
sample_batch = sample_batch.to(device)
print(f"\n--- Layer-by-Layer Tensor Shapes [Roll No: CH.SC.U4AIE24058] ---")
print(f"Input shape:             {sample_batch.shape}")
x1 = model.conv1(sample_batch)
print(f"After Conv1 (32 filters): {x1.shape}")
x2 = model.pool1(model.relu(x1))
print(f"After MaxPool1:          {x2.shape}")
x3 = model.conv2(x2)
print(f"After Conv2 (64 filters): {x3.shape}")
x4 = model.pool2(model.relu(x3))
print(f"After MaxPool2:          {x4.shape}")
x5 = x4.view(x4.size(0), -1)
print(f"After Flatten:           {x5.shape}")
x6 = model.relu(model.fc1(x5))
print(f"After FC1 (128 neurons):  {x6.shape}")
out = model.fc2(x6)
print(f"After FC2 (Output):       {out.shape}")
print("-" * 55)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training for 3 epochs
epochs = 3
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Roll No: CH.SC.U4AIE24058] Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = 100 * correct / total
print("\n" + "=" * 60)
print(f"Final Test Accuracy [Roll No: CH.SC.U4AIE24058]: {acc:.2f}%")
print("=" * 60)
