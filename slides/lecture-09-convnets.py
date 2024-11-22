## BME i9400
## Fall 2024
### Convolutional Neural Networks

## Introduction
- In previous lectures, we covered feedforward neural networks (multilayer perceptrons)
- Today we will cover convolutional neural networks (CNNs)
- CNNs are particularly well-suited for processing data with spatial structure
- Examples in biomedical engineering:
    - Medical image analysis
    - ECG/EEG signal processing
    - Protein structure prediction
    - Drug discovery

## The Convolution Operation
- The fundamental operation in CNNs is the convolution
- A convolution is a mathematical operation between two functions
- In the context of neural networks, we convolve an input with a kernel (or filter)

```python
import numpy as np
import matplotlib.pyplot as plt

# 1D Convolution Example
def conv1d(signal, kernel):
    return np.convolve(signal, kernel, mode='valid')

# Create a sample signal and kernel
t = np.linspace(0, 10, 1000)
signal = np.sin(2*np.pi*t) + np.random.normal(0, 0.1, len(t))
kernel = np.ones(50)/50  # Moving average filter

# Apply convolution
filtered_signal = conv1d(signal, kernel)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t[:len(filtered_signal)], filtered_signal, label='Filtered')
plt.plot(t[:len(filtered_signal)], signal[:len(filtered_signal)], alpha=0.5, label='Original')
plt.title('1D Convolution Example')
plt.legend()
plt.show()
```

## 2D Convolution
- In 2D, the kernel slides over both dimensions of the input
- Common in image processing applications

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Create a sample image
image = np.zeros((10, 10))
image[4:7, 4:7] = 1

# Create an edge detection kernel
kernel = np.array([[-1, -1, -1],
                  [-1,  8, -1],
                  [-1, -1, -1]])

# Apply 2D convolution
conv_result = signal.convolve2d(image, kernel, mode='valid')

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.imshow(image)
ax1.set_title('Original Image')
ax2.imshow(conv_result)
ax2.set_title('After Convolution')
plt.show()
```

## Why Use Convolutions in Neural Networks?
1. Weight Sharing
   - Same kernel is applied across the entire input
   - Reduces number of parameters
   - Translation invariance

2. Local Features
   - Each output value depends only on nearby input values
   - Captures spatial relationships
   - Hierarchical feature learning

## 1D Convolutional Layers
- Used for processing sequential data
- Examples in biomedical engineering:
    - ECG signals
    - EEG recordings
    - Blood pressure time series

```python
import torch
import torch.nn as nn

# Example 1D CNN for ECG processing
class ECG_CNN(nn.Module):
    def __init__(self):
        super(ECG_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(32 * 29, 2)  # Example output size
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 29)
        return self.fc(x)
```

## 2D Convolutional Layers
- Used for processing image data
- Examples in biomedical engineering:
    - X-ray images
    - MRI scans
    - Histology slides
    - Microscopy images

```python
class MedicalImageCNN(nn.Module):
    def __init__(self):
        super(MedicalImageCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 6 * 6, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 6 * 6)
        return self.fc(x)
```

## Pooling Layers
- Reduce spatial dimensions
- Types:
    - Max pooling (most common)
    - Average pooling
- Benefits:
    - Reduces computation
    - Provides some translation invariance
    - Helps prevent overfitting

```python
# Demonstrate pooling
import numpy as np
import matplotlib.pyplot as plt

def max_pool2d(input_array, pool_size=2):
    h, w = input_array.shape
    output_h, output_w = h//pool_size, w//pool_size
    output = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            output[i,j] = np.max(input_array[i*pool_size:(i+1)*pool_size, 
                                           j*pool_size:(j+1)*pool_size])
    return output

# Create sample data
data = np.random.rand(6, 6)
pooled_data = max_pool2d(data)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.imshow(data)
ax1.set_title('Original')
ax2.imshow(pooled_data)
ax2.set_title('After Max Pooling')
plt.show()
```

## Striding
- Controls how the kernel moves across the input
- Larger stride = reduced output size
- Can be used instead of or in addition to pooling

## Applications of CNNs in Biomedical Engineering
1. 1D CNNs:
   - ECG classification
   - Sleep stage scoring
   - Seizure detection

2. 2D CNNs:
   - Medical image segmentation
   - Disease classification
   - Cell detection

3. 3D CNNs:
   - Volumetric medical imaging (CT, MRI)
   - Drug-protein interaction prediction
   - Motion analysis in medical videos

## Practical Example: MNIST Digit Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 320)
        return self.fc(x)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Training function
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Test function
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')

# Train the model
for epoch in range(1, 4):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)
```

## Summary
- Convolutional neural networks use weight sharing and local connectivity
- Key components: convolution layers, pooling layers, stride
- Particularly effective for data with spatial structure
- Wide range of applications in biomedical engineering
- Easy to implement using modern deep learning frameworks like PyTorch

## Questions?
