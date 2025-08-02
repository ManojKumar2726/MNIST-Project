import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image


# Hyper Parameters
input_size = 784  # 28x28 images
hidden_size = 500
num_classes = 10  # Digits 0-9

# Reinitialize Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNetwork(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load("model.pth"))
model.eval()

#  Load a sample from MNIST test set
'''transform = transforms.ToTensor()
mnist_test = datasets.MNIST(root='.', train=False, download=True, transform=transform)
sample_idx = 10  # Change this index to see different samples
img_tensor, label = mnist_test[sample_idx]'''

# Load our own data 
img = Image.open("Sample\sample1.png").convert("L").resize((28, 28))
transform = transforms.ToTensor()
img_tensor = transform(img)
img_tensor = img_tensor.view(1, -1)

# Visualize the image
plt.imshow(img_tensor.view(28, 28).numpy().squeeze(), cmap="gray")
plt.title(f"MNIST Sample")
plt.axis('off')
plt.show()

# Prediction
with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)
    predicted = torch.argmax(probs, dim=1)
    print(f"\nPredicted Digit: {predicted.item()}")
    print("Confidence per digit:")
    for i, prob in enumerate(probs[0]):
        print(f" {i} -> {prob.item()*100:.2f}%")
