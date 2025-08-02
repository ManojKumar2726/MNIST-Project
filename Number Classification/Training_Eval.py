import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import sys
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist")

#Gpu suppprt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper Parameters

input_size = 784  # 28x28 images
hidden_size = 500
num_classes = 10  # Digits 0-9
num_epochs = 2
batch_size = 100
l_rate = 0.001

# Load MNIST dataset

train_dataset = torchvision.datasets.MNIST(root='./data', train = True, transform = transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset , batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#example

examples = iter(train_loader)
samples, labels = next(examples)

for i in range (1):
    plt.subplot (2,3,i+1)
    plt.imshow(samples[i][0], cmap='gray')
#plt.show()

img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_img',img_grid)

# Define the neural network model

class NeuralNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNetwork,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

writer.add_graph(model, samples.reshape(-1, 28*28).to(device))
writer.close()
#sys.exit()

running_loss = 0.0
running_correct = 0.0
running_samples = 0.0

n_total_steps = len(train_loader)

# Train the model

for epoch in range (num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images = images.reshape(-1 , 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _,pred = torch.max(outputs.data,1)

        running_loss += loss.item()
        running_correct += (pred == labels).sum().item()
        running_samples += labels.size(0)

        if (i+1) % 100 == 0:
            avg_loss = running_loss / 100
            avg_acc = running_correct / running_samples 

            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')
    
            writer.add_scalar('Training Loss', avg_loss, epoch * n_total_steps + i)
            writer.add_scalar('Accuracy', avg_acc, epoch * n_total_steps + i)
    
            running_loss = 0.0
            running_correct = 0.0
            running_samples = 0



# Save the model

FILE = "model.pth"
#torch.save(model.state_dict(),FILE)

# Test the model

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _,pred = torch.max(outputs, 1)

        n_samples += labels.size(0)
        n_correct += (pred == labels).sum().item()

acc = n_correct/n_samples * 100
print(f'Accuracy : {acc:.2f}')



