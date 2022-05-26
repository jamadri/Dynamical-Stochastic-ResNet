import timing
import torch
from torch import nn
from torch.utils.data import DataLoader # Wraps an iterable around the Dataset
from torch.utils.data import Subset
from torchvision import datasets 
from torchvision.transforms import ToTensor



from models.resnet_light_MNIST import PreAct_ResNet_MNIST, PreActBasicBlock

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False, # test_data will correspond to the testing part f CIFAR10,
    download=True,
    transform=ToTensor(),
)
training_data = Subset(training_data,torch.arange(10000))
test_data = Subset(test_data, torch.arange(1000))
BATCH_SIZE = 20

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

for X,y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# Get cpu device for training.
DEVICE = "cpu"
print(f"Using {DEVICE} device")

# Optimizing the Model Parameters
MODEL = PreAct_ResNet_MNIST(PreActBasicBlock,[6,8,12,6]).to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    """Trains the specified model"""
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    """Evaluates the specified model."""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, MODEL, loss_fn, optimizer)
    test(test_dataloader, MODEL, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")