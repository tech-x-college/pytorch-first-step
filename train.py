from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()   # put the model into training mode
    for batch_idx, (data, target) in enumerate(train_loader):   # iterate through batches
        data, target = data.to(device), target.to(device)   # shift data to devices
        optimizer.zero_grad()   # set gradients to zero
        output = model(data)    # forward pass
        loss = F.nll_loss(output, target)   # compute the loss
        loss.backward()     # backwar propagation
        optimizer.step()    # next steps
        if batch_idx % 10 == 0:   # let's print it from time to time
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()   # put the model into evaluation mode
    correct = 0
    with torch.no_grad():   # gradients are not needed
        for data, target in test_loader:  # iterate through test data
            data, target = data.to(device), target.to(device)   # shift data to devices
            output = model(data)    # forward pass
            pred = output.max(1, keepdim=True)[1] # get the index of the best prediction
            correct += pred.eq(target.view_as(pred)).sum().item()   # see if target and prediction are the same

    print('\nTest set: Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))

def main():
    device = torch.device("cpu")    # use the CPU for this
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1000, shuffle=True)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)    # set up optimizer
    epochs = 10     # compute epochs
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


if __name__ == '__main__':
    main()