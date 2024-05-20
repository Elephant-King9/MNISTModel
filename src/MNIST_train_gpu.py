import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from MNISTModel import *

train_dataset = torchvision.datasets.MNIST(root='../datasets', train=True, download=True,
                                           transform=torchvision.transforms.ToTensor())

test_dataset = torchvision.datasets.MNIST(root='../datasets', train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

writer = SummaryWriter('../logs')


train_loader = DataLoader(train_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

model = MNISTModel().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
learn_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

epoch = 10
total_train_step = 0
start_time = time.time()
for i in range(epoch):
    pre_train_step = 0
    pre_train_loss = 0
    model.train()
    for data in train_loader:
        # print(data)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        pre_train_step += 1
        total_train_step += 1
        pre_train_loss += loss.item()
        if pre_train_step % 100 == 0:
            end_time = time.time()
            print(f'Epoch:{i+1},pre_train_loss:{pre_train_loss / pre_train_step},time = {end_time - start_time}')
            writer.add_scalar('train_loss', pre_train_loss/pre_train_step, total_train_step)

    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_accuracy += outputs.argmax(1).eq(labels).sum().item()

    print(f'Epoch:{i+1},Test Accuracy: {total_accuracy / len(test_dataset)}')
    writer.add_scalar('test_accuracy', total_accuracy / len(test_dataset), i)

    torch.save(model, f'../models/MNISTModel_{i}.pth')

writer.close()



