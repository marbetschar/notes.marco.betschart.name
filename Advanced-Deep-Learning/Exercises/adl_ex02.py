#
# Exercise 02 for advanced deep learning course
#

#
# Construct a deep CNN model for Pet Classification
#


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchinfo import summary
from torcheval.metrics import MulticlassAccuracy

import numpy as np


def get_data_set(batch_size):
    #
    # CenterCrop is one possibility, but you can also try to resize the image
    #
    transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.CenterCrop(256)])
    data_train = torchvision.datasets.OxfordIIITPet(root='data/OxfordIIITPet', download=True, transform=transform)
    data_test = torchvision.datasets.OxfordIIITPet(root='data/OxfordIIITPet', split='test', download=True,
                                                   transform=transform)
    len_train = (int)(0.8 * len(data_train))
    len_val = len(data_train) - len_train
    data_train_subset, data_val_subset = torch.utils.data.random_split(
            data_train, [len_train, len_val])

    data_train_loader = torch.utils.data.DataLoader(dataset=data_train_subset, shuffle=True, batch_size=batch_size)
    data_val_loader = torch.utils.data.DataLoader(dataset=data_val_subset, shuffle=True, batch_size=batch_size)
    data_test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size)

    print(len_train, len_val, len(data_train))

    return data_train_loader, data_val_loader, data_test_loader


class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        # to complete

    def forward(self, x):
        # to complete

        return x

#
# This version does not use wandb, but tensorboard or wandb are recommended
#
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    metrics = MulticlassAccuracy(num_classes=37)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        metrics.reset()
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            metrics.update(predicted, labels)
            train_acc = metrics.compute()

            if (step+1) % 10 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], '
                       f'Step [{step+1}/{total_step}], '
                       f'Loss: {loss.item(): .4f}, '
                       f'Accuracy: {train_acc: .2f}')
        model.eval()
        with torch.no_grad():
            metrics.reset()
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                metrics.update(predicted, labels)
            val_acc = metrics.compute()

            print(f'Val Accuracy: {val_acc: .2f}')

    return model

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # test if it worked
        x = torch.ones(1, device=device)
        print('Using CUDA device')

    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        x = torch.ones(1, device=device)
        print('Using MPS device')
    else:
        print('Using CPU')
        device = torch.device('cpu')
    return device

def main():
    batch_size = 64
    train_loader, val_loader, test_loader = get_data_set(batch_size)
    # to complete



if __name__ == '__main__':
    main()
