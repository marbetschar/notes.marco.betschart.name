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

import wandb

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

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            #
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            #
            nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=640, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(640),
            nn.ReLU(),
            nn.Conv2d(in_channels=640, out_channels=768, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            #
            nn.Flatten(),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 37),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

#
# This version does not use wandb, but tensorboard or wandb are recommended
#
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    wandb.init(project="advdelearn-OxfordIIITPet", config={'epochs': num_epochs, 'batch_size': train_loader.batch_size})
    metrics = MulticlassAccuracy(num_classes=37)

    total_step = len(train_loader)
    wandb_step = 1

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

            train_metrics = {'train/train_loss:': loss,
                             'train/train_acc': train_acc,
                             'train/epoch': epoch}

            wandb_step += 1
            wandb.log(train_metrics, wandb_step)

            if (step+1) % 10 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], '
                       f'Step [{step+1}/{total_step}], '
                       f'Loss: {loss.item(): .4f}, '
                       f'Accuracy: {train_acc: .2f}')
        model.eval()
        with torch.no_grad():
            metrics.reset()
            for step, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                metrics.update(predicted, labels)

                val_acc = metrics.compute()
                val_metrics = {'val/val_loss': loss,
                               'val/val_acc': val_acc}

                wandb_step += 1
                wandb.log(val_metrics, wandb_step)

            print(f'Val Accuracy: {val_acc: .2f}')

    return model

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # test if it worked
        _ = torch.ones(1, device=device)
        print('Using CUDA device')

    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        _ = torch.ones(1, device=device)
        print('Using MPS device')
    else:
        print('Using CPU')
        device = torch.device('cpu')
    return device

def get_input_size(data_loader: torch.utils.data.DataLoader) -> tuple:
    return tuple(next(iter(data_loader))[0].shape)

def main():
    batch_size = 64
    device = get_device()

    train_loader, val_loader, test_loader = get_data_set(batch_size)

    input_size = get_input_size(train_loader)
    print(f"Input size: {input_size}")

    model = DeepCNN()
    summary(model, input_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    num_epochs = 1000

    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)


if __name__ == '__main__':
    wandb.login()
    main()
