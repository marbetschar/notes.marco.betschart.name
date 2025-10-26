#
# Exercise for advanced deep learning course
#

#
# Construct a deep CNN model for Pet Classification
#
# (no jupyter notebook)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchinfo import summary
from torcheval.metrics import MulticlassAccuracy

import numpy as np

import wandb

def get_data_set(batch_size):
    transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.CenterCrop(256)])
    transform_augment = torchvision.transforms.Compose(
            [torchvision.transforms.AutoAugment(),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.CenterCrop(256)])
    data_train = torchvision.datasets.OxfordIIITPet(root='data/OxfordIIITPet', download=True, transform=transform_augment)
    data_val = torchvision.datasets.OxfordIIITPet(root='data/OxfordIIITPet', download=True, transform=transform)
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
        # Initial 3x3
        self.conv1 = nn.Conv2d(3, 16, (3,3), (1, 1), 1)

        # block at same resolution (256x256)
        self.bn1_1 = self.bottleneck_block(16, 64, 1, 1)
        # 1x1 conv to increase channels for the skip connection
        self.up1 = nn.Conv2d(16, 64, (1, 1), (1, 1), 0)

        # decrease resolution and add one more block (original res uses more)
        # (128x128)
        self.down2 = nn.Conv2d(64, 64, (1, 1), (2, 2), 0)
        self.bn2_1 = self.bottleneck_block(64, 64, 2, 1)
        self.bn2_2 = self.bottleneck_block(64, 64, 1, 1)

        # decrease resolution and add one more block (original res uses more)
        # (64x64)
        self.down3 = nn.Conv2d(64, 64, (1, 1), (2, 2), 0)
        self.bn3_1 = self.bottleneck_block(64, 64, 2, 1)
        self.bn3_2 = self.bottleneck_block(64, 64, 1, 1)

        # decrease resolution and add one more block (original res uses more)
        # (32x32
        self.down4 = nn.Conv2d(64, 128, (1, 1), (2, 2), 0)
        self.bn4_1 = self.bottleneck_block(64, 128, 2, 1)
        self.bn4_2 = self.bottleneck_block(128, 128, 1, 1)

        # decrease resolution and add one more block (original res uses more)
        # (16x16)
        self.down5 = nn.Conv2d(128, 128, (1, 1), (2, 2), 0)
        self.bn5_1 = self.bottleneck_block(128, 128, 2, 1)
        self.bn5_2 = self.bottleneck_block(128, 128, 1, 1)

        # decrease resolution and add one more block (original res uses more)
        # (8x8)
        self.down6 = nn.Conv2d(128, 128, (1, 1), (2, 2), 0)
        self.bn6_1 = self.bottleneck_block(128, 128, 2, 1)
        self.bn6_2 = self.bottleneck_block(128, 128, 1, 1)

        self.b7 = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.AvgPool2d((8, 8)),
                nn.Flatten())

        self.fc = nn.Linear(128, 37)

    def pre_activation_res_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        )

    def bottleneck_block(self, in_channels, out_channels, stride, padding):
        bottleneck_channels = out_channels // 4
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),

            # conv 1x1
            nn.Conv2d(in_channels, bottleneck_channels, (1, 1), stride, 0),

            # conv 3x3
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, (3, 3), (1, 1), padding),

            # conv 1x1
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(True),
            nn.Conv2d(bottleneck_channels, out_channels, (1, 1), (1, 1), 0))

    def res_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Block 1 with 1 res module
        x = self.conv1(x)
        x1 = self.bn1_1(x)
        x2 = self.up1(x)
        x = x1 + x2

        # Block 2 with 2 res modules and half size
        x1 = self.down2(x)
        x2 = self.bn2_1(x)
        x = x1 + x2

        x = x + self.bn2_2(x)


        # Block 3 with 2 res modules and half size
        x1 = self.down3(x)
        x2 = self.bn3_1(x)
        x = x1 + x2
        x = x + self.bn3_2(x)

        # Block 4 with 2 res modules and half size
        x1 = self.down4(x)
        x2 = self.bn4_1(x)
        x = x1 + x2
        x = x + self.bn4_2(x)

        # Block 5 with 2 res modules and half size
        x1 = self.down5(x)
        x2 = self.bn5_1(x)
        x = x1 + x2
        x = x + self.bn5_2(x)

        # Block 6 with 2 res modules and half size
        x1 = self.down6(x)
        x2 = self.bn6_1(x)
        x = x1 + x2
        x = x + self.bn6_2(x)

        x = self.b7(x)
        x = self.fc(x)

        return x



def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model_stats = summary(model, (train_loader.batch_size, 3, 256, 256), verbose=0)
    summary_str = str(model_stats)
    run = wandb.init(project="petimage-cnn", config={'epochs'    : num_epochs,
                                                     'batch_size': train_loader.batch_size,
                                                     'weight_decay': optimizer.param_groups[0]['weight_decay'],
                                                     'model': summary_str}
                     )
    metrics = MulticlassAccuracy(num_classes=37)
    total_step = len(train_loader)
    model.to(device)
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

            train_metrics = \
                {'train/train_loss:': loss.item(),
                 'train/train_acc'        : train_acc,
                 'train/epoch'            : epoch+1}

            if (step+1) % 10 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], '
                       f'Step [{step+1}/{total_step}], '
                       f'Loss: {loss.item(): .4f}, '
                       f'Accuracy: {train_acc: .2f}')
            wandb.log(train_metrics)


        model.eval()
        with torch.no_grad():
            metrics.reset()
            val_loss = []
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                metrics.update(predicted, labels)

                loss = criterion(outputs, labels)
                val_loss.append(loss.item())

            val_loss_mean = np.mean(val_loss)
            val_acc = metrics.compute()
            val_metrics = {'val/val_loss' : val_loss_mean,
                           'val/val_acc'       : val_acc}
            wandb.log(val_metrics)

            print(f'Val Accuracy : {val_acc: .2f}  Val Loss : {val_loss_mean: .4f}')

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
    # Hyperparameters
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001
    weight_decay = 0.01

    # Model, Loss, Optimizer
    model = DeepCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Data loaders
    train_loader, val_loader, test_loader = get_data_set(batch_size)

    summary(model, input_size=(64, 3, 256, 256))

    device = get_device()
    model.to(device)
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    torch.save({
        'epoch'               : num_epochs,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'petimage_cnn.pth')

    wandb.save('petimage_cnn.pth')
    wandb.finish()

if __name__ == '__main__':
    main()
