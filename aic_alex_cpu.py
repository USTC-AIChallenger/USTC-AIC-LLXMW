import numpy as np
import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from skimage import io, transform
import sys
sys.path.append('..')
from torchvision import transforms, models
import torch.backends.cudnn as cudnn

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        img = transform.resize(image, (32, 32))
        return {'image': img, 'label': label}

class ToTensor(object):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': label}

class Mydataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.label_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.label_frame)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.label_frame.iloc[idx, 1])
        image = io.imread(img_name)
        label = self.label_frame.iloc[idx, 0]
        sample = {'image': image, 'label': label}
        #print(label)
        #print(img_name)
        if self.transform:
            sample = self.transform(sample)
        return sample

transformed_traindataset = Mydataset(csv_file='./train.csv',
                                           root_dir='/media/hai/Thinkpad/train/',
                                           transform=transforms.Compose([Rescale(32), ToTensor()]))
train_loader = DataLoader(transformed_traindataset, batch_size=64, shuffle=True)

transformed_validdataset = Mydataset(csv_file='./valid.csv',
                                           root_dir='/media/hai/Thinkpad/valid/',
                                           transform=transforms.Compose([Rescale(32), ToTensor()]))
valid_loader = DataLoader(transformed_validdataset, batch_size=64, shuffle=True)


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

net = AlexNet(num_classes=61)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1)
losses = []
acces = []
valid_losses = []
valid_acces = []

for e in range(200):
    train_loss = 0
    train_acc = 0
    net.train()
    for i, sample_batched in enumerate(train_loader):
        im = Variable(sample_batched['image'].float())
        label = Variable(sample_batched['label'].float())
        out = net(im)
        label = label.long()
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))


    valid_loss = 0
    valid_acc = 0
    net.eval()
    for i, sample_batched in enumerate(valid_loader):
        im = Variable(sample_batched['image'].float())
        label = Variable(sample_batched['label'].float())
        out = net(im)
        label = label.long()
        loss = criterion(out, label)
        valid_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        valid_acc += acc

    valid_losses.append(valid_loss / len(valid_loader))
    valid_acces.append(valid_acc / len(valid_loader))

    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, valid Loss: {:.6f}, valid Acc: {:.6f}'
          .format(e, train_loss / len(train_loader), train_acc / len(train_loader),
                  valid_loss / len(valid_loader), valid_acc / len(valid_loader)))
