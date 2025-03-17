import argparse
import io
from io import BytesIO
import gzip
import numpy as np
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import fsspec
import time
from dotenv import load_dotenv

load_dotenv()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def load_mnist_data_from_s3(bucket, s3_endpoint, train=True):
    if train:
        images_url = f's3://{bucket}/mnist/train-images-idx3-ubyte.gz'
        labels_url = f's3://{bucket}/mnist/train-labels-idx1-ubyte.gz'
    else:
        images_url = f's3://{bucket}/mnist/t10k-images-idx3-ubyte.gz'
        labels_url = f's3://{bucket}/mnist/t10k-labels-idx1-ubyte.gz'

    s3_options = {'anon': True}
    if s3_endpoint:
        s3_options['client_kwargs'] = {'endpoint_url': s3_endpoint}

    with fsspec.open(images_url, 'rb', **s3_options) as img_f, fsspec.open(labels_url, "rb", **s3_options) as lbl_f:
        with gzip.open(img_f, 'rb') as img_gz, gzip.open(lbl_f, 'rb') as lbl_gz:
            img_data = img_gz.read()
            lbl_data = lbl_gz.read()
            images = np.frombuffer(img_data[16:], dtype=np.uint8).reshape(-1, 28, 28)
            labels = np.frombuffer(lbl_data[8:], dtype=np.uint8)

    return images, labels


class S3MNIST(torch.utils.data.Dataset):
    def __init__(self, s3_bucket, s3_endpoint, train=True, transform=None):
        self.data, self.targets = load_mnist_data_from_s3(s3_bucket, s3_endpoint=s3_endpoint, train=train)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img, mode='L')
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.targets)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--s3-path', type=str, default=None,
                        help='S3 bucket name to stream MNIST data from')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    mean, std = 0.1307, 0.3081

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    if args.s3_path:
        s3_endpoint = os.environ.get('S3_ENDPOINT','https://s3-west.nrp-nautilus.io')
        print(f"Using S3 endpoint: {s3_endpoint}")
        dataset1 = S3MNIST(s3_bucket=args.s3_path, train=True, transform=transform, s3_endpoint=s3_endpoint)
        dataset2 = S3MNIST(s3_bucket=args.s3_path, train=False, transform=transform, s3_endpoint=s3_endpoint)
    else:
        dataset1 = datasets.MNIST('../data', train=True, download=True,
                                  transform=transform)
        dataset2 = datasets.MNIST('../data', train=False, download=True,
                                  transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_filename = f"mnist_cnn_{timestamp}.pt"
        torch.save(model.state_dict(), model_filename)


if __name__ == '__main__':
    main()