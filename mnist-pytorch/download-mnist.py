import torchvision.datasets as datasets

# Download MNIST dataset
mnist_dataset = datasets.MNIST(root="./data", train=True, download=True)
