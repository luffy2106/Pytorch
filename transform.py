import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda



"""
Data does not always come in its final processed form that is required for training machine learning algorithms. 
We use transforms to perform some manipulation of the data and make it suitable for training.

All TorchVision datasets have two parameters -transform to modify the features and target_transform to modify the labels - 
that accept callables containing the transformation logic. The torchvision.transforms module offers several commonly-used transforms out of the box.

The FashionMNIST features are in PIL Image format, and the labels are integers. 
For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors. 
To make these transformations, we use ToTensor and Lambda.

reference:
https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
"""


ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)