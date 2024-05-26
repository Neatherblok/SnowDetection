# To be up-to-date on the most current version of this code. Check out our GitHub repository: https://github.com/Neatherblok/SnowDetection

# pip install datasets
# pip install --upgrade --force-reinstall huggingface_hub


import torch
import torchvision
from torchvision.transforms import transforms, Lambda, Resize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

'''
Creates a class object CustomDataLoader
************
__init__
  This function initializes a CustomDataLoader object.
  It loads a part of the snow dataset specified by the user.
  Options for dataset_type are 'train' and 'test'.

  Args:
    data_path (str): Path to the dataset.
    batch_size (int): Batch size for data loading.
    dataset_type (str): Specifies whether the dataset is for training or testing.

************
__getitem__
  This function returns two variables containing each a variable present 
  in the original snow dataset.
  Before returning these variables, it splits up and transforms the image 
  into a fixed resolution of 48 by 48 pixels.
  
  Returns:
    data: The transformed image data.
    label: The label corresponding to the image.
************
__len__
  Returns: the length of the loaded dataset.
'''


class CustomDataLoader:
    def __init__(self, data_path, batch_size, dataset_type, mean=None, std=None):
        self.data_path = data_path
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.mean = mean
        self.std = std

        if mean is None or std is None:
            self.transform = transforms.Compose([
                transforms.Resize(128),  # Resize to size 128x128,
                transforms.ToTensor()
            ])

        else: 
            self.transform = transforms.Compose([
                transforms.Resize(128),  # Resize to size 128x128,
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])

        self.dataset = torchvision.datasets.ImageFolder(root=f"{self.data_path}/{dataset_type}",
                                                        transform=self.transform)

        if self.dataset_type == 'train':
            shuffle = True
        else:
            shuffle = False

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle,
                                      num_workers=os.cpu_count(),
                                      pin_memory=True if torch.cuda.is_available() else False)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
