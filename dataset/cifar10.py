import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import pickle
import os
from PIL import Image

def row_to_image(row):
    """
    Convert a row from the CIFAR-10 dataset to a 32x32 RGB image.
    
    Parameters:
        row (numpy array): A 1D numpy array of shape (3072, ) containing the pixel values.
    
    Returns:
        numpy array: A 32x32x3 RGB image.
    """
    red = row[:1024].reshape(32, 32)
    green = row[1024:2048].reshape(32, 32)
    blue = row[2048:].reshape(32, 32)
    
    image = np.stack([red, green, blue], axis=-1)
    return image

def display_image(row):
    """Display the CIFAR-10 image from a row."""
    image = row_to_image(row)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def unpickle(file):
    """Unpickle a CIFAR-10 batch file."""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10Dataset(Dataset):
    """Custom PyTorch Dataset for CIFAR-10."""
    def __init__(self, data_dir, train=True, transform=None, predict=False):
        """
        Args:
            data_dir (str): Path to CIFAR-10 batches.
            train (bool): Load training data if True, else load test data.
            transform (callable, optional): Optional transform to apply to images.
        """
        self.transform = transform
        self.predict = predict
        self.data = []
        self.labels = []
        
        if train:
            batches = [f"data_batch_{i}" for i in range(1, 6)]
        elif predict:
            batches = ["cifar_test_nolabel.pkl"] # filename of the dataset for prediction 
        else:
            batches = ["test_batch"]
        
        for batch in batches:
            batch_path = os.path.join(data_dir, batch)
            batch_dict = unpickle(batch_path)
            self.data.append(batch_dict[b'data'])
            if not predict:
                self.labels.extend(batch_dict[b'labels'])
        
        self.data = np.vstack(self.data)  # Shape (50000, 3072) or (10000, 3072)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.predict:
            image = self.data[idx]
        else:
            image = row_to_image(self.data[idx])
            
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        if self.predict:
            return image  # No label in prediction mode
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
