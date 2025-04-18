import cv2
from sympy import totient
from torchvision import  transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import logging
logging.getLogger('PIL').setLevel(logging.INFO)

class SupConDataset(Dataset):

    def __init__(self, root_dir,
                 mode,
                 img_dir="images",
                 transform=None
                 ):

        self.root_dir = root_dir
        self.mode = mode
        # self.imgs_dir = os.path.join(root_dir, mode).replace('\\', '/')
        self.imgs_dir = os.path.join(root_dir, mode).replace('\\', '/')
        # self.list_of_images = os.path.join(root_dir, mode, f'{mode}.txt').replace('\\', '/')
        self.list_of_images = os.path.join(root_dir, mode, f'{mode}.txt').replace('\\', '/')

        self.get_list_of_dataset()
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        orig_img_name, lbl = self.dataset[index].strip().split(':')

        img = Image.open(os.path.join(self.imgs_dir, orig_img_name), mode='r').convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        lbl = int(lbl)

        return {
            'image' : img,
            'name' : str(os.path.join(self.imgs_dir, orig_img_name)),
            'label' : lbl
        }




    def get_list_of_dataset(self):
        lines_list = []

        with open(self.list_of_images, 'r') as file:
            for line in file:
                lines_list.append(line.strip())

        self.dataset = lines_list



