# CMU 16-726 Learning-Based Image Synthesis / Spring 2023, Assignment 3
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip

import glob
import os

import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from torchvision.transforms.autoaugment import AutoAugmentPolicy


class CustomDataSet(Dataset):
    """Load images under folders"""

    def __init__(self, main_dir, ext='*.png', transform=None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = glob.glob(os.path.join(main_dir, ext))
        self.total_imgs = all_imgs
        print(os.path.join(main_dir, ext))
        print(len(self))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


def get_data_loader(dataset, opts):
    """Create training and test data loaders."""
    basic_transform = transforms.Compose([
        transforms.Resize(opts.image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    deluxe_transform = transforms.Compose([
        transforms.Resize(opts.image_size, Image.BICUBIC),
        # transforms.RandomResizedCrop(opts.image_size, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomResizedCrop(opts.image_size, interpolation=Image.BICUBIC,scale=(0.8,1.0),ratio=(0.7,1.1)),
        # transforms.Resize([int(1.1 * opts.image_size), int(1.1 * opts.image_size)], Image.BICUBIC),
        # transforms.RandomCrop(opts.image_size),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(p=0.1),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        # transforms.RandomInvert(p=0.5),
        # transforms.RandomPerspective(distortion_scale=0.1,p=0.1),
        # transforms.RandomRotation(degrees=(0,100)),
        # transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.RandomErasing(p=0.1),
    ])

    if opts.data_preprocess == 'basic':
        train_transform = basic_transform
    elif opts.data_preprocess == 'deluxe':
        # todo: add your code here: below are some ideas for your reference
        # load_size = int(1.1 * opts.image_size)
        # osize = [load_size, load_size]
        # transforms.Resize(osize, Image.BICUBIC)
        # transforms.RandomCrop(opts.image_size)
        # transforms.RandomHorizontalFlip()
        # pass
        train_transform = deluxe_transform

    # dataset = CustomDataSet(
    #     os.path.join('data/', data_path), opts.ext, train_transform
    # )
    dloader = DataLoader(
        dataset=dataset, batch_size=opts.batch_size,
        shuffle=True, num_workers=opts.num_workers
    )

    return dloader
