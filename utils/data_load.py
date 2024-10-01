
import sys, os, os.path as osp
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import Dataset
import torchvision.transforms as tr
import monai.transforms as t
import pandas as pd
from PIL import Image

def intensities(x):
    brightness, contrast, saturation, hue = 0.15, 0.15, 0.15, 0.05
    jitter = tr.ColorJitter(brightness, contrast, saturation, hue)
    sharpness = tr.RandomAdjustSharpness(sharpness_factor=0.25, p=0.5)
    return jitter(sharpness(x))

def normalize(x):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # imagenet stats
    normalize = tr.Normalize(mean, std)
    return normalize(x)

def print_shape(x):
    print(x.shape)
    return x
def get_seg_transforms(im_size):
    p_app, pr_geom = 0.25, 0.25
    tr_transforms = t.Compose([
        t.CropForegroundd(keys=('img', 'mask'), source_key='img'),  # 0 is the default threshold
        t.ScaleIntensityd(keys=('img',)),
        t.Resized(spatial_size=im_size, keys=('img', 'mask'), mode=('bilinear', 'nearest')),
        t.RandFlipd(keys=('img', 'mask'), prob=pr_geom, spatial_axis=0),
        t.RandFlipd(keys=('img', 'mask'), prob=pr_geom, spatial_axis=1),
        t.RandRotate90d(keys=('img', 'mask'), prob=pr_geom, max_k=2),
        t.Lambda(lambda d: {'img': intensities(d['img']), 'mask': d['mask']}),
        t.Lambda(lambda d: {'img': normalize(d['img']), 'mask': d['mask']}),
    ])

    vl_transforms = t.Compose([
        t.CropForegroundd(keys=('img', 'mask'), source_key='img'),  # 0 is the default threshold
        t.ScaleIntensityd(keys=('img',)),
        t.Resized(spatial_size=im_size, keys=('img', 'mask'), mode=('bilinear', 'nearest')),
        t.Lambda(lambda d: {'img': normalize(d['img']), 'mask': d['mask']}),
    ])

    return tr_transforms, vl_transforms

def get_test_seg_transforms(im_size):
    # only images here
    test_transforms = t.Compose([
        t.ScaleIntensityd(keys=('img',)),
        t.Resized(spatial_size=im_size, keys=('img'), mode=('bilinear')),
        t.Lambda(lambda d: {'img': normalize(d['img'])}),
    ])

    return test_transforms

class SegDataset(Dataset):
    def __init__(self, csv_path, transforms):
        # assumes in the csv first column is file name, second column is target
        self.csv_path = csv_path
        df = pd.read_csv(self.csv_path)
        self.im_list = df['im_path'].values
        self.mask_list = df['mask_path'].values
        self.transforms = transforms

    def __getitem__(self, index):
        # load image and targets
        img = Image.open(self.im_list[index])
        mask = Image.open(self.mask_list[index])
        d = {'img': tr.functional.pil_to_tensor(img),
             'mask': tr.functional.pil_to_tensor(mask)[0].unsqueeze(0).bool()}
        d = self.transforms(d)
        return d['img'], d['mask']

    def __len__(self):
        return len(self.im_list)

class TestSegDataset(Dataset):
    def __init__(self, csv_path, transforms):
        # assumes in the csv first column is file name, second column is target
        self.csv_path = csv_path
        df = pd.read_csv(self.csv_path)
        self.im_list = df['im_path'].values
        self.transforms = transforms

    def __getitem__(self, index):
        # load image and targets
        im_name = self.im_list[index]
        img = Image.open(im_name)
        d = {'img': tr.functional.pil_to_tensor(img)}
        d = self.transforms(d)
        return d['img'], im_name

    def __len__(self):
        return len(self.im_list)


def get_test_seg_loader(csv_path_test, batch_size, im_size, num_workers=0):
    test_seg_transforms = get_test_seg_transforms(im_size)

    test_ds = TestSegDataset(csv_path_test, transforms=test_seg_transforms)

    test_loader = DataLoader(dataset=test_ds, batch_size=2*batch_size, num_workers=num_workers)

    return test_loader
def get_train_val_seg_loaders(csv_path_tr, batch_size, im_size, num_workers=0, tr_percentage=1., vl_percentage=1.):
    tr_transforms, vl_transforms = get_seg_transforms(im_size)

    tr_ds = SegDataset(csv_path_tr, transforms=tr_transforms)
    csv_path_vl = csv_path_tr.replace('tr_', 'vl_')
    vl_ds = SegDataset(csv_path_vl, transforms=vl_transforms)
    ovft_ds = SegDataset(csv_path_tr, transforms=vl_transforms)

    if tr_percentage < 1.:
        print(40*'-')
        n_tr_examples = len(tr_ds)
        subset_size = int(tr_percentage*n_tr_examples)
        subset_idxs = torch.randperm(len(tr_ds))[:subset_size]
        tr_ds = Subset(tr_ds, subset_idxs)
        print('Reducing training data from {} items to {}'.format(n_tr_examples, len(tr_ds)))
        print(40 * '-')
    if vl_percentage < 1.:
        print(40 * '-')
        n_vl_examples = len(vl_ds)
        subset_size = int(vl_percentage * n_vl_examples)
        subset_idxs = torch.randperm(len(vl_ds))[:subset_size]
        vl_ds = Subset(vl_ds, subset_idxs)
        print('Reducing validation data from {} items to {}'.format(n_vl_examples, len(vl_ds)))
        print(40 * '-')

    subset_size = len(vl_ds)
    subset_idxs = torch.randperm(len(ovft_ds))[:subset_size]
    ovft_ds = Subset(ovft_ds, subset_idxs)

    tr_loader = DataLoader(dataset=tr_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    vl_loader = DataLoader(dataset=vl_ds, batch_size= 2 *batch_size, num_workers=num_workers)
    ovft_loader= DataLoader(dataset=ovft_ds, batch_size= 2 *batch_size, num_workers=num_workers)

    # x = next(iter(tr_loader))
    # sys.exit('there')

    return tr_loader, ovft_loader, vl_loader

class ClassTestDataset(Dataset):
    def __init__(self, csv_path, transforms):
        self.csv_path = csv_path
        df = pd.read_csv(self.csv_path)
        self.im_list = df.image.values
        self.transforms = transforms

    def __getitem__(self, index):
        # load image and targets
        img = Image.open(self.im_list[index])
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.im_list)
