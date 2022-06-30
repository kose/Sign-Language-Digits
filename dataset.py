# -*- mode: python -*-

import os

import torch
from torchvision import transforms

import numpy as np
import pandas as pd
from PIL import Image
import cv2 as cv

from tqdm import tqdm

from enum import Flag, auto
class Flip(Flag):
    none = auto()
    flip = auto()
    both = auto()

dbdir="db"
image_size = 64
image_size_db = image_size + 6

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, csvfile="train_master.tsv", flip=Flip.none, transform=False, repeat=1):

        csvfile = os.path.join(dbdir, csvfile)
        self.repeat = repeat

        if transform:
            self.transform = transform_expand = transforms.Compose([
                transforms.RandomChoice([
                    transforms.RandomAffine(5),
                    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                ]),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.3),
                # transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 10.0)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.GaussianBlur(kernel_size=3, sigma=5.0),
                transforms.ToTensor(),
            ])

        dataset_image = np.empty([0, image_size_db, image_size_db, 3], np.uint8)
        dataset_info = np.array([], dtype=np.int32)
        
        df = pd.read_csv(csvfile, header=None)

        with tqdm(total=len(df)) as pbar:
            for filename, sign in df.values:
                # print(id, dictionary[expression])
                path = os.path.join(dbdir, filename)
                image_in = cv.imread(path, 1)
                image_in = cv.resize(image_in, (image_size_db, image_size_db)) # 100x100 -> image_size_db

                # cv.imshow("image", image_in)
                # cv.waitKey(100)

                if flip == Flip.both or flip == Flip.none:
                    dataset_image = np.append(dataset_image, [image_in], axis=0)
                    dataset_info = np.append(dataset_info, [sign])

                if flip == Flip.both or flip == Flip.flip:
                    image_in = cv.flip(image_in, 1)
                    dataset_image = np.append(dataset_image, [image_in], axis=0)
                    dataset_info = np.append(dataset_info, [sign])

                pbar.update(1)


        self.dataset_image = dataset_image
        self.dataset_info = dataset_info
        self.df = df


    def __len__(self):
        return len(self.dataset_image) * self.repeat


    def __getitem__(self, idx):

        i = int(idx / self.repeat)

        image = self.transform(Image.fromarray(self.dataset_image[i]))
        info = self.dataset_info[i]

        return image, info


    def df(self):
        return self.df

##
## main function
##
if __name__ == '__main__':

    sx = 20
    sy = 20
    
    dataset = MyDataset(csvfile="test.csv", flip=Flip.none, transform=False, repeat=1)
    # dataset = MyDataset(csvfile="train.csv", flip=Flip.both, transform=True, repeat=1)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=sx*sy, shuffle=False)
    dataset, info = dataloader.__iter__().next()

    from visualizer import visualize

    visualize(sy, sx, dataset)


# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###

