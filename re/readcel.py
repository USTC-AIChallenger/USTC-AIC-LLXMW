# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Data Loading and Processing Tutorial
====================================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

A lot of effort in solving any machine learning problem goes in to
preparing the data. PyTorch provides many tools to make data loading
easy and hopefully, to make your code more readable. In this tutorial,
we will see how to load and preprocess/augment data from a non trivial
dataset.

To run this tutorial, please make sure the following packages are
installed:

-  ``scikit-image``: For image io and transforms
-  ``pandas``: For easier csv parsing

"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.autograd import Variable
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#from vggmodel import vgg16
import torch.nn as nn
from PIL import Image
# Ignore warnings

import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

######################################################################
# Dataset class
# -------------
#
# ``torch.utils.data.Dataset`` is an abstract class representing a
# dataset.
# Your custom dataset should inherit ``Dataset`` and override the following
# methods:
#
# -  ``__len__`` so that ``len(dataset)`` returns the size of the dataset.
# -  ``__getitem__`` to support the indexing such that ``dataset[i]`` can
#    be used to get :math:`i`\ th sample
#
# Let's create a dataset class for our face landmarks dataset. We will
# read the csv in ``__init__`` but leave the reading of images to
# ``__getitem__``. This is memory efficient because all the images are not
# stored in the memory at once but read as required.
#
# Sample of our dataset will be a dict
# ``{'image': image, 'landmarks': landmarks}``. Our datset will take an
# optional argument ``transform`` so that any required processing can be
# applied on the sample. We will see the usefulness of ``transform`` in the
# next section.
#

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file,header = None)
        ## read from the firsrt line!!!
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        #print(img_name)
        #image = io.imread(img_name)
        image = Image.open(img_name).convert('RGB')
        #print((image.shape))
        #image = np.array([image,image,image])
        image = np.array(image)
        #image = image.transpose((1, 2, 0))
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks, 'img_name': img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


######################################################################
# Transforms
# ----------
#
# One issue we can see from the above is that the samples are not of the
# same size. Most neural networks expect the images of a fixed size.
# Therefore, we will need to write some prepocessing code.
# Let's create three transforms:
#
# -  ``Rescale``: to scale the image
# -  ``RandomCrop``: to crop from image randomly. This is data
#    augmentation.
# -  ``ToTensor``: to convert the numpy images to torch images (we need to
#    swap axes).
#
# We will write them as callable classes instead of simple functions so
# that parameters of the transform need not be passed everytime it's
# called. For this, we just need to implement ``__call__`` method and
# if required, ``__init__`` method. We can then use a transform like this:
#
# ::
#
#     tsfm = Transform(params)
#     transformed_sample = tsfm(sample)
#
# Observe below how these transforms had to be applied both on the image and
# landmarks.
#

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}
'''
class Random_erase:
    def __init__(self, p, area_ratio_range, min_aspect_ratio, max_attempt):
        self.p = p
        self.sl, self.sh = area_ratio_range
        self.rl, self.rh = min_aspect_ratio, 1. / min_aspect_ratio
        self.max_attempt = max_attempt
        
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        #image = np.asarray(image).copy()
        
        #if np.random.random() > self.p:
        #    return {'image': image, 'landmarks': landmarks}
        
        h, w = image.shape[:2]
        image_area = h * w

        for _ in range(self.max_attempt):
            if np.random.random() > self.p:
                break
            mask_area = np.random.uniform(self.sl, self.sh) * image_area
            aspect_ratio = np.random.uniform(self.rl, self.rh)
            mask_h = int(np.sqrt(mask_area * aspect_ratio))
            mask_w = int(np.sqrt(mask_area / aspect_ratio))

            if mask_w < w and mask_h < h:
                x0 = np.random.randint(0, w - mask_w)
                y0 = np.random.randint(0, h - mask_h)
                x1 = x0 + mask_w
                y1 = y0 + mask_h
                image[y0:y1, x0:x1] = np.random.uniform(0, 1)#np.random.randint(0,100)#np.random.uniform(0, 1)
                #break
        
        return {'image': image, 'landmarks': landmarks}
'''


class Random_erase:
    def __init__(self, p, area_ratio_range, min_aspect_ratio, max_attempt):
        self.p = p
        self.sl, self.sh = area_ratio_range
        self.rl, self.rh = min_aspect_ratio, 1. / min_aspect_ratio
        self.max_attempt = max_attempt
        
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        #image = np.asarray(image).copy()
        
        #if np.random.random() > self.p:
        #    return {'image': image, 'landmarks': landmarks}
        
        h, w = image.shape[:2]
        image_area = h * w

        for _ in range(self.max_attempt):
            if np.random.random() > self.p:
                mask_area = np.random.uniform(self.sl, self.sh) * image_area
                aspect_ratio = np.random.uniform(self.rl, self.rh)
                mask_h = int(np.sqrt(mask_area * aspect_ratio))
                mask_w = int(np.sqrt(mask_area / aspect_ratio))

                if mask_w < w and mask_h < h:
                    x0 = np.random.randint(0, w - mask_w)
                    y0 = np.random.randint(0, h - mask_h)
                    x1 = x0 + mask_w
                    y1 = y0 + mask_h
                    image[y0:y1, x0:x1] = np.random.uniform(0, 1)#np.random.randint(0,100)#np.random.uniform(0, 1)
                #break
        
        return {'image': image, 'landmarks': landmarks}


class CenterCrop(object):
    """Crop center of the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = int((h - new_h)/2)
        left =int((w - new_w)/2)

        #image = image[top: top + new_h,
        #              left: left + new_w]

        image = image[top: top + new_h,
                      left: left + new_w]
        #image = image[::-1,::-1,:]
        
        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}
    
class Flipver():
    """Crop center of the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

#    def __init__(self, probability):
        


    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
#        new_h, new_w = self.output_size

#        top = int((h - new_h)/2)
#        left =int((w - new_w)/2)


        image = image.transpose(Image.FLIP_LEFT_RIGHT)

        
        landmarks = landmarks

        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}



transformed_dataset = FaceLandmarksDataset(csv_file='./file.csv',
                                           root_dir='./data/TrainingData/',
                                           transform=transforms.Compose([
                                               Rescale(600),
                                               Random_erase(0.5,[0.02,0.2], 0.3,2),
                                               #RandomCrop(288),
                                               #Rescale(224),
                                               #CenterCrop(560),
                                               #Flipver(),
                                               #RandomCrop(448),
                                               #Rescale(448),
                                               ToTensor()
                                               #Random_erase(0,[0.02,0.2], 0.3,2)
                                           ]))

dataloader = DataLoader(transformed_dataset, batch_size=1,
                        shuffle=True, num_workers=4)


print ('\nLength of batch per epoch is %d \n'%(len(dataloader)))

#model = vgg16()
#model = model.double()
#criterion = nn.MSELoss()

# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')


np.random.seed(None)
for i_batch, sample_batched in enumerate(dataloader):
    #print(i_batch, sample_batched['image'].size(),
    #      sample_batched['landmarks'].size())
    
    #print(sample_batched['landmarks'])
    #sample_batched['landmarks'] = sample_batched['landmarks'].reshape(-1)
    #sample_batched['landmarks'] = sample_batched['landmarks'].reshape(-1,12)
    '''
    images = Variable(sample_batched['image'].float())
    labels = Variable(sample_batched['landmarks'].float())
    labels = labels.view(-1,12)
    outputs = model(images)
    
    loss = criterion(outputs, labels)
    print(sample_batched['landmarks'])
    print(outputs)
    print(loss)
    '''
#    print(sample_batched['landmarks'])

    #print(sample_batched['img_name'])
    # observe 4th batch and stop.
    if i_batch == 1:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
