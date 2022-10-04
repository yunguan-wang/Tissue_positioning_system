import os
import sys
import math
import time

import pandas as pd
from collections import defaultdict, OrderedDict
from torchvision.transforms import ToTensor

from DIPModels.utils_g.utils_image import *
from DIPModels.utils_g.torch_layers import *
from DIPModels.utils_g import utils_pytorch
from DIPModels.utils_g import utils_data

SEED = None

CLASSES = OrderedDict({'cv': [255, 0, 0], 'pv': [0, 255, 0]})
GLOBAL_MEAN = np.array([0.5, 0.5, 0.5])
GLOBAL_STD = 1.0


def load_data(image_ids, image_path, masks_path=None, **kwargs):
    for image_id in image_ids:
        image_file = image_path.format(image_id=image_id)
        masks_file = masks_path.format(image_id=image_id) if masks_path is not None else None
        yield image_id, (image_file, masks_file), {**kwargs}


def process_image_and_masks(image, masks, dapi_only=True):
    ## Read in rois from lung segmentation
    # image_id = kwargs['pid']
    # region = kwargs['region']
    image = img_as('float')(image)
    if dapi_only:
        image = image[..., 2:]
    if masks is not None:
        masks = img_as('float')(masks)
        cv = masks[..., 0]
        pv = masks[..., 1]
        return image, ([cv, pv], [1, 2])
    else:
        return image, (None, [1,2])
    



class SegDataset(utils_data.ImageDataset):
    def __getitem__(self, idx):
        image_info = self.images[idx]
        ## Read in image and masks
        (image_file, masks_file), kwargs = image_info['data'], image_info['kwargs']
        image = kwargs['image_reader'](image_file)
        masks = (kwargs['masks_reader'](masks_file) 
                 if masks_file is not None and os.path.exists(masks_file) else None)
        
        image, (masks, labels) = process_image_and_masks(image, masks, kwargs['dapi_only'])
        image, (masks, labels) = self.processor(image, (masks, labels), **kwargs)
        
        image = ToTensor()(image.copy()).type(torch.float)
        if masks is not None:
            masks = ToTensor()(np.stack(masks, axis=-1)).type(torch.float)
        assert image.shape == kwargs['output_shape']
        return image, masks
    
    def global_stats(self, transform=img_as('float')):
        x = np.stack([image_info['kwargs']['image_reader'](image_info['data'][0]) 
                      for image_info in self.images])
        if transform is not None:
            x = transform(x)
        return np.mean(x, axis=(0, 1, 2)), np.std(x, axis=(0, 1, 2))
    
    def display(self, indices=None):
        if indices is None:
            indices = range(len(self))
        for idx in indices:
            image, masks = self[idx]
            image, N = image.numpy(), 1
            if masks is not None:
                masks = masks.numpy()
                N += len(masks)
            
            print(self.images[idx]['image_id'])
            print(image_stats(image))
            print(image_stats(masks))
            
            fig, ax = plt.subplots(1, N, figsize=(4*N, 4))
            if self.images[idx]['kwargs']['dapi_only']:
                ax[0].imshow(image[0])
            else:
                ax[0].imshow(np.moveaxis(image, 0, -1))
            ax[0].set_title("image")
            if masks is not None:
                for i, mask in enumerate(masks, 1):
                    ax[i].imshow(mask)
                    ax[i].set_title(self.labels[i])
            plt.show()


def train_processor(image, objects, **kwargs):
    c, h, w = kwargs['output_shape']
    rescale_factor = kwargs['rescale_factor']
    
    ## image augmentation: randomly adjust image intensity
    image *= np.random.uniform(0.9, 1/np.max(image))
    rescale_size = (image.shape[0]//rescale_factor, image.shape[1]//rescale_factor)
    
    ## augment image and masks together
    masks, labels = objects
    slides = [image] + masks
    slides = Resize(size=rescale_size, order=3)(slides)
    slides = Crop(size=(h, w), pos='random')(slides)
    slides = RandomTransform(size=(h, w), rotation=30, translate=(50, 50), 
                             scale=(0.2, 0.1), shear=10, projection=(0.0002, 0.0002), 
                             order=3, p=0.7)(slides)
    slides = RandomHorizontalFlip(0.5)(slides)
    slides = RandomVerticalFlip(0.5)(slides)
    image, masks = slides[0], slides[1:]
    
    return image, (masks, labels)


def val_processor(image, objects, **kwargs):
    c, h, w = kwargs['output_shape']
    rescale_factor = kwargs['rescale_factor']
    
    ## process image and masks
    masks, labels = objects
    rescale_size = (image.shape[0]//rescale_factor, image.shape[1]//rescale_factor)
    
    slides = [image] + masks
    slides = Resize(size=rescale_size, order=3)(slides)
    slides = Pad(size=(h, w), mode='constant', position='center')(slides)
    image, masks = slides[0], slides[1:]
    
    return image, (masks, labels)


def test_processor(image, objects, **kwargs):
    c, h, w = kwargs['output_shape']
    rescale_factor = kwargs['rescale_factor']
    
    ## process image and masks
    masks, labels = objects
    rescale_size = (image.shape[0]//rescale_factor, image.shape[1]//rescale_factor)
    
    slides = [image] + masks
    slides = Resize(size=rescale_size, order=3)(slides)
    slides = Pad(size=(h, w), mode='constant', position='center')(slides)
    image, masks = slides[0], slides[1:]
    
    return image, (masks, labels)

def predict_processor(image, objects, **kwargs):
    c, h, w = kwargs['output_shape']
    rescale_factor = kwargs['rescale_factor']

    # Handle conditions when the input is larger than 2048 x 4096
    print(image.shape)
    img_h, img_w, c = image.shape
    if img_h > 2*h or img_w > 2*w:
        image = image[:2*h,:2*w,:]

    ## process image and masks
    masks, labels = objects
    rescale_size = (image.shape[0]//rescale_factor, image.shape[1]//rescale_factor)
    
    slides = [image]
    slides = Resize(size=rescale_size, order=3)(slides)
    slides = Pad(size=(h, w), mode='constant', position='center')(slides)
    image = slides[0]
    
    return image, (None, labels)