import os
import torch
import warnings
from torchvision.transforms import ToTensor
import pandas as pd
from utils import *

CLASSES = {'cv': [255, 0, 0], 'pv': [0, 255, 0]}

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


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, data, root='./', processor=None, **kwargs):
        self.processor = processor
        self.kwargs = kwargs
        self.images = []
        if isinstance(data, str):
            data = pd.read_csv(data)
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')
        for entry in data:
            image_file = os.path.join(root, entry['image_path'])
            masks_file = os.path.join(root, entry['mask_path']) if entry.get('mask_path') else None
            entry_info = {"image_id": image_file, "data": (image_file, masks_file), "kwargs": {}}
            self.images.append(entry_info)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_file, masks_file = image_info['data']
        kwargs = {**self.kwargs, **image_info['kwargs']}

        image = kwargs['image_reader'](image_file)
        masks = (kwargs['masks_reader'](masks_file) 
                 if masks_file is not None and os.path.exists(masks_file) else None)
        
        image, (masks, labels) = process_image_and_masks(image, masks, kwargs['dapi_only'])
        if self.processor is not None:
            image, (masks, labels) = self.processor(image, (masks, labels), **kwargs)

        image = ToTensor()(image.copy()).type(torch.float)
        if masks is not None:
            masks = ToTensor()(np.stack(masks, axis=-1)).type(torch.float)
        assert image.shape == kwargs['output_shape']
        
        return image, masks

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
            if self.kwargs['dapi_only']:
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
    rescale_size = (int(image.shape[0]*rescale_factor), int(image.shape[1]*rescale_factor))
    
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
    rescale_size = (int(image.shape[0]*rescale_factor), int(image.shape[1]*rescale_factor))
    
    slides = [image] + masks
    slides = Resize(size=rescale_size, order=3)(slides)
    slides = Pad(size=(h, w), mode='constant', position='center')(slides)
    image, masks = slides[0], slides[1:]
    
    return image, (masks, labels)


def inference_processor(image, objects, **kwargs):
    c, h, w = kwargs['output_shape']
    rescale_factor = kwargs['rescale_factor']
    
    ## process image and masks
    masks, labels = objects
    rescale_size = (int(image.shape[0]*rescale_factor), int(image.shape[1]*rescale_factor))
    if rescale_size[0] > h or rescale_size[1] > w:
        warnings.warn(f"Input image after rescale has size {rescale_size}, will be cropped to size {(h, w)}. ")

    slides = [image]
    slides = Resize(size=rescale_size, order=3)(slides)
    slides = Crop(size=(h, w), mode='constant', position='center')(slides)
    slides = Pad(size=(h, w), mode='constant', position='center')(slides)
    image = slides[0]

    return image, (None, labels)
