import os
import numpy as np
import tifffile as tiff
import random
import torch

from data.copy_paste import extract_bboxes
from torch.utils.data import Dataset

class SWED(Dataset):
  def __init__(self, root_dir, image_processor, transform=None, bands='rgb', split='train'):
    self.root_dir = os.path.join(root_dir, split)
    self.image_processor = image_processor
    self.id2label = {"0": "not water", "1": "water"}
    self.bands = bands
    self.images = sorted(os.listdir(os.path.join(self.root_dir, 'images')))
    self.transform = transform
    self.split = split

  def __len__(self):
    return len(self.images)

  def select_bands(self, img):
    if self.bands == 'rgb':
      img = img[:, :, [3,2,1]]  # Red, Green, Blue
    elif self.bands == 'color_ir':
      img = img[:, :, [7,3,2]]  # NIR, Red, Green
    elif self.bands == 'ndwi':  #  (Green – NIR) / (Green + NIR)
      b8 = img[:, :, 7]  # NIR
      b3 = img[:, :, 2]  # Green
      b8 = (b8 - np.min(b8)) / (np.max(b8) - np.min(b8))
      b3 = (b3 - np.min(b3)) / (np.max(b3) - np.min(b3))
      img = (b3 - b8) / (b3 + b8)
      img = np.expand_dims(img, axis=-1) 
    else:
      assert "Invalid band selection!"
    return img

  def normalize(self, img):
    img = img.astype(np.float32)
    for i in range(img.shape[-1]):
      m = np.min(img[:,:,i])
      M = np.max(img[:,:,i])
      img[:,:,i] = (img[:,:,i]-m)/(M-m)
    img[~np.isfinite(img)] = 0
    return img

  def read_image(self, idx):
    # Get image paths
    im_path = os.path.join(self.root_dir, 'images', self.images[idx])
    to_replace = 'label' if self.split == 'test' else 'chip'
    label_path = os.path.join(self.root_dir, 'labels', self.images[idx].replace('image', to_replace))
    
    # Load images 
    if self.split == 'train':
      img = np.load(im_path)
      label = np.load(label_path).squeeze()
    else:
      img = tiff.imread(im_path).transpose(1, 2, 0)
      label = tiff.imread(label_path)

    # Select the bands accordingly
    img = self.select_bands(img)
    
    # Normalize the images
    img = self.normalize(img)
    return img, label


  def __getitem__(self,idx):
    # Load images 
    img, label = self.read_image(idx)

    if self.transform is not None:
      # Read a random image to paste
      paste_idx = random.randint(0, self.__len__() - 1)
      paste_img, paste_label = self.read_image(paste_idx)
      
      # Invert 0 and 1 in order to paste areas without water
      paste_label_inversed = np.where((paste_label==0)|(paste_label==1), paste_label^1, paste_label)
      out = self.transform(image = img, masks = np.expand_dims(label, 0),
                           bboxes = extract_bboxes(np.expand_dims(label, 0)),
                           paste_image = paste_img, paste_masks = np.expand_dims(paste_label_inversed, 0),
                           paste_bboxes = extract_bboxes(np.expand_dims(paste_label_inversed, 0)))
      encoded_inputs = self.image_processor(out['image'], return_tensors="pt")
      
      # Replace with the new segmentation map
      label = np.array(out['masks'][0])
    else:
      encoded_inputs = self.image_processor(img, return_tensors="pt")

    for k,v in encoded_inputs.items():
      encoded_inputs[k] = v.squeeze(0) # remove batch dimension

    # Add the original image and labels as well
    encoded_inputs["original_image"] = img.astype(np.uint8)
    encoded_inputs["original_labels"] = torch.from_numpy(label).long()

    return encoded_inputs