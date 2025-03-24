import os
import numpy as np
import random
import torch

from data.copy_paste import CopyPaste, extract_bboxes
from torch.utils.data import Dataset

class SNOWED(Dataset):
  def __init__(self, root_dir, image_processor, bands = 'rgb', transform = None):
    self.path = root_dir
    self.image_processor = image_processor

    self.bands = bands
    self.transform = transform
  
    self.images = sorted(os.listdir(self.path)) 

  def __len__(self):
    return len(self.images)
  
  def read_image(self, sample_dir):
    # Retrieve data
    sample_2A = np.load(os.path.join(sample_dir, 'sample_2A.npy')).astype(np.float32)
    label = np.load(os.path.join(sample_dir, 'label.npy'))
    img = None

    # Select the bands
    if self.bands == 'rgb':
      img = sample_2A[:,:,[3,2,1]]  # Red, Green, Blue
    elif self.bands == 'color_ir':
      img = sample_2A[:,:,[7,3,2]]  # NIR, Red, Green
    elif self.bands == 'ndwi':  #  (Green – NIR) / (Green + NIR)
      b8 = img[:, :, 7]  # NIR
      b3 = img[:, :, 2]  # Green
      b8 = (b8 - np.min(b8)) / (np.max(b8) - np.min(b8))
      b3 = (b3 - np.min(b3)) / (np.max(b3) - np.min(b3))
      img = (b3 - b8) / (b3 + b8)
      img = np.expand_dims(img, axis=-1) 
    else:
      assert "Invalid band selection!"

    # Normalize channels
    for i in range(img.shape[-1]):
      m = np.min(img[:,:,i])
      M = np.max(img[:,:,i])
      img[:,:,i] = (img[:,:,i]-m)/(M-m)
    
    # Remove inf or NaN
    img[~np.isfinite(img)] = 0
    return img, label

  def __getitem__(self,idx):
    sample_dir = os.path.join(self.path, self.images[idx])
    img, label = self.read_image(sample_dir)
    
    if self.transform is not None:
      # read random image to paste
      paste_idx = random.randint(0, self.__len__() - 1)
      paste_img, paste_label = self.read_image(os.path.join(self.path, self.images[paste_idx]))
      # invert 0 and 1 in order to paste areas without water
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

    encoded_inputs["original_image"] = img.astype(np.uint8)
    encoded_inputs["original_labels"] = torch.from_numpy(label).long()

    return encoded_inputs