import os
import numpy as np
import random

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
    sample_2A = np.load(os.path.join(sample_dir, 'sample_2A.npy'))
    label = np.load(os.path.join(sample_dir, 'label.npy'))
    img = None
    if self.bands == 'rgb':
      img = sample_2A[:,:,[3,2,1]]
    elif self.bands == 'color_ir':
      img = sample_2A[:,:,[7,3,2]]
    for i in range(3):
      m = np.min(img[:,:,i])
      M = np.max(img[:,:,i])
      img[:,:,i] = (img[:,:,i]-m)/(M-m)*255
    return img, label

  def __getitem__(self,idx):
    sample_dir = os.path.join(self.path, self.images[idx])
    img, label = self.read_image(sample_dir)
    
    if self.transform is not None:
      # read random image to paste
      paste_idx = random.randint(0, self.__len__() - 1)
      paste_img, paste_label = self.read_image(os.path.join(self.path, self.images[paste_idx]))
      # invert 0 and 1 in order to paste areas without water
      paste_label = np.where((paste_label==0)|(paste_label==1), paste_label^1, paste_label)
  
      out = self.transform(image = img, masks = np.expand_dims(label, 0),
                           bboxes = extract_bboxes(np.expand_dims(label, 0)),
                           paste_image = paste_img, paste_masks = np.expand_dims(paste_label, 0),
                           paste_bboxes = extract_bboxes(np.expand_dims(paste_label, 0)))
      encoded_inputs = self.image_processor(out['image'], np.array(out['masks'][0]), return_tensors="pt")
    
    else:
      encoded_inputs = self.image_processor(img, label, return_tensors="pt")

    for k,v in encoded_inputs.items():
      encoded_inputs[k].squeeze_() # remove batch dimension

    encoded_inputs["original_image"] = img.astype(np.uint8)

    return encoded_inputs