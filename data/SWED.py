import os
import numpy as np

from torch.utils.data import Dataset
class SWED(Dataset):
  def __init__(self, root_dir, image_processor, split = 'train'):
    self.root_dir = root_dir
    self.image_processor = image_processor
    self.id2label = {"0": "not water", "1": "water"}

    self.images = sorted(os.listdir(os.path.join(self.root_dir, split, 'images')))

  def __len__(self):
    return len(self.images)

  def __getitem__(self,idx):
    im_path = os.path.join(self.root_dir, 'train', 'images', self.images[idx])
    label_path = os.path.join(self.root_dir, 'train', 'labels', self.images[idx].replace('image', 'chip'))
    img = np.load(im_path)
    label = np.load(label_path)[0]
    img = img [:,:,[3,2,1]]
    for i in range(3):
      m = np.min(img[:,:,i])
      M = np.max(img[:,:,i])
      img[:,:,i] = (img[:,:,i]-m)/(M-m)*255

    encoded_inputs = self.image_processor(img, label, return_tensors="pt")

    for k,v in encoded_inputs.items():
      encoded_inputs[k].squeeze_() # remove batch dimension

    encoded_inputs["original_image"] = img.astype(np.uint8)

    return encoded_inputs
