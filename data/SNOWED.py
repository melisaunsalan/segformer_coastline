import os
import numpy as np

from torch.utils.data import Dataset

class SNOWED(Dataset):
  def __init__(self, root_dir, image_processor):
    self.path = root_dir
    self.image_processor = image_processor

    self.images = sorted(os.listdir(self.path)) 

  def __len__(self):
    return len(self.images)

  def __getitem__(self,idx):
    sample_dir = os.path.join(self.path, self.images[idx])
    sample_2A = np.load(os.path.join(sample_dir, 'sample_2A.npy'))
    label = np.load(os.path.join(sample_dir, 'label.npy'))

    img = sample_2A[:,:,[3,2,1]]
    for i in range(3):
      m = np.min(img[:,:,i])
      M = np.max(img[:,:,i])
      img[:,:,i] = (img[:,:,i]-m)/(M-m)*255

    encoded_inputs = self.image_processor(img, label, return_tensors="pt")

    for k,v in encoded_inputs.items():
      encoded_inputs[k].squeeze_() # remove batch dimension

    encoded_inputs["original_image"] = img.astype(np.uint8)

    return encoded_inputs