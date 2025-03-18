import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import argparse
import albumentations as A
import yaml

from torch.utils.data import Dataset, DataLoader, random_split
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from datasets import disable_caching

from data.copy_paste import CopyPaste
from data.SNOWED import SNOWED
from data.SWED import SWED

if __name__ == '__main__':

    disable_caching()
    torch.manual_seed(0)
    np.random.seed(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default= "config.yaml")
    parser.add_argument("--ckpt_path")
    parser.add_argument("--data_path")
    args = parser.parse_args() 

    with open(args.config) as file:
        cfg = yaml.safe_load(file)

    # Load dataset
    image_processor = SegformerImageProcessor(reduce_labels=True)

    if cfg['DATASET_PARAMS']['db_name'] == 'swed':
        dataset = SWED(root_dir=cfg['DATASET_PARAMS']['db_path'], image_processor=image_processor)
        # TODO: update test dataset 
        # TODO: add color ir
        # TODO: add copy paste augmentation
        train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
    else:
        dataset = SNOWED(root_dir=cfg['DATASET_PARAMS']['db_path'], image_processor=image_processor, bands=cfg['DATASET_PARAMS']['bands'], transform = None)
        train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])

    valid_dataloader = DataLoader(valid_dataset, batch_size=1)

    # Label and id
    id2label = {"0": "not water", "1": "water"}
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-" + cfg['MODEL_PARAMS']['model_config'],
                                                            num_labels=2,
                                                            id2label=id2label,
                                                            label2id=label2id) 
      
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    model.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device('cpu')))
  
    model.to(device)

  
    # Inference

    os.makedirs(cfg['OUTPUT_PATH']['inference_path'], exist_ok = True)

    # Color palette 
    palette = np.array([[0,0,0], [255,255,255], [255, 0 , 0], [0, 0, 255]]) # TN, TP , FP, FN

    red_patch = mpatches.Patch(color='red', label='FP')
    blue_patch = mpatches.Patch(color = 'blue', label = 'FN')
    white_patch = mpatches.Patch(color = 'white', label = 'TP')
    black_patch = mpatches.Patch(color = 'black', label = 'TN')
 
    model.eval()
    for idx, batch in enumerate(tqdm(valid_dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        labels = np.array(labels.squeeze())[::2,::2]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)


        predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(256,256)])[0]
        predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
       
        # 0 and 1 might be reversed? 
        predicted_segmentation_map = np.logical_not(predicted_segmentation_map)
        labels = np.logical_not(labels)


        color_seg = np.zeros((predicted_segmentation_map.shape[0],
                            predicted_segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
        color_seg[np.logical_or(labels, predicted_segmentation_map) == 0] = palette[0] # TN
        color_seg[np.logical_and(labels, predicted_segmentation_map) == 1] = palette[1] # TP
        color_seg[np.logical_or(labels, np.logical_not(predicted_segmentation_map)) == 0] = palette[2] # FP
        color_seg[np.logical_and(labels, np.logical_not(predicted_segmentation_map)) == 1] = palette[3] # FN

   

        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.array(batch["original_image"].squeeze()))
        plt.title('Original image')
        plt.subplot(1,2,2)
        plt.imshow(color_seg)
        plt.title('Segmentation')
        plt.legend(handles = [red_patch, blue_patch, black_patch, white_patch])

        plt.savefig(cfg['OUTPUT_PATH']['inference_path'] + '/output_' + str(idx)+'.png')
        plt.close()
    

      



