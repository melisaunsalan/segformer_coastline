import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import argparse
import albumentations as A
import yaml

from torch.utils.data import Dataset, DataLoader, random_split
from transformers import SegformerConfig, SegformerImageProcessor, SegformerForSemanticSegmentation
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
    parser.add_argument("--ckpt_path")
    parser.add_argument("--dataset")
    parser.add_argument("--data_path")
    parser.add_argument("--bands")
    parser.add_argument("--output")
    args = parser.parse_args() 

    # Define image processor
    # image_processor = SegformerImageProcessor(reduce_labels=False)
    image_processor = SegformerImageProcessor(do_rescale=False,
                                              do_normalize=False)

    # Load dataset
    if args.dataset == 'swed':
        valid_dataset = SWED(root_dir=args.data_path, 
                             image_processor=image_processor,
                             transform=None,
                             bands=args.bands,
                             split='test')
    else:
        dataset = SNOWED(root_dir=args.data_path, 
                         image_processor=image_processor, 
                         bands=args.bands, 
                         transform = None,
                         test=True)
        _, valid_dataset = random_split(dataset, [0.8, 0.2])

    # Create validation loader
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)

    # Label and id
    id2label = {"0": "not water", "1": "water"}
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    # Define the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )
    
    # If ndwi band is selected, replace the overlapping patch embeddings
    if args.bands == 'ndwi':
        model.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(in_channels=1, 
                                                                           out_channels=model.config.hidden_sizes[0],
                                                                           kernel_size=model.config.patch_sizes[0], 
                                                                           stride=model.config.strides[0])

    # Load state dict
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device), strict=True)
    model.to(device)

    # Inference
    inference_path = args.output
    os.makedirs(inference_path, exist_ok = True)

    # Color palette 
    palette = np.array([[0,0,0], [255,255,255], [255, 0 , 0], [0, 0, 255]]) # TN, TP , FP, FN

    red_patch = mpatches.Patch(color='red', label='FP')
    blue_patch = mpatches.Patch(color = 'blue', label = 'FN')
    white_patch = mpatches.Patch(color = 'white', label = 'TP')
    black_patch = mpatches.Patch(color = 'black', label = 'TN')
 
    model.eval()
    for idx, batch in enumerate(tqdm(valid_dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["original_labels"].unsqueeze(dim=0).numpy()

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(256,256)])[0]
        predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
       
        # Create color seg array
        color_seg = np.zeros((predicted_segmentation_map.shape[0],
                            predicted_segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
        color_seg[(np.logical_or(labels, predicted_segmentation_map) == 0).squeeze()] = palette[0] # TN
        color_seg[(np.logical_and(labels, predicted_segmentation_map) == 1).squeeze()] = palette[1] # TP
        color_seg[(np.logical_or(labels, np.logical_not(predicted_segmentation_map)) == 0).squeeze()] = palette[2] # FP
        color_seg[(np.logical_and(labels, np.logical_not(predicted_segmentation_map)) == 1).squeeze()] = palette[3] # FN

        # Plot and save
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.array(batch["original_image"].squeeze() * 255, dtype=np.uint8))
        plt.title('Original image')
        plt.subplot(1,2,2)
        plt.imshow(color_seg)
        plt.title('Segmentation')
        plt.legend(handles = [red_patch, blue_patch, black_patch, white_patch])
        plt.savefig(os.path.join(inference_path, 'output_' + str(idx)+'.png'))
        plt.close()
    

      



