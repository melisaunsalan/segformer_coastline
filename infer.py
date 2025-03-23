import os
import numpy as np
import torch
import argparse
import albumentations as A
import matplotlib.pyplot as plt
import evaluate 
import json 

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from tqdm import tqdm
from datasets import disable_caching
from data.SNOWED import SNOWED
from data.SWED import SWED

def main():
    disable_caching()
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["swed", "snowed"], default='snowed')
    parser.add_argument("-o", "--output", default='output')
    parser.add_argument("-p", "--path", help="Path to dataset", default="SNOWED_v02/SNOWED")
    parser.add_argument("-c", "--checkpoint", help="Path to model checkpoint", required=True)
    parser.add_argument("-b", "--bands", choices=['rgb', 'color_ir'], default='rgb', help='Sentinel-2 band combination')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    image_processor = SegformerImageProcessor(reduce_labels=True)

    dataset_cls = SWED if args.dataset == "swed" else SNOWED
    dataset = dataset_cls(root_dir=args.path, image_processor=image_processor, bands=args.bands)
    _, valid_dataset = random_split(dataset, [0.8, 0.2])
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)

    id2label = {0: "land", 1: "water"}
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", 
                                                             num_labels=2,
                                                             id2label=id2label,
                                                             label2id=label2id)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs(args.output, exist_ok=True)

    # Color palette 
    palette = np.array([[255, 255, 0], [255, 0, 255]])

    metric = evaluate.load("mean_iou")
    all_predictions, all_references = [], []
    
    for idx, batch in enumerate(tqdm(valid_dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["original_labels"]
        labels = labels.squeeze().cpu().numpy()

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

            upsampled_logits = nn.functional.interpolate(outputs.logits.cpu(), 
                                                         size=labels.shape[-2:], 
                                                         mode="bilinear", 
                                                         align_corners=False)
            predicted = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
        
        # Append results
        all_predictions.append(predicted)
        all_references.append(labels)

        predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(256,256)])[0]
        predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()

        h, w = predicted_segmentation_map.shape
        color_seg = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[predicted_segmentation_map == label, :] = color

        img = np.array(batch["original_image"].squeeze())
        output_path = os.path.join(args.output, f"output_orig_{idx}.png")
        # plt.imsave(output_path, img.astype(np.uint8))

        img = np.array(batch["original_image"].squeeze()) * 0.8 + color_seg * 0.2
        img = img.astype(np.uint8)
        
        output_path = os.path.join(args.output, f"output_{idx}.png")
        plt.imsave(output_path, img)
    
    metrics = metric._compute(
                        predictions=np.array(all_predictions),
                        references=np.array(all_references),
                        num_labels=len(id2label),
                        ignore_index=255,
                        reduce_labels=False,
                    )
    
    # Log
    print("Validation Metrics:", metrics)
    metrics_serializable = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in metrics.items()}
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics_serializable, f, indent=4)

if __name__ == "__main__":
    main()