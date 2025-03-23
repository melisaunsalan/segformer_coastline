import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import evaluate
import json
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

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    disable_caching()
    torch.manual_seed(0)
    np.random.seed(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "config.yaml")
    args = parser.parse_args() 

    with open(args.config) as file:
        cfg = yaml.safe_load(file)

    # Load dataset
    image_processor = SegformerImageProcessor(reduce_labels=True)

    # Init TensorBoard logger
    experiments_path = "experiments"
    os.makedirs(experiments_path, exist_ok=True)
    exp_path = os.path.join(experiments_path, cfg['EXPERIMENT']['name'])
    writer = SummaryWriter(log_dir=exp_path)

    # Load dataset
    image_processor = SegformerImageProcessor(do_rescale=False,
                                              do_normalize=False)


    if cfg['DATASET_PARAMS']['use_copypaste_aug']: 
        transform = A.Compose([
        A.RandomScale(scale_limit=(0.1, 1), p=cfg['DATASET_PARAMS']['probability_of_aug']), 
        A.PadIfNeeded(256, 256, border_mode=0),
        A.RandomCrop(256, 256),
        CopyPaste(blend=True, sigma=1, pct_objects_paste=None, p=cfg['DATASET_PARAMS']['probability_of_aug']) 
    ], bbox_params=A.BboxParams(format="coco",min_visibility=0.05, label_fields = [])
    )
    else:
        transform = None

    if cfg['DATASET_PARAMS']['db_name'] == 'swed':
        dataset = SWED(root_dir=cfg['DATASET_PARAMS']['db_path'], image_processor=image_processor)
        # TODO: update test dataset 
        # TODO: add color ir
        # TODO: add copy paste augmentation
        train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
    else:
        dataset = SNOWED(root_dir=cfg['DATASET_PARAMS']['db_path'], image_processor=image_processor, bands=cfg['DATASET_PARAMS']['bands'], transform = transform)
        train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['TRAIN_PARAMS']['batch_size'], shuffle=True, num_workers=cfg['TRAIN_PARAMS']['num_workers'])
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)

    # Label and id
    id2label = {"0": "not water", "1": "water"}
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    # Devine the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: ', device)
    ckp_path = getattr(args, "ckp", None)
    if ckp_path and os.path.isfile(ckp_path):
        print(f"Loading model from checkpoint: {ckp_path}")
        model = SegformerForSemanticSegmentation(num_labels=2, id2label=id2label, label2id=label2id)
        model.load_state_dict(torch.load(ckp_path, map_location=device), strict=True)
    else:
        print("No valid checkpoint provided, loading pretrained model.")
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=2, id2label=id2label, label2id=label2id
        )

    # If ndwi band is selected, replace the overlapping patch embeddings
    if cfg['DATASET_PARAMS']['bands'] == 'ndwi':
        model.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(in_channels=1, 
                                                                           out_channels=model.config.hidden_sizes[0],
                                                                           kernel_size=model.config.patch_sizes[0], 
                                                                           stride=model.config.strides[0])

    # Move model to device
    model.to(device)   

    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['TRAIN_PARAMS']['lr'])
    os.makedirs(cfg['OUTPUT_PATH']['weights_path'], exist_ok=True)

    # Best metric and model
    best_iou = 0
    best_model = None

    # Define metrics
    cache_dir = f"exp/{cfg['EXPERIMENT']['name']}/cache"
    os.makedirs(cache_dir, exist_ok=True)
    iou_metric = evaluate.load("mean_iou", cache_dir=cache_dir) 
    val_iou_metric = evaluate.load("mean_iou", cache_dir=cache_dir) 

    # Create the output path and color pallete for predictions
    os.makedirs(cfg['OUTPUT_PATH']['inference_path'], exist_ok = True)
    palette = np.array([[255, 255, 0], [255, 0, 255]])

    # Training
    model.train()
    for epoch in range(cfg['TRAIN_PARAMS']['num_epochs']):  
        print("Epoch:", epoch)
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # get the inputs;
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["original_labels"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits

            loss.backward()
            optimizer.step()

            # Compute training metrics
            with torch.no_grad():
                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)

                # Predict arrays
                pred_arr = predicted.detach().cpu().numpy()
                labels_arr = labels.detach().cpu().numpy()

                # Add batch
                iou_metric.add_batch(predictions=pred_arr, 
                                 references=labels_arr)

        # Compute the score
        iou_metrics = iou_metric.compute(num_labels=2, ignore_index=255)

        print("Loss:", loss.item())
        print("Mean IOU:", iou_metrics["mean_iou"])
        print("Mean accuracy:", iou_metrics["mean_accuracy"])

        # Log to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('Train_Metrics/mean_iou', iou_metrics["mean_iou"], epoch)
        writer.add_scalar('Train_Metrics/mean_acc', iou_metrics["mean_accuracy"], epoch)

        # Evaluate after each epoch
        model.eval()
        val_loss = 0.0
        for val_idx, val_batch in enumerate(tqdm(valid_dataloader)):
            with torch.no_grad():
                pixel_values = val_batch["pixel_values"].to(device)
                labels = val_batch["original_labels"].to(device)

                # Forward pass
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss, logits = outputs.loss, outputs.logits
                val_loss += loss.item()

                # Upsample logits
                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)

                # Convert preds and labels to numpy 
                pred_arr = predicted.detach().cpu().numpy()
                labels_arr = labels.detach().cpu().numpy()

                if epoch == cfg['TRAIN_PARAMS']['num_epochs'] - 1:
                    # If last epoch => save preditions
                    predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(256,256)])[0]
                    predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()

                    h, w = predicted_segmentation_map.shape
                    color_seg = np.zeros((h, w, 3), dtype=np.uint8)
                    for label, color in enumerate(palette):
                        color_seg[predicted_segmentation_map == label, :] = color
                    
                    # Original image
                    # img = np.array(batch["original_image"].squeeze())
                    # plt.imsave(cfg['OUTPUT_PATH']['inference_path'] + '/output_' + str(idx)+'_orig.png', img)

                    # Segmentation map
                    img = np.array(batch["original_image"].squeeze()) * 0.5 + color_seg * 0.5
                    img = img.astype(np.uint8)
                    plt.imsave(cfg['OUTPUT_PATH']['inference_path'] + '/output_' + str(idx)+'.png', img)

                # Add batch
                val_iou_metric.add_batch(predictions=pred_arr, references=labels_arr)

        # Compute and log loss and metrics
        iou_val_metrics = val_iou_metric.compute(num_labels=2, ignore_index=255)
        print("Validation Loss:", val_loss / len(valid_dataloader))
        print("Validation Mean IoU:", iou_val_metrics["mean_iou"])
        print("Validation Mean accuracy:", iou_val_metrics["mean_accuracy"])

        writer.add_scalar('Loss/val', val_loss / len(valid_dataloader), epoch)
        writer.add_scalar('Validation_Metrics/mean_iou', iou_val_metrics["mean_iou"], epoch)
        writer.add_scalar('Validation_Metrics/mean_acc', iou_val_metrics["mean_accuracy"], epoch)

        # Save best model
        if iou_val_metrics["mean_iou"] > best_iou:
            best_iou = iou_val_metrics["mean_iou"]
            best_model_path = os.path.join(exp_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch: {epoch+1}. Best model saved at {best_model_path}.")
    
        model.train()

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(exp_path, 'checkpoint_' + str(epoch+1) + '.pth'))

    # Save last model
    torch.save(model.state_dict(), os.path.join(exp_path, 'last_model.pth'))

    
        




