import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import evaluate
import json
import argparse

from torch.utils.data import Dataset, DataLoader, random_split
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from data.SNOWED import SNOWED
from data.SWED import SWED

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["swed", "snowed"], default = 'snowed')
    parser.add_argument("--path", help = "Path to dataset", default="data/SNOWED_v02/SNOWED")
    parser.add_argument("--num_epochs", default=100, type=int)
    args = parser.parse_args()

    # Load dataset
    image_processor = SegformerImageProcessor(reduce_labels=True)

    if args.dataset == "swed":
        dataset = SWED(root_dir=args.path, image_processor=image_processor)
        # TODO: update test dataset 
        train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
    else:
        dataset = SNOWED(root_dir=args.path, image_processor=image_processor)
        train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2)

    # Label and id
    id2label = {"0": "not water", "1": "water"}
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                            num_labels=2,
                                                            id2label=id2label,
                                                            label2id=label2id) 
     
    metric = evaluate.load("mean_iou") 

    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for epoch in range(args.num_epochs):  
        print("Epoch:", epoch)
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # get the inputs;
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits

            loss.backward()
            optimizer.step()

            # evaluate
            with torch.no_grad():
                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)

                # note that the metric expects predictions + labels as numpy arrays
                metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

            # let's print loss and metrics every 100 batches
            if idx % 100 == 0:
            # currently using _compute instead of compute
            # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
                metrics = metric._compute(
                        predictions=predicted.cpu(),
                        references=labels.cpu(),
                        num_labels=len(id2label),
                        ignore_index=255,
                        reduce_labels=False, # we've already reduced the labels ourselves
                    )

                print("Loss:", loss.item())
                print("Mean_iou:", metrics["mean_iou"])
                print("Mean accuracy:", metrics["mean_accuracy"])

        if (epoch+1)%10 == 0:
            torch.save(model.state_dict(), 'checkpoint_' + str(epoch+1) + '.pth')

    torch.save(model.state_dict(), 'model.pth')
        




