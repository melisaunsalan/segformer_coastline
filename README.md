# Finetune SegFormer for coastline detection


## Installation


```bash
pip install -r requirements.txt
```

## Usage

SNOWED dataset, RGB images, no augmentation
```bash
python main.py --path /path/to/dataset
```
SNOWED dataset, Color IR images, no augmentation
```bash
python main.py --path /path/to/dataset --bands color_ir
```

SNOWED dataset, RGB images, copy paste augmentation
```bash
python main.py --path /path/to/dataset --copypaste
```
