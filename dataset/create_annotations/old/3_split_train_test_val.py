import os
import json
import random
import shutil
from collections import defaultdict

random.seed(42)

os.chdir('whales_from_space_matched')

# === CONFIG ===
root_dir = os.getcwd()  # adjust to your working dir
annotations_file = os.path.join(root_dir, '../new_annotations.json')
output_dir = os.path.join(root_dir, '0_dataset')  # final folder

# Train/val/test ratios (must sum to 1)
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

splits = ['train', 'val', 'test']

# === CREATE OUTPUT STRUCTURE ===
images_dir = os.path.join(output_dir, 'images')
annotations_dir = os.path.join(output_dir, 'annotations')

for split in splits:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# === LOAD COCO ===
with open(annotations_file) as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']

# === SHUFFLE & SPLIT ===
random.shuffle(images)
n = len(images)
train_cut = int(n * train_ratio)
val_cut = int(n * (train_ratio + val_ratio))

split_images = {
    'train': images[:train_cut],
    'val': images[train_cut:val_cut],
    'test': images[val_cut:]
}

# === MAP image_id -> annotations ===
image_id_to_anns = defaultdict(list)
for ann in annotations:
    image_id_to_anns[ann['image_id']].append(ann)

# === PROCESS ===
for split in splits:
    imgs = split_images[split]
    ids = set(img['id'] for img in imgs)
    anns = []
    for img_id in ids:
        anns.extend(image_id_to_anns[img_id])

    # Copy images into images/{split}/
    new_imgs = []  # store fixed images
    for img in imgs:
        src = os.path.join(root_dir, img['file_name'])
        dst = os.path.join(images_dir, split, os.path.basename(img['file_name']))
        shutil.copy2(src, dst)

    
        img_copy = img.copy()
        img_copy['file_name'] = os.path.basename(img['file_name'])
        new_imgs.append(img_copy)

    # Save COCO annotations in annotations/ folder
    split_coco = {
        'images': new_imgs,  # use updated filenames
        'annotations': anns,
        'categories': coco.get('categories', []),
        'licenses': coco.get('licenses', [])
    }
    with open(os.path.join(annotations_dir, f'instances_{split}.json'), 'w') as f:
        json.dump(split_coco, f)

print(f"Done! Structure created in: {output_dir}")
