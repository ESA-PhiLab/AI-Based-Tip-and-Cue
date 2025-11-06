import json, sys

def merge_coco_json(files: list[str], output_path: str) -> dict:
    """Merge multiple COCO-style JSON files into one; adjusts image & annotation IDs."""
    merged = {"images": [], "annotations": [], "categories": []}
    img_offset, ann_offset = 0, 0

    for i, path in enumerate(files):
        with open(path, "r") as f:
            data = json.load(f)

        # Ensure categories only added once
        if i == 0 and "categories" in data:
            merged["categories"] = data["categories"]

        # Reindex images and annotations to avoid ID collisions
        for img in data["images"]:
            img["id"] += img_offset
            merged["images"].append(img)

        for ann in data["annotations"]:
            ann["id"] += ann_offset
            ann["image_id"] += img_offset
            merged["annotations"].append(ann)

        img_offset = max([img["id"] for img in merged["images"]], default=0) + 1
        ann_offset = max([ann["id"] for ann in merged["annotations"]], default=0) + 1

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    return merged

if __name__ == "__main__":
    files = [
        "instances_train_mapped.json",
        "instances_val_mapped.json",
        "instances_test_mapped.json",
    ]
    merge_coco_json(files, "instances_merged.json")
    print("Merged JSON saved to instances_merged.json")
