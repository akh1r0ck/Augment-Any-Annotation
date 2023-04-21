import cv2
import sys
import argparse
import json
import numpy as np
from copy import deepcopy
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any, Dict, List
import random
from collections import defaultdict

from sam_segment import predict_masks_with_sam
from stable_diffusion_inpaint import fill_img_with_sd
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask


def get_coords(x, y, cornor_num=4):
    center = [x, y]
    bottom = [x, y+h]; top = [x, y-h]
    left = [x-w, y]; right = [x+w, y]
    if cornor_num==8:
        top_left = [x-w, y-h]; top_right = [x+w, y-h]
        bottom_left = [x-w, y+h]; bottom_right = [x+w, y+h]
        return [center, bottom, top, left, right, top_left, top_right, bottom_left, bottom_right], [1, 0, 0, 0, 0, 0, 0, 0, 0]

    else:
        return [center, bottom, top, left, right], [1, 0, 0, 0, 0]


def torch_fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def xywh2center(bbox):
    x, y, w, h = bbox
    x_out = (x + w / 2)
    y_out = (y + h / 2)
    return x_out, y_out, w//2, h//2


def show_points(ax, coords: List[List[float]], labels: List[int], size=375):
    coords = np.array(coords)
    labels = np.array(labels)

    color_table = {0: 'red', 1: 'green'}
    # for label_value, color in color_table.items():
    for index in range(len(labels)):
        points = coords[:, index]
        label = labels[index]
        color = color_table[label]

        ax.scatter(points[:, 0], points[:, 1], color=color, marker='*',
                s=size, edgecolor='white', linewidth=1.25)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-desc", type=str, default="natural scene")
    parser.add_argument("--dilate-kernel-size", type=int, default=50)
    parser.add_argument("--sam-model-type", type=str, choices=['vit_h', 'vit_l', 'vit_b'], default="vit_h")
    parser.add_argument("--sam-ckpt", type=str)
    parser.add_argument("--annotation-path", type=str)
    parser.add_argument("--cornor-num", type=int, choices=[4, 8], default=4)
    parser.add_argument("--seed", type=int, default=1814141513)

    return parser.parse_args()



if __name__ == "__main__":
    """Example usage:
    python a3_SingleImage_byCOCO.py \
        --dataset-desc animal \
        --sam-ckpt ../../sam_local/segment-anything-main/ckpt/sam_vit_h_4b8939.pth \
        --annotation-path ./annotations_1.json
    """

    args = arg_parse()
    # Argments
    dataset_desc = args.dataset_desc
    dilate_kernel_size = args.dilate_kernel_size
    sam_model_type = args.sam_model_type
    sam_ckpt = args.sam_ckpt    
    annotation_path = args.annotation_path
    cornor_num = args.cornor_num
    seed = args.seed


    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get json path form image path, provided that they are the same file name
    json_open = open(annotation_path, "r")
    
    # load json = annotation data
    json_load = json.load(json_open)
    
    # augmented annotation data
    new_json = deepcopy(json_load)

    # analyze annotation data
    category_names = {}
    for category in json_load["categories"]:
        category_names[category["id"]] = category["name"]

    # get imageId_imageInfo key: image id, value: annotation information
    imageId_imageInfo = {}
    last_image_id = 0
    for images in json_load["images"]:
        imageId_imageInfo[images["id"]] = images
        if last_image_id<images["id"]:
            last_image_id=images["id"]

    # get imageId_annoInfo key: image id, value: image information
    imageId_annoInfo = defaultdict(list)
    last_anno_id = 0
    for annotation in json_load["annotations"]:
        imageId_annoInfo[annotation["image_id"]].append(annotation)
        if last_anno_id<annotation["id"]:
            last_anno_id=annotation["id"]


    image_ids = list(imageId_annoInfo.keys())
    
    # image loop
    image_id = image_ids[0]
    # get annotation information using image id
    annotations = imageId_annoInfo[image_id]

    # image_id = annotation["image_id"]    
    image_name = imageId_imageInfo[image_id]["file_name"]

    ag_image_name = image_name.replace(".", "_ag.")
    # annotation information loop in an image 
    for anno_id, annotation in enumerate(annotations):
        if anno_id==0: # original image
            input_img = image_name
        else: # overwrite on augmented image
            input_img = ag_image_name

        # get bbox information
        x, y, w, h = xywh2center(annotation["bbox"])

        # get class id
        class_id = annotation["category_id"]
        # get class name
        class_name = category_names[class_id]
        
        # get each points; object and background
        point_coords, point_labels = get_coords(x, y, cornor_num)

        # set prompt for SD
        text_prompt = f"{class_name} in the context of {dataset_desc}"

        # load image
        img = load_img_to_array(input_img)

        # SAM prediction
        masks, scores, _ = predict_masks_with_sam(
            img,
            point_coords,
            point_labels,
            model_type=sam_model_type,
            ckpt_p=sam_ckpt,
            device=device,
        )
        maximum_index = np.argmax(scores)
        masks = masks.astype(np.uint8) * 255

        # dilate mask to avoid unmasked edge effect
        if dilate_kernel_size is not None:
            masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]


        # get maximum sroced mask
        idx = maximum_index; mask = masks[idx]


        # for confirmation
        # save the mask
        save_array_to_img(mask, image_name.replace(".", f"_{anno_id}_mask."))

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        # with point
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [point_coords], point_labels,
                    size=(width*0.04)**2)
        plt.savefig(image_name.replace(".", f"_{anno_id}_withPoints."), bbox_inches='tight', pad_inches=0)
        
        # with mask
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(image_name.replace(".", f"_{anno_id}_withMask."), bbox_inches='tight', pad_inches=0)
        plt.close()

        # inpainitng        
        if seed==1814141513:
            seed = random.randint(0, 512)
            torch_fix_seed(seed)
        img_filled = fill_img_with_sd(
            img, mask, text_prompt, device=device)
        save_array_to_img(img_filled, ag_image_name)


        # add annotation information
        last_anno_id+=1
        new_annoInfo = deepcopy(annotation)
        new_annoInfo["id"] = last_anno_id
        new_annoInfo["image_id"] = last_image_id+1
        new_json["annotations"].append(new_annoInfo)

    # add image information
    last_image_id += 1
    new_imageInfo = deepcopy(imageId_imageInfo[image_id])
    new_imageInfo["id"] = last_image_id
    new_imageInfo["file_name"] = ag_image_name
    new_json["images"].append(new_imageInfo)

    with open(annotation_path.replace(".json", "_ag.json"), mode="w") as F:
        json.dump(new_json, F, indent=4)