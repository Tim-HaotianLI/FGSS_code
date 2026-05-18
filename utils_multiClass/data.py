import os
import json
import random
import torch
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, img_path_root, ann_file, labels2ids, dataType=None, transform=None):
        self.img_path_root = img_path_root
        self.ann_file = ann_file
        self.labels2ids = labels2ids
        self.transform = transform
        self.data = []  # 用于存储所有的数据
        self.dataType = dataType
        if self.dataType == "train":
            self.process_train_data()
        elif self.dataType == "test":
            self.process_test_data()

    def process_train_data(self):
        # 训练的时候，使用ground truth的结果进行训练，因为数据在输入之后，会进行人为修正
        with open(self.ann_file, "r") as fp:
            raw_data = json.loads(fp.read())

        imgId_imgInfo = {item["id"]: item for item in raw_data["images"]}
        annotations = raw_data["annotations"]

        for ann in annotations:
            imgId = ann["image_id"]
            imgInfo = imgId_imgInfo[imgId]
            img_path = os.path.join(self.img_path_root, imgInfo["file_name"])
            bbox = ann["bbox"] # xywh
            labels = ann["category_id"]
            if labels == [2,2]:
                continue
            self.data.append({
                "img_path": img_path,
                "bbox": bbox,
                "label": labels
            })

    def process_test_data(self):
        # 推理的时候，使用vgms的predictions结果进行推理，因为推理阶段的过程中，是没有人为修正这个步骤的。
        with open(self.ann_file, "r") as fp:
            raw_data = json.loads(fp.read())

        for k, v in raw_data.items():
            img_name = f"{k}.jpg"
            img_path = os.path.join(self.img_path_root, img_name)

            for i, ann in v.items():
                x1,y1,x2,y2 = ann["bbox"] #xyxy
                w = x2 - x1
                h = y2 - y1
                labels = ann["label"]
                if "status unknown" in labels:
                    continue

                self.data.append({
                    "img_path": img_path,
                    "bbox": [x1, y1, w, h],
                    "label": [self.labels2ids[i][label] for i, label in enumerate(labels)]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        img_path = sample["img_path"]
        image = Image.open(img_path).convert("RGB")
        img_W, img_H = image.size
        x1, y1, w, h = sample["bbox"]


        if self.dataType == "train":
            p_jitter = 0.6
            if random.random() < p_jitter:
                # 根据目标大小，自适应抖动幅度
                area_ratio = (w * h) / (img_W * img_H)
                if area_ratio < 0.02:  # 小目标
                    scale_range = (-0.02, 0.02)
                    shift_range = (-0.015, 0.015)
                elif area_ratio < 0.08:  # 中目标
                    scale_range = (-0.04, 0.04)
                    shift_range = (-0.025, 0.025)
                else:  # 大目标
                    scale_range = (-0.05, 0.05)
                    shift_range = (-0.03, 0.03)


                jitter_scale = random.uniform(*scale_range)
                jitter_shift = random.uniform(*shift_range)

                orig_bbox = [x1, y1, w, h]

                # 中心坐标
                cx = x1 + w / 2
                cy = y1 + h / 2

                # 抖动中心与尺寸
                cx += jitter_shift * w
                cy += jitter_shift * h
                w = w * (1 + jitter_scale)
                h = h * (1 + jitter_scale)

                # 重新计算左上角坐标
                x1 = cx - w / 2
                y1 = cy - h / 2

                # 防止越界
                min_size = 8
                x1 = max(0, min(x1, img_W - min_size))
                y1 = max(0, min(y1, img_H - min_size))
                w = max(min(w, img_W - x1), min_size)
                h = max(min(h, img_H - y1), min_size)

                # IoU保护：若偏差太大则回退原 bbox
                def compute_iou(box1, box2):
                    xA = max(box1[0], box2[0])
                    yA = max(box1[1], box2[1])
                    xB = min(box1[0] + box1[2], box2[0] + box2[2])
                    yB = min(box1[1] + box1[3], box2[1] + box2[3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
                    return inter / union if union > 0 else 0.0

                iou = compute_iou([x1, y1, w, h], orig_bbox)
                if iou < 0.8:
                    x1, y1, w, h = orig_bbox  # 回退


        x2 = x1 + w
        y2 = y1 + h
        cropped_region = image.crop((x1, y1, x2, y2))

        if self.transform:
            image = self.transform(image)
            cropped_region = self.transform(cropped_region)

        bbox_normed = torch.tensor([
            x1 / img_W,
            y1 / img_H,
            x2 / img_W,
            y2 / img_H
        ], dtype=torch.float32)

        return {
            "original": image,
            "crop": cropped_region,
            "label": sample["label"],
            "bbox": bbox_normed
        }