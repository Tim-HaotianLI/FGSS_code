from utils.data import MyDataset
from utils.loss import CombinedLoss, FocalLossWithMining, OptimalCombinedLoss, FocalLoss, FixedCombinedLoss
from utils.model1 import ObjectStateNet,BiCrossAttention, EnhancedClassifier
from utils.train_test1 import train, progressive_unfreeze_vit_with_scheduler

import json
import torch
import numpy as np
import random
import pandas as pd
from torch.utils.data import DataLoader, Subset
from collections import Counter
import torch.nn as nn
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight

def set_seed(seed=42):
    random.seed(seed)                      # Python 内置随机数
    np.random.seed(seed)                   # NumPy 随机数
    torch.manual_seed(seed)                # PyTorch CPU 随机数
    torch.cuda.manual_seed(seed)           # 当前 GPU
    torch.cuda.manual_seed_all(seed)       # 所有 GPU（多卡训练时）

    # 确保 CuDNN 的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_class_weight(dataloader):
    all_train_labels = []
    for batch in dataloader:
        labels = batch["label"].numpy()
        all_train_labels.extend(labels)
    train_labels = np.array(all_train_labels)

    classes = np.unique(train_labels)
    weights = compute_class_weight("balanced", classes=classes, y=train_labels)
    # print(f"Classes: {classes}")
    # print(f"Weights: {weights}")
    class_weights = torch.tensor(weights, dtype=torch.float).cuda()
    return class_weights

def get_class_weight_v2(dataloader):
    from collections import Counter
    counts = Counter()
    for batch in dataloader:
        for y in batch["label"].tolist():
            counts[int(y)] += 1

    max_cls = max(counts)
    class_counts = torch.tensor(
        [counts[i] for i in range(max_cls + 1)],
        dtype=torch.float,
        device="cuda"
    )

    # ⚙️ 温和权重：count^-0.25 适用于轻度不平衡
    weights = torch.pow(class_counts.clamp(min=1.0), -0.25)
    weights = weights / weights.mean()  # 保持均值为 1
    print(f"Class counts: {class_counts.tolist()}  →  weights: {weights.tolist()}")
    return weights

def create_balanced_subset(dataset, subset_ratio=0.3, seed=42):
    """从 dataset 中按类别比例采样，返回一个子集 (Subset)。"""
    random.seed(seed)
    labels = [d["label"] for d in dataset.data]
    label_counter = Counter(labels)
    subset_indices = []

    print("📊 原始类别分布:")
    for cls, count in label_counter.items():
        target = max(1, int(count * subset_ratio))
        all_idx = [i for i, lbl in enumerate(labels) if lbl == cls]
        random.shuffle(all_idx)
        subset_indices.extend(all_idx[:target])
        print(f"  类别 {cls}: {count} → 抽样 {target}")

    print(f"✅ 子集抽样完成，共 {len(subset_indices)} / {len(dataset)} 样本 "
          f"({subset_ratio * 100:.1f}%)\n")

    return Subset(dataset, subset_indices)

def get_dataloader(ds_id, train_transform, test_transform, use_subset=False):
    datasets = [
        "1. gas cylinder",
        "2. working at height",
        "3. worker with hardhat & vest",
        "4. working with eye protection",
        "5. vehicle"
    ]

    dataset = datasets[ds_id]

    train_img_path = f"./config_file/{dataset}/trainImgs/"
    train_ann_file = f"./config_file/{dataset}/train.json"
    test_img_path = f"./config_file/{dataset}/testImgs/"
    test_ann_file = f"./config_file/{dataset}/testImgs/bboxes_{ds_id + 1}.json"
    labels_file = f"./config_file/{dataset}/testImgs/labels_{ds_id + 1}.json"
    with open(labels_file, "r") as fp:
        label2id = json.loads(fp.read())
    num_classes = len(label2id)
    id2label = {v: k for k, v in label2id.items()}



    train_dataset = MyDataset(
        img_path_root=train_img_path,
        ann_file=train_ann_file,
        label2id=label2id,
        dataType="train",
        transform=train_transform,
    )
    test_dataset = MyDataset(
        img_path_root=test_img_path,
        ann_file=test_ann_file,
        label2id=label2id,
        dataType="test",
        transform=test_transform
    )

    seed = 42

    # ✅ 如果启用子集
    if use_subset:
        subset_ratio = 0.7
        print(f"⚙️ 正在对子集进行采样 (subset_ratio={subset_ratio})...")
        train_dataset = create_balanced_subset(train_dataset, subset_ratio=subset_ratio, seed=seed)
        # test_dataset = create_balanced_subset(test_dataset, subset_ratio=subset_ratio, seed=seed)


    g = torch.Generator()
    g.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=g)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_dataloader, test_dataloader, num_classes, id2label



train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomRotation(10),
    # transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # <======== 加了这个内容，效果有显著提升。=======
])

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    columns = ["dataset", "backbone", "attn", "precision", "recall", "F1", "model_weight_path"]
    save_results = pd.DataFrame(columns=columns)

    dataset_id = [4] #0, 1, 2, 3, 4
    backbone_set = ["resnet101"] # , "vit_b_16", "resnet101", "swin_v2_b",

    attn_types = ["cross"] # , "None", "element", "cross"
    epoches = 45


    for attn in attn_types:
        for backbone in backbone_set:
            for ds_id in dataset_id:
                set_seed(42)
                print(f"Backbone: {backbone} ---- Attn Block: {attn} ---- Dataset: {ds_id + 1}")
                # 1. 定义数据集
                train_dataloader, test_dataloader, num_classes, id2label = get_dataloader(
                    ds_id = ds_id,
                    train_transform = train_transform,
                    test_transform = test_transform,
                    use_subset=False
                )
                print(f"Length of train_dataloaderl: {len(train_dataloader)}")
                print(f"Length of test_dataloader: {len(test_dataloader)}")

                # 2. 定义损失函数
                class_names = list(id2label.values())
                # class_weights = get_class_weight(train_dataloader)
                # class_weights = torch.sqrt(class_weights)

                class_weights = get_class_weight_v2(train_dataloader)

                criterion = FixedCombinedLoss(
                    alpha=class_weights,
                    gamma=1.6,#1.5
                    smoothing=0.08,#
                    focal_weight=0.5,
                    smooth_weight=0.5
                ) # ==> F1 = 0    0.7645 ± 0.0760

                # 3. 定义模型
                model = ObjectStateNet(
                    backbone_name = backbone,
                    num_classes = num_classes,
                    pretrained=True,
                    attn_type=attn,
                    use_residual=True,
                    fusion="object"
                    # patchLevel=True
                ).to(device)

                # 4.冻结backbone的权重
                for param in model.object_extractor.parameters():
                    param.requires_grad = False

                if hasattr(model, 'global_extractor'):
                    for param in model.global_extractor.parameters():
                        param.requires_grad = False

                save_results = train(
                    model=model,
                    backbone=backbone,
                    attn=attn,

                    criterion=criterion,

                    num_classes=num_classes,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,

                    num_epochs=epoches,
                    ds_id=ds_id,

                    save_results=save_results,
                )
                print(f"Backbone: {backbone} ---- Attn Block: {attn} ---- Dataset: {ds_id + 1}")
                print(f"\n----------------------------------------------------------------------")
                print(f"-------------↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Next Model ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓-------------")
                print(f"----------------------------------------------------------------------")
        #         break
        #     break
        # break


    save_results.to_csv("./config_file/result_convnext_bicross_bbtrainable.csv", encoding="utf-8-sig")
