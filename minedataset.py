import torch
import torch.utils.data as data
import albumentations as A
import cv2
import numpy as np
import os
import random
from albumentations.pytorch import ToTensorV2
import tifffile as tiff

class Mine_Dataset(data.Dataset):
    def __init__(self, crop_size=512, seed=None, type=None, height_mask_out=False):
        
        self.type = type
        self.crop_size = crop_size
        self.height_mask_out = height_mask_out
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if self.type == 'train':
            self.imglist = os.listdir("/data1/Miningset/image_v2/train/")
        else:
            self.imglist = os.listdir("/data1/Miningset/image_v2/test/")

        self.imglist.sort()
        random.shuffle(self.imglist)

        self.transform = A.Compose([
            A.Resize(crop_size,crop_size),
            # A.RandomCrop(width=crop_size, height=crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])

    def __getitem__(self, index):
        if self.type == 'train':
            img = tiff.imread(os.path.join("/data1/Miningset/image_v2/train/", self.imglist[index]))
            label = tiff.imread(os.path.join("/data1/Miningset/annotation_v2/train/", self.imglist[index].replace('.tif', '_mask.tif')))//255
        else:
            img = tiff.imread(os.path.join("/data1/Miningset/image_v2/test/", self.imglist[index]))
            label = tiff.imread(os.path.join("/data1/Miningset/annotation_v2/test/", self.imglist[index].replace('.tif', '_mask.tif')))//255

        if self.height_mask_out:
            # 计算距离变换，得到每个前景像素到背景的距离
            dist_transform = cv2.distanceTransform(label, cv2.DIST_L2, 5)
            # dist_transform = cv2.distanceTransform(label, cv2.DIST_L1, 5)
            # 归一化距离变换结果，使其在0到1之间
            dist_transform_normalized = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
            # 矿山高度模拟：背景高度为1，矿山高度根据距离变换结果递减
            # 离边界越远，高度越低，最低为0
            height_label = 1 - dist_transform_normalized
            # 将背景区域的高度设置为1
            height_label[label == 0] = 1
            if self.type == 'train':
                augmented = self.transform(image=img, masks=[label, height_label])
                img = augmented['image']
                labels = augmented['masks']
                labels = {'label':torch.tensor(labels[0]).float().unsqueeze(0), 'height_label':torch.tensor(labels[1]).float().unsqueeze(0)}
                # if label.max()==1:
                #     import matplotlib.pyplot as plt
                #     import numpy as np
                #     # 创建一个画布，分为 1 行 3 列
                #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                #     # 可视化原始影像
                #     axes[0].imshow(np.transpose(img, (1, 2, 0)))
                #     axes[0].set_title('Original Image')
                #     axes[0].axis('off')  # 关闭坐标轴

                #     # 可视化前景二值掩膜
                #     axes[1].imshow(labels['label'][0], cmap='gray')
                #     axes[1].set_title('Binary Mask (Label)')
                #     axes[1].axis('off')  # 关闭坐标轴

                #     # 可视化高度标签
                #     height_map = axes[2].imshow(labels['height_label'][0], cmap='viridis')  # 使用颜色映射
                #     axes[2].set_title('Height Label')
                #     axes[2].axis('off')  # 关闭坐标轴

                #     # 添加颜色条
                #     fig.colorbar(height_map, ax=axes[2], fraction=0.046, pad=0.04)

                #     # 调整布局
                #     plt.tight_layout()

                #     # 保存图像到指定路径
                #     output_path = '/data1/yry22/temp/Mine_rsl/code/temp.png'
                #     plt.savefig(output_path, bbox_inches='tight', dpi=300)
                
                return img, labels

        if self.type == 'train':
            augmented = self.transform(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask'].float().unsqueeze(0)
        else:
            # resized = A.Resize(self.crop_size,self.crop_size)(image=img, mask=label)
            # img=resized['image']
            # label = resized['mask']
            img = A.Normalize(mean=self.mean, std=self.std)(image=img)['image']
            img = ToTensorV2()(image=img)['image']
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        if self.type == 'train':
            return img, label
        else:
            return img, label, self.imglist[index]

    def __len__(self):
        return len(self.imglist)