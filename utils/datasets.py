import glob
import random
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    """
    功能：将img图像使用pad_value值填充，填充到（ max(w, h) ， max(w, h))大小
    :param img:
    :param pad_value:
    :return:
    """
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)  # （w, w, h, h)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    """
    功能：将image resize 到（size， size）大小
    :param image:
    :param size:
    :return:
    """
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        # self.img_file 图片路径
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        # self.label_files 标签路径
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

        # self.img_size 图片大小
        self.img_size = img_size
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        # self.max_objects
        self.max_objects = 100
        # self.augment 数据增强
        self.augment = augment
        # self.multiscale 多尺度
        self.multiscale = multiscale
        # self.normalized_labels 标签格式是否标准化标签的
        self.normalized_labels = normalized_labels
        # self.batch_count  batch计数，在多尺度时使用
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        # 获取图片路径img_path
        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor （Image转成tensor格式）
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution，（填充像素）
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        # 标签路径 label_path
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            # 获取一张图片中，多个bbox
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

            # Extract coordinates for unpadded + unscaled image，对未padding图片 解压 标签坐标
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2) # 左上角x1
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2) # 左上角y1
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2) # 右下角x2
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2) # 右下角y2

            # Adjust for added padding
            x1 += pad[0]    # 扩充左上角x1
            y1 += pad[2]    # 扩充左上角y1
            x2 += pad[1]    # 扩充左上角x2
            y2 += pad[3]    # 扩充左上角y2

            # Returns (x, y, w, h)，返回标准化的（cx,cy,w,h), 这几个数都是0，1之间的
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            # 扩充boxs<N,5>为target<N,6>, 其中，6比5多个位，表示batchsize中某个样本的下表
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0) # target type is tensor， shape（119，6），例如：[[0,...],[1, ...], ..., [19,...]

        # Selects new image size every tenth batch, 如果self.multiscale为真，且batch_count计数为10的倍数
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
