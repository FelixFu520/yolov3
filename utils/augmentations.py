import torch


def horisontal_flip(images, targets):
    """
    功能：水平反转图片
    :param images:
    :param targets:
    :return:
    """
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets
