import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random

from fastcore.basics import store_attr


class ExtraTransforms:
    """Additional transforms based on kaggle discussion.
    https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065#latest-624210
    """

    def __init__(self, contrast_range=0.1, brightness_range=0.1, hue_range=0.1,
                 saturation_range=0.1, sharpness_factor=0.1, blur_sigma=0.1, kernel_size=1,
                 rot_angles=np.arange(0, 100, 20), do_mirror=True):
        contrast_range = random.uniform(1 - contrast_range, 1 + contrast_range)
        brightness_range = random.uniform(1 - brightness_range, 1 + brightness_range)
        hue_range = random.uniform(-hue_range, hue_range)
        saturation_range = random.uniform(1 - saturation_range, 1 + saturation_range)
        sharpness_factor = random.uniform(1 - sharpness_factor, 1 + sharpness_factor)
        rot_angle = float(random.choice(rot_angles))
        store_attr('contrast_range,brightness_range,hue_range,saturation_range, \
                   blur_sigma, kernel_size, sharpness_factor,rot_angle,do_mirror')

    def __call__(self, x):
        x = TF.adjust_contrast(x, self.contrast_range)
        x = TF.adjust_brightness(x, self.brightness_range)
        x = TF.adjust_hue(x, self.hue_range)
        x = TF.adjust_saturation(x, self.saturation_range)
        x = TF.adjust_sharpness(x, self.sharpness_factor)
        x = TF.gaussian_blur(x, self.kernel_size, self.blur_sigma)
        x = TF.rotate(x, self.rot_angle)
        if self.do_mirror:
            x = TF.hflip(x)
        return x


def train_augs(resize_sz, mean, std):
    """Training augmentation"""
    return T.Compose([
        ExtraTransforms(),
        T.ToTensor(),
        T.Resize(resize_sz),
        T.Normalize(mean=mean, std=std), ])


def valid_augs(resize_sz, mean, std):
    """Validation augmentation"""
    return T.Compose([
        T.ToTensor(),
        T.Resize(resize_sz),
        T.Normalize(mean=mean, std=std), ])
