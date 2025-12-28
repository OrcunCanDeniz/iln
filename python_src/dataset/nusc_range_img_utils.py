
import os
import PIL
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder, DatasetFolder
import torch

import torch.utils.data as data
import torch.nn.functional as F
from pyquaternion import Quaternion
import mmcv
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from PIL import Image

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.dataset import ImageDataset
import numpy as np

import os
import os.path
import random, math
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
import pdb
from pathlib import Path
import quaternion  # pip install numpy-quaternion

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.jpx')
NPY_EXTENSIONS = ('.npy', '.rimg', '.bin')
dataset_list = {}

from dataset.dataset_utils import register_dataset

ELEV_DEG_PER_RING_NUCSENES = np.array([-30.67, -29.33, -28., -26.66, -25.33, -24., -22.67, -21.33,
                               -20., -18.67, -17.33, -16., -14.67, -13.33, -12., -10.67,
                                -9.33, -8., -6.66, -5.33, -4., -2.67, -1.33, 0.,
                                1.33, 2.67, 4., 5.33, 6.67, 8., 9.33, 10.67], dtype=np.float32)
ELEV_DEG_PER_RING_NUCSENES_RAD = np.deg2rad(ELEV_DEG_PER_RING_NUCSENES)

offset_lut = np.array([48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0])

azimuth_lut = np.array([4.23,1.43,-1.38,-4.18,4.23,1.43,-1.38,-4.18,4.24,1.43,-1.38,-4.18,4.24,1.42,-1.38,-4.19,4.23,1.43,-1.38,-4.19,4.23,1.43,-1.39,-4.19,4.23,1.42,-1.39,-4.2,4.23,1.43,-1.39,-4.19,4.23,1.42,-1.4,-4.2,4.23,1.42,-1.4,-4.2,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.39,-4.2,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.41,-4.21,4.22,1.41,-1.41,-4.21,4.21,1.4,-1.41,-4.21,4.21,1.41,-1.41,-4.21,4.22,1.41,-1.42,-4.22,4.22,1.4,-1.41,-4.22,4.21,1.41,-1.42,-4.22,4.22,1.4,-1.41,-4.22,4.21,1.4,-1.41,-4.23,4.21,1.4,-1.42,-4.23,4.21,1.4,-1.42,-4.22,4.21,1.39,-1.42,-4.22,4.21,1.4,-1.42,-4.21,4.21,1.4,-1.42,-4.22,4.2,1.4,-1.41,-4.22,4.2,1.4,-1.42,-4.22,4.2,1.4,-1.42,-4.22])

elevation_lut = np.array([21.42,21.12,20.81,20.5,20.2,19.9,19.58,19.26,18.95,18.65,18.33,18.02,17.68,17.37,17.05,16.73,16.4,16.08,15.76,15.43,15.1,14.77,14.45,14.11,13.78,13.45,13.13,12.79,12.44,12.12,11.77,11.45,11.1,10.77,10.43,10.1,9.74,9.4,9.06,8.72,8.36,8.02,7.68,7.34,6.98,6.63,6.29,5.95,5.6,5.25,4.9,4.55,4.19,3.85,3.49,3.15,2.79,2.44,2.1,1.75,1.38,1.03,0.68,0.33,-0.03,-0.38,-0.73,-1.07,-1.45,-1.8,-2.14,-2.49,-2.85,-3.19,-3.54,-3.88,-4.26,-4.6,-4.95,-5.29,-5.66,-6.01,-6.34,-6.69,-7.05,-7.39,-7.73,-8.08,-8.44,-8.78,-9.12,-9.45,-9.82,-10.16,-10.5,-10.82,-11.19,-11.52,-11.85,-12.18,-12.54,-12.87,-13.2,-13.52,-13.88,-14.21,-14.53,-14.85,-15.2,-15.53,-15.84,-16.16,-16.5,-16.83,-17.14,-17.45,-17.8,-18.11,-18.42,-18.72,-19.06,-19.37,-19.68,-19.97,-20.31,-20.61,-20.92,-21.22])

origin_offset = 0.015806

lidar_to_sensor_z_offset = 0.03618

angle_off = math.pi * 4.2285/180.

def npy_loader(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        range_intensity_map = np.load(f)
        range_map = range_intensity_map[..., 0]
    return range_map.astype(np.float32)

def img_to_pcd_nuscenes(range_img,
                        maximum_range: float = 100.0,
                        flip_vertical: bool = True,
                        eval: bool = True) -> np.ndarray:
    """
    Simplest consistent inverse for HDL-32E NuScenes range images.
    range_img: (H, W) 32x1024
    flip_vertical: wheter the range image top is highest elevation
    eval: is this being called for evaluation? if so denormalize the range image from 0-1 to 0-maximum_range
    """
    if range_img.ndim == 3 and range_img.shape[2] == 1:
        range_img = range_img[..., 0]
    H, W = range_img.shape

    # Valid pixels
    R = range_img.astype(np.float32, copy=True)
    if eval:
        R *= maximum_range # eval 
    rr = np.repeat(np.arange(H), W) # [000000,111111,222222,333333,444444,555555,666666,777777]
    cc = np.tile(np.arange(W), H) # [01234567,01234567,01234567,01234567,01234567,01234567,01234567,01234567]

    # Undo vertical flip from forward pass if needed
    row_idx = (H - 1 - rr) if flip_vertical else rr

    # Elevation directly from row index
    el = ELEV_DEG_PER_RING_NUCSENES_RAD[row_idx]
    az = ((cc.astype(np.float32) + 0.5) / W) * (2*np.pi) - np.pi
    r = R[rr, cc]

    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)

    pc = np.stack([x, y, z], axis=1)
    return pc.astype(np.float32)