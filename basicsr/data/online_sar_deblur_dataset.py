import sys
import cv2
import math
import numpy as np
import random
import os
import torch
from torch.utils import data as data
from basicsr.data.transforms import augment
sys.path.append(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from basicsr.utils import FileClient, imfrombytes, scandir
from basicsr.utils.registry import DATASET_REGISTRY
from typing import Optional, Dict, List, Tuple
from basicsr.utils import FileClient, imfrombytes, img2tensor

def bivariate_Gaussian(kernel_size: int, sig_x: float, sig_y: float, theta: float, grid: Optional[np.ndarray] = None, isotropic: bool = True) -> np.ndarray:
    if grid is None: grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]]) if isotropic else sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    if np.sum(kernel) == 0:
        kernel = np.zeros((kernel_size, kernel_size)); kernel[kernel_size // 2, kernel_size // 2] = 1.0
        return kernel
    return kernel / np.sum(kernel)

def bivariate_generalized_Gaussian(kernel_size: int, sig_x: float, sig_y: float, theta: float, beta: float, grid: Optional[np.ndarray] = None, isotropic: bool = True) -> np.ndarray:
    if grid is None: grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]]) if isotropic else sigma_matrix2(sig_x, sig_y, theta)
    try: inverse_sigma = np.linalg.inv(sigma_matrix)
    except np.linalg.LinAlgError: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
    exponent_base = np.sum(np.dot(grid, inverse_sigma) * grid, 2)
    exponent_base[exponent_base < 1e-10] = 1e-10
    try: powered_term = np.power(exponent_base, beta)
    except ValueError: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
    kernel = np.exp(-0.5 * powered_term)
    if np.sum(kernel) == 0: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
    return kernel / np.sum(kernel)

def bivariate_plateau(kernel_size: int, sig_x: float, sig_y: float, theta: float, beta: float, grid: Optional[np.ndarray] = None, isotropic: bool = True) -> np.ndarray:
    if grid is None: grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]]) if isotropic else sigma_matrix2(sig_x, sig_y, theta)
    try: inverse_sigma = np.linalg.inv(sigma_matrix)
    except np.linalg.LinAlgError: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
    exponent_base = np.sum(np.dot(grid, inverse_sigma) * grid, 2)
    exponent_base[exponent_base < 1e-10] = 1e-10
    try: powered_term = np.power(exponent_base, beta)
    except ValueError: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
    kernel = np.reciprocal(powered_term + 1)
    if np.sum(kernel) == 0: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
    return kernel / np.sum(kernel)

def sigma_matrix2(sig_x: float, sig_y: float, theta: float) -> np.ndarray:
    sig_x, sig_y = max(sig_x, 1e-6), max(sig_y, 1e-6)
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

def mesh_grid(kernel_size: int):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack([xx.reshape((-1, 1)), yy.reshape((-1, 1))]).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy

def pdf2(sigma_matrix: np.ndarray, grid: np.ndarray) -> np.ndarray:
    try: inverse_sigma = np.linalg.inv(sigma_matrix)
    except np.linalg.LinAlgError: return np.zeros_like(grid[:,:,0])
    return np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))

def random_bivariate_Gaussian(kernel_size: int, sigma_x_range: List[float], sigma_y_range: List[float], rotation_range: List[float], isotropic: bool = True) -> np.ndarray:
    assert kernel_size % 2 == 1
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    sigma_y, rotation = (sigma_x, 0) if isotropic else (np.random.uniform(sigma_y_range[0], sigma_y_range[1]), np.random.uniform(rotation_range[0], rotation_range[1]))
    return bivariate_Gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic)

def random_bivariate_generalized_Gaussian(kernel_size: int, sigma_x_range: List[float], sigma_y_range: List[float], rotation_range: List[float], beta_range: List[float], isotropic: bool = True) -> np.ndarray:
    assert kernel_size % 2 == 1
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    sigma_y, rotation = (sigma_x, 0) if isotropic else (np.random.uniform(sigma_y_range[0], sigma_y_range[1]), np.random.uniform(rotation_range[0], rotation_range[1]))
    beta = np.random.uniform(beta_range[0], beta_range[1])
    return bivariate_generalized_Gaussian(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)

def random_bivariate_plateau(kernel_size: int, sigma_x_range: List[float], sigma_y_range: List[float], rotation_range: List[float], beta_range: List[float], isotropic: bool = True) -> np.ndarray:
    assert kernel_size % 2 == 1
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    sigma_y, rotation = (sigma_x, 0) if isotropic else (np.random.uniform(sigma_y_range[0], sigma_y_range[1]), np.random.uniform(rotation_range[0], rotation_range[1]))
    beta = np.random.uniform(beta_range[0], beta_range[1])
    return bivariate_plateau(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)

@DATASET_REGISTRY.register()
class OnlineSAR_Deblur_Dataset(data.Dataset):
    """
    一个执行“在线”单次退化模糊生成的PyTorch数据集类，适配BasicSR框架。
    """

    def __init__(self, opt):
        super(OnlineSAR_Deblur_Dataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.is_train = 'train' in opt['name'].lower()
        self.kernel_options = opt['kernel_options']
        self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))
        assert self.paths, f'No images found in {self.gt_folder}'
        
        self.dataset_name = self.opt['name']
        self.dataset_unique_id = abs(hash(self.dataset_name)) % (10**8)

        self.save_lq_folder = opt.get('save_lq_folder', None)
        if self.save_lq_folder:
            os.makedirs(self.save_lq_folder, exist_ok=True)
            print(f"INFO: LQ images from dataset '{self.dataset_name}' will be saved to: {self.save_lq_folder}")

        print(f"Dataset '{self.dataset_name}' initialized in {'TRAIN (Random Blur)' if self.is_train else 'VAL/TEST (Reproducible & Isolated Blur)'} mode.")

        kernel_list_info = self.kernel_options.get('kernel_list')
        kernel_prob_info = self.kernel_options.get('kernel_prob')
        if kernel_prob_info:
            print(f"[INFO] Kernel info for '{self.dataset_name}': {kernel_list_info} with prob {kernel_prob_info}")
        else:
            print(f"[INFO] Kernel for '{self.dataset_name}': '{kernel_list_info}'")

        self.debug_printed_indices = set()

    def _generate_random_kernel(self):
        # (此函数内部逻辑无需修改)
        ko = self.kernel_options
        kernel_size = random.choice(range(ko['kernel_size_range'][0], ko['kernel_size_range'][1] + 1, 2))
        
        if isinstance(ko.get('kernel_list'), str):
            kernel_type = ko['kernel_list']
        else:
            kernel_type = np.random.choice(ko['kernel_list'], p=ko['kernel_prob'])

        isotropic = ko.get('isotropic_options', {}).get(kernel_type, False)
        
        try:
            if 'gaussian' in kernel_type.lower() or 'aniso' in kernel_type.lower():
                return random_bivariate_Gaussian(kernel_size, ko['sigma_x_range'], ko['sigma_y_range'], ko['rotation_range'], isotropic=isotropic)
            elif 'generalized' in kernel_type.lower():
                return random_bivariate_generalized_Gaussian(kernel_size, ko['sigma_x_range'], ko['sigma_y_range'], ko['rotation_range'], ko['betag_range'], isotropic=isotropic)
            else: # 'plateau'
                return random_bivariate_plateau(kernel_size, ko['sigma_x_range'], ko['sigma_y_range'], ko['rotation_range'], ko['betap_range'], isotropic=isotropic)
        except Exception as e:
            # 这里的错误回退逻辑在您的代码中可能存在变量未定义的bug，已修正
            print(f"Kernel generation failed: {e}. Falling back to default Gaussian.")
            return bivariate_Gaussian(21, 3, 3, 0, isotropic=True)


    def __getitem__(self, index):
        if not self.is_train:
            seed = index + self.dataset_unique_id
            np.random.seed(seed)
            random.seed(seed)

        gt_path = self.paths[index]
        img_bytes = FileClient().get(gt_path, 'gt')
        img_gt_numpy = imfrombytes(img_bytes, flag='grayscale', float32=True)
        
        kernel = self._generate_random_kernel()
        img_lq_numpy = cv2.filter2D(img_gt_numpy, -1, kernel)

        if self.save_lq_folder:
            basename = os.path.basename(gt_path)
            save_path = os.path.join(self.save_lq_folder, basename)
            img_lq_to_save = np.uint8((img_lq_numpy.clip(0, 1) * 255.).round())
            cv2.imwrite(save_path, img_lq_to_save)
            
        # img_bytes = FileClient().get(img_lq_to_save, 'gt')
        # img_lq1_numpy = imfrombytes(img_bytes, flag='grayscale', float32=True)
        
        img_gt_numpy_expanded = np.expand_dims(img_gt_numpy, axis=2)
        img_lq_numpy_expanded = np.expand_dims(img_lq_numpy, axis=2)

        if self.is_train and (self.opt.get('use_flip', False) or self.opt.get('use_rot', False)):

            img_gt_numpy_expanded, img_lq_numpy_expanded = augment([img_gt_numpy_expanded, img_lq_numpy_expanded], self.opt['use_flip'], self.opt['use_rot'])
        
        img_gt_tensor = img2tensor(img_gt_numpy_expanded, bgr2rgb=False, float32=True)
        img_lq_tensor = img2tensor(img_lq_numpy_expanded, bgr2rgb=False, float32=True)


        return {
            'lq': img_lq_tensor, 
            'gt': img_gt_tensor,
            'lq_path': gt_path, 
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
