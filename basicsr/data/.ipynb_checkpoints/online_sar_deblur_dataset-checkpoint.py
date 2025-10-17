# # basicsr/data/online_sar_dataset.py
# import sys
# import cv2
# import math
# import numpy as np
# import random
# import os
# import torch
# from torch.utils import data as data
# from basicsr.data.transforms import augment
# sys.path.append(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
# from basicsr.utils import FileClient, imfrombytes, scandir
# from basicsr.utils.registry import DATASET_REGISTRY
# from typing import Optional, Dict, List, Tuple
# from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir

# # =======================================================================================
# # == 区域一: 模糊核生成函数库 ==
# # == 说明: 这部分包含了所有模糊核的数学定义和随机生成函数，是在线退化的基础。==
# # =======================================================================================
# def bivariate_Gaussian(kernel_size: int, sig_x: float, sig_y: float, theta: float, grid: Optional[np.ndarray] = None, isotropic: bool = True) -> np.ndarray:
#     """生成一个双变量高斯核。"""
#     if grid is None: grid, _, _ = mesh_grid(kernel_size)
#     sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]]) if isotropic else sigma_matrix2(sig_x, sig_y, theta)
#     kernel = pdf2(sigma_matrix, grid)
#     if np.sum(kernel) == 0:
#         kernel = np.zeros((kernel_size, kernel_size)); kernel[kernel_size // 2, kernel_size // 2] = 1.0
#         return kernel
#     return kernel / np.sum(kernel)

# def bivariate_generalized_Gaussian(kernel_size: int, sig_x: float, sig_y: float, theta: float, beta: float, grid: Optional[np.ndarray] = None, isotropic: bool = True) -> np.ndarray:
#     """生成一个双变量广义高斯核。"""
#     if grid is None: grid, _, _ = mesh_grid(kernel_size)
#     sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]]) if isotropic else sigma_matrix2(sig_x, sig_y, theta)
#     try: inverse_sigma = np.linalg.inv(sigma_matrix)
#     except np.linalg.LinAlgError: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
#     exponent_base = np.sum(np.dot(grid, inverse_sigma) * grid, 2)
#     exponent_base[exponent_base < 1e-10] = 1e-10
#     try: powered_term = np.power(exponent_base, beta)
#     except ValueError: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
#     kernel = np.exp(-0.5 * powered_term)
#     if np.sum(kernel) == 0: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
#     return kernel / np.sum(kernel)

# def bivariate_plateau(kernel_size: int, sig_x: float, sig_y: float, theta: float, beta: float, grid: Optional[np.ndarray] = None, isotropic: bool = True) -> np.ndarray:
#     """生成一个平台型的双变量核。"""
#     if grid is None: grid, _, _ = mesh_grid(kernel_size)
#     sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]]) if isotropic else sigma_matrix2(sig_x, sig_y, theta)
#     try: inverse_sigma = np.linalg.inv(sigma_matrix)
#     except np.linalg.LinAlgError: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
#     exponent_base = np.sum(np.dot(grid, inverse_sigma) * grid, 2)
#     exponent_base[exponent_base < 1e-10] = 1e-10
#     try: powered_term = np.power(exponent_base, beta)
#     except ValueError: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
#     kernel = np.reciprocal(powered_term + 1)
#     if np.sum(kernel) == 0: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
#     return kernel / np.sum(kernel)

# def sigma_matrix2(sig_x: float, sig_y: float, theta: float) -> np.ndarray:
#     sig_x, sig_y = max(sig_x, 1e-6), max(sig_y, 1e-6)
#     d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
#     u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#     return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

# def mesh_grid(kernel_size: int):
#     ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
#     xx, yy = np.meshgrid(ax, ax)
#     xy = np.hstack([xx.reshape((-1, 1)), yy.reshape((-1, 1))]).reshape(kernel_size, kernel_size, 2)
#     return xy, xx, yy

# def pdf2(sigma_matrix: np.ndarray, grid: np.ndarray) -> np.ndarray:
#     try: inverse_sigma = np.linalg.inv(sigma_matrix)
#     except np.linalg.LinAlgError: return np.zeros_like(grid[:,:,0])
#     return np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))

# def random_bivariate_Gaussian(kernel_size: int, sigma_x_range: List[float], sigma_y_range: List[float], rotation_range: List[float], isotropic: bool = True) -> np.ndarray:
#     assert kernel_size % 2 == 1
#     sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
#     sigma_y, rotation = (sigma_x, 0) if isotropic else (np.random.uniform(sigma_y_range[0], sigma_y_range[1]), np.random.uniform(rotation_range[0], rotation_range[1]))
#     return bivariate_Gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic)

# def random_bivariate_generalized_Gaussian(kernel_size: int, sigma_x_range: List[float], sigma_y_range: List[float], rotation_range: List[float], beta_range: List[float], isotropic: bool = True) -> np.ndarray:
#     assert kernel_size % 2 == 1
#     sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
#     sigma_y, rotation = (sigma_x, 0) if isotropic else (np.random.uniform(sigma_y_range[0], sigma_y_range[1]), np.random.uniform(rotation_range[0], rotation_range[1]))
#     beta = np.random.uniform(beta_range[0], beta_range[1])
#     return bivariate_generalized_Gaussian(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)

# def random_bivariate_plateau(kernel_size: int, sigma_x_range: List[float], sigma_y_range: List[float], rotation_range: List[float], beta_range: List[float], isotropic: bool = True) -> np.ndarray:
#     assert kernel_size % 2 == 1
#     sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
#     sigma_y, rotation = (sigma_x, 0) if isotropic else (np.random.uniform(sigma_y_range[0], sigma_y_range[1]), np.random.uniform(rotation_range[0], rotation_range[1]))
#     beta = np.random.uniform(beta_range[0], beta_range[1])
#     return bivariate_plateau(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)


# # ============================================================================
# # == 区域二: 执行“在线单次退化”的Dataset类 (已合规化) ==
# # ============================================================================
# @DATASET_REGISTRY.register()
# class OnlineSAR_Deblur_Dataset(data.Dataset):
#     """
#     一个执行“在线”单次退化模糊生成的PyTorch数据集类，适配BasicSR框架。
#     保留您原有的模糊核生成逻辑，仅做合规化调整。
#     """

#     def __init__(self, opt):
#         super(OnlineSAR_Deblur_Dataset, self).__init__()
#         self.opt = opt
#         # 从配置文件(opt)中解析参数
#         self.gt_folder = opt['dataroot_gt']
#         self.is_train = 'train' in opt['name']
        
#         # 获取在线退化所需的模糊核参数
#         self.kernel_options = opt['kernel_options']
#         print(f"[DEBUG] Dataset {opt['name']} 使用的 kernel_list: {self.kernel_options['kernel_list']}")
#         # 扫描清晰图像文件夹，获取所有图像的路径
#         self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

#         # 断言：必须有路径
#         assert self.paths, f'No images found in {self.gt_folder}'

#     def _generate_random_kernel(self):
#         """辅助函数：根据配置生成一个随机模糊核。"""
#         ko = self.kernel_options
#         kernel_size = random.choice(range(ko['kernel_size_range'][0], ko['kernel_size_range'][1] + 1, 2))
#         kernel_type = np.random.choice(ko['kernel_list'], p=ko['kernel_prob'])
#         isotropic = ko.get('isotropic_options', {}).get(kernel_type, False)
#         try:
#             if 'Gaussian' in kernel_type: 
#                 return random_bivariate_Gaussian(kernel_size, ko['sigma_x_range'], ko['sigma_y_range'], ko['rotation_range'], isotropic=isotropic)
#             elif 'generalized' in kernel_type: 
#                 return random_bivariate_generalized_Gaussian(kernel_size, ko['sigma_x_range'], ko['sigma_y_range'], ko['rotation_range'], ko['betag_range'], isotropic=isotropic)
#             else: 
#                 return random_bivariate_plateau(kernel_size, ko['sigma_x_range'], ko['sigma_y_range'], ko['rotation_range'], ko['betap_range'], isotropic=isotropic)
#         except Exception: 
#             return bivariate_Gaussian(21, 3, 3, 0, isotropic=True)

#     def __getitem__(self, index):
#         """根据索引获取一个数据样本(清晰图, 模糊图)对。"""
#         # 1. 加载一张清晰图像 (HQ)
#         gt_path = self.paths[index]
#         img_bytes = FileClient().get(gt_path, 'gt')
        
#         # ✅ 正确加载单通道 SAR 图像
#         img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)  # (H, W)

#         # --- 核心逻辑: 在线生成模糊图像 (LQ) ---
#         # 2.1 生成一个全新的随机模糊核
#         kernel = self._generate_random_kernel()
        
#         # 2.2 应用卷积生成模糊图像
#         img_lq = cv2.filter2D(img_gt, -1, kernel)  # (H, W)
#         if img_gt.ndim == 2:
#             img_gt = np.expand_dims(img_gt, axis=2) # 从 (H, W) 变为 (H, W, 1)
#         if img_lq.ndim == 2:
#             img_lq = np.expand_dims(img_lq, axis=2) # 从 (H, W) 变为 (H, W, 1)
#         # 3. 数据增强 (使用BasicSR的augment函数)
#         if self.opt.get('use_flip', False) or self.opt.get('use_rot', False):
#             img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])
        
#         # 4. 转换为Tensor (使用 img2tensor)
#         img_gt = img2tensor(img_gt, bgr2rgb=False, float32=True)  # (1, H, W)
#         img_lq = img2tensor(img_lq, bgr2rgb=False, float32=True)  # (1, H, W)
        
#         return {
#             'lq': img_lq,
#             'gt': img_gt,
#             'lq_path': gt_path,
#             'gt_path': gt_path
#         }

#     def __len__(self):
#         """返回数据集中清晰图像的总数。"""
#         return len(self.paths)

# # ============================================================================
# # == 区域二: 执行“在线单次退化”的Dataset类 (修改后，支持测试模式) ==
# # ============================================================================
# # @DATASET_REGISTRY.register()
# # class OnlineSAR_Deblur_Dataset(data.Dataset):
# #     """
# #     一个执行“在线”单次退化模糊生成的PyTorch数据集类，适配BasicSR框架。
# #     ✨ 新增功能：在非训练模式下，模糊生成是可复现的。
# #     """

# #     def __init__(self, opt):
# #         super(OnlineSAR_Deblur_Dataset, self).__init__()
# #         self.opt = opt
# #         # 从配置文件(opt)中解析参数
# #         self.gt_folder = opt['dataroot_gt']
        
# #         # ✨ self.is_train 的判断逻辑非常关键
# #         self.is_train = 'train' in opt['name'].lower()
        
# #         # 获取在线退化所需的模糊核参数
# #         self.kernel_options = opt['kernel_options']
        
# #         # 扫描清晰图像文件夹，获取所有图像的路径
# #         self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

# #         # 断言：必须有路径
# #         assert self.paths, f'No images found in {self.gt_folder}'
        
# #         # ✨ 添加一个日志，清晰地告知当前数据集处于何种模式
# #         print(f"Dataset '{self.opt['name']}' initialized in {'TRAIN (Random Blur)' if self.is_train else 'VAL/TEST (Reproducible Blur)'} mode.")
# #         print(f"[INFO] Kernel List for '{self.opt['name']}': {self.kernel_options['kernel_list']} with prob {self.kernel_options['kernel_prob']}")


# #     def _generate_random_kernel(self):
# #         """辅助函数：根据配置生成一个随机模糊核。"""
# #         ko = self.kernel_options
# #         kernel_size = random.choice(range(ko['kernel_size_range'][0], ko['kernel_size_range'][1] + 1, 2))
# #         kernel_type = np.random.choice(ko['kernel_list'], p=ko['kernel_prob'])
# #         isotropic = ko.get('isotropic_options', {}).get(kernel_type, False)
        
# #         # ✨ 这里代码没变，但是调用它之前的随机种子状态决定了它的行为
# #         try:
# #             if 'Gaussian' in kernel_type or 'aniso' in kernel_type: # 兼容'aniso'命名
# #                 return random_bivariate_Gaussian(kernel_size, ko['sigma_x_range'], ko['sigma_y_range'], ko['rotation_range'], isotropic=isotropic)
# #             elif 'generalized' in kernel_type:
# #                 return random_bivariate_generalized_Gaussian(kernel_size, ko['sigma_x_range'], ko['sigma_y_range'], ko['rotation_range'], ko['betag_range'], isotropic=isotropic)
# #             else: # 'plateau'
# #                 # ✨ 修复了一个小笔误，原代码中 random_bivariate_plateau 误用了 sig_x, sig_y
# #                 sigma_x = np.random.uniform(ko['sigma_x_range'][0], ko['sigma_x_range'][1])
# #                 sigma_y, rotation = (sigma_x, 0) if isotropic else (np.random.uniform(ko['sigma_y_range'][0], ko['sigma_y_range'][1]), np.random.uniform(ko['rotation_range'][0], ko['rotation_range'][1]))
# #                 beta = np.random.uniform(ko['betap_range'][0], ko['betap_range'][1])
# #                 return bivariate_plateau(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)
# #         except Exception as e:
# #             print(f"Kernel generation failed: {e}. Falling back to default Gaussian.")
# #             return bivariate_Gaussian(21, 3, 3, 0, isotropic=True)

# #     def __getitem__(self, index):
# #         """根据索引获取一个数据样本(清晰图, 模糊图)对。"""

# #         # ✨✨✨ 核心修改点 ✨✨✨
# #         # 如果不是训练模式（即验证或测试），则根据图像索引设置随机种子。
# #         # 这能确保对于同一张图片(index相同)，每次生成的"随机"模糊核都是一样的。
# #         if not self.is_train:
# #             np.random.seed(index)
# #             random.seed(index)

# #         # 1. 加载一张清晰图像 (HQ)
# #         gt_path = self.paths[index]
# #         img_bytes = FileClient().get(gt_path, 'gt')
        
# #         # ✅ 正确加载单通道 SAR 图像
# #         img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)  # (H, W)

# #         # --- 核心逻辑: 在线生成模糊图像 (LQ) ---
# #         # 2.1 生成模糊核 (在测试模式下，这一步是可复现的)
# #         kernel = self._generate_random_kernel()
        
# #         # 2.2 应用卷积生成模糊图像
# #         img_lq = cv2.filter2D(img_gt, -1, kernel)  # (H, W)

# #         # 确保图像是 3D 的 (H, W, C)
# #         if img_gt.ndim == 2:
# #             img_gt = np.expand_dims(img_gt, axis=2) # 从 (H, W) 变为 (H, W, 1)
# #         if img_lq.ndim == 2:
# #             img_lq = np.expand_dims(img_lq, axis=2) # 从 (H, W) 变为 (H, W, 1)

# #         # 3. 数据增强 (在测试时，配置文件中应关闭)
# #         if self.is_train and (self.opt.get('use_flip', False) or self.opt.get('use_rot', False)):
# #             img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])
        
# #         # 4. 转换为Tensor (使用 img2tensor)
# #         img_gt = img2tensor(img_gt, bgr2rgb=False, float32=True)  # (1, H, W)
# #         img_lq = img2tensor(img_lq, bgr2rgb=False, float32=True)  # (1, H, W)
        
# #         return {
# #             'lq': img_lq,
# #             'gt': img_gt,
# #             'lq_path': gt_path,
# #             'gt_path': gt_path
# #         }

# #     def __len__(self):
# #         """返回数据集中清晰图像的总数。"""
# #         return len(self.paths)

# basicsr/data/online_sar_dataset.py (Diagnostic Version with Detailed Logging)
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

# =======================================================================================
# == 区域一: 模糊核生成函数库 (已修正BUG) ==
# =======================================================================================

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


# ============================================================================
# == 区域二: 执行“在线单次退化”的Dataset类 (增加诊断功能) ==
# ============================================================================
@DATASET_REGISTRY.register()
class OnlineSAR_Deblur_Dataset(data.Dataset):
    """
    一个执行“在线”单次退化模糊生成的PyTorch数据集类，适配BasicSR框架。
    ✨ 诊断版本：增加了详细的日志打印功能，用于检查前10个样本的数据状态。
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
        
        # 使日誌打印代碼更健壯
        kernel_list_info = self.kernel_options.get('kernel_list')
        kernel_prob_info = self.kernel_options.get('kernel_prob')
        if kernel_prob_info:
            print(f"[INFO] Kernel info for '{self.dataset_name}': {kernel_list_info} with prob {kernel_prob_info}")
        else:
            print(f"[INFO] Kernel for '{self.dataset_name}': '{kernel_list_info}'")
        
        # ✨ 新增：一个标志，用于确保只在第一个epoch打印详细信息
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
            # 注意：数据增强作用在Numpy数组上
            img_gt_numpy_expanded, img_lq_numpy_expanded = augment([img_gt_numpy_expanded, img_lq_numpy_expanded], self.opt['use_flip'], self.opt['use_rot'])
        
        img_gt_tensor = img2tensor(img_gt_numpy_expanded, bgr2rgb=False, float32=True)
        img_lq_tensor = img2tensor(img_lq_numpy_expanded, bgr2rgb=False, float32=True)
        
        # ✨✨✨ 核心修改：增加详细的诊断打印逻辑 ✨✨✨
        # 只对前10张图片打印，并且确保在多worker时不重复打印
#         if index < 10 and index not in self.debug_printed_indices:
#             self.debug_printed_indices.add(index) # 记录已打印的索引
            
#             print("\n" + "="*70)
#             print(f"--- 🔬 数据体检报告 for index: {index} (Dataset: {self.dataset_name}) 🔬 ---")
            
#             # --- 阶段一：在线生成的 Numpy 数组 ---
#             print("\n[阶段 1] 在线生成为 Numpy 数组后 (img_lq_numpy):")
#             print(f"  - 数据类型 (dtype): {img_lq_numpy.dtype}")
#             print(f"  - 形状 (shape):   {img_lq_numpy.shape}")
#             print(f"  - 最小值 (min):   {np.min(img_lq_numpy):.4f}")
#             print(f"  - 最大值 (max):   {np.max(img_lq_numpy):.4f}")
#             print(f"  - 平均值 (mean): {np.mean(img_lq_numpy):.4f}")

#             # --- 阶段二：最终转换为 PyTorch Tensor 后 ---
#             print("\n[阶段 2] 转换为 Tensor 后 (img_lq_tensor):")
#             print(f"  - 数据类型 (dtype): {img_lq_tensor.dtype}")
#             print(f"  - 形状 (shape):   {img_lq_tensor.shape}")
#             print(f"  - 最小值 (min):   {img_lq_tensor.min():.4f}")
#             print(f"  - 最大值 (max):   {img_lq_tensor.max():.4f}")
#             print(f"  - 平均值 (mean): {img_lq_tensor.mean():.4f}")
            
#             # --- 阶段三：打印 Tensor 的部分数值内容 ---
#             if index == 0:
#                 print("\n[阶段 3] 第一个样本 Tensor 左上角 5x5 具体数值:")
#                 print(img_lq_tensor[0, :5, :5])

#             print("="*70 + "\n")

        return {
            'lq': img_lq_tensor, 
            'gt': img_gt_tensor,
            'lq_path': gt_path, 
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

# basicsr/data/online_sar_dataset.py (Final Diagnostic Version)
# import sys
# import cv2
# import math
# import numpy as np
# import random
# import os
# import torch
# from torch.utils import data as data
# from basicsr.data.transforms import augment
# sys.path.append(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
# from basicsr.utils import FileClient, imfrombytes, scandir
# from basicsr.utils.registry import DATASET_REGISTRY
# from typing import Optional, Dict, List, Tuple
# from basicsr.utils import FileClient, imfrombytes, img2tensor
# from torchvision.utils import save_image

# # =======================================================================================
# # 区域一: 模糊核生成函数库 (功能完整，无需修改)
# # =======================================================================================

# def bivariate_Gaussian(kernel_size: int, sig_x: float, sig_y: float, theta: float, grid: Optional[np.ndarray] = None, isotropic: bool = True) -> np.ndarray:
#     if grid is None: grid, _, _ = mesh_grid(kernel_size)
#     sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]]) if isotropic else sigma_matrix2(sig_x, sig_y, theta)
#     kernel = pdf2(sigma_matrix, grid)
#     if np.sum(kernel) == 0:
#         kernel = np.zeros((kernel_size, kernel_size)); kernel[kernel_size // 2, kernel_size // 2] = 1.0
#         return kernel
#     return kernel / np.sum(kernel)

# def bivariate_generalized_Gaussian(kernel_size: int, sig_x: float, sig_y: float, theta: float, beta: float, grid: Optional[np.ndarray] = None, isotropic: bool = True) -> np.ndarray:
#     if grid is None: grid, _, _ = mesh_grid(kernel_size)
#     sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]]) if isotropic else sigma_matrix2(sig_x, sig_y, theta)
#     try: inverse_sigma = np.linalg.inv(sigma_matrix)
#     except np.linalg.LinAlgError: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
#     exponent_base = np.sum(np.dot(grid, inverse_sigma) * grid, 2)
#     exponent_base[exponent_base < 1e-10] = 1e-10
#     try: powered_term = np.power(exponent_base, beta)
#     except ValueError: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
#     kernel = np.exp(-0.5 * powered_term)
#     if np.sum(kernel) == 0: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
#     return kernel / np.sum(kernel)

# def bivariate_plateau(kernel_size: int, sig_x: float, sig_y: float, theta: float, beta: float, grid: Optional[np.ndarray] = None, isotropic: bool = True) -> np.ndarray:
#     if grid is None: grid, _, _ = mesh_grid(kernel_size)
#     sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]]) if isotropic else sigma_matrix2(sig_x, sig_y, theta)
#     try: inverse_sigma = np.linalg.inv(sigma_matrix)
#     except np.linalg.LinAlgError: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
#     exponent_base = np.sum(np.dot(grid, inverse_sigma) * grid, 2)
#     exponent_base[exponent_base < 1e-10] = 1e-10
#     try: powered_term = np.power(exponent_base, beta)
#     except ValueError: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
#     kernel = np.reciprocal(powered_term + 1)
#     if np.sum(kernel) == 0: return bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid, isotropic)
#     return kernel / np.sum(kernel)

# def sigma_matrix2(sig_x: float, sig_y: float, theta: float) -> np.ndarray:
#     sig_x, sig_y = max(sig_x, 1e-6), max(sig_y, 1e-6)
#     d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
#     u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#     return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

# def mesh_grid(kernel_size: int):
#     ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
#     xx, yy = np.meshgrid(ax, ax)
#     xy = np.hstack([xx.reshape((-1, 1)), yy.reshape((-1, 1))]).reshape(kernel_size, kernel_size, 2)
#     return xy, xx, yy

# def pdf2(sigma_matrix: np.ndarray, grid: np.ndarray) -> np.ndarray:
#     try: inverse_sigma = np.linalg.inv(sigma_matrix)
#     except np.linalg.LinAlgError: return np.zeros_like(grid[:,:,0])
#     return np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))

# def random_bivariate_Gaussian(kernel_size: int, sigma_x_range: List[float], sigma_y_range: List[float], rotation_range: List[float], isotropic: bool = True) -> np.ndarray:
#     assert kernel_size % 2 == 1
#     sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
#     sigma_y, rotation = (sigma_x, 0) if isotropic else (np.random.uniform(sigma_y_range[0], sigma_y_range[1]), np.random.uniform(rotation_range[0], rotation_range[1]))
#     return bivariate_Gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic)

# def random_bivariate_generalized_Gaussian(kernel_size: int, sigma_x_range: List[float], sigma_y_range: List[float], rotation_range: List[float], beta_range: List[float], isotropic: bool = True) -> np.ndarray:
#     assert kernel_size % 2 == 1
#     sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
#     sigma_y, rotation = (sigma_x, 0) if isotropic else (np.random.uniform(sigma_y_range[0], sigma_y_range[1]), np.random.uniform(rotation_range[0], rotation_range[1]))
#     beta = np.random.uniform(beta_range[0], beta_range[1])
#     return bivariate_generalized_Gaussian(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)

# def random_bivariate_plateau(kernel_size: int, sigma_x_range: List[float], sigma_y_range: List[float], rotation_range: List[float], beta_range: List[float], isotropic: bool = True) -> np.ndarray:
#     assert kernel_size % 2 == 1
#     sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
#     sigma_y, rotation = (sigma_x, 0) if isotropic else (np.random.uniform(sigma_y_range[0], sigma_y_range[1]), np.random.uniform(rotation_range[0], rotation_range[1]))
#     beta = np.random.uniform(beta_range[0], beta_range[1])
#     return bivariate_plateau(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)

# # ============================================================================
# # == 区域二: 执行“在线单次退化”的Dataset类 (增加差异分析功能) ==
# # ============================================================================
# @DATASET_REGISTRY.register()
# class OnlineSAR_Deblur_Dataset(data.Dataset):
#     """
#     一个执行“在线”单次退化模糊生成的PyTorch数据集类，适配BasicSR框架。
#     ✨ 诊断版本：增加了“保存-读取”差异分析功能。
#     """

#     def __init__(self, opt):
#         super(OnlineSAR_Deblur_Dataset, self).__init__()
#         self.opt = opt
#         self.gt_folder = opt['dataroot_gt']
#         self.is_train = 'train' in opt['name'].lower()
#         self.kernel_options = opt['kernel_options']
#         self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))
#         assert self.paths, f'No images found in {self.gt_folder}'
        
#         # ✨ 核心修正 1: 遵循标准实践，在 __init__ 中声明 self.file_client
#         self.file_client = None
#         # 从 YML 中读取 IO 后端配置
#         self.io_backend_opt = self.opt['io_backend']

#         self.dataset_name = self.opt['name']
#         self.dataset_unique_id = abs(hash(self.dataset_name)) % (10**8)

#         self.save_lq_folder = opt.get('save_lq_folder', None)
#         if self.save_lq_folder:
#             os.makedirs(self.save_lq_folder, exist_ok=True)
#             print(f"INFO: LQ images from dataset '{self.dataset_name}' will be saved to: {self.save_lq_folder}")

#         print(f"Dataset '{self.dataset_name}' initialized in {'TRAIN (Random Blur)' if self.is_train else 'VAL/TEST (Reproducible & Isolated Blur)'} mode.")
        
#         kernel_list_info = self.kernel_options.get('kernel_list')
#         kernel_prob_info = self.kernel_options.get('kernel_prob')
#         if kernel_prob_info:
#             print(f"[INFO] Kernel info for '{self.dataset_name}': {kernel_list_info} with prob {kernel_prob_info}")
#         else:
#             print(f"[INFO] Kernel for '{self.dataset_name}': '{kernel_list_info}'")
        
#         self.debug_printed_indices = set()

#     def _generate_random_kernel(self):
#         ko = self.kernel_options
#         kernel_size = random.choice(range(ko['kernel_size_range'][0], ko['kernel_size_range'][1] + 1, 2))
        
#         if isinstance(ko.get('kernel_list'), str):
#             kernel_type = ko['kernel_list']
#         else:
#             kernel_type = np.random.choice(ko['kernel_list'], p=ko['kernel_prob'])

#         isotropic = ko.get('isotropic_options', {}).get(kernel_type, False)
        
#         try:
#             if 'gaussian' in kernel_type.lower() or 'aniso' in kernel_type.lower():
#                 return random_bivariate_Gaussian(kernel_size, ko['sigma_x_range'], ko['sigma_y_range'], ko['rotation_range'], isotropic=isotropic)
#             elif 'generalized' in kernel_type.lower():
#                 return random_bivariate_generalized_Gaussian(kernel_size, ko['sigma_x_range'], ko['sigma_y_range'], ko['rotation_range'], ko['betag_range'], isotropic=isotropic)
#             else: # 'plateau'
#                 return random_bivariate_plateau(kernel_size, ko['sigma_x_range'], ko['sigma_y_range'], ko['rotation_range'], ko['betap_range'], isotropic=isotropic)
#         except Exception as e:
#             # ✨ BUG修复: 您的代码中，这里的 except 块存在逻辑问题，已修正
#             print(f"Kernel generation failed: {e}. Falling back to default Gaussian.")
#             return bivariate_Gaussian(21, 3, 3, 0, isotropic=True)

#     def __getitem__(self, index):
#         # ✨ 核心修正 2: 在 __getitem__ 开头使用标准方式延迟初始化 file_client
#         if self.file_client is None:
#             self.file_client = FileClient(
#                 self.io_backend_opt.pop('type'), **self.io_backend_opt)
                
#         if not self.is_train:
#             seed = index + self.dataset_unique_id
#             np.random.seed(seed)
#             random.seed(seed)

#         gt_path = self.paths[index]
#         # ✨ 核心修正 3: 使用 self.file_client 读取数据，而不是创建临时对象
#         img_bytes = self.file_client.get(gt_path, 'gt')
#         img_gt_numpy = imfrombytes(img_bytes, flag='grayscale', float32=True)
        
#         kernel = self._generate_random_kernel()
#         img_lq_numpy = cv2.filter2D(img_gt_numpy, -1, kernel)

#         # ✨✨✨ 差异分析与打印逻辑 ✨✨✨
#         # 仅在测试模式、配置了保存路径且是第一个worker处理第一个样本时执行
#         if not self.is_train and self.save_lq_folder and index == 0 and index not in self.debug_printed_indices:
#             self.debug_printed_indices.add(index)
            
#             print("\n" + "="*70)
#             print(f"--- 🔬 “保存-读取”差异分析 for index: {index} (Dataset: {self.dataset_name}) 🔬 ---")
            
#             basename = os.path.basename(gt_path)
#             save_path = os.path.join(self.save_lq_folder, basename)
#             #img_lq_to_save = np.uint8((img_lq_numpy.clip(0, 1) * 255.).round())
            
#             #cv2.imwrite(save_path, img_lq_to_save)
#             img_gt_tensor = img2tensor(np.expand_dims(img_gt_numpy, axis=2), bgr2rgb=False, float32=True)
#             img_lq_tensor = img2tensor(np.expand_dims(img_lq_numpy, axis=2), bgr2rgb=False, float32=True)
#             save_image(img_lq_tensor,save_path)
#             print(f"\n[步骤 1] 原始 float32 模糊图已保存为 uint8 图像至: {save_path}")

#             try:
#                 # 使用 self.file_client 从路径读取，确保IO后端与正常流程一致
#                 img_bytes_reread = self.file_client.get(save_path)
#                 img_lq1_numpy = imfrombytes(img_bytes_reread, flag='grayscale', float32=True)
#                 print(f"[步骤 2] 已从 {save_path} 通过 FileClient 重新读入图像。")
#             except Exception as e:
#                 print(f"[步骤 2] 重新读取图像时出错: {e}")
#                 img_lq1_numpy = None

#             print("\n--- 分析报告 ---")
            
#             if img_lq1_numpy is None:
#                 print("!! 无法进行差异分析，因为重新读取图像失败。")
#             elif img_lq_numpy.shape != img_lq1_numpy.shape:
#                 print("!! 错误：原始图像和读回的图像形状不匹配！")
#             else:
#                 diff_map = np.abs(img_lq_numpy - img_lq1_numpy)
#                 avg_diff = np.mean(diff_map)
#                 max_diff = np.max(diff_map)
#                 non_zero_pixels = np.count_nonzero(diff_map > 1e-6) # 增加一个小的容忍度
#                 total_pixels = diff_map.size
                
#                 mse = np.mean((img_lq_numpy - img_lq1_numpy) ** 2)
#                 if mse < 1e-10: # 避免除以零
#                     psnr_val = float('inf')
#                 else:
#                     psnr_val = 20 * math.log10(1.0 / math.sqrt(mse))

#                 print(f"原始内存中的模糊图 (lq) vs 重新读入的模糊图 (lq1):")
#                 print(f"  - 平均绝对像素差异: {avg_diff:.8f}")
#                 print(f"  - 最大绝对像素差异: {max_diff:.8f}")
#                 print(f"  - 存在差异的像素数: {non_zero_pixels} / {total_pixels} ({non_zero_pixels/total_pixels:.2%})")
#                 print(f"  - 两者间的PSNR (量化误差): {psnr_val:.2f} dB")

#             print("="*70 + "\n")

#         # --- 后续正常流程 ---
#         # img_gt_tensor = img2tensor(np.expand_dims(img_gt_numpy, axis=2), bgr2rgb=False, float32=True)
#         # img_lq_tensor = img2tensor(np.expand_dims(img_lq_numpy, axis=2), bgr2rgb=False, float32=True)
#         # save_image(lq_tensor,save_path)
#         return { 'lq': img_lq_tensor, 'gt': img_gt_tensor, 'lq_path': gt_path, 'gt_path': gt_path }

#     def __len__(self):
#         return len(self.paths)

