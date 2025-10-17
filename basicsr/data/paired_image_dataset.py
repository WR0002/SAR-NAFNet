# # # ------------------------------------------------------------------------
# # # Copyright (c) 2022 megvii-model. All Rights Reserved.
# # # ------------------------------------------------------------------------
# # # Modified from BasicSR (https://github.com/xinntao/BasicSR)
# # # Copyright 2018-2020 BasicSR Authors
# # # ------------------------------------------------------------------------
# # from torch.utils import data as data
# # from torchvision.transforms.functional import normalize

# # from basicsr.data.data_util import (paired_paths_from_folder,
# #                                     paired_paths_from_lmdb,
# #                                     paired_paths_from_meta_info_file)
# # from basicsr.data.transforms import augment, paired_random_crop
# # from basicsr.utils import FileClient, imfrombytes, img2tensor, padding


# # class PairedImageDataset(data.Dataset):
# #     """Paired image dataset for image restoration.

# #     Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
# #     GT image pairs.

# #     There are three modes:
# #     1. 'lmdb': Use lmdb files.
# #         If opt['io_backend'] == lmdb.
# #     2. 'meta_info_file': Use meta information file to generate paths.
# #         If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
# #     3. 'folder': Scan folders to generate paths.
# #         The rest.

# #     Args:
# #         opt (dict): Config for train datasets. It contains the following keys:
# #             dataroot_gt (str): Data root path for gt.
# #             dataroot_lq (str): Data root path for lq.
# #             meta_info_file (str): Path for meta information file.
# #             io_backend (dict): IO backend type and other kwarg.
# #             filename_tmpl (str): Template for each filename. Note that the
# #                 template excludes the file extension. Default: '{}'.
# #             gt_size (int): Cropped patched size for gt patches.
# #             use_flip (bool): Use horizontal flips.
# #             use_rot (bool): Use rotation (use vertical flip and transposing h
# #                 and w for implementation).

# #             scale (bool): Scale, which will be added automatically.
# #             phase (str): 'train' or 'val'.
# #     """

# #     def __init__(self, opt):
# #         super(PairedImageDataset, self).__init__()
# #         self.opt = opt
# #         # file client (io backend)
# #         self.file_client = None
# #         self.io_backend_opt = opt['io_backend']
# #         self.mean = opt['mean'] if 'mean' in opt else None
# #         self.std = opt['std'] if 'std' in opt else None

# #         self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
# #         if 'filename_tmpl' in opt:
# #             self.filename_tmpl = opt['filename_tmpl']
# #         else:
# #             self.filename_tmpl = '{}'

# #         if self.io_backend_opt['type'] == 'lmdb':
# #             self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
# #             self.io_backend_opt['client_keys'] = ['lq', 'gt']
# #             self.paths = paired_paths_from_lmdb(
# #                 [self.lq_folder, self.gt_folder], ['lq', 'gt'])
# #         elif 'meta_info_file' in self.opt and self.opt[
# #                 'meta_info_file'] is not None:
# #             self.paths = paired_paths_from_meta_info_file(
# #                 [self.lq_folder, self.gt_folder], ['lq', 'gt'],
# #                 self.opt['meta_info_file'], self.filename_tmpl)
# #         else:
# #             self.paths = paired_paths_from_folder(
# #                 [self.lq_folder, self.gt_folder], ['lq', 'gt'],
# #                 self.filename_tmpl)

# #     def __getitem__(self, index):
# #         if self.file_client is None:
# #             self.file_client = FileClient(
# #                 self.io_backend_opt.pop('type'), **self.io_backend_opt)

# #         scale = self.opt['scale']

# #         # Load gt and lq images. Dimension order: HWC; channel order: BGR;
# #         # image range: [0, 1], float32.
# #         gt_path = self.paths[index]['gt_path']
# #         # print('gt path,', gt_path)
# #         img_bytes = self.file_client.get(gt_path, 'gt')
# #         try:
# #             img_gt = imfrombytes(img_bytes, float32=True)
# #         except:
# #             raise Exception("gt path {} not working".format(gt_path))

# #         lq_path = self.paths[index]['lq_path']
# #         # print(', lq path', lq_path)
# #         img_bytes = self.file_client.get(lq_path, 'lq')
# #         try:
# #             img_lq = imfrombytes(img_bytes, float32=True)
# #         except:
# #             raise Exception("lq path {} not working".format(lq_path))


# #         # augmentation for training
# #         if self.opt['phase'] == 'train':
# #             gt_size = self.opt['gt_size']
# #             # padding
# #             img_gt, img_lq = padding(img_gt, img_lq, gt_size)

# #             # random crop
# #             img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
# #                                                 gt_path)
# #             # flip, rotation
# #             img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
# #                                      self.opt['use_rot'])

# #         # TODO: color space transform
# #         # BGR to RGB, HWC to CHW, numpy to tensor
# #         img_gt, img_lq = img2tensor([img_gt, img_lq],
# #                                     bgr2rgb=True,
# #                                     float32=True)
# #         # normalize
# #         if self.mean is not None or self.std is not None:
# #             normalize(img_lq, self.mean, self.std, inplace=True)
# #             normalize(img_gt, self.mean, self.std, inplace=True)

# #         return {
# #             'lq': img_lq,
# #             'gt': img_gt,
# #             'lq_path': lq_path,
# #             'gt_path': gt_path
# #         }

# #     def __len__(self):
# #         return len(self.paths)
# # ------------------------------------------------------------------------
# # Copyright (c) 2022 megvii-model. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Modified from BasicSR (https://github.com/xinntao/BasicSR  )
# # Copyright 2018-2020 BasicSR Authors
# # ------------------------------------------------------------------------
# from torch.utils import data as data
# from torchvision.transforms.functional import normalize

# from basicsr.data.data_util import (paired_paths_from_folder,
#                                     paired_paths_from_lmdb,
#                                     paired_paths_from_meta_info_file)
# from basicsr.data.transforms import augment, paired_random_crop
# from basicsr.utils import FileClient, imfrombytes, img2tensor, padding


# class PairedImageDataset(data.Dataset):
#     """Paired image dataset for image restoration.

#     Read LQ (Low Quality) and GT (Ground Truth) image pairs.
#     特别适配单通道灰度图（如 SAR 图像），不会转为三通道。

#     Args:
#         opt (dict): Config for train/test datasets. It contains the following keys:
#             dataroot_gt (str): Data root path for gt images.
#             dataroot_lq (str): Data root path for lq images.
#             meta_info_file (str): Path for meta information file (optional).
#             io_backend (dict): IO backend type and other kwargs.
#             filename_tmpl (str): Template for each filename. Default: '{}'.
#             gt_size (int): Cropped patch size for gt patches.
#             use_flip (bool): Use horizontal flips (default: False for test).
#             use_rot (bool): Use rotation (default: False for test).
#             scale (int): Scale factor (default: 1 for deblur).
#             phase (str): 'train' or 'val'.
#     """

#     def __init__(self, opt):
#         super(PairedImageDataset, self).__init__()
#         self.opt = opt
#         # file client (io backend)
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']
#         self.mean = opt['mean'] if 'mean' in opt else None
#         self.std = opt['std'] if 'std' in opt else None

#         self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
#         if 'filename_tmpl' in opt:
#             self.filename_tmpl = opt['filename_tmpl']
#         else:
#             self.filename_tmpl = '{}'

#         # 根据 io_backend 类型加载路径
#         if self.io_backend_opt['type'] == 'lmdb':
#             self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
#             self.io_backend_opt['client_keys'] = ['lq', 'gt']
#             self.paths = paired_paths_from_lmdb(
#                 [self.lq_folder, self.gt_folder], ['lq', 'gt'])
#         elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
#             self.paths = paired_paths_from_meta_info_file(
#                 [self.lq_folder, self.gt_folder], ['lq', 'gt'],
#                 self.opt['meta_info_file'], self.filename_tmpl)
#         else:
#             self.paths = paired_paths_from_folder(
#                 [self.lq_folder, self.gt_folder], ['lq', 'gt'],
#                 self.filename_tmpl)

#     def __getitem__(self, index):
#         if self.file_client is None:
#             self.file_client = FileClient(
#                 self.io_backend_opt.pop('type'), **self.io_backend_opt)

#         scale = self.opt['scale']

#         # Load gt and lq images
#         gt_path = self.paths[index]['gt_path']
#         img_bytes = self.file_client.get(gt_path, 'gt')
#         try:
#             # ✅ 强制以灰度模式读取，返回 (H, W)
#             img_gt = imfrombytes(img_bytes, float32=True, flag='grayscale')
#         except Exception as e:
#             raise Exception(f"Failed to read GT image {gt_path}: {e}")

#         lq_path = self.paths[index]['lq_path']
#         img_bytes = self.file_client.get(lq_path, 'lq')
#         try:
#             # ✅ 强制以灰度模式读取，返回 (H, W)
#             img_lq = imfrombytes(img_bytes, float32=True, flag='grayscale')
#         except Exception as e:
#             raise Exception(f"Failed to read LQ image {lq_path}: {e}")

#         # augmentation for training (test 时通常不启用)
#         if self.opt['phase'] == 'train':
#             gt_size = self.opt['gt_size']
#             # padding
#             img_gt, img_lq = padding(img_gt, img_lq, gt_size)
#             # random crop
#             img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
#             # flip, rotation
#             img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

#         # BGR to RGB, HWC to CHW, numpy to tensor
#         # ✅ 关键：bgr2rgb=False，因为是单通道，无需颜色转换
#         img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
#         # normalize
#         if self.mean is not None or self.std is not None:
#             normalize(img_lq, self.mean, self.std, inplace=True)
#             normalize(img_gt, self.mean, self.std, inplace=True)

#         return {
#             'lq': img_lq,      # (1, H, W)
#             'gt': img_gt,      # (1, H, W)
#             'lq_path': lq_path,
#             'gt_path': gt_path
#         }

#     def __len__(self):
#         return len(self.paths)

# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
# ✨ This version is specifically adapted for single-channel (grayscale) image restoration tasks.
# ========================================================================
import cv2 # 导入 cv2 用于可能的调试
import numpy as np # 导入 numpy
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                     paired_paths_from_lmdb,
                                     paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
from basicsr.utils.registry import DATASET_REGISTRY # ✨ 引入注册器

@DATASET_REGISTRY.register() # ✨ 使用注册器，方便YML文件通过类名调用
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality) and GT (Ground Truth) image pairs.
    ✨ 特别适配单通道灰度图（如 SAR 图像），确保数据以单通道形式处理。

    Args:
        opt (dict): Config for train/test datasets.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None) # ✨ 使用 .get() 增加健壮性
        self.std = opt.get('std', None)

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}') # ✨ 使用 .get()

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt.get('meta_info_file') is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt.get('scale', 1)

        # ----------------- Load gt and lq images -----------------
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        
        # ✨ 核心修改 1: 强制以单通道灰度图模式读取 ✨
        # imfrombytes 会自动将 uint8 [0,255] 转为 float32 [0,1]
        img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        
        # ✨ 核心修改 1: 同样强制以单通道灰度图模式读取 ✨
        img_lq = imfrombytes(img_bytes, flag='grayscale', float32=True)

        # ----------------- augmentation for training -----------------
        if self.opt['phase'] == 'train':
            gt_size = self.opt.get('gt_size')
            if gt_size is not None:
                # padding
                img_gt, img_lq = padding(img_gt, img_lq, gt_size)
                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            
            # flip, rotation
            img_gt, img_lq = augment(
                [img_gt, img_lq], self.opt.get('use_flip', False), self.opt.get('use_rot', False))

        # ----------------- HWC to CHW, numpy to tensor -----------------
        # ✨ 核心修改 2: 明确告知 img2tensor 输入的是单通道数据，无需进行 BGR 到 RGB 的转换 ✨
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                      bgr2rgb=False,
                                      float32=True)
                                      
        # ----------------- normalize -----------------
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,      # 最终 shape: (1, H, W)
            'gt': img_gt,      # 最终 shape: (1, H, W)
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)