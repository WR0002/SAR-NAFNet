# # basicsr/data/paired_grayscale_dataset.py (A new, simplified, and robust version)

# import cv2
# import numpy as np
# import torch
# from torch.utils import data as data

# from basicsr.data.data_util import paired_paths_from_folder
# from basicsr.utils import FileClient, imfrombytes
# from basicsr.utils.registry import DATASET_REGISTRY

# @DATASET_REGISTRY.register()
# class PairedGrayscaleImageDataset(data.Dataset):
#     """
#     專為配對的、單通道灰度圖像（如SAR圖像）設計的Dataset類。
#     加載LQ（低質量）和GT（真值）圖像對。
#     流程經過簡化和優化，確保數據處理的正確性。
#     """

#     def __init__(self, opt):
#         super(PairedGrayscaleImageDataset, self).__init__()
#         self.opt = opt
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']

#         # 獲取LQ和GT文件夾路徑
#         self.gt_folder = opt['dataroot_gt']
#         self.lq_folder = opt['dataroot_lq']
        
#         # 獲取圖像路徑列表
#         # 確保您的 LQ 和 GT 圖像文件名完全一樣
#         self.paths = paired_paths_from_folder(
#             [self.lq_folder, self.gt_folder], ['lq', 'gt'],
#             '{}' # 文件名模板
#         )

#     def __getitem__(self, index):
#         if self.file_client is None:
#             self.file_client = FileClient(
#                 self.io_backend_opt.pop('type'), **self.io_backend_opt)

#         # ----------------- 讀取 GT 和 LQ 圖像 -----------------
#         gt_path = self.paths[index]['gt_path']
#         img_bytes = self.file_client.get(gt_path, 'gt')
#         # ✨ 關鍵點1: 明確以單通道灰度圖讀取，並轉為 float32
#         img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)

#         lq_path = self.paths[index]['lq_path']
#         img_bytes = self.file_client.get(lq_path, 'lq')
#         # ✨ 關鍵點1: 明確以單通道灰度圖讀取，並轉為 float32
#         img_lq = imfrombytes(img_bytes, flag='grayscale', float32=True)

#         # ----------------- 將 Numpy 轉換為 Tensor -----------------
#         # (H, W) -> (H, W, 1)
#         img_gt = np.expand_dims(img_gt, axis=2)
#         img_lq = np.expand_dims(img_lq, axis=2)

#         # (H, W, C) -> (C, H, W)
#         img_gt = torch.from_numpy(np.transpose(img_gt, (2, 0, 1))).float()
#         img_lq = torch.from_numpy(np.transpose(img_lq, (2, 0, 1))).float()

#         return {
#             'lq': img_lq,      # 最終 shape: (1, H, W)
#             'gt': img_gt,      # 最終 shape: (1, H, W)
#             'lq_path': lq_path,
#             'gt_path': gt_path
#         }

#     def __len__(self):
#         return len(self.paths)

# basicsr/data/paired_grayscale_dataset.py (Final Recommended Version)

# basicsr/data/paired_grayscale_dataset.py (Fixed Sorting Logic)

# import cv2
# import numpy as np
# import torch
# from torch.utils import data as data

# from basicsr.data.data_util import paired_paths_from_folder
# from basicsr.utils import FileClient, imfrombytes, img2tensor
# from basicsr.utils.registry import DATASET_REGISTRY

# @DATASET_REGISTRY.register()
# class PairedGrayscaleImageDataset(data.Dataset):
#     """
#     專為配對的、單通道灰度圖像（如SAR圖像）設計的Dataset類。
#     加載LQ（低質量）和GT（真值）圖像對。
#     流程經過簡化和優化，確保數據處理的正確性。
#     """

#     def __init__(self, opt):
#         super(PairedGrayscaleImageDataset, self).__init__()
#         self.opt = opt
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']

#         # 獲取LQ和GT文件夾路徑
#         self.gt_folder = opt['dataroot_gt']
#         self.lq_folder = opt['dataroot_lq']
        
#         # ✨✨✨ 核心修正 ✨✨✨
#         # 1. 首先，從文件夾獲取未排序的路徑列表
#         paths_list = paired_paths_from_folder(
#             [self.lq_folder, self.gt_folder], ['lq', 'gt'],
#             '{}' # 文件名模板
#         )
        
#         # 2. 然後，明確地根據 'gt_path' 的值對這個列表進行排序
#         #    key=lambda x: x['gt_path'] 的意思是：
#         #    对于列表中的每一个元素x（它是一个字典），都提取出 x['gt_path'] 的值，并以此为依据进行排序。
#         self.paths = sorted(paths_list, key=lambda x: x['gt_path'])
        

#     def __getitem__(self, index):
#         if self.file_client is None:
#             self.file_client = FileClient(
#                 self.io_backend_opt.pop('type'), **self.io_backend_opt)

#         # ----------------- 讀取 GT 和 LQ 圖像 -----------------
#         gt_path = self.paths[index]['gt_path']
#         img_bytes = self.file_client.get(gt_path, 'gt')
#         img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)

#         lq_path = self.paths[index]['lq_path']
#         img_bytes = self.file_client.get(lq_path, 'lq')
#         img_lq = imfrombytes(img_bytes, flag='grayscale', float32=True)

#         # ----------------- 將 Numpy 轉換為 Tensor -----------------
#         img_gt = np.expand_dims(img_gt, axis=2)
#         img_lq = np.expand_dims(img_lq, axis=2)

#         img_gt = torch.from_numpy(np.transpose(img_gt, (2, 0, 1))).float()
#         img_lq = torch.from_numpy(np.transpose(img_lq, (2, 0, 1))).float()

#         return {
#             'lq': img_lq,      # 最終 shape: (1, H, W)
#             'gt': img_gt,      # 最終 shape: (1, H, W)
#             'lq_path': lq_path,
#             'gt_path': gt_path
#         }

#     def __len__(self):
#         return len(self.paths)

# basicsr/data/paired_grayscale_dataset.py (Diagnostic Version with Detailed Logging)

import cv2
import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.utils import FileClient, imfrombytes
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class PairedGrayscaleImageDataset(data.Dataset):
    """
    专为配对的、单通道灰度图像设计的Dataset类。
    ✨ 诊断版本：增加了详细的日志打印功能，用于检查前10个样本的数据状态。
    """

    def __init__(self, opt):
        super(PairedGrayscaleImageDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        
        paths_list = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            '{}'
        )
        self.paths = sorted(paths_list, key=lambda x: x['gt_path'])
        
        print(f"--- Dataset '{self.opt['name']}' Initialized ---")
        print(f"Found {len(self.paths)} image pairs.")
        # ✨ 新增：一个标志，用于确保只在第一个epoch打印详细信息
        self.debug_printed_indices = set()


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # ----------------- 1. 读取 GT 和 LQ 图像 -----------------
        gt_path = self.paths[index]['gt_path']
        img_bytes_gt = self.file_client.get(gt_path, 'gt')
        # 强制以单通道灰度图读取，并转为 float32 [0, 1]
        img_gt_numpy = imfrombytes(img_bytes_gt, flag='grayscale', float32=True)

        lq_path = self.paths[index]['lq_path']
        img_bytes_lq = self.file_client.get(lq_path, 'lq')
        img_lq_numpy = imfrombytes(img_bytes_lq, flag='grayscale', float32=True)

        # ----------------- 2. 将 Numpy 转换为 Tensor -----------------
        img_gt_numpy_expanded = np.expand_dims(img_gt_numpy, axis=2)
        img_lq_numpy_expanded = np.expand_dims(img_lq_numpy, axis=2)

        img_gt_tensor = torch.from_numpy(np.transpose(img_gt_numpy_expanded, (2, 0, 1))).float()
        img_lq_tensor = torch.from_numpy(np.transpose(img_lq_numpy_expanded, (2, 0, 1))).float()

        # ✨✨✨ 核心修改：增加详细的诊断打印逻辑 ✨✨✨
        # 只对前10张图片打印，并且确保在多worker时不重复打印
#         if index < 10 and index not in self.debug_printed_indices:
#             self.debug_printed_indices.add(index) # 记录已打印的索引
            
#             print("\n" + "="*70)
#             print(f"--- 🔬 数据体检报告 for index: {index} 🔬 ---")
            
#             # --- 阶段一：刚从文件读取为 Numpy 数组后 ---
#             print("\n[阶段 1] 读取为 Numpy 数组后 (img_lq_numpy):")
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
#             # 只打印第一个样本的左上角 5x5 区域，以窥探具体数值
#             if index == 0:
#                 print("\n[阶段 3] 第一个样本 Tensor 左上角 5x5 具体数值:")
#                 print(img_lq_tensor[0, :5, :5])

#             print("="*70 + "\n")

        return {
            'lq': img_lq_tensor,
            'gt': img_gt_tensor,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)