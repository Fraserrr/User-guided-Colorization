import numpy as np
import cv2
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve


class ColorizationSolver:
    def __init__(self, original_image_path):
        """
        初始化求解器
        :param original_image_path: 原始黑白图片的路径
        """
        # 读取原始图片
        raw_img = cv2.imread(original_image_path)
        if raw_img is None:
            raise ValueError(f"无法读取图片: {original_image_path}")

        # 强制转为黑白 (3通道)
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        self.orig_bgr_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 记录原图的真实高分辨率尺寸
        self.orig_rows, self.orig_cols = self.orig_bgr_img.shape[:2]

    def solve(self, marked_img_bgr):
        """
        核心求解函数 (结合 Levin 原版扩散 与 降采样加速)
        :param marked_img_bgr: 用户涂鸦后的图片 (BGR格式)
        :return: 上色完成的 BGR 图片
        """
        # ==========================================
        # 必须在高分辨率下计算 Mask
        # 防止因降采样插值导致的背景差异被误认为是灰色涂鸦
        # ==========================================
        diff = np.abs(self.orig_bgr_img.astype(np.int16) - marked_img_bgr.astype(np.int16))
        # 生成二值化的原始 Mask
        is_colored_orig = (np.sum(diff, axis=2) > 10).astype(np.uint8) * 255

        # 设定最大计算分辨率阈值 (例如长边最大 400 像素)
        max_dim = 400
        scale_factor = 1.0

        if max(self.orig_rows, self.orig_cols) > max_dim:
            scale_factor = max_dim / max(self.orig_rows, self.orig_cols)
            new_w = int(self.orig_cols * scale_factor)
            new_h = int(self.orig_rows * scale_factor)

            # 降采样图片
            work_bgr = cv2.resize(self.orig_bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # 涂鸦图和 Mask 必须使用最近邻插值，保证颜色纯度
            work_marked = cv2.resize(marked_img_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            work_mask = cv2.resize(is_colored_orig, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # 根据准确缩放后的 Mask 确定着色点
            is_colored = work_mask > 0
        else:
            work_bgr = self.orig_bgr_img.copy()
            work_marked = marked_img_bgr.copy()
            is_colored = is_colored_orig > 0

        # 提取工作分辨率下的 YUV 和 Y 通道
        work_yuv = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2YUV)
        work_Y = work_yuv[:, :, 0].astype(np.float64) / 255.0
        work_rows, work_cols = work_Y.shape
        num_pixels = work_rows * work_cols

        # 提取用户指定的 U, V 值
        marked_yuv = cv2.cvtColor(work_marked, cv2.COLOR_BGR2YUV)
        U_marked = marked_yuv[:, :, 1].astype(np.float64) / 255.0
        V_marked = marked_yuv[:, :, 2].astype(np.float64) / 255.0

        # 构建稀疏线性方程组 Ax = b (使用原版 Levin 全局方差逻辑)
        A = self._get_laplacian_matrix(work_Y)

        # 应用约束条件
        colored_indices = np.nonzero(is_colored.ravel())[0]

        lambda_const = 1e4
        D = sparse.lil_matrix((num_pixels, num_pixels))
        D[colored_indices, colored_indices] = 1
        D = D.tocsc()

        A_final = A + lambda_const * D

        # 分别求解 U 和 V 通道
        b_u = np.zeros(num_pixels)
        b_v = np.zeros(num_pixels)

        flat_U = U_marked.ravel()
        flat_V = V_marked.ravel()

        b_u[colored_indices] = flat_U[colored_indices] * lambda_const
        b_v[colored_indices] = flat_V[colored_indices] * lambda_const

        print(f"正在求解线性方程组 (工作分辨率: {work_cols}x{work_rows})...")
        new_u = spsolve(A_final, b_u)
        new_v = spsolve(A_final, b_v)
        print("求解完成!")

        # 将一维结果转回二维
        work_U_2d = new_u.reshape(work_rows, work_cols)
        work_V_2d = new_v.reshape(work_rows, work_cols)

        # 升采样色度，并与原图无损高频亮度融合
        if scale_factor < 1.0:
            final_U = cv2.resize(work_U_2d, (self.orig_cols, self.orig_rows), interpolation=cv2.INTER_CUBIC)
            final_V = cv2.resize(work_V_2d, (self.orig_cols, self.orig_rows), interpolation=cv2.INTER_CUBIC)
        else:
            final_U = work_U_2d
            final_V = work_V_2d

        # 合成最终图像
        orig_yuv = cv2.cvtColor(self.orig_bgr_img, cv2.COLOR_BGR2YUV)
        result_yuv = np.zeros_like(orig_yuv)

        # 【画质核心】原图的高分辨率 Y (亮度纹理) 保持绝对不变
        result_yuv[:, :, 0] = orig_yuv[:, :, 0]
        result_yuv[:, :, 1] = (final_U * 255).clip(0, 255).astype(np.uint8)
        result_yuv[:, :, 2] = (final_V * 255).clip(0, 255).astype(np.uint8)

        # 转回 BGR 用于显示
        result_bgr = cv2.cvtColor(result_yuv, cv2.COLOR_YUV2BGR)
        return result_bgr

    def _get_laplacian_matrix(self, Y_channel):
        """
        构建稀疏权重矩阵 (回归 Levin et al. 原版逻辑)
        """
        rows, cols = Y_channel.shape
        num_pixels = rows * cols

        window_size = 1
        inds_M = np.arange(num_pixels).reshape((rows, cols))

        data = []
        row_inds = []
        col_inds = []

        # 遍历 3x3 邻域 (8 个邻居)
        for dy in range(-window_size, window_size + 1):
            for dx in range(-window_size, window_size + 1):
                if dx == 0 and dy == 0:
                    continue

                row_start, row_end = max(0, -dy), min(rows, rows - dy)
                col_start, col_end = max(0, -dx), min(cols, cols - dx)

                center_inds = inds_M[row_start:row_end, col_start:col_end]
                neighbor_inds = inds_M[row_start + dy: row_end + dy, col_start + dx: col_end + dx]

                Y_center = Y_channel[row_start:row_end, col_start:col_end]
                Y_neighbor = Y_channel[row_start + dy: row_end + dy, col_start + dx: col_end + dx]

                # 回退到原始的方差计算逻辑
                diff = Y_center - Y_neighbor
                variance = np.mean(diff ** 2)
                if variance < 1e-6:
                    variance = 1e-6

                # 原始权重公式
                weights = np.exp(- (diff ** 2) / (2 * variance))

                data.append(-weights.flatten())
                row_inds.append(center_inds.flatten())
                col_inds.append(neighbor_inds.flatten())

        data = np.concatenate(data)
        row_inds = np.concatenate(row_inds)
        col_inds = np.concatenate(col_inds)

        A = sparse.coo_matrix((data, (row_inds, col_inds)), shape=(num_pixels, num_pixels))
        A = A.tocsr()

        sum_rows = -np.array(A.sum(axis=1)).flatten()
        diag = sparse.diags(sum_rows)
        A = A + diag

        return A