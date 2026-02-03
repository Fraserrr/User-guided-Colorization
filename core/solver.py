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
        # 必须与 GUI 逻辑保持严格一致，否则后续计算涂鸦差异时，会因为背景像素值不匹配而导致算法失效。
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        self.bgr_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 转换为 YUV 颜色空间
        # 此时 U 和 V 通道理论上应该是 128 (灰色)，只用 Y 通道做亮度参考
        self.yuv_img = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2YUV)

        # 提取 Y 通道并归一化到 [0, 1]，用于计算权重
        self.Y = self.yuv_img[:, :, 0].astype(np.float64) / 255.0
        self.rows, self.cols = self.Y.shape
        self.num_pixels = self.rows * self.cols

    def solve(self, marked_img_bgr):
        """
        核心求解函数
        :param marked_img_bgr: 用户涂鸦后的图片 (BGR格式)
        :return: 上色完成的 BGR 图片
        """
        # 1. 找出用户涂色的像素位置 (Mask)
        # 如果像素的 RGB 值与原图差异较大，说明被用户画过了
        # 设定一个阈值，避免 JPEG 压缩噪声的干扰
        diff = np.abs(self.bgr_img.astype(np.int16) - marked_img_bgr.astype(np.int16))
        is_colored = np.sum(diff, axis=2) > 10  # 形状 (rows, cols)

        # 2. 将涂鸦图也转为 YUV，提取用户指定的 U, V 值
        marked_yuv = cv2.cvtColor(marked_img_bgr, cv2.COLOR_BGR2YUV)
        U_marked = marked_yuv[:, :, 1].astype(np.float64) / 255.0
        V_marked = marked_yuv[:, :, 2].astype(np.float64) / 255.0

        # 3. 构建稀疏线性方程组 Ax = b
        # A 是 (N, N) 的稀疏矩阵，N 是像素总数
        # b 是 (N, ) 的向量

        # 获取邻居索引和权重矩阵
        # 这是一个计算密集型操作，核心逻辑封装在 _get_weights 中
        A = self._get_laplacian_matrix()

        # 4. 应用约束条件 (Constraint)
        # 对于被用户涂色的像素，修改 A 和 b，强制 x 等于用户指定的值
        # 找到被涂色像素的一维索引
        colored_indices = np.nonzero(is_colored.ravel())[0]

        # 构建对角约束矩阵 (Diagonal Constraint Matrix)
        # 这种方法比修改 A 的行更高效
        # 系统变为: (A + lambda * D) * x = (lambda * D * values)
        # lambda 是一个很大的权重，保证约束被满足
        lambda_const = 1e4
        D = sparse.lil_matrix((self.num_pixels, self.num_pixels))
        D[colored_indices, colored_indices] = 1
        D = D.tocsc()

        # 最终的系数矩阵
        A_final = A + lambda_const * D

        # 5. 分别求解 U 和 V 通道
        # 这里的 b 向量只在被涂色的位置有值 (用户指定的颜色 * lambda)
        b_u = np.zeros(self.num_pixels)
        b_v = np.zeros(self.num_pixels)

        flat_U = U_marked.ravel()
        flat_V = V_marked.ravel()

        b_u[colored_indices] = flat_U[colored_indices] * lambda_const
        b_v[colored_indices] = flat_V[colored_indices] * lambda_const

        print("正在求解线性方程组 (这可能需要几秒钟)...")
        # 使用 scipy 的稀疏求解器
        new_u = spsolve(A_final, b_u)
        new_v = spsolve(A_final, b_v)
        print("求解完成!")

        # 6. 合成最终图像
        # 将求出的 1D 数组 reshape 回 2D 图像
        result_yuv = np.zeros_like(self.yuv_img)
        result_yuv[:, :, 0] = self.yuv_img[:, :, 0]  # 原图的 Y (亮度) 保持不变
        result_yuv[:, :, 1] = (new_u.reshape(self.rows, self.cols) * 255).clip(0, 255).astype(np.uint8)
        result_yuv[:, :, 2] = (new_v.reshape(self.rows, self.cols) * 255).clip(0, 255).astype(np.uint8)

        # 转回 BGR 用于显示
        result_bgr = cv2.cvtColor(result_yuv, cv2.COLOR_YUV2BGR)
        return result_bgr

    def _get_laplacian_matrix(self):
        """
        构建稀疏权重矩阵 (Levin et al.)
        逻辑：如果两个相邻像素的亮度(Y)相似，它们应该具有相似的颜色(U/V)。
        """
        # 使用 3x3 窗口 (neighbors)
        window_size = 1
        len_window = (2 * window_size + 1) ** 2

        num_pixels = self.num_pixels
        inds_M = np.arange(num_pixels).reshape((self.rows, self.cols))

        # 准备稀疏矩阵的数据容器
        # data: 权重值, row_inds: 行索引, col_inds: 列索引
        data = []
        row_inds = []
        col_inds = []

        # 遍历窗口内的每个邻居偏移量 (dx, dy)
        # 例如: (-1, -1), (-1, 0), ..., (1, 1)
        for dy in range(-window_size, window_size + 1):
            for dx in range(-window_size, window_size + 1):
                if dx == 0 and dy == 0:
                    continue

                # 找到当前邻居对应的像素索引
                # 通过切片操作实现快速向量化计算，避免慢速 for 循环

                # 确定中心像素的有效范围
                row_start, row_end = max(0, -dy), min(self.rows, self.rows - dy)
                col_start, col_end = max(0, -dx), min(self.cols, self.cols - dx)

                # 中心像素索引
                center_inds = inds_M[row_start:row_end, col_start:col_end]

                # 邻居像素索引
                neighbor_inds = inds_M[row_start + dy: row_end + dy, col_start + dx: col_end + dx]

                # 获取 Y 值
                Y_center = self.Y[row_start:row_end, col_start:col_end]
                Y_neighbor = self.Y[row_start + dy: row_end + dy, col_start + dx: col_end + dx]

                # 计算权重: w_ij = exp( - (Y_i - Y_j)^2 / (2 * sigma^2) )
                # 这里简化处理，直接使用方差的逆作为相关性
                # 实际 Levin 论文中使用了局部方差，这里为了速度使用全局参数或简化版
                # 只要亮度越接近，diff 越小，val 越大
                diff = Y_center - Y_neighbor
                variance = np.mean(diff ** 2)
                if variance < 1e-6: variance = 1e-6  # 防止除零

                weights = np.exp(- (diff ** 2) / (2 * variance))

                # 记录数据
                data.append(-weights.flatten())  # 注意是负数，移项后 Ax=b
                row_inds.append(center_inds.flatten())
                col_inds.append(neighbor_inds.flatten())

        # 将列表转换为 numpy 数组
        data = np.concatenate(data)
        row_inds = np.concatenate(row_inds)
        col_inds = np.concatenate(col_inds)

        # 构建稀疏矩阵 A
        # A[i, j] = -w_ij (如果 i, j 是邻居)
        A = sparse.coo_matrix((data, (row_inds, col_inds)), shape=(num_pixels, num_pixels))

        # 转换为 CSR 格式以便快速计算
        A = A.tocsr()

        # 填充对角线
        # A[i, i] = sum(w_ij)
        # 这样保证每一行的和为 0 (对于未约束点)
        sum_rows = -np.array(A.sum(axis=1)).flatten()
        diag = sparse.diags(sum_rows)
        A = A + diag

        return A