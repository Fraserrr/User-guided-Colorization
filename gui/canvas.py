from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QRect
import cv2


# 继承 QWidget 而不是 QLabel，因为要自己画图
class ImageCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(False)
        self.drawing = False
        self.brush_size = 3
        self.brush_color = QColor("#faad14")  # 默认颜色

        # 核心数据
        self.original_cv_img = None  # 原始 CV2 图像 (BGR)
        self.mask_img = None  # 涂鸦记录层 (BGR)
        self.display_pixmap = None  # 当前用于显示的 Qt 图像缓存

        # 告诉布局管理器：不要管我的建议尺寸，有多少空间就给我多少
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def load_image(self, img_path):
        """加载图片"""
        self.original_cv_img = cv2.imread(img_path)
        if self.original_cv_img is None:
            return False

        # 初始化 Mask (拷贝一份原图)
        self.mask_img = self.original_cv_img.copy()

        # 更新显示缓存
        self.update_display_from_cv(self.mask_img)
        return True

    def update_display_from_cv(self, cv_img):
        """更新显示缓存 (不直接操作界面，只更新数据)"""
        if cv_img is None: return

        # CV2 (BGR) -> Qt (RGB)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, c = rgb_img.shape
        bytes_per_line = rgb_img.strides[0]

        # 创建 QImage (使用 .copy() 防止内存崩溃)
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self.display_pixmap = QPixmap.fromImage(q_img)

        # 触发界面重绘 (调用 paintEvent)
        self.update()

    def set_brush_color(self, color_hex):
        """设置画笔颜色"""
        self.brush_color = QColor(color_hex)

    def paintEvent(self, event):
        """
        所有的绘制都发生在这里
        Qt 会自动处理窗口大小变化，只需要把图片画在正中间即可
        """
        painter = QPainter(self)
        # 图片周围的填充背景
        painter.fillRect(self.rect(), QColor("#ffffff"))

        if self.display_pixmap and not self.display_pixmap.isNull():
            # 计算图片应该画在哪 (保持比例居中)
            target_rect = self._get_displayed_rect()

            # 绘制图片
            # PyQt 会自动处理缩放采样，Result: 平滑且不闪烁
            painter.drawPixmap(target_rect, self.display_pixmap)

    def _get_displayed_rect(self):
        """计算图片在窗口中的实际显示区域 (Rect)"""
        if not self.display_pixmap: return QRect()

        W_widget, H_widget = self.width(), self.height()
        W_img, H_img = self.display_pixmap.width(), self.display_pixmap.height()

        # 计算缩放比例 (KeepAspectRatio)
        ratio_w = W_widget / W_img
        ratio_h = H_widget / H_img
        scale = min(ratio_w, ratio_h)

        # 计算显示尺寸
        display_w = int(W_img * scale)
        display_h = int(H_img * scale)

        # 计算居中偏移
        x = (W_widget - display_w) // 2
        y = (H_widget - display_h) // 2

        return QRect(x, y, display_w, display_h)

    # ================= 鼠标事件 =================

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.original_cv_img is not None:
            self.drawing = True
            self.last_point = self._map_to_image_coords(event.pos())

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing and self.original_cv_img is not None:
            current_point = self._map_to_image_coords(event.pos())

            # 在数据层画线 (OpenCV)
            cv2.line(self.mask_img, self.last_point, current_point,
                     (self.brush_color.blue(), self.brush_color.green(), self.brush_color.red()),
                     self.brush_size * 2)

            # 更新显示缓存 & 触发重绘
            self.update_display_from_cv(self.mask_img)

            self.last_point = current_point

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def _map_to_image_coords(self, widget_pos):
        """将鼠标坐标映射回原图像素坐标"""
        if not self.display_pixmap: return (0, 0)

        # 获取当前图片在窗口里的矩形区域
        rect = self._get_displayed_rect()

        # 减去偏移量
        x_rel = widget_pos.x() - rect.x()
        y_rel = widget_pos.y() - rect.y()

        # 除以缩放比例
        scale = rect.width() / self.display_pixmap.width()
        if scale == 0: return (0, 0)

        x_img = int(x_rel / scale)
        y_img = int(y_rel / scale)

        # 边界限制 (防止越界)
        h_orig, w_orig, _ = self.original_cv_img.shape
        x_img = max(0, min(x_img, w_orig - 1))
        y_img = max(0, min(y_img, h_orig - 1))

        return (x_img, y_img)

    def get_mask(self):
        """获取用于 Solver 的标记图"""
        return self.mask_img

    def clear(self):
        """清空画布 (用于重置)"""
        self.original_cv_img = None
        self.mask_img = None
        self.display_pixmap = None
        self.update()
        self.setText("")

    def setText(self, text):
        pass  # QWidget 没有 setText，留空兼容