import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QColorDialog, QLabel, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from gui.canvas import ImageCanvas
from core.solver import ColorizationSolver


# ==========================================
# 定义后台工作线程 (Worker Thread)
# ==========================================
class ColorizationWorker(QThread):
    # 定义两个信号：一个用于传输结果(numpy数组)，一个用于传输错误信息(str)
    result_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, img_path, mask_img):
        super().__init__()
        self.img_path = img_path
        self.mask_img = mask_img

    def run(self):
        """线程入口函数"""
        try:
            # 初始化求解器
            solver = ColorizationSolver(self.img_path)

            # 执行计算 (这是最耗时的步骤)
            # 注意：mask_img 是我们在 canvas 里画出来的 BGR 图片
            result_bgr = solver.solve(self.mask_img)

            # 发送结果信号
            self.result_ready.emit(result_bgr)

        except Exception as e:
            # 捕获所有算法异常，发回主界面显示
            import traceback
            traceback.print_exc()  # 在控制台打印详细报错
            self.error_occurred.emit(str(e))


# ==========================================
# 主窗口逻辑
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python Colorization App (User-Guided)")
        self.setGeometry(100, 100, 1200, 800)

        # 核心数据
        self.current_img_path = None
        self.current_result_img = None  # 保存计算结果，用于 Save 功能
        self.worker = None  # 以此保持对线程的引用

        self._init_ui()

    def _init_ui(self):
        """初始化界面布局 (与之前基本一致)"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # --- 画板区域 ---
        canvas_layout = QHBoxLayout()

        self.input_canvas = ImageCanvas()
        self.input_canvas.setStyleSheet("border: 2px dashed gray; background-color: #f0f0f0;")

        self.result_canvas = ImageCanvas()
        self.result_canvas.setStyleSheet("border: 2px solid gray; background-color: #f0f0f0;")
        self.result_canvas.setMouseTracking(False)
        self.result_canvas.setEnabled(False)  # 结果区不可交互

        canvas_layout.addWidget(self.input_canvas, stretch=1)
        canvas_layout.addWidget(self.result_canvas, stretch=1)
        main_layout.addLayout(canvas_layout, stretch=1)

        # --- 工具栏 ---
        toolbar_layout = QHBoxLayout()

        self.btn_load = QPushButton("📂 Load Image")
        self.btn_load.clicked.connect(self.on_load_image)

        self.btn_color = QPushButton("🎨 Pick Color")
        self.btn_color.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        self.btn_color.clicked.connect(self.on_pick_color)

        self.btn_run = QPushButton("🚀 Run Colorization")
        self.btn_run.clicked.connect(self.on_run)
        self.btn_run.setEnabled(False)

        self.btn_save = QPushButton("💾 Save Result")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_save.setEnabled(False)

        # 按钮高度
        for btn in [self.btn_load, self.btn_color, self.btn_run, self.btn_save]:
            btn.setMinimumHeight(40)

        toolbar_layout.addWidget(self.btn_load)
        toolbar_layout.addWidget(self.btn_color)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.btn_run)
        toolbar_layout.addWidget(self.btn_save)

        main_layout.addLayout(toolbar_layout)

        # 状态栏
        self.status_label = QLabel("Ready. Please load a B&W image.")
        self.statusBar().addWidget(self.status_label)

    # === 事件处理 (Slots) ===

    def on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            success = self.input_canvas.load_image(path)
            if success:
                self.current_img_path = path
                self.btn_run.setEnabled(True)
                self.status_label.setText(f"Loaded: {os.path.basename(path)}")

                # 重置状态
                self.result_canvas.clear()
                self.current_result_img = None
                self.btn_save.setEnabled(False)

    def on_pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.input_canvas.set_brush_color(color.name())
            # 更新按钮背景色，字体根据亮度反色
            text_color = "black" if color.lightness() > 128 else "white"
            self.btn_color.setStyleSheet(f"background-color: {color.name()}; color: {text_color}; font-weight: bold;")

    def on_run(self):
        if not self.current_img_path: return

        # 获取用户涂鸦后的 Mask
        mask_img = self.input_canvas.get_mask()

        # UI 状态更新
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Running... (Please Wait)")
        self.status_label.setText("Solving linear equations... This may take a few seconds.")
        self.input_canvas.setEnabled(False)  # 暂时禁止涂鸦

        # 启动后台线程
        self.worker = ColorizationWorker(self.current_img_path, mask_img)
        self.worker.result_ready.connect(self.on_result_ready)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.start()

    def on_result_ready(self, result_img):
        """当线程计算完成时被调用"""
        # 保存结果数据
        self.current_result_img = result_img

        # 显示结果
        self.result_canvas.update_display_from_cv(result_img)

        # 恢复 UI 状态
        self.btn_run.setEnabled(True)
        self.btn_run.setText("🚀 Run Colorization")
        self.btn_save.setEnabled(True)
        self.input_canvas.setEnabled(True)
        self.status_label.setText("Done! Time to save.")

        # 释放线程
        self.worker = None

    def on_worker_error(self, error_msg):
        """当线程出错时被调用"""
        QMessageBox.critical(self, "Error", f"Algorithm failed:\n{error_msg}")
        self.btn_run.setEnabled(True)
        self.btn_run.setText("🚀 Run Colorization")
        self.input_canvas.setEnabled(True)
        self.status_label.setText("Error occurred.")
        self.worker = None

    def on_save(self):
        if self.current_result_img is None: return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "result.png", "Images (*.png *.jpg)")
        if save_path:
            # OpenCV 保存是 BGR 格式，result_img 也是 BGR，直接保存即可
            cv2.imwrite(save_path, self.current_result_img)
            self.status_label.setText(f"Saved to: {save_path}")
            QMessageBox.information(self, "Success", "Image saved successfully!")