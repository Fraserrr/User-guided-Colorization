import os
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QColorDialog, QLabel, QMessageBox,
                             QFrame, QProgressBar, QStackedWidget, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from gui.canvas import ImageCanvas
from core.solver import ColorizationSolver
from PyQt5.QtCore import QThread


# ==========================================
# 核心工作线程
# ==========================================
class ColorizationWorker(QThread):
    result_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, img_path, mask_img):
        super().__init__()
        self.img_path = img_path
        self.mask_img = mask_img

    def run(self):
        try:
            solver = ColorizationSolver(self.img_path)
            result_bgr = solver.solve(self.mask_img)
            self.result_ready.emit(result_bgr)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))


# ==========================================
# UI 主窗口
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Easy Colorization --- by Fraserrr")
        self.resize(1300, 850)

        # 核心数据
        self.current_img_path = None
        self.current_result_img = None
        self.worker = None

        # 应用全局样式 (Modern Flat UI)
        self._apply_styles()

        # 初始化界面
        self._init_ui()

    def _apply_styles(self):
        """定义主题 QSS 样式表 (Light Mode)"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f4f6f9; /* 柔和的灰白背景 */
            }
            QWidget {
                color: #333333; /* 深灰字体，保证阅读对比度 */
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
            }
            /* 顶部工具栏容器 */
            QFrame#ToolbarFrame {
                background-color: #ffffff; /* 纯白工具栏 */
                border-bottom: 1px solid #dcdcdc; /* 浅灰分割线 */
            }
            /* 按钮通用样式 */
            QPushButton {
                background-color: #597ef7; /* 蓝 */
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2f54eb;
            }
            QPushButton:pressed {
                background-color: #2f54eb;
            }
            QPushButton:disabled {
                background-color: #e0e0e0;
                color: #a0a0a0;
            }
            /* 颜色选择按钮特殊样式 */
            QPushButton#BtnColor {
                border: 2px solid #e0e0e0; /* 浅灰边框 */
            }
            /* 画布容器 (卡片式设计) */
            QFrame#CanvasContainer {
                background-color: #ffffff; /* 纯白卡片 */
                border-radius: 10px;
                border: 1px solid #e0e0e0; /* 极其微弱的边框 */
            }
            QLabel#CanvasTitle {
                color: #888888; /* 浅灰标题 */
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 1px;
                padding-bottom: 10px;
                border-bottom: 1px solid #f0f0f0; /* 标题下方的分割线 */
                margin-bottom: 10px;
            }
            /* 进度条 */
            QProgressBar {
                border: 1px solid #dcdcdc;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #007aff; /* 与按钮同色 */
                border-radius: 4px;
            }
        """)

    def _set_btn_color(self, btn, color_hex):
        """
        自动给按钮设置颜色，并生成变暗的 Hover/Pressed 效果
        无需手动写复杂的 QSS
        """
        # 解析 Hex 颜色
        color_hex = color_hex.lstrip('#')
        r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)

        # 计算变暗的颜色 (Hover变暗 10%, Pressed变暗 20%)
        def darken(v, factor): return max(0, int(v * factor))

        hover_hex = f"#{darken(r, 0.9):02x}{darken(g, 0.9):02x}{darken(b, 0.9):02x}"
        press_hex = f"#{darken(r, 0.8):02x}{darken(g, 0.8):02x}{darken(b, 0.8):02x}"

        # 应用完整样式 (保留圆角等通用属性，只改变背景色)
        style = f"""
            QPushButton {{
                background-color: #{color_hex};
                color: white;
                border: none;
                border-radius: 5px; /* 这里保持和全局样式一致 */
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_hex};
            }}
            QPushButton:pressed {{
                background-color: {press_hex};
            }}
            QPushButton:disabled {{
                background-color: #8c8c8c;
                color: #fafafa;
            }}
        """
        btn.setStyleSheet(style)

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 主布局 (垂直: 顶部工具栏 -> 中间画布区 -> 底部状态栏)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_widget.setLayout(main_layout)

        # === 顶部工具栏区域 ===
        toolbar_frame = QFrame()
        toolbar_frame.setObjectName("ToolbarFrame")
        toolbar_layout = QHBoxLayout(toolbar_frame)
        toolbar_layout.setContentsMargins(20, 15, 20, 15)
        toolbar_layout.setSpacing(15)

        # 左侧功能组
        self.btn_load = QPushButton("📂 Open Image")
        self.btn_load.clicked.connect(self.on_load_image)

        self.btn_color = QPushButton("🎨 Pick Color")
        self._set_btn_color(self.btn_color, "#faad14") # 橙色
        self.btn_color.clicked.connect(self.on_pick_color)

        # 中间弹簧
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # 右侧操作组
        self.btn_run = QPushButton("🚀 Generate")
        self.btn_run.setMinimumWidth(120)
        self._set_btn_color(self.btn_run, "#a0d911") # 绿色
        self.btn_run.clicked.connect(self.on_run)
        self.btn_run.setEnabled(False)

        self.btn_save = QPushButton("💾 Save")
        self.btn_save.clicked.connect(self.on_save)
        self._set_btn_color(self.btn_save, "#9254de") # 紫色
        self.btn_save.setEnabled(False)

        toolbar_layout.addWidget(self.btn_load)
        toolbar_layout.addWidget(self.btn_color)
        toolbar_layout.addWidget(spacer)
        toolbar_layout.addWidget(self.btn_run)
        toolbar_layout.addWidget(self.btn_save)

        main_layout.addWidget(toolbar_frame)

        # === 画布核心区域 ===
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(20)

        # --- 左侧：输入区 ---
        input_container = QFrame()
        input_container.setObjectName("CanvasContainer")
        input_layout = QVBoxLayout(input_container)

        lbl_input_title = QLabel("INPUT / SCRIBBLE")
        lbl_input_title.setObjectName("CanvasTitle")
        lbl_input_title.setAlignment(Qt.AlignCenter)
        lbl_input_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.input_canvas = ImageCanvas()

        input_layout.addWidget(lbl_input_title)
        # 添加 stretch=1，强制画布占据剩余所有空间，这样 Title 只占顶部一点点，Canvas 占满下面
        input_layout.addWidget(self.input_canvas, 1)

        # --- 右侧：结果区 (带堆叠层，用于显示进度条) ---
        result_container = QFrame()
        result_container.setObjectName("CanvasContainer")
        result_layout = QVBoxLayout(result_container)

        lbl_result_title = QLabel("COLORIZED RESULT")
        lbl_result_title.setObjectName("CanvasTitle")
        lbl_result_title.setAlignment(Qt.AlignCenter)

        # 堆叠挂件：Index 0 是画布，Index 1 是加载动画
        self.right_stack = QStackedWidget()

        # 页面 0: 结果画布
        self.result_canvas = ImageCanvas()
        self.result_canvas.setMouseTracking(False)
        self.result_canvas.setEnabled(False)
        self.right_stack.addWidget(self.result_canvas)

        # 页面 1: 加载界面
        loading_widget = QWidget()
        loading_layout = QVBoxLayout(loading_widget)
        loading_layout.setAlignment(Qt.AlignCenter)

        self.pbar = QProgressBar()
        self.pbar.setFixedWidth(300)
        self.pbar.setRange(0, 0)  # 关键：设为 0,0 会触发“忙碌”动画 (Indeterminate)
        self.pbar.setTextVisible(False)

        lbl_loading = QLabel("Solving Linear Equations...\n(It might needs a few minutes)")
        lbl_loading.setAlignment(Qt.AlignCenter)
        lbl_loading.setStyleSheet("color: #8c8c8c; margin-top: 10px;")

        loading_layout.addWidget(self.pbar)
        loading_layout.addWidget(lbl_loading)
        self.right_stack.addWidget(loading_widget)

        result_layout.addWidget(lbl_result_title)
        result_layout.addWidget(self.right_stack, 1)

        # 添加到内容层
        content_layout.addWidget(input_container, stretch=1)
        content_layout.addWidget(result_container, stretch=1)

        main_layout.addLayout(content_layout)

        # === C. 底部状态栏 ===
        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("color: #8c8c8c; padding: 5px;")
        self.statusBar().addWidget(self.status_label)

    # === 逻辑处理 (Slots) ===

    def on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            success = self.input_canvas.load_image(path)
            if success:
                self.current_img_path = path
                self.btn_run.setEnabled(True)
                self.status_label.setText(f"Loaded: {os.path.basename(path)}")

                # 重置右侧
                self.result_canvas.clear()
                self.right_stack.setCurrentIndex(0)  # 显示画布页
                self.btn_save.setEnabled(False)

    def on_pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.input_canvas.set_brush_color(color.name())
            # 动态改变按钮颜色以反馈当前选择
            # 计算反色文字以保证可读性
            text_col = "black" if color.lightness() > 128 else "white"
            self.btn_color.setStyleSheet(
                f"background-color: {color.name()}; color: {text_col}; border: 2px solid #555;")

    def on_run(self):
        if not self.current_img_path: return

        # UI 切换到“加载中”状态
        self.right_stack.setCurrentIndex(1)  # 切换到进度条页
        self.btn_run.setEnabled(False)
        self.input_canvas.setEnabled(False)
        self.status_label.setText("Computing... Please wait.")

        # 启动线程
        mask_img = self.input_canvas.get_mask()
        self.worker = ColorizationWorker(self.current_img_path, mask_img)
        self.worker.result_ready.connect(self.on_result_ready)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.start()

    def on_result_ready(self, result_img):
        self.current_result_img = result_img
        self.result_canvas.update_display_from_cv(result_img)

        # UI 切换回“结果”状态
        self.right_stack.setCurrentIndex(0)

        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.input_canvas.setEnabled(True)
        self.status_label.setText("Done. Time cost varies by resolution.")

    def on_worker_error(self, error_msg):
        self.right_stack.setCurrentIndex(0)  # 切回画布（虽然是空的）
        QMessageBox.critical(self, "Algorithm Error", error_msg)
        self.btn_run.setEnabled(True)
        self.input_canvas.setEnabled(True)
        self.status_label.setText("Error occurred.")

    def on_save(self):
        if self.current_result_img is None: return
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "result.png", "Images (*.png *.jpg)")
        if save_path:
            import cv2
            cv2.imwrite(save_path, self.current_result_img)
            self.status_label.setText(f"Saved to: {save_path}")