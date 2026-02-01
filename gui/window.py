from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QColorDialog, QLabel, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from gui.canvas import ImageCanvas
from core.solver import ColorizationSolver
import cv2
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python Colorization App")
        self.setGeometry(100, 100, 1200, 800)

        # 主容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 布局
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # === 画板区域 ===
        canvas_layout = QHBoxLayout()

        # 左边：操作画布
        self.input_canvas = ImageCanvas()
        self.input_canvas.setStyleSheet("border: 2px dashed gray; background-color: #f0f0f0;")
        # self.input_canvas.setAlignment(Qt.AlignCenter)  <-- 【已删除此行】

        # 右边：结果展示
        self.result_canvas = ImageCanvas()
        self.result_canvas.setStyleSheet("border: 2px solid gray; background-color: #f0f0f0;")
        # self.result_canvas.setAlignment(Qt.AlignCenter) <-- 【已删除此行】
        self.result_canvas.setMouseTracking(False)
        self.result_canvas.setEnabled(False)

        canvas_layout.addWidget(self.input_canvas, stretch=1)
        canvas_layout.addWidget(self.result_canvas, stretch=1)

        main_layout.addLayout(canvas_layout, stretch=1)

        # === 工具栏区域 ===
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

        toolbar_layout.addWidget(self.btn_load)
        toolbar_layout.addWidget(self.btn_color)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.btn_run)
        toolbar_layout.addWidget(self.btn_save)

        main_layout.addLayout(toolbar_layout)

        # 状态栏
        self.status_label = QLabel("Ready. Please load an image.")
        self.statusBar().addWidget(self.status_label)

        # 保存当前的图片路径，供 Solver 使用
        self.current_img_path = None

    # === 槽函数 ===

    def on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            success = self.input_canvas.load_image(path)
            if success:
                self.current_img_path = path
                self.btn_run.setEnabled(True)
                self.status_label.setText(f"Loaded: {path}")
                # 清空结果区
                self.result_canvas.clear()

    def on_pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.input_canvas.set_brush_color(color.name())
            self.btn_color.setStyleSheet(f"background-color: {color.name()}; color: black; font-weight: bold;")

    def on_run(self):
        self.status_label.setText("Running... (Logic not connected yet)")
        QMessageBox.information(self, "Info", "UI fixed! Ready for Phase 3.")

    def on_save(self):
        pass