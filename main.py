import sys
from PyQt5.QtWidgets import QApplication
from gui.window import MainWindow


def main():
    app = QApplication(sys.argv)

    # 设置全局字体
    # font = app.font()
    # font.setPointSize(10)
    # app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()