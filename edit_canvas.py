import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QColorDialog, QSlider, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint
from data_generator.canvas import Canvas

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 3*512, 3*512)  # 设置窗口初始位置和大小
        self.canvas = Canvas()
        # self.canvas.move(400,400)
        self.setCentralWidget(self.canvas)
        # self.adjustCanvasPosition(512, 512)

        saveButton = QPushButton("Save Mask as JPG", self)
        saveButton.setGeometry(10, 50, 20, 30)
        saveButton.clicked.connect(self.saveMask)

        # self.maskButton = saveButton
        # self.adjustButtonSize(self.maskButton,30,30)
        # self.adjustButtonPosition(self.maskButton, 100,200)


        colorButton = QPushButton("Choose Color", self)
        colorButton.setGeometry(10, 10, 20, 30)
        colorButton.clicked.connect(self.chooseColor)

        # self.colorButton = colorButton
        # self.adjustButtonSize(self.colorButton,30,30)
        # self.adjustButtonPosition(self.colorButton, 100,300)



        penWidthSlider = QSlider(Qt.Horizontal)
        penWidthSlider.setRange(1, 20)  # 设置笔迹粗细滑块的取值范围
        penWidthSlider.setValue(5)  # 设置默认值
        penWidthSlider.valueChanged.connect(self.setPenWidth)

        penWidthLabel = QLabel("Pen Width")
        layout = QVBoxLayout()
        layout.addWidget(self.canvas, 1)
        layout.addStretch(1)
        layout.addWidget(colorButton)
        layout.addWidget(penWidthLabel)
        layout.addWidget(penWidthSlider)
        layout.addWidget(saveButton)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.setWindowTitle("Drawing Application")


    def saveMask(self):
        self.canvas.saveMask()

    def chooseColor(self):
        self.canvas.chooseColor()

    def setPenWidth(self, width):
        self.canvas.setPenWidth(width)

    def adjustCanvasPosition(self, x, y):
        self.canvas.move(x, y)

    def adjustButtonPosition(self, button, x, y):
        button.move(x, y)

    def adjustButtonSize(self, button, width, height):
        button.resize(width, height)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())