import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QColorDialog, QSlider, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint


class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.background = QPixmap("./render_img/fig_test.jpg")
        if self.background.isNull():
            print("Error: Unable to load image")
        self.setFixedSize(self.background.size())
        self.lastPoint = QPoint()
        self.pixmap = QPixmap(self.size())
        self.pixmap.fill(Qt.transparent)
        self.penColor = Qt.black
        self.penWidth = 5  # 默认笔迹粗细为5
        self.colorIndex = 1  # 初始化颜色索引

    def paintEvent(self, event):
        if not self.background.isNull():
            painter = QPainter(self)
            painter.drawPixmap(0, 0, self.background)
            painter.drawPixmap(self.rect(), self.pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton:
            painter = QPainter(self.pixmap)
            painter.setPen(
                QPen(self.penColor, self.penWidth, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))  # 使用选择的颜色和笔迹粗细
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def saveMask(self):
        self.pixmap.save("./render_img/mask.jpg")
        self.clearPixmap()

    def clearPixmap(self):
        self.pixmap.fill(Qt.transparent)
        self.update()

    def chooseColor(self):
        self.penColor = QColorDialog.getColor()
        if not self.penColor.isValid():
            self.penColor = Qt.black

    def setPenWidth(self, width):
        self.penWidth = width


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.canvas = Canvas()
        saveButton = QPushButton("Save Mask as JPG")
        saveButton.clicked.connect(self.saveMask)
        colorButton = QPushButton("Choose Color")
        colorButton.clicked.connect(self.chooseColor)
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

    def saveMask(self):
        self.canvas.saveMask()

    def chooseColor(self):
        self.canvas.chooseColor()

    def setPenWidth(self, width):
        self.canvas.setPenWidth(width)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())