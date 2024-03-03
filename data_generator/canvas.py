import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QColorDialog, QSlider, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint


class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        # self.setGeometry(100, 100, 1024 + 256, 1024 + 256)
        self.background = QPixmap("./render_img/fig_test.jpg")
        if self.background.isNull():
            print("Error: Unable to load image")
        self.setFixedSize(self.background.size())
        # self.setGeometry(0,0,512,512)

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
