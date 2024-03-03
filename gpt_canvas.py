import sys  
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget  
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor  
from PyQt5.QtCore import Qt, QPoint  
  
class Canvas(QWidget):  
    def __init__(self):  
        super().__init__()  
        self.setFixedSize(800, 600)  
        self.lastPoint = QPoint()  
  
    def paintEvent(self, event):  
        painter = QPainter(self)  
        painter.drawPixmap(self.rect(), self.pixmap)  
  
    def mousePressEvent(self, event):  
        if event.button() == Qt.LeftButton:  
            self.lastPoint = event.pos()  
  
    def mouseMoveEvent(self, event):  
        if event.buttons() and Qt.LeftButton:  
            painter = QPainter(self.pixmap)  
            painter.setPen(QPen(QColor(0, 0, 0), 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))  
            painter.drawLine(self.lastPoint, event.pos())  
            self.lastPoint = event.pos()  
            self.update()  
  
class MainWindow(QMainWindow):  
    def __init__(self):  
        super().__init__()  
        self.canvas = Canvas()  
        self.setCentralWidget(self.canvas)  
        self.canvas.pixmap = QPixmap(self.canvas.size())  
        self.canvas.pixmap.fill(Qt.white)  
  
if __name__ == '__main__':  
    app = QApplication(sys.argv)  
    window = MainWindow()  
    window.show()  
    sys.exit(app.exec_())  
