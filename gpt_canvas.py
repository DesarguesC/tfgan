# 3D Gaussian Splatting
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.models as models
from torch import nn
import splatting.my_renderer.loading_volume as vol
import splatting.my_renderer.cameras as camera
import splatting.my_renderer.mapping as mapping
import splatting.my_renderer.rendering as rendering
import splatting.my_renderer.my_renderer as my_renderer
from splatting.texture import *

import sys, os, torch
from PyQt5.QtWidgets import (
                            QApplication, QMainWindow, QWidget, QPushButton,
                            QVBoxLayout, QHBoxLayout, QColorDialog, QSlider, QLabel,
                            QComboBox
                             )
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint

from matplotlib import pyplot as plt

file_path = "./volumedata/tooth_103x94x161_uint8.raw"
dimensions = (161, 94, 103)  # 体数据维度
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tf_resolution = 64
colors = 0.5 * torch.rand(tf_resolution, 4).to(device)
# 这里以后可以改成前端输入


volumes = vol.read_raw_volume(file_path, dimensions) # np.array
voxels = torch.from_numpy(vol.volume_to_voxels(volumes, use_tag=True, tag_volume=get_tag_voxels(dimensions))).to(device)
# TODO: 在get_tag_voxels中加入学长写的降维后选中数据点的方法, 打上tag即可
print(voxels.shape)
R, T = camera.look_at_view_transform(20, 10, 270, device=device)
cameras = camera.FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01, scale_xyz=((1.5, 1.5, 1.5),))
rendering_settings = rendering.RenderingSettings(
    image_width=512,
    image_height=512,
    radius=75
)
drawing = rendering.Rendering(cameras=cameras, rendering_settings=rendering_settings)

texture_tf = TFMapping(tf_resolution, colors, device=device, lock=True).to(device)
texture_tf.flash(voxels)
# focus: texture_tf.tag_list


def do_vatiatinal_edit():
    """
        按照之前写的流程做可微分渲染即可, 目前按照如下方式存储, 暂时非实时渲染：
            上传的体数据, 初始视角的当前渲染结果在ref_test.jpg中,
            手画编辑结果存在color.jpg和mask.jpg中,
            降维体数据的二维图在points.jpg中,
            编辑结果当前视角渲染结果在output.jpg中
    """
    pass

def draw_texture(tf_func, tag):
    """
        tf_func: LocalTF mapping function, from texture_tf
        暂时使用Gaussian去画, 有个图先

    """
    color_matrix = tf_func.colors.clone().detach()
    x_list = range(1, 4*64+1) # Gaussian图是按照透明度alpha最后变化的 ? (认为先遍历列?) ->
    y_list = [color_matrix[i%64-1][(i-1)//64] for i in x_list]
    plt.xticks(np.arange(0, 300, 100), size=15)
    plt.yticks(np.arange(0, 1.1, 0.2), size=15)
    plt.plot(x_list, y_list, color='blue', label=f'Gaussian LocalTF-{tag}')
    plt.legend()
    plt.savefig(f'./render_img/figure/Local-{tag}.jpg')





class TFCurve(QWidget):
    def __init__(self, texture_mapping):
        super().__init__()
        self.tf = texture_mapping
        self.show_TFcurve(tag=0)
    def update_tf(self, texture_mapping):
        self.tf = texture_mapping
        self.show_TFcurve(tag=texture_mapping[-1])
    def show_TFcurve(self, tag=0):
        image_label = QLabel(self)
        if tag not in self.tf.tag_list:
            pixmap = QPixmap('./render_img/none.jpg')
        else:
            LocalTF = self.tf.LocalTF[tag]
            draw_texture(LocalTF, tag)
            pixmap = QPixmap(f'./render_img/figure/LocalTF-{tag}.jpg')
        image_label.setPixmap(pixmap)
        layout = QVBoxLayout()
        layout.addWidget(image_label)
        self.setLayout(layout)



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
        cnt = len(os.listdir('./render_img/masks/'))
        self.pixmap.save(f"./render_img/masks/mask-{cnt}.jpg")
        self.clearPixmap()
        do_vatiatinal_edit() # 做可微分体数据编辑


    def clearPixmap(self):
        self.pixmap.fill(Qt.transparent)
        self.update()

    def chooseColor(self):
        self.penColor = QColorDialog.getColor()
        if not self.penColor.isValid():
            self.penColor = Qt.black

    def setPenWidth(self, width):
        self.penWidth = width


class ImageViewer(QWidget):
    def __init__(self, img_path='./render_img/ref_test.jpg'):
        super().__init__()
        self.img_path = img_path
        self.initUI()

    def initUI(self):
        image_label = QLabel(self)
        if not os.path.exists(self.img_path):
            self.img_path = './render_img/none.jpg' # 处理后面还没render完成的情况
        pixmap = QPixmap(self.img_path)
        image_label.setPixmap(pixmap)

        layout = QVBoxLayout()
        layout.addWidget(image_label)
        self.setLayout(layout)

    def update_img(self):
        self.initUI()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 3 * 512, 3 * 512)
        """
            上传的体数据, 初始视角的当前渲染结果在ref_test.jpg中, 手画编辑结果存在color.jpg和mask.jpg中, 
            降维体数据的二维图在points.jpg中, 编辑结果当前视角渲染结果在output.jpg中
        """
        self.rendered = ImageViewer('./render_img/ref_test.jpg') # 初始渲染结果, 这个后面再变成实时渲染
        self.reducted = ImageViewer('./render_img/points.jpg')
        self.output = ImageViewer('./render_img/output.jpg') # 最终编辑结果, 这个后main再变成实时渲染

        self.canvas = Canvas()
        self.TFCurve = TFCurve(texture_tf) # 当作ImageViewer类来处理, 一个图形显示器

        self.saveButton = QPushButton("Save as Mask")
        self.saveButton.clicked.connect(self.saveMask)

        self.colorButton = QPushButton("Choose Color")
        self.colorButton.clicked.connect(self.chooseColor)

        penWidthSlider = QSlider(Qt.Horizontal)
        penWidthSlider.setRange(1, 20)  # 设置笔迹粗细滑块的取值范围
        penWidthSlider.setValue(5)  # 设置默认值
        penWidthSlider.valueChanged.connect(self.setPenWidth)
        penWidthLabel = QLabel("Pen Width")
        self.penWidthSlider = penWidthSlider
        self.penWidthLabel = penWidthLabel

        self.combo_box = QComboBox()
        self.combo_box.addItem(str(texture_tf.tag_list[-1]))
        self.combo_box.currentIndexChanged.connect(self.on_combobox_changed) # 切换选项

        # show render result in previous step, which is saved in './render_img/ref_test.jpg'
        h_layout = QHBoxLayout() # TODO: For Canvas (水平布局, 理解为指明其中元素为水平关系)
        v_layout = QVBoxLayout() # TODO: For Column (垂直布局, 理解为指明其中元素为垂直关系)

        h_layout.addWidget(self.colorButton)
        h_layout.addWidget(self.saveButton)

        v_layout.addWidget(self.canvas)
        v_layout.addWidget(penWidthSlider)
        v_layout.addLayout(h_layout)
        # TODO: CANVAS end !
        """
            v_layout => CANVAS module
        """

        widget = QWidget()
        widget.setLayout(v_layout)
        self.setCentralWidget(widget)

        # layout = QVBoxLayout()
        # layout.addWidget(self.canvas, 1)
        # layout.addStretch(1)
        # layout.addWidget(colorButton)
        # layout.addWidget(penWidthLabel)
        # layout.addWidget(penWidthSlider)
        # layout.addWidget(saveButton)
        # widget = QWidget()
        # widget.setLayout(layout)
        self.setCentralWidget(widget)

    def saveMask(self):
        self.canvas.saveMask()
        texture_tf.flash(voxels)
        # 这里应该还要加一个更新voxels中tag的方法(加入2,3,...的tag)
        self.combo_box.addItem(texture_tf.tag_list[-1])

    def chooseColor(self):
        self.canvas.chooseColor()

    def setPenWidth(self, width):
        self.canvas.setPenWidth(width)

    def on_combobox_changed(self, index):
        if not isinstance(index, str):
            index = str(index)
        selected_option = self.combo_box.itemText(index)
        self.TFCurve.update_tf(texture_tf) # 里面带有show, 已经完成绘图和保存




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())