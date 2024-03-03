from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout  
  
# 创建应用程序和窗口  
app = QApplication([])  
window = QWidget()  
  
# 创建水平布局  
h_layout1 = QHBoxLayout()  
h_layout2 = QHBoxLayout()  
  
# 创建垂直布局  
v_layout1 = QVBoxLayout()  
v_layout2 = QVBoxLayout()  
  
# 创建按钮  
button1 = QPushButton("Button 1")  
button2 = QPushButton("Button 2")  
button3 = QPushButton("Button 3")  
button4 = QPushButton("Button 4")  
  
# 将按钮添加到水平布局1  
h_layout1.addWidget(button1)  
h_layout1.addWidget(button2)  
  
# 将按钮添加到水平布局2  
h_layout2.addWidget(button3)  
h_layout2.addWidget(button4)  
  
# 将水平布局1添加到垂直布局1  
v_layout1.addLayout(h_layout1)  
  
# 将水平布局2添加到垂直布局2  
v_layout2.addLayout(h_layout2)  
  
# 将垂直布局1添加到窗口  
v_layout2.addLayout(v_layout1)  
  
# 将垂直布局2设置为窗口的布局  
window.setLayout(v_layout2)  
  
# 显示窗口  
window.show()  
app.exec_()  
