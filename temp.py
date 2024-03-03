from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QLabel  
  
# 创建应用程序和窗口  
app = QApplication([])  
window = QWidget()  
  
# 创建下拉选框列表  
combo_box = QComboBox()  
combo_box.addItem("Option 1")  
combo_box.addItem("Option 2")  
combo_box.addItem("Option 3")  
  
# 创建标签  
label = QLabel("")  
  
# 创建垂直布局  
layout = QVBoxLayout()  
  
# 将下拉选框列表和标签添加到布局  
layout.addWidget(combo_box)  
layout.addWidget(label)  
  
# 将布局设置为窗口的布局  
window.setLayout(layout)  
  
# 当下拉选框列表的选项发生变化时执行的内容  
def on_combobox_changed(index):  
    selected_option = combo_box.itemText(index)  
    label.setText(f"Selected option: {selected_option}")  
    # 在这里添加根据选择的选项执行的内容  
  
# 连接下拉选框列表的currentIndexChanged信号到on_combobox_changed槽函数  
combo_box.currentIndexChanged.connect(on_combobox_changed)  
  
# 显示窗口  
window.show()  
app.exec_()  
