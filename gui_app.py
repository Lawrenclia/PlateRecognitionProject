import sys
import os
import cv2
import torch
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QSizePolicy, QMessageBox, QFrame
)
from PySide6.QtGui import QPixmap, QImage, QFont, QCursor
from PySide6.QtCore import Qt, Signal

# --- 从你的项目中导入必要的模块 ---
try:
    from models.experimental import attempt_load
    from plate_recognition.plate_rec import init_model, cv_imread
    from detect_plate import detect_Recognition_plate, load_model
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保此脚本放置在 'chinese_license_plate_detection_recognition' 项目的根目录下,")
    print("并且已经安装了所有 'requirements.txt' 中的依赖项。")
    sys.exit(1)

# --- 关键函数：从 detect_plate.py 中复制 ---
def four_point_transform(image, pts):
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    if maxWidth == 0 or maxHeight == 0:
        print("警告: 透视变换的宽度或高度为0")
        return None 

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# --- 自定义美化组件 ---

class ClickableImageFrame(QLabel):
    """
    一个可点击的QLabel, 用于加载图片 (v2)
    """
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QLabel {
                background-color: #F0F2F5;
                border: 2px dashed #B0B9C2;
                border-radius: 10px;
                color: #707982;
                font-size: 18px;
            }
            QLabel:hover {
                background-color: #E6E9ED;
                border-color: #0078D7;
            }
        """)
        self.setText("点击加载图片")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

    def setPixmap(self, pixmap: QPixmap):
        self.original_pixmap = pixmap
        self.update_display()

    def clear(self):
        self.original_pixmap = None
        self.setText("点击加载图片")
        self.setStyleSheet(self.styleSheet()) # 刷新样式

    def update_display(self):
        if not self.original_pixmap:
            self.setText("点击加载图片")
            return
        
        scaled_pixmap = self.original_pixmap.scaled(
            self.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        super().setPixmap(scaled_pixmap)
        self.setStyleSheet("border: none; border-radius: 10px;")

    def resizeEvent(self, event):
        if self.original_pixmap:
            self.update_display()
        super().resizeEvent(event)


class PlateResultWidget(QFrame):
    """
    用于显示格式化车牌结果
    """
    PLATE_COLOR_MAP = {
        "蓝色": "#0055AA",
        "黄色": "#F6A600",
        "绿色": "#008800",
        "黑色": "#222222",
        "白色": "#FFFFFF",
        "未知": "#707982" 
    }
    LIGHT_BACKGROUNDS = ["#F6A600", "#FFFFFF"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80) 
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
        
        self.layout = QHBoxLayout(self)
        self.layout.setAlignment(Qt.AlignCenter)
        self.layout.setSpacing(5) 

        self.plate_font = QFont("黑体", 24, QFont.Bold)
        
        # 状态 (status) 字体现在主要由样式表控制 (16px)
        # 但我们保留 setFont 以防万一样式表被覆盖
        self.status_font = QFont(self.font()) 
        self.status_font.setPointSize(16) # (这个 16pt 会被 16px 覆盖) 
        
        # 预先创建标签, 方便切换
        self.prefix_label = QLabel()
        self.prefix_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.prefix_label.setFont(self.plate_font)

        self.dot_label = QLabel("·")
        self.dot_label.setAlignment(Qt.AlignCenter)
        self.dot_label.setFont(self.plate_font)

        self.suffix_label = QLabel()
        self.suffix_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.suffix_label.setFont(self.plate_font)
        
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(self.status_font) 
        
        self.clear() # 设置初始状态

    def _show_plate_labels(self, show: bool):
        """ 辅助函数, 用于切换显示车牌号还是状态 """
        self.prefix_label.setVisible(show)
        self.dot_label.setVisible(show)
        self.suffix_label.setVisible(show)
        self.status_label.setVisible(not show)

    def set_result(self, text, color_name):
        hex_color = self.PLATE_COLOR_MAP.get(color_name, self.PLATE_COLOR_MAP["未知"])
        
        text_color = "#FFFFFF" 
        if hex_color in self.LIGHT_BACKGROUNDS:
            text_color = "#111111"

        prefix_text = "N/A"
        dot_text = ""
        suffix_text = ""
        
        is_plate = False
        if text and len(text) == 7: # 标准7位
            prefix_text = text[0:2] # 苏A
            dot_text = "·"
            suffix_text = text[2:]  # 88888
            is_plate = True
        elif text and len(text) == 8: # 8位新能源
            prefix_text = text[0:2]
            dot_text = ""           # 新能源无点
            suffix_text = text[2:]
            is_plate = True
        else:
            prefix_text = text # 备用, "未检测到"
        
        # 清理旧布局
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().setVisible(False)
        
        self.layout.addStretch(1)
        if is_plate:
            self.layout.addWidget(self.prefix_label)
            self.layout.addWidget(self.dot_label)
            self.layout.addWidget(self.suffix_label)
            self._show_plate_labels(True)
            self.dot_label.setVisible(bool(dot_text))
        else:
            self.layout.addWidget(self.status_label)
            self._show_plate_labels(False)
            self.status_label.setText(prefix_text)

            self.status_label.setStyleSheet(f"""
                color: {text_color}; 
                background-color: transparent; 
                border: none;
                font-size: 16px; 
            """)
        self.layout.addStretch(1)

        
        self.prefix_label.setText(prefix_text)
        self.dot_label.setText(dot_text)
        self.suffix_label.setText(suffix_text)
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {hex_color};
                border: 3px solid white; 
                border-radius: 10px;
            }}
            QLabel {{
                color: {text_color};
                background-color: transparent;
                border: none;
                padding-top: 5px; 
                padding-bottom: 5px;
            }}
        """)
        
    def clear(self):
        # 清理旧布局
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().setVisible(False)

        self._show_plate_labels(False)
        
        self.layout.addStretch(1)
        self.layout.addWidget(self.status_label)
        self.layout.addStretch(1)
        
        self.status_label.setText("等待识别")
        
        # --- 修改开始：统一 "等待" 状态的样式 (添加了 font-size) ---
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #F8F9FA; /* 统一为浅灰色背景 */
                border: 1px solid #E0E4E8; /* 统一为细边框 */
                border-radius: 10px;
            }}
            QLabel {{
                color: #707982; /* 统一文本颜色 */
                background-color: transparent;
                border: none;
                padding-top: 5px;
                padding-bottom: 5px;
                font-size: 16px; /* <-- 修正点：确保字体大小为 16px */
            }}
        """)
        # --- 修改结束 ---
        
# --- 主 GUI 应用程序 ---

class PlateRecognizerGUI(QWidget):
    def __init__(self):
        super().__init__()
        
        self.DETECT_MODEL_PATH = 'weights/plate_detect.pt'
        self.REC_MODEL_PATH = 'weights/plate_rec_color.pth'
        self.IMG_SIZE = 640
        self.IS_COLOR = True 
        
        self.device = None
        self.detect_model = None
        self.plate_rec_model = None
        
        self.plate_label_pixmap = None # 存储原始的截图 Pixmap
        
        self.init_ai_models()
        self.init_ui()

    def init_ai_models(self):
        print("正在加载模型，请稍候...")
        try:
            if not os.path.exists(self.DETECT_MODEL_PATH) or not os.path.exists(self.REC_MODEL_PATH):
                raise FileNotFoundError("未找到 'weights' 文件夹中的模型文件。")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用设备: {self.device}")

            self.detect_model = load_model(self.DETECT_MODEL_PATH, self.device)
            self.plate_rec_model = init_model(self.device, self.REC_MODEL_PATH, is_color=self.IS_COLOR)
            print("模型加载完毕。")

        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", f"加载 AI 模型时出错: {e}\n程序即将退出。")
            sys.exit(1)


    def init_ui(self):
        self.setWindowTitle("车牌识别系统!!!!!")
        self.resize(1000, 600)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #F0F2F5;
                font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            }
            QFrame#LeftPanel {
                background-color: #FFFFFF;
                border-radius: 10px;
                padding: 15px;
            }
            QLabel.TitleLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }
        """)

        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 2. 左侧
        left_panel = QFrame()
        left_panel.setObjectName("LeftPanel")
        left_layout = QVBoxLayout(left_panel)
        
        self.image_label = ClickableImageFrame()
        self.image_label.clicked.connect(self.load_and_process_image)
        left_layout.addWidget(self.image_label)
        main_layout.addWidget(left_panel, 2) 

        # 3. 右侧
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15) 

        # 3.1 车牌截图
        plate_title = QLabel("车牌截图")
        plate_title.setProperty("class", "TitleLabel") 
        
        self.plate_label = QLabel("等待加载")
        self.plate_label.setObjectName("PlateLabel")
        self.plate_label.setAlignment(Qt.AlignCenter)
        self.plate_label.setMinimumHeight(80) 
        self.plate_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
        
        self.plate_label.setStyleSheet("""
            QLabel#PlateLabel {
                background-color: #F8F9FA;
                border-radius: 10px; 
                color: #707982;
                border: 1px solid #E0E4E8;
                font-size: 16px;
            }
        """)
        
        right_layout.addWidget(plate_title)
        right_layout.addWidget(self.plate_label) 

        # 3.2 识别结果
        result_title = QLabel("识别结果")
        result_title.setProperty("class", "TitleLabel")
        self.result_widget = PlateResultWidget() 
        
        right_layout.addWidget(result_title)
        right_layout.addWidget(self.result_widget) 
        
        right_layout.addStretch(1) # 将所有内容推到顶部

        main_layout.addLayout(right_layout, 1) 


    def load_and_process_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return

        # 重置UI
        self.plate_label_pixmap = None 
        self.plate_label.setPixmap(QPixmap()) 
        
        # 设置为 "处理中..." 并刷新样式, 保持 16px
        self.plate_label.setText("处理中...") 
        self.plate_label.setStyleSheet(self.plate_label.styleSheet())
        
        self.result_widget.clear()
        QApplication.processEvents() 

        try:
            cv_image = cv_imread(file_path)
            if cv_image is None:
                raise Exception(f"无法读取图片: {file_path}")
            
            if cv_image.shape[-1] == 4:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)

            original_pixmap = self.convert_cv_to_pixmap(cv_image)
            self.image_label.setPixmap(original_pixmap)
            
            dict_list = detect_Recognition_plate(
                self.detect_model, cv_image, self.device, 
                self.plate_rec_model, self.IMG_SIZE, is_color=self.IS_COLOR
            )
            
            if dict_list:
                result = dict_list[0] 
                plate_number = result.get('plate_no', 'N/A')
                plate_color = result.get('plate_color', '未知')
                landmarks = result.get('landmarks')

                # 3.1 显示车牌号
                self.result_widget.set_result(plate_number, plate_color)

                # 3.2 显示截图
                if landmarks:
                    landmarks_np = np.array(landmarks)
                    cropped_plate_img = four_point_transform(cv_image, landmarks_np)
                    
                    if cropped_plate_img is not None:
                        self.plate_label_pixmap = self.convert_cv_to_pixmap(cropped_plate_img)
                        self.display_scaled_image(self.plate_label, self.plate_label_pixmap)
                    else:
                        self.plate_label_pixmap = None
                        self.plate_label.setPixmap(QPixmap()) 
                        self.plate_label.setText("截图失败")
                else:
                    self.plate_label_pixmap = None
                    self.plate_label.setPixmap(QPixmap()) 
                    self.plate_label.setText("无角点信息")
            else:
                self.plate_label_pixmap = None
                self.result_widget.set_result("未检测到", "未知")
                self.plate_label.setPixmap(QPixmap())
                self.plate_label.setText("未检测到")

        except Exception as e:
            QMessageBox.warning(self, "处理失败", f"处理图片时出错: {e}")
            self.image_label.clear()
            self.result_widget.clear()
            self.plate_label_pixmap = None
            self.plate_label.setPixmap(QPixmap())
            self.plate_label.setText("处理失败")

    def display_scaled_image(self, label: QLabel, pixmap: QPixmap):
        """
        辅助函数：在QLabel中显示QPixmap，并保持纵横比
        """
        if pixmap is None or pixmap.isNull():
            label.clear()
            return
            
        label.setPixmap(pixmap.scaled(
            label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))

    def convert_cv_to_pixmap(self, cv_img) -> QPixmap:
        """
        辅助工具: 将 OpenCV 图像 (BGR 或 灰度) 转换为 QPixmap
        """
        if cv_img is None:
            return QPixmap()
            
        try:
            if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            elif len(cv_img.shape) == 2:
                h, w = cv_img.shape
                bytes_per_line = w
                q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            else:
                return QPixmap()
                
            return QPixmap.fromImage(q_img)
        except Exception as e:
            print(f"转换 CV -> QPixmap 出错: {e}")
            return QPixmap()

    def resizeEvent(self, event):
        """
        当窗口大小改变时, 重新缩放截图
        """
        super().resizeEvent(event)
        if hasattr(self, 'plate_label_pixmap') and self.plate_label_pixmap and not self.plate_label_pixmap.isNull():
             self.display_scaled_image(self.plate_label, self.plate_label_pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    try:
        window = PlateRecognizerGUI()
        if window.detect_model is None or window.plate_rec_model is None:
            print("模型未加载，程序退出。")
            sys.exit(1)
            
        window.show()
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"启动 GUI 时发生未知错误: {e}")
        sys.exit(1)