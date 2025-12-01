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
    from detect_plate import detect_Recognition_plate, load_model, detect_Recognition_plate_multi_models
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保此脚本放置在 'chinese_license_plate_detection_recognition' 项目的根目录下,")
    print("并且已经安装了所有 'requirements.txt' 中的依赖项。")
    sys.exit(1)

BG_LIGHT = "#f0f4f9"      # 主背景 (浅蓝灰) 
CARD_LIGHT = "#ffffff"      # 卡片/面板背景 (纯白) 
ACCENT_BLUE = "#0b57d0"    # 强调色 (亮蓝) 

INPUT_BG = "#F8F9FA"      # 输入框/占位符背景 (浅灰)
BORDER_LIGHT = "#E0E4E8"   # 边框 (标准浅灰)
TEXT_PRIMARY = "#202124"  # 主文字 (深灰/近黑)
TEXT_MUTED = "#5f6368"    # 辅助文字 (中灰)
HOVER_LIGHT = "#F4F8FE"    # 悬停 (极浅蓝)


def resource_path(relative_path):
    """ 获取资源绝对路径，用于 PyInstaller 打包 """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 会将资源解压到 sys._MEIPASS 临时目录
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


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


class ClickableImageFrame(QLabel):
    """
    一个可点击的QLabel, 用于加载图片 (v2 - Light Theme)
    """
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        
        # --- [配色修改] ---
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {BG_LIGHT};
                border: 2px dashed {BORDER_LIGHT};
                border-radius: 10px;
                color: {TEXT_MUTED};
                font-size: 18px;
            }}
            QLabel:hover {{
                background-color: {HOVER_LIGHT};
                border-color: {ACCENT_BLUE};
            }}
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
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {BG_LIGHT};
                border: 2px dashed {BORDER_LIGHT};
                border-radius: 10px;
                color: {TEXT_MUTED};
                font-size: 18px;
            }}
            QLabel:hover {{
                background-color: {HOVER_LIGHT};
                border-color: {ACCENT_BLUE};
            }}
        """)

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
        self.setStyleSheet(f"border: none; border-radius: 10px; background-color: {CARD_LIGHT};")

    def resizeEvent(self, event):
        if self.original_pixmap:
            self.update_display()
        super().resizeEvent(event)


class PlateResultWidget(QFrame):
    """
    用于显示格式化车牌结果 (Light Theme)
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
        
        self.status_font = QFont(self.font()) 
        self.status_font.setPointSize(16)
        
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
        
        # 默认使用白色文字 (适用于 蓝/绿/黑 车牌)
        text_color = "#FFFFFF" 
        if hex_color in self.LIGHT_BACKGROUNDS:
            text_color = "#111111" # 浅色背景(黄/白)使用黑色文字

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
                border: 3px solid {CARD_LIGHT}; /* 用卡片色(白色)做边框 */
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
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {INPUT_BG}; /* 统一为输入框背景 */
                border: 1px solid {BORDER_LIGHT}; /* 统一为细边框 */
                border-radius: 10px;
            }}
            QLabel {{
                color: {TEXT_MUTED}; /* 统一文本颜色 */
                background-color: transparent;
                border: none;
                padding-top: 5px;
                padding-bottom: 5px;
                font-size: 16px; /* 确保字体大小为 16px */
            }}
        """)
        
# --- 主 GUI 应用程序 ---

class PlateRecognizerGUI(QWidget):
    def __init__(self):
        super().__init__()
        
        # 定义多个备选模型路径
        self.detect_model_paths = [
            resource_path(os.path.join('weights', 'norm1.pt')),
            resource_path(os.path.join('weights', 'norm2.pt')),
            resource_path(os.path.join('weights', 'green.pt')),
            resource_path(os.path.join('weights', 'tilt.pt')),
            resource_path(os.path.join('weights', 'hard.pt')),
            # 可以添加更多备选模型
        ]
        
        self.REC_MODEL_PATH = resource_path(os.path.join('weights', 'plate_rec_color.pth'))
        self.IMG_SIZE = 640
        self.IS_COLOR = True 
        
        self.device = None
        self.detect_models = []  # 存储多个检测模型
        self.plate_rec_model = None
        
        self.plate_label_pixmap = None # 存储原始的截图 Pixmap
        
        self.init_ai_models()
        self.init_ui()

    def init_ai_models(self):
        print("正在加载模型，请稍候...")
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用设备: {self.device}")

            # 加载多个检测模型
            self.detect_models = []
            for model_path in self.detect_model_paths:
                if os.path.exists(model_path):
                    try:
                        print(f"尝试加载检测模型: {os.path.basename(model_path)}")
                        model = load_model(model_path, self.device)
                        self.detect_models.append(model)
                        print(f"成功加载检测模型: {os.path.basename(model_path)}")
                    except Exception as e:
                        print(f"加载检测模型 {os.path.basename(model_path)} 失败: {e}")
                else:
                    print(f"检测模型文件不存在: {model_path}")

            if not self.detect_models:
                raise FileNotFoundError("没有可用的检测模型文件。")

            # 加载车牌识别模型
            if not os.path.exists(self.REC_MODEL_PATH):
                raise FileNotFoundError(f"未找到车牌识别模型文件: {self.REC_MODEL_PATH}")

            self.plate_rec_model = init_model(self.device, self.REC_MODEL_PATH, is_color=self.IS_COLOR)
            print("所有模型加载完毕。")

        except Exception as e:
            # --- [配色修改] ---
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("模型加载失败")
            msg_box.setText(f"加载 AI 模型时出错: {e}\n程序即将退出。")
            msg_box.setStyleSheet(f"""
                QMessageBox {{ background-color: {CARD_LIGHT}; }}
                QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    background-color: {ACCENT_BLUE}; color: white; 
                    padding: 5px 10px; border-radius: 5px; 
                }}
            """)
            msg_box.exec_()
            sys.exit(1)


    def init_ui(self):
        self.setWindowTitle("车牌识别系统")
        self.resize(1000, 600)
        
        # --- [配色修改] ---
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_LIGHT};
                font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            }}
            QFrame#LeftPanel {{
                background-color: {CARD_LIGHT};
                border-radius: 10px;
                padding: 15px;
            }}
            QLabel.TitleLabel {{
                font-size: 18px;
                font-weight: bold;
                color: {TEXT_PRIMARY};
                margin-bottom: 10px;
            }}
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
        
        # --- [配色修改] ---
        self.plate_label.setStyleSheet(f"""
            QLabel#PlateLabel {{
                background-color: {INPUT_BG};
                border-radius: 10px; 
                color: {TEXT_MUTED};
                border: 1px solid {BORDER_LIGHT};
                font-size: 16px;
            }}
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
        
        self.plate_label.setText("处理中...") 
        self.plate_label.setStyleSheet(f"""
            QLabel#PlateLabel {{
                background-color: {INPUT_BG};
                border-radius: 10px; 
                color: {TEXT_MUTED};
                border: 1px solid {BORDER_LIGHT};
                font-size: 16px;
            }}
        """)
        
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
            
            # 首先对整张图片进行180度旋转
            cv_image = self.rotate_image_180(cv_image)
            
            # 使用多模型检测
            dict_list, used_model_index = detect_Recognition_plate_multi_models(
                self.detect_models, cv_image, self.device, 
                self.plate_rec_model, self.IMG_SIZE, is_color=self.IS_COLOR
            )
            
            # 显示使用的模型信息
            if used_model_index >= 0:
                model_name = os.path.basename(self.detect_model_paths[used_model_index])
                print(f"检测成功！使用的模型: {model_name}")
            
            if dict_list:
                result = dict_list[0] 
                plate_number = result.get('plate_no', 'N/A')
                plate_color = result.get('plate_color', '未知')
                landmarks = result.get('landmarks')

                # 显示车牌号
                self.result_widget.set_result(plate_number, plate_color)

                # 显示截图
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
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("处理失败")
            msg_box.setText(f"处理图片时出错: {e}")
            msg_box.setStyleSheet(f"""
                QMessageBox {{ background-color: {CARD_LIGHT}; }}
                QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    background-color: {ACCENT_BLUE}; color: white; 
                    padding: 5px 10px; border-radius: 5px; 
                }}
            """)
            msg_box.exec_()
            
            self.image_label.clear()
            self.result_widget.clear()
            self.plate_label_pixmap = None
            self.plate_label.setPixmap(QPixmap())
            self.plate_label.setText("处理失败")

    def rotate_image_180(self, image):
        """
        将图像旋转180度
        """
        if image is None:
            return None
        try:
            # 使用OpenCV的rotate函数进行180度旋转
            rotated = cv2.rotate(image, cv2.ROTATE_180)
            return rotated
        except Exception as e:
            print(f"图像旋转失败: {e}")
            return image  # 如果旋转失败，返回原图

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
        label.setStyleSheet(f"border: none; border-radius: 10px; background-color: {INPUT_BG};")


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
        if not window.detect_models or window.plate_rec_model is None:
            print("模型未加载，程序退出。")
            sys.exit(1)
            
        window.show()
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"启动 GUI 时发生未知错误: {e}")
        sys.exit(1)