# core/detector.py
import torch
from ultralytics import YOLO

# --- 在这里配置你的模型 ---
MODEL_PATH = 'models/detect.pt' # 这是你们必须自己训练出来的检测模型
CONFIDENCE_THRESHOLD = 0.4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# --------------------------

# 1. 加载模型 (在程序启动时只加载一次)
try:
    print(f"正在加载检测模型: {MODEL_PATH} (设备: {DEVICE})")
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    print("检测模型加载成功。")
except Exception as e:
    print(f"错误: 无法加载检测模型 '{MODEL_PATH}'.")
    print("请确保你已经按教程训练了模型，并将其放置在正确路径。")
    model = None

def detect_plates(cv_image):
    """
    [真实代码] - 使用 YOLOv8 模型进行车牌检测

    输入: 
        cv_image: 原始 OpenCV 图像 (BGR)
    输出: 
        boxes: 边界框列表 [(x1, y1, x2, y2), ...]
    """
    if model is None:
        return [] # 模型加载失败，返回空
        
    # 2. 运行推理
    results = model(cv_image, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    # 3. 解析结果
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    return boxes