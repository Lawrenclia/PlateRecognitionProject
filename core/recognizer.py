# core/recognizer.py
import torch
import cv2
import numpy as np
from core.lprnet_model import LPRNet
from core.lprnet_utils import NUM_CHARS, ctc_decode

# --- 在这里配置你的模型 ---
MODEL_PATH = 'models/recognize.pt' # 这是你们必须自己训练出来的识别模型
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_WIDTH = 94
INPUT_HEIGHT = 24
# --------------------------

# 1. 加载模型 (在程序启动时只加载一次)
try:
    print(f"正在加载识别模型: {MODEL_PATH} (设备: {DEVICE})")
    # LPRNet 的类别数 = 真实字符数 + 1 (留给CTC的 'blank' 符)
    model = LPRNet(num_classes=NUM_CHARS + 1)
    
    # 加载你训练好的权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # 必须设置为评估模式
    print("识别模型加载成功。")
    
except Exception as e:
    print(f"错误: 无法加载识别模型 '{MODEL_PATH}'.")
    print("请确保你已经按教程训练了模型，并将其放置在正确路径。")
    print(f"详细错误: {e}")
    model = None

def _preprocess(cv_image):
    """
    将裁切的图像预处理为 LPRNet 所需的格式
    """
    # 1. 缩放到 (94, 24)
    resized_img = cv2.resize(cv_image, (INPUT_WIDTH, INPUT_HEIGHT))
    
    # 2. 归一化
    norm_img = resized_img.astype(np.float32) / 255.0
    
    # 3. 转换为 Tensor (H, W, C) -> (C, H, W)
    tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1)
    
    # 4. 增加 Batch 维度 (C, H, W) -> (1, C, H, W)
    batch_img = tensor_img.unsqueeze(0).to(DEVICE)
    
    return batch_img

def _detect_color(cv_image):
    """
    [占位符] - 简单的车牌颜色检测
    """
    # TODO: 实现更鲁棒的颜色检测 (例如转到 HSV 空间)
    avg_color = np.mean(cv_image, axis=(0, 1))
    
    if avg_color[0] > avg_color[2] and avg_color[0] > 100: # B > R
        return "蓝色"
    if avg_color[1] > avg_color[2] and avg_color[1] > 100: # G > R
        return "绿色 (新能源)"
    
    return "未知"


def recognize_plate(cropped_plate_img):
    """
    [真实代码] - 使用 LPRNet 模型进行字符识别

    输入: 
        cropped_plate_img: 裁切后的车牌图像 (BGR)
    输出: 
        (result_text, color, char_count)
    """
    if model is None:
        return "模型加载失败", "N/A", 0
        
    try:
        # 1. 预处理
        input_tensor = _preprocess(cropped_plate_img)
        
        # 2. 运行推理
        with torch.no_grad():
            preds = model(input_tensor) # (seq_len, 1, num_classes)
            
        # 3. CTC 解码
        # (seq_len, 1, num_classes) -> (seq_len, num_classes)
        preds = preds.squeeze(1) 
        result_text = ctc_decode(preds)
        
        # 4. 识别颜色
        color = _detect_color(cropped_plate_img)
        
        # 5. 统计字符数
        char_count = len(result_text)
        
        return (result_text, color, char_count)

    except Exception as e:
        print(f"识别过程中发生错误: {e}")
        return "识别失败", "N/A", 0