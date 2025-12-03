import os
import random
from shutil import copyfile
from PIL import Image

# -------------------------
# 你要训练哪些子集？
# 想训练全部就全写进去
# -------------------------
SUBSETS = [
    "ccpd_rotate",
    "ccpd_tilt",
]

# -------------------------
# 输出目录（YOLO 数据集）
# -------------------------
DATA_DIR = "D:\\OpenCV\\data\\CCPD2019"
OUTPUT_DIR = "dataset"
TRAIN_IMG_DIR = os.path.join(OUTPUT_DIR, "images/train")
VAL_IMG_DIR = os.path.join(OUTPUT_DIR, "images/val")
TRAIN_LABEL_DIR = os.path.join(OUTPUT_DIR, "labels/train")
VAL_LABEL_DIR = os.path.join(OUTPUT_DIR, "labels/val")

for path in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LABEL_DIR, VAL_LABEL_DIR]:
    os.makedirs(path, exist_ok=True)

# -------------------------
# 收集所有图片路径
# -------------------------
all_imgs = []
for subset in SUBSETS:
    subset_path = os.path.join(DATA_DIR, subset)
    if not os.path.exists(subset_path):
        print(f"⚠ 跳过不存在的子集: {subset_path}")
        continue
    
    imgs = os.listdir(subset_path)
    jpgs = [os.path.join(subset_path, f) for f in imgs if f.endswith(".jpg")]
    print(f"📁 子集 {subset_path} 加载到 {len(jpgs)} 张图片")
    all_imgs.extend(jpgs)

random.shuffle(all_imgs)
print(f"\n📌 总计加载 {len(all_imgs)} 张 CCPD 图片\n")

# -------------------------
# 划分 train / val
# -------------------------
split_ratio = 0.85
split_idx = int(len(all_imgs) * split_ratio)
train_imgs = all_imgs[:split_idx]
val_imgs = all_imgs[split_idx:]

print(f"训练集: {len(train_imgs)} 张")
print(f"验证集: {len(val_imgs)} 张\n")


# -------------------------
# CCPD 文件名解析 → YOLO 格式（边界框 + 四个角点）
# -------------------------

def parse_filename_to_yolo(filename, img_path):
    """
    CCPD 文件名示例：
    0221132662835-90_89-240&524_513&620-515&616_243&619_244&520_516&517-0_0_7_27_32_33_19-183-35.jpg
    
    格式解析：
    - 文件名由 '-' 分割成多个部分
    - 第3部分：240&524_513&620 是车牌区域的左上和右下坐标（用于计算边界框）
    - 第4部分：515&616_243&619_244&520_516&517 是四个角点坐标
    
    我们需要生成13列的YOLO格式：
    class x_center y_center width height x1 y1 x2 y2 x3 y3 x4 y4
    """
    base = os.path.basename(filename).replace(".jpg", "")

    # 分割文件名
    parts = base.split('-')
    if len(parts) < 5:
        raise ValueError(f"文件名格式错误，应有至少5部分: {filename}")
    
    # 第3部分是边界框坐标（左上和右下）
    bbox_part = parts[2]  # 例如：240&524_513&620
    bbox_strs = bbox_part.split('_')
    
    if len(bbox_strs) != 2:
        raise ValueError(f"边界框部分应该包含2个点，实际得到 {len(bbox_strs)} 个: {bbox_part}")
    
    # 解析边界框坐标
    bbox_points = []
    for bbox_str in bbox_strs:
        x, y = map(int, bbox_str.split('&'))
        bbox_points.append((x, y))
    
    x_min, y_min = bbox_points[0]  # 左上点
    x_max, y_max = bbox_points[1]  # 右下点
    
    # 第4部分是四个角点坐标
    corners_part = parts[3]  # 例如：515&616_243&619_244&520_516&517
    corner_strs = corners_part.split('_')
    
    if len(corner_strs) != 4:
        raise ValueError(f"角点部分应该包含4个点，实际得到 {len(corner_strs)} 个: {corners_part}")

    # 解析四个角点
    corners = []
    for corner_str in corner_strs:
        x, y = map(int, corner_str.split('&'))
        corners.append((x, y))
    
    # 图片宽高
    W, H = Image.open(img_path).size

    # 计算边界框的中心点和宽高（归一化）
    x_center = (x_min + x_max) / 2 / W
    y_center = (y_min + y_max) / 2 / H
    width = (x_max - x_min) / W
    height = (y_max - y_min) / H

    # 转换为YOLO格式：class x_center y_center width height x1 y1 x2 y2 x3 y3 x4 y4
    yolo_label = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    # 添加四个角点坐标（归一化）
    for x, y in corners:
        x_norm = x / W
        y_norm = y / H
        yolo_label += f" {x_norm:.6f} {y_norm:.6f}"
    
    return yolo_label + "\n"


# -------------------------
# 转换函数：复制图片 + 写标签
# -------------------------

def convert_and_copy(img_list, img_dest_dir, label_dest_dir):
    success_count = 0
    error_count = 0
    
    for img_path in img_list:
        filename = os.path.basename(img_path)
        label_path = os.path.join(label_dest_dir, filename.replace(".jpg", ".txt"))

        # 写 YOLO 标签
        try:
            yolo_label = parse_filename_to_yolo(filename, img_path)
            with open(label_path, "w") as f:
                f.write(yolo_label)

            # 拷贝图片
            copyfile(img_path, os.path.join(img_dest_dir, filename))
            success_count += 1
            
            # 每处理100张图片打印一次进度
            if success_count % 100 == 0:
                print(f"  已处理 {success_count} 张图片...")
                
        except Exception as e:
            error_count += 1
            print(f"❌ 解析失败 {filename}: {e}")
    
    return success_count, error_count


# -------------------------
# 执行转换
# -------------------------

print("⌛ 正在生成 YOLO 格式训练集...")
train_success, train_errors = convert_and_copy(train_imgs, TRAIN_IMG_DIR, TRAIN_LABEL_DIR)

print("⌛ 正在生成 YOLO 格式验证集...")
val_success, val_errors = convert_and_copy(val_imgs, VAL_IMG_DIR, VAL_LABEL_DIR)

print(f"\n🎉 转换完成！")
print(f"训练集: 成功 {train_success}, 失败 {train_errors}")
print(f"验证集: 成功 {val_success}, 失败 {val_errors}")
print(f"总计: 成功 {train_success + val_success}, 失败 {train_errors + val_errors}")

# -------------------------
# 验证标签格式
# -------------------------
def verify_labels():
    print("\n🔍 验证标签格式...")
    sample_labels = os.listdir(TRAIN_LABEL_DIR)[:3]  # 检查前3个标签文件
    
    for label_file in sample_labels:
        label_path = os.path.join(TRAIN_LABEL_DIR, label_file)
        with open(label_path, 'r') as f:
            content = f.read().strip()
            parts = content.split()
            print(f"{label_file}: {len(parts)}列 - 格式: {content}")

verify_labels()

# -------------------------
# 创建数据集配置文件
# -------------------------
dataset_yaml = f"""
# 车牌检测数据集配置
path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val

# 类别数
nc: 1
# 类别名称
names: ['license_plate']

# 关键点配置（四个角点）
kpt_shape: [4, 2]  # 4个点，每个点有x,y两个坐标
flip_idx: [1, 0, 3, 2]  # 水平翻转时关键点的对应关系
"""

with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), "w") as f:
    f.write(dataset_yaml)

print(f"📄 数据集配置文件已生成: {os.path.join(OUTPUT_DIR, 'dataset.yaml')}")