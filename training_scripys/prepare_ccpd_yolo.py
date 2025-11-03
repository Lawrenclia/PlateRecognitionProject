# prepare_ccpd_yolo.py
import os
import cv2
from tqdm import tqdm

def parse_ccpd_filename(filename):
    # 示例: 025-95_113-154&383_386&473-386&473_177&454_154&383_316&488-0.jpg
    # 我们只需要 154&383_386&473 (边界框)
    try:
        parts = filename.split('-')
        bbox_str = parts[2].split('_')

        p1 = bbox_str[0].split('&')
        p2 = bbox_str[1].split('&')

        x1 = int(p1[0])
        y1 = int(p1[1])
        x2 = int(p2[0])
        y2 = int(p2[1])

        return [x1, y1, x2, y2]
    except Exception:
        return None

def convert_to_yolo(bbox, img_w, img_h):
    # (x1, y1, x2, y2) -> (class_id, x_center, y_center, width, height)
    class_id = 0 # 只有一类: 'license_plate'

    x_center = ((bbox[0] + bbox[2]) / 2) / img_w
    y_center = ((bbox[1] + bbox[3]) / 2) / img_h
    width = (bbox[2] - bbox[0]) / img_w
    height = (bbox[3] - bbox[1]) / img_h

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_ccpd(ccpd_dir, output_dir):
    # CCPD 包含 'ccpd_base', 'ccpd_fn', 'ccpd_rotate' 等
    subfolders = ['ccpd_base', 'ccpd_fn', 'ccpd_rotate', 'ccpd_db', 'ccpd_blur']

    # 创建输出目录
    img_train_dir = os.path.join(output_dir, 'images/train')
    lbl_train_dir = os.path.join(output_dir, 'labels/train')
    # (为了简单，我们都放到 train 里，YOLO 会自己划分验证集)

    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(lbl_train_dir, exist_ok=True)

    print("开始转换 CCPD -> YOLO 格式...")
    for folder in subfolders:
        src_img_dir = os.path.join(ccpd_dir, folder)
        if not os.path.isdir(src_img_dir):
            print(f"跳过: {src_img_dir}")
            continue

        for filename in tqdm(os.listdir(src_img_dir), desc=f"Processing {folder}"):
            if not filename.endswith('.jpg'):
                continue

            bbox = parse_ccpd_filename(filename)
            if bbox is None:
                continue

            # 读取图像获取尺寸
            img_path = os.path.join(src_img_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            yolo_label = convert_to_yolo(bbox, w, h)

            # 复制图像和标签
            # (为了防止文件名冲突，我们用 文件夹名+文件名 的方式重命名)
            new_filename_base = f"{folder}_{filename[:-4]}"

            # 复制图像
            new_img_path = os.path.join(img_train_dir, new_filename_base + ".jpg")
            cv2.imwrite(new_img_path, img)

            # 写入标签
            label_path = os.path.join(lbl_train_dir, new_filename_base + ".txt")
            with open(label_path, 'w') as f:
                f.write(yolo_label)

    print("转换完成!")

if __name__ == "__main__":
    # 1. 修改为你解压后的 CCPD 根目录
    CCPD_DATA_PATH = "D:/Files/openCV/dataset/CCPD2019"
    # 2. 修改为你希望的输出目录 (YOLO 格式)
    YOLO_DATA_PATH = "D:/Files/openCV/datasets/CCPD_YOLO"

    process_ccpd(CCPD_DATA_PATH, YOLO_DATA_PATH)