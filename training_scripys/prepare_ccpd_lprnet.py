# prepare_ccpd_lprnet.py
import os
import cv2
from tqdm import tqdm

# CCPD 省份简称
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "使", "领"]
# 字母和数字 (移除 'O' 和 'I')
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ads = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] + alphabets

def parse_ccpd_filename_for_lpr(filename):
    # 示例: 025-95_113-154&383_386&473-386&473_177&454_154&383_316&488-0.jpg
    try:
        parts = filename.split('-')

        # 1. 获取 BBox
        bbox_str = parts[2].split('_')
        p1 = bbox_str[0].split('&')
        p2 = bbox_str[1].split('&')
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        bbox = [x1, y1, x2, y2]

        # 2. 获取车牌号
        plate_str = parts[4]
        plate_indices = [int(i) for i in plate_str.split('_')]

        plate_text = ""
        plate_text += provinces[plate_indices[0]] # 省份
        plate_text += alphabets[plate_indices[1]] # 城市
        for i in range(2, 7): # 后 5 位
            plate_text += ads[plate_indices[i]]

        return bbox, plate_text

    except Exception:
        return None, None

def process_ccpd_for_lpr(ccpd_dir, output_dir):
    subfolders = ['ccpd_base', 'ccpd_fn', 'ccpd_rotate', 'ccpd_db', 'ccpd_blur']

    # LPRNet_Pytorch 需要的目录结构
    img_train_dir = os.path.join(output_dir, 'data/train')
    img_val_dir = os.path.join(output_dir, 'data/val')
    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(img_val_dir, exist_ok=True)

    train_labels = []
    val_labels = []

    print("开始转换 CCPD -> LPRNet 格式...")
    count = 0
    for folder in subfolders:
        src_img_dir = os.path.join(ccpd_dir, folder)
        if not os.path.isdir(src_img_dir):
            continue

        for filename in tqdm(os.listdir(src_img_dir), desc=f"Processing {folder}"):
            if not filename.endswith('.jpg'):
                continue

            bbox, plate_text = parse_ccpd_filename_for_lpr(filename)
            if bbox is None:
                continue

            img_path = os.path.join(src_img_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 裁切车牌
            x1, y1, x2, y2 = bbox
            cropped_plate = img[y1:y2, x1:x2]

            if cropped_plate.size == 0:
                continue

            # 划分训练/验证集 (9:1)
            new_filename = f"{count:07d}.jpg"
            if count % 10 == 0:
                save_dir = img_val_dir
                label_list = val_labels
            else:
                save_dir = img_train_dir
                label_list = train_labels

            # 保存裁切图
            cv2.imwrite(os.path.join(save_dir, new_filename), cropped_plate)

            # 记录标签
            # (LPRNet_Pytorch 的 txt 标签是相对于 data 目录的)
            relative_path = os.path.join(os.path.basename(save_dir), new_filename).replace("\\", "/")
            label_list.append(f"{relative_path} {plate_text}")

            count += 1

    # 写入 LPRNet_Pytorch 所需的 txt 文件
    with open(os.path.join(output_dir, 'data/train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_labels))
    with open(os.path.join(output_dir, 'data/val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_labels))

    print("LPRNet 数据准备完成!")

if __name__ == "__main__":
    # 1. 你的 CCPD 根目录
    CCPD_DATA_PATH = 'D:/datasets/CCPD'
    # 2. 你的输出目录 (将数据直接输出到 LPRNet_Pytorch 库中)
    LPR_DATA_PATH = 'D:/projects/LPRNet_Pytorch/'

    process_ccpd_for_lpr(CCPD_DATA_PATH, LPR_DATA_PATH)