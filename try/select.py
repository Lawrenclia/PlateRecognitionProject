import os
import shutil
import random
from pathlib import Path

def create_subset_dataset(original_dataset_path, output_dataset_path, train_size=40000, val_size=4000, test_size=2000):
    """
    从原始数据集中创建子集
    
    参数:
        original_dataset_path: 原始数据集路径
        output_dataset_path: 输出子集路径
        train_size: 新训练集大小
        val_size: 新验证集大小
        test_size: 新测试集大小
    """
    
    # 原始数据集路径
    original_images_train = os.path.join(original_dataset_path, "images", "train")
    original_labels_train = os.path.join(original_dataset_path, "labels", "train")
    
    # 输出数据集路径
    output_images_train = os.path.join(output_dataset_path, "images", "train")
    output_images_val = os.path.join(output_dataset_path, "images", "val")
    output_images_test = os.path.join(output_dataset_path, "images", "test")
    output_labels_train = os.path.join(output_dataset_path, "labels", "train")
    output_labels_val = os.path.join(output_dataset_path, "labels", "val")
    output_labels_test = os.path.join(output_dataset_path, "labels", "test")
    
    # 创建输出目录
    for path in [output_images_train, output_images_val, output_images_test, 
                 output_labels_train, output_labels_val, output_labels_test]:
        os.makedirs(path, exist_ok=True)
    
    # 获取所有训练图片文件
    all_images = [f for f in os.listdir(original_images_train) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"原始训练集图片数量: {len(all_images)}")
    
    # 检查是否有足够的样本
    total_needed = train_size + val_size + test_size
    if len(all_images) < total_needed:
        print(f"警告: 原始数据集只有 {len(all_images)} 个样本，但要求 {total_needed} 个样本")
        print("将使用所有可用样本")
        # 按比例重新分配
        train_ratio = train_size / total_needed
        val_ratio = val_size / total_needed
        test_ratio = test_size / total_needed
        
        train_size = int(len(all_images) * train_ratio)
        val_size = int(len(all_images) * val_ratio)
        test_size = len(all_images) - train_size - val_size
    
    print(f"新训练集大小: {train_size}")
    print(f"新验证集大小: {val_size}")
    print(f"新测试集大小: {test_size}")
    
    # 随机打乱并选择样本
    random.shuffle(all_images)
    
    # 选择训练集、验证集和测试集样本
    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size + val_size:train_size + val_size + test_size]
    
    # 复制训练集
    print("正在复制训练集...")
    for i, img_name in enumerate(train_images):
        # 复制图片
        src_img = os.path.join(original_images_train, img_name)
        dst_img = os.path.join(output_images_train, img_name)
        shutil.copy2(src_img, dst_img)
        
        # 复制对应的标签文件
        label_name = img_name.rsplit('.', 1)[0] + '.txt'
        src_label = os.path.join(original_labels_train, label_name)
        dst_label = os.path.join(output_labels_train, label_name)
        
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        
        if (i + 1) % 1000 == 0:
            print(f"已复制 {i + 1}/{len(train_images)} 训练样本")
    
    # 复制验证集
    print("正在复制验证集...")
    for i, img_name in enumerate(val_images):
        # 复制图片
        src_img = os.path.join(original_images_train, img_name)
        dst_img = os.path.join(output_images_val, img_name)
        shutil.copy2(src_img, dst_img)
        
        # 复制对应的标签文件
        label_name = img_name.rsplit('.', 1)[0] + '.txt'
        src_label = os.path.join(original_labels_train, label_name)
        dst_label = os.path.join(output_labels_val, label_name)
        
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        
        if (i + 1) % 500 == 0:
            print(f"已复制 {i + 1}/{len(val_images)} 验证样本")
    
    # 复制测试集
    print("正在复制测试集...")
    for i, img_name in enumerate(test_images):
        # 复制图片
        src_img = os.path.join(original_images_train, img_name)
        dst_img = os.path.join(output_images_test, img_name)
        shutil.copy2(src_img, dst_img)
        
        # 复制对应的标签文件
        label_name = img_name.rsplit('.', 1)[0] + '.txt'
        src_label = os.path.join(original_labels_train, label_name)
        dst_label = os.path.join(output_labels_test, label_name)
        
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        
        if (i + 1) % 500 == 0:
            print(f"已复制 {i + 1}/{len(test_images)} 测试样本")
    
    # 创建数据集配置文件
    create_dataset_yaml(output_dataset_path, len(train_images), len(val_images), len(test_images))
    
    print(f"\n子集创建完成!")
    print(f"训练集: {len(train_images)} 个样本")
    print(f"验证集: {len(val_images)} 个样本")
    print(f"测试集: {len(test_images)} 个样本")
    print(f"输出路径: {output_dataset_path}")

def create_dataset_yaml(dataset_path, train_size, val_size, test_size):
    """创建YOLO格式的数据集配置文件"""
    yaml_content = f"""# 车牌检测数据集子集
# 从原始数据集创建的子集
# 训练集: {train_size} 个样本
# 验证集: {val_size} 个样本
# 测试集: {test_size} 个样本

path: {os.path.abspath(dataset_path)}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['license_plate']
"""
    
    yaml_path = os.path.join(dataset_path, "plate.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"创建数据集配置文件: {yaml_path}")

def check_dataset_structure(dataset_path):
    """检查数据集结构是否正确"""
    required_dirs = [
        "images/train",
        "labels/train"
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_path, dir_path)
        if not os.path.exists(full_path):
            print(f"错误: 目录不存在: {full_path}")
            return False
    
    # 检查图片和标签文件
    images_dir = os.path.join(dataset_path, "images/train")
    labels_dir = os.path.join(dataset_path, "labels/train")
    
    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    print(f"图片文件数量: {len(images)}")
    print(f"标签文件数量: {len(labels)}")
    
    if len(images) == 0:
        print("错误: 没有找到图片文件")
        return False
    
    return True

if __name__ == "__main__":
    # 配置参数
    original_dataset = "D:\\OpenCV\\try\\dataset"  # 原始数据集路径
    output_dataset = "D:\\OpenCV\\try\\datasets"  # 输出子集路径
    train_samples = 20000  # 训练集样本数
    val_samples = 2000     # 验证集样本数
    test_samples = 2000     # 测试集样本数
    
    print("开始创建数据集子集...")
    print(f"原始数据集: {original_dataset}")
    print(f"输出子集: {output_dataset}")
    print(f"训练集大小: {train_samples}")
    print(f"验证集大小: {val_samples}")
    print(f"测试集大小: {test_samples}")
    print("-" * 50)
    
    # 检查原始数据集结构
    if not check_dataset_structure(original_dataset):
        print("原始数据集结构不正确，请检查路径和目录结构")
        exit(1)
    
    # 创建子集
    create_subset_dataset(original_dataset, output_dataset, train_samples, val_samples, test_samples)
    
    print("\n使用说明:")
    print(f"1. 新的数据集路径: {output_dataset}")
    print(f"2. 数据集配置文件: {output_dataset}/plate.yaml")
    print(f"3. 在训练时使用: data='{output_dataset}/plate.yaml'")