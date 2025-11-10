import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm  

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            if temp.endswith(".jpg"):
                allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)

def order_points(pts):
    pts=pts[:4,:]
    rect = np.zeros((5, 2), dtype = "float32")
 
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    return rect


def get_rect_and_landmarks(img_path):
   file_name = img_path.split("/")[-1].split("-")
   landmarks_np =np.zeros((5,2))
   rect = file_name[2].split("_")
   landmarks=file_name[3].split("_")
   rect_str = "&".join(rect)
   landmarks_str= "&".join(landmarks)
   rect= rect_str.split("&")
   landmarks=landmarks_str.split("&")
   rect=[int(x) for x in rect]
   landmarks=[int(x) for x in landmarks]
   for i in range(4):
        landmarks_np[i][0]=landmarks[2*i]
        landmarks_np[i][1]=landmarks[2*i+1]
#    middle_landmark_w =int((landmarks[4]+landmarks[6])/2) 
#    middle_landmark_h =int((landmarks[5]+landmarks[7])/2) 
#    landmarks.append(middle_landmark_w)
#    landmarks.append(middle_landmark_h)
   landmarks_np_new=order_points(landmarks_np)
#    landmarks_np_new[4]=np.array([middle_landmark_w,middle_landmark_h])
   return rect,landmarks,landmarks_np_new

def x1x2y1y2_yolo(rect,landmarks,img):
    h,w,c =img.shape
    rect[0] = max(0, rect[0])
    rect[1] = max(0, rect[1])
    rect[2] = min(w - 1, rect[2]-rect[0])
    rect[3] = min(h - 1, rect[3]-rect[1])
    annotation = np.zeros((1, 14))
    annotation[0, 0] = (rect[0] + rect[2] / 2) / w  # cx
    annotation[0, 1] = (rect[1] + rect[3] / 2) / h  # cy
    annotation[0, 2] = rect[2] / w  # w
    annotation[0, 3] = rect[3] / h  # h

    annotation[0, 4] = landmarks[0] / w  # l0_x
    annotation[0, 5] = landmarks[1] / h  # l0_y
    annotation[0, 6] = landmarks[2] / w  # l1_x
    annotation[0, 7] = landmarks[3] / h  # l1_y
    annotation[0, 8] = landmarks[4] / w  # l2_x
    annotation[0, 9] = landmarks[5] / h # l2_y
    annotation[0, 10] = landmarks[6] / w  # l3_x
    annotation[0, 11] = landmarks[7] / h  # l3_y
    # annotation[0, 12] = landmarks[8] / w  # l4_x
    # annotation[0, 13] = landmarks[9] / h  # l4_y
    return annotation

def xywh2yolo(rect,landmarks_sort,img):
    h,w,c =img.shape
    rect[0] = max(0, rect[0])
    rect[1] = max(0, rect[1])
    rect[2] = min(w - 1, rect[2]-rect[0])
    rect[3] = min(h - 1, rect[3]-rect[1])
    annotation = np.zeros((1, 12))
    annotation[0, 0] = (rect[0] + rect[2] / 2) / w  # cx
    annotation[0, 1] = (rect[1] + rect[3] / 2) / h  # cy
    annotation[0, 2] = rect[2] / w  # w
    annotation[0, 3] = rect[3] / h  # h

    annotation[0, 4] = landmarks_sort[0][0] / w  # l0_x
    annotation[0, 5] = landmarks_sort[0][1] / h  # l0_y
    annotation[0, 6] = landmarks_sort[1][0] / w  # l1_x
    annotation[0, 7] = landmarks_sort[1][1] / h  # l1_y
    annotation[0, 8] = landmarks_sort[2][0] / w  # l2_x
    annotation[0, 9] = landmarks_sort[2][1] / h # l2_y
    annotation[0, 10] = landmarks_sort[3][0] / w  # l3_x
    annotation[0, 11] = landmarks_sort[3][1] / h  # l3_y
    # annotation[0, 12] = landmarks_sort[4][0] / w  # l4_x
    # annotation[0, 13] = landmarks_sort[4][1] / h  # l4_y
    return annotation

def write_lable(file_path):
    pass


if __name__ == '__main__':
    file_root = r"D:/Files/openCV/dataset"
    file_list=[]
    allFilePath(file_root,file_list)
    
    # 使用 tqdm(file_list...) 
    for img_path in tqdm(file_list, desc="Processing Images"): 
        
        try: 
            text_path= img_path.replace(".jpg",".txt")
            img =cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue
                
            rect,landmarks,landmarks_sort=get_rect_and_landmarks(img_path)
            annotation=xywh2yolo(rect,landmarks_sort,img)
            str_label = "0 "
            for i in range(len(annotation[0])):
                    str_label = str_label + " " + str(annotation[0][i])
            str_label = str_label.replace('[', '').replace(']', '')
            str_label = str_label.replace(',', '') + '\n'
            
            with open(text_path,"w") as f:
                    f.write(str_label)
                    
        except Exception as e: 
            # 捕获所有可能的错误 (比如文件名解析失败)
            print(f"\nError processing file {img_path}: {e}. Skipping.")
        
        
        
        