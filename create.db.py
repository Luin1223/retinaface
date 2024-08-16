import cv2
import dlib
import numpy as np
import sqlite3
import io
import os
from retinaface import RetinaFace

# 使用 RetinaFace 進行臉部偵測
def face_detect(image_path):
    detector = RetinaFace()
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.predict(img_rgb)
    return img_rgb, detections

# 使用 dlib 進行臉部校正
def face_align(img_rgb, detection):
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    
    rect = dlib.rectangle(detection['x1'], detection['y1'], detection['x2'], detection['y2'])
    shape = predictor(img_rgb, rect)
    
    aligned_face = dlib.get_face_chip(img_rgb, shape, size=150)  # 校正臉部影像的大小設為 150x150
    return aligned_face

# 使用 dlib 進行特徵提取
def get_embeddings(aligned_face):
    face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
    face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
    
    shape = dlib.rectangle(0, 0, aligned_face.shape[1], aligned_face.shape[0])
    face_descriptor = face_rec_model.compute_face_descriptor(aligned_face)
    embeddings = np.array(face_descriptor)
    return embeddings

# 數據庫處理
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# 載入文件
def load_file(file_path):
    file_data = {}
    for person_name in os.listdir(file_path):
        person_file = os.path.join(file_path, person_name)
        if os.path.isdir(person_file):
            total_pictures = []
            for picture in os.listdir(person_file):
                picture_path = os.path.join(person_file, picture)
                total_pictures.append(picture_path)
            file_data[person_name] = total_pictures
    return file_data

# 註冊 NumPy 陣列的轉換器到 SQLite
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)

# 連接到 SQLite 數據庫並創建表格
conn_db = sqlite3.connect('database.db')
conn_db.execute("CREATE TABLE IF NOT EXISTS face_info (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, embedding ARRAY NOT NULL)")

# 主程序
def main(file_path):
    if os.path.exists(file_path):
        file_data = load_file(file_path)
        for person_name in file_data.keys():
            picture_paths = file_data[person_name]
            sum_embeddings = np.zeros(128)  # 確保這與嵌入向量的大小一致
            num_pictures = len(picture_paths)
            for picture_path in picture_paths:
                try:
                    img_rgb, detections = face_detect(picture_path)
                    for detection in detections:
                        aligned_face = face_align(img_rgb, detection)
                        embeddings = get_embeddings(aligned_face)
                        sum_embeddings += embeddings
                except Exception as e:
                    print(f"處理 {picture_path} 時發生錯誤: {e}")
            if num_pictures > 0:
                final_embedding = sum_embeddings / num_pictures
                conn_db.execute("INSERT INTO face_info (name, embedding) VALUES (?, ?)", (person_name, final_embedding))
        conn_db.commit()
    conn_db.close()

# 主程序入口
if __name__ == "__main__":
    file_path = 'database'
    main(file_path)
