import cv2
import dlib
import numpy as np
import sqlite3
import io
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from retinaface import RetinaFace

# 初始化dlib模型
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# 設置數據庫相關函數
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def compare_embeddings(db_path, embeddings, threshold=0.35):
    conn_db = sqlite3.connect(db_path)
    cursor = conn_db.execute("SELECT * FROM face_info")
    db_data = cursor.fetchall()
    conn_db.close()

    results = []
    for data in db_data:
        name = data[1]
        db_embeddings = convert_array(data[2])
        
        if isinstance(db_embeddings, np.ndarray) and isinstance(embeddings, np.ndarray):
            if db_embeddings.shape != embeddings.shape:
                continue
            
            distance = np.linalg.norm(db_embeddings - embeddings)
            results.append((name, distance))
    
    if not results:
        return "Unknown Person", float('inf')
    
    results.sort(key=lambda x: x[1])
    name, distance = results[0]
    if distance < threshold:
        return name, distance
    else:
        return "Unknown Person", distance

# 檢測和對齊人臉
def face_detect(image):
    detector = RetinaFace()
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.predict(img_rgb)
    return detections

def face_align(img_rgb, detection):
    rect = dlib.rectangle(int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2']))
    shape = predictor(img_rgb, rect)
    aligned_face = dlib.get_face_chip(img_rgb, shape, size=150)
    return aligned_face

def get_embeddings(aligned_face):
    face_descriptor = face_rec_model.compute_face_descriptor(aligned_face)
    embeddings = np.array(face_descriptor)
    return embeddings

# 更新圖像顯示
def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    detections = face_detect(frame)
    for detection in detections:
        aligned_face = face_align(frame, detection)
        embeddings = get_embeddings(aligned_face)
        name, distance = compare_embeddings(db_path, embeddings)
        
        x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2'])
        if name == "Unknown Person":
            color = (0, 0, 255)  # 紅色
        else:
            color = (0, 255, 0)  # 綠色
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} ({distance:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    panel.imgtk = imgtk
    panel.config(image=imgtk)
    
    window.after(10, update_frame)

# 初始化GUI
window = tk.Tk()
window.title("Face Recognition System")

panel = Label(window)
panel.pack(padx=10, pady=10)

# 打開攝像頭
cap = cv2.VideoCapture(0)
db_path = 'database.db'

update_frame()
window.mainloop()

cap.release()
cv2.destroyAllWindows()
