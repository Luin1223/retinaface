import cv2
import dlib
import numpy as np
import sqlite3
import io
from retinaface import RetinaFace

# 使用 RetinaFace 進行人臉檢測
def face_detect(image_path):
    detector = RetinaFace()
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.predict(img_rgb)
    return img_rgb, detections

# 使用 dlib 進行人臉對齊
def face_align(img_rgb, detection):
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    
    rect = dlib.rectangle(int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2']))
    shape = predictor(img_rgb, rect)
    
    aligned_face = dlib.get_face_chip(img_rgb, shape, size=150)  # 設置尺寸為150x150
    return aligned_face

# 使用 dlib 進行特徵提取
def get_embeddings(aligned_face):
    face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
    face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
    
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

# 比較特徵嵌入
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
                print(f"Shape mismatch: db_embeddings.shape={db_embeddings.shape}, embeddings.shape={embeddings.shape}")
                continue
            
            distance = np.linalg.norm(db_embeddings - embeddings)
            results.append((name, distance))
    
    if not results:
        return "Unknown", float('inf')
    
    results.sort(key=lambda x: x[1])
    name, distance = results[0]
    if distance < threshold:
        return name, distance
    else:
        return "Unknown", distance

# 畫出結果
def draw_results(image_path, results, output_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    for (x1, y1, x2, y2), name, distance in results:
        # 根據識別結果選擇框框顏色
        if name == "Unknown":
            color = (0, 0, 255)  # 紅色框框
        else:
            color = (0, 255, 0)  # 綠色框框

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{name} ({distance:.2f})" if name != "Unknown" else f"Unknown ({distance:.2f})"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Result", img)  # 顯示圖片
    cv2.waitKey(0)  # 等待按鍵
    cv2.destroyAllWindows()  # 關閉所有窗口
    cv2.imwrite(output_path, img)  # 保存圖片
    print(f"Image saved to {output_path}")

# 獲取照片中的嵌入特徵
def get_photo_embeddings(image_path):
    img_rgb, detections = face_detect(image_path)
    if not detections:
        raise ValueError("No faces detected in the image")
    
    results = []
    for detection in detections:
        aligned_face = face_align(img_rgb, detection)
        embeddings = get_embeddings(aligned_face)
        results.append((detection, embeddings))
    
    return results

# 主函數
def main(file_path, db_path, output_image_path):
    detection_and_embeddings = get_photo_embeddings(file_path)
    results = []

    for detection, embeddings in detection_and_embeddings:
        name, distance = compare_embeddings(db_path, embeddings)
        
        # 調整邊界框座標
        x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2'])
        results.append(((x1, y1, x2, y2), name, distance))
    
    draw_results(file_path, results, output_image_path)

if __name__ == "__main__":
    image_path = 'images/all.jpg'  # 替換為你要測試的照片路徑
    db_path = 'database.db'
    output_image_path = 'images/all_results.jpg'
    
    main(image_path, db_path, output_image_path)
