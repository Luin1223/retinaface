import numpy as np
import sqlite3
import io
import os

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def compare_embeddings(db_path, embeddings, threshold=0.2):
    """
    比對輸入的特徵嵌入與資料庫中的嵌入，找出最匹配的人臉。
    
    參數：
    db_path (str): 資料庫路徑。
    embeddings (np.ndarray): 輸入的特徵嵌入。
    threshold (float): 距離閾值，低於此值認定為同一人。
    
    返回：
    tuple: 最匹配的人臉名稱、距離、所有比對結果的字典。
    """
    conn_db = sqlite3.connect(db_path)
    cursor = conn_db.execute("SELECT * FROM face_info")
    db_data = cursor.fetchall()
    
    total_distances = []
    total_names = []
    for data in db_data:
        total_names.append(data[1])
        db_embeddings = convert_array(data[2])
        distance = round(np.linalg.norm(db_embeddings - embeddings), 2)
        total_distances.append(distance)
    
    total_result = dict(zip(total_names, total_distances))
    idx_min = np.argmin(total_distances)
    distance, name = total_distances[idx_min], total_names[idx_min]
    
    conn_db.close()
    
    if distance < threshold:
        return name, distance, total_result
    else:
        name = "Unknown Person"
        return name, distance, total_result

# 示例調用
db_path = 'database.db'
input_embeddings = np.random.rand(1, 128)  # 示例嵌入，實際應從你的模型獲取
name, distance, result = compare_embeddings(db_path, input_embeddings)
print(f"Name: {name}, Distance: {distance}")
print(f"All results: {result}")
