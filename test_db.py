import sqlite3
import numpy as np
import io

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# 註冊 numpy array 轉換器
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

# 連接到資料庫
conn_db = sqlite3.connect('database.db', detect_types=sqlite3.PARSE_DECLTYPES)

# 執行 SQL 查詢
cursor = conn_db.execute("SELECT * FROM face_info")
db_data = cursor.fetchall()

# 打印結果
for data in db_data:
    print(f"ID: {data[0]}, Name: {data[1]}, Embedding: {data[2]}")

# 關閉資料庫連接
conn_db.close()
