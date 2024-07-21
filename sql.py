import pyodbc
import os

server = os.getenv('SQL_SERVER')
database = os.getenv('SQL_DATABASE_NAME')
username = os.getenv('SQL_USERNAME')
password = os.getenv('SQL_PWD')

# 建立數據庫連接
conn = pyodbc.connect(
    f"DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
)
cursor = conn.cursor()


cursor.execute('''
INSERT INTO icare_user (username, account, pwd) VALUES (?, ?, ?)
''', ('john_doe', 'john@example.com', '123'))

cursor.execute('''
INSERT INTO conversation_record (timestamp, message, respond, user_emotion) VALUES (?, ?, ?, ?)
''', ('2023-07-22 12:34:56', 'Hello, how are you?', 'hi', '開心'))

conn.commit()

cursor.close()
conn.close()
