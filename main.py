import os
import time
import threading
import queue
import azure.cognitiveservices.speech as speechsdk
import openai
import requests
import json
import cv2
import pyodbc    
import pyaudio
import wave
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from deepface import DeepFace
from dotenv import load_dotenv
from datetime import datetime
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cdist
from gtts import gTTS
import sqlite3
import yaml
import subprocess

#import VALL-E-X 
from VALLEX.utils.prompt_making import make_prompt
from VALLEX.utils.generation import generate_audio, preload_models
from pydub import AudioSegment
from scipy.io.wavfile import write as write_wav
from werkzeug.utils import secure_filename
import datetime

load_dotenv()

subscription_key = os.getenv("AZURE_KEY")
region = os.getenv("AZURE_REGION")
openai_api_key = os.getenv("OPENAI_API_KEY")
server = os.getenv('SQL_SERVER')
database = os.getenv('SQL_DATABASE_NAME')
username = os.getenv('SQL_USERNAME')
password = os.getenv('SQL_PWD')

app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY") 
UPLOAD_FOLDER = '.'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

global img_url 
result_queue = queue.Queue()
distances = []
user_ids=[]
def store_tracks(audio_path):
    connection = f'DRIVER={{SQL SERVER}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(connection)
    cursor = conn.cursor()
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    inference = Inference(model, window="whole")
    track_embedding = inference(audio_path).reshape(1, -1)
    result, ids = calculate_similarity(audio_path)
    threshold = 0.4
    for dist in result:
        if dist < threshold:
            return render_template('login.html', message = "使用者已建立")
        
    query = "INSERT INTO icares (Embedding) VALUES (?)"
    cursor.execute(query, track_embedding.tobytes())
    conn.commit()
    cursor.close()
    conn.close()
    return render_template('login.html', message="帳號註冊成功")

def retrieve_tracks():
    connection = f'DRIVER={{SQL SERVER}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(connection)
    cursor = conn.cursor()
    query = "SELECT ID, Embedding FROM icares"
    cursor.execute(query)
    results = cursor.fetchall()
    
    tracks_dict = {row[0]: np.frombuffer(row[1], dtype=np.float32).reshape(1, -1) for row in results}
    return tracks_dict

def calculate_similarity(audio_path):
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    inference = Inference(model, window="whole")
    tracks_dict = retrieve_tracks()
    new_track = inference(audio_path).reshape(1, -1)
    distances = []
    user_ids = []
    
    for track_id, old_track in tracks_dict.items():
        old_track = old_track.reshape(1, -1)
        distance = cdist(old_track, new_track, metric="cosine")[0, 0]
        distances.append(distance)
        user_ids.append(track_id)
    
    return distances, user_ids

def call_gpt(text, role, emotion):
    openai.api_key = openai_api_key
    print(emotion)
    if role == '照護員':
        prompt = f"你的名字是小妤，使用者為年長者，你須扮演一個陪伴年長者的角色,如果使用者有提及病痛相關的內容，請回應「好的，我會幫您向您的家人反應」，再依據 user 內容及{emotion}給予適當回應，請以較口語化的方式回應，並請勿反駁使用者所說的內容，結果請以繁體中文"
    elif role == '平輩':
        prompt = f"你的名字是小妤，使用者為年長者，你須扮演一個朋友或同齡人的角色，內容須表現出共感的感覺，依據 user 內容及{emotion}給予適當回應，請以較口語化的方式回應，並請勿反駁使用者所說的內容，結果請以繁體中文"
    elif role == '晚輩':
        prompt = f"你的名字是小妤，使用者為年長者，你須扮演一個晚輩或孫輩的角色，依據 user 內容及{emotion}給予適當回應，請以較口語化的方式回應，並請勿反駁使用者所說的內容，結果請以繁體中文"
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content


def send_role():
    data = request.json
    role = data.get('role')
    print(f"User role selected: {role}")
    return jsonify({'status': 'success', 'role': role})

def update_yaml_image_path(yaml_path, image_path, mp3_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    print(f"Updating YAML with image_path: {image_path} and mp3_path: {mp3_path}")  # 調試輸出
    config['task_0']['video_path'] = image_path
    config['task_0']['audio_path'] = mp3_path

    with open(yaml_path, 'w') as file:
        yaml.safe_dump(config, file)

def run_inference(yaml_path):
    command = f"python -m scripts.inference --inference_config {yaml_path}"
    print(f"Running command: {command}")

    try:
        exit_code = os.system(command)
        if exit_code != 0:
            print("Inference failed.")
        else:
            print("Inference succeeded.")
    except Exception as e:
        print(f"Unexpected error: {e}")

class ContinuousRecognizer:
    def __init__(self, role,image_path, user_id):
        self.speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
        self.speech_config.speech_recognition_language="zh-TW"
        self.audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        self.recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=self.audio_config)
        self.last_speech_time = time.time()
        self.timer = None
        self.audio_path = None
        self.current_emotion = "未知"
        self.role = role
        self.image_path = image_path
        self.user_id = user_id
        print(f"Initialized with image_path: {self.image_path}")

    def start(self):
        self.recognizer.recognized.connect(self.recognized)
        self.recognizer.start_continuous_recognition()
        threading.Thread(target=self.detect_emotion).start()

    def recognized(self, args):
        if args.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            self.last_speech_time = time.time()
            print("Recognized: {}".format(args.result.text))
            if self.timer:
                self.timer.cancel()
            self.timer = threading.Timer(3.0, self.respond, [args.result.text])
            self.timer.start()

    def respond(self, text):
        try:
            if text.strip():
                print(f"Responding to text: {text}") 
                response_text = call_gpt(text, self.role, self.current_emotion)
                print("GPT Response: {}".format(response_text))  
                currentDateAndTime = datetime.datetime.now()
                emotion = self.current_emotion
                print(self.user_id)
                connection = f'DRIVER={{SQL SERVER}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
                conn = pyodbc.connect(connection)
                cursor = conn.cursor()
                print(self.user_id)
                cursor.execute('''
                    INSERT INTO conversation_record (timestamp, user_id, message, respond, user_emotion) 
                    VALUES (?, ?, ?, ?, ?)
                ''', (currentDateAndTime, self.user_id, text, response_text, emotion))
                print(currentDateAndTime, self.user_id, text, response_text, emotion)

                conn.commit()
                cursor.close()
                conn.close()

                preload_models()
                text_prompt = response_text
                audio_array = generate_audio(text_prompt, prompt=self.user_id)
                write_wav(f"response_{currentDateAndTime.strftime('%Y%m%d_%H%M%S')}.wav", 16000, audio_array)
                audio = AudioSegment.from_wav(f"response_{currentDateAndTime.strftime('%Y%m%d_%H%M%S')}.wav")
                mp3_filename = f"response_{currentDateAndTime.strftime('%Y%m%d_%H%M%S')}.mp3"
                audio.export(mp3_filename, format="mp3")

                print(f"MP3 file saved as {mp3_filename}")
                print(f"Using image_path: {self.image_path}")  
                yaml_path = 'configs/inference/test.yaml'
                update_yaml_image_path(yaml_path, self.image_path, mp3_filename)
                run_inference(yaml_path)
        except Exception as e:
            print(f"Error in respond: {e}")  
    
    def detect_emotion(self):
        text_obj = {
            'angry': '生氣',
            'disgust': '噁心',
            'fear': '害怕',
            'happy': '開心',
            'sad': '難過',
            'surprise': '驚訝',
            'neutral': '正常'
        }
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame")
                break
            img = cv2.resize(frame, (384, 240))
            try:
                analyze = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                if isinstance(analyze, list):
                    analyze = analyze[0]  
                self.current_emotion = text_obj.get(analyze['dominant_emotion'], '未知')  
            except Exception as e:
                print(f"Error: {e}")
                pass
            cv2.imshow('Emotion Recognition', img)
            if cv2.waitKey(10) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

def save_to_database(user_id):
        data=request.json
        print(user_id)
        connection = f'DRIVER={{SQL SERVER}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
        conn = pyodbc.connect(connection)
        cursor = conn.cursor()

        query_check = "SELECT COUNT(*) FROM medication_reminders WHERE icare_id = ?"
        cursor.execute(query_check, (user_id,))
        exists = cursor.fetchone()[0] > 0

        if exists:
            # 如果记录存在，更新所有天的数据
            query_update = """
            UPDATE medication_reminders SET 
            Monday_morning = ?, Monday_noon = ?, Monday_evening = ?, Monday_morning_note = ?, Monday_noon_note = ?, Monday_evening_note = ?,
            Tuesday_morning = ?, Tuesday_noon = ?, Tuesday_evening = ?, Tuesday_morning_note = ?, Tuesday_noon_note = ?, Tuesday_evening_note = ?,
            Wednesday_morning = ?, Wednesday_noon = ?, Wednesday_evening = ?, Wednesday_morning_note = ?, Wednesday_noon_note = ?, Wednesday_evening_note = ?,
            Thursday_morning = ?, Thursday_noon = ?, Thursday_evening = ?, Thursday_morning_note = ?, Thursday_noon_note = ?, Thursday_evening_note = ?,
            Friday_morning = ?, Friday_noon = ?, Friday_evening = ?, Friday_morning_note = ?, Friday_noon_note = ?, Friday_evening_note = ?,
            Saturday_morning = ?, Saturday_noon = ?, Saturday_evening = ?, Saturday_morning_note = ?, Saturday_noon_note = ?, Saturday_evening_note = ?,
            Sunday_morning = ?, Sunday_noon = ?, Sunday_evening = ?, Sunday_morning_note = ?, Sunday_noon_note = ?, Sunday_evening_note = ?
            WHERE icare_id = ?
            """
            values = []
            for day, day_data in data.items():
                values.extend([
                    day_data['morning'], day_data['noon'], day_data['evening'],
                    day_data['morning_note'], day_data['noon_note'], day_data['evening_note']
                ])
            values.append(user_id)
            cursor.execute(query_update, tuple(values))
        else:
            # 如果记录不存在，插入新的记录
            query_insert = """
            INSERT INTO medication_reminders (
                icare_id, 
                Monday_morning, Monday_noon, Monday_evening, Monday_morning_note, Monday_noon_note, Monday_evening_note,
                Tuesday_morning, Tuesday_noon, Tuesday_evening, Tuesday_morning_note, Tuesday_noon_note, Tuesday_evening_note,
                Wednesday_morning, Wednesday_noon, Wednesday_evening, Wednesday_morning_note, Wednesday_noon_note, Wednesday_evening_note,
                Thursday_morning, Thursday_noon, Thursday_evening, Thursday_morning_note, Thursday_noon_note, Thursday_evening_note,
                Friday_morning, Friday_noon, Friday_evening, Friday_morning_note, Friday_noon_note, Friday_evening_note,
                Saturday_morning, Saturday_noon, Saturday_evening, Saturday_morning_note, Saturday_noon_note, Saturday_evening_note,
                Sunday_morning, Sunday_noon, Sunday_evening, Sunday_morning_note, Sunday_noon_note, Sunday_evening_note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?,?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            values = [user_id]
            for day, day_data in data.items():
                values.extend([
                    day_data['morning'], day_data['noon'], day_data['evening'],
                    day_data['morning_note'], day_data['noon_note'], day_data['evening_note']
                ])
            cursor.execute(query_insert, tuple(values))

        conn.commit()
        conn.close()

def get_user_medication_reminders(user_id):
    connection = f'DRIVER={{SQL SERVER}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(connection)
    cursor = conn.cursor()

    # 查詢用戶的藥物提醒資料
    query = "SELECT * FROM medication_reminders WHERE icare_id = ?"
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()

    conn.close()

    # 中英文字段对照表
    field_translation = {
        'Monday_morning': '星期一早上',
        'Monday_noon': '星期一中午',
        'Monday_evening': '星期一晚上',
        'Monday_morning_note': '星期一早上，服用之藥物',
        'Monday_noon_note': '星期一中午，服用之藥物',
        'Monday_evening_note': '星期一晚上，服用之藥物',
        'Tuesday': '星期二',
        'Tuesday_morning': '星期二早上',
        'Tuesday_noon': '星期二中午',
        'Tuesday_evening': '星期二晚上',
        'Tuesday_morning_note': '星期二早上，服用之藥物',
        'Tuesday_noon_note': '星期二中午，服用之藥物',
        'Tuesday_evening_note': '星期二晚上，服用之藥物',
        'Wednesday': '星期三',
        'Wednesday_morning': '星期三早上',
        'Wednesday_noon': '星期三中午',
        'Wednesday_evening': '星期三晚上',
        'Wednesday_morning_note': '星期三早上，服用之藥物',
        'Wednesday_noon_note': '星期三中午，服用之藥物',
        'Wednesday_evening_note': '星期三晚上，服用之藥物',
        'Thursday': '星期四',
        'Thursday_morning': '星期四早上',
        'Thursday_noon': '星期四中午',
        'Thursday_evening': '星期四晚上',
        'Thursday_morning_note': '星期四早上，服用之藥物',
        'Thursday_noon_note': '星期四中午，服用之藥物',
        'Thursday_evening_note': '星期四晚上，服用之藥物',
        'Friday': '星期五',
        'Friday_morning': '星期五早上',
        'Friday_noon': '星期五中午',
        'Friday_evening': '星期五晚上',
        'Friday_morning_note': '星期五早上，服用之藥物',
        'Friday_noon_note': '星期五中午，服用之藥物',
        'Friday_evening_note': '星期五晚上，服用之藥物',
        'Saturday': '星期六',
        'Saturday_morning': '星期六早上',
        'Saturday_noon': '星期六中午',
        'Saturday_evening': '星期六晚上',
        'Saturday_morning_note': '星期六早上，服用之藥物',
        'Saturday_noon_note': '星期六中午，服用之藥物',
        'Saturday_evening_note': '星期六晚上，服用之藥物',
        'Sunday': '星期日',
        'Sunday_morning': '星期日早上',
        'Sunday_noon': '星期日中午',
        'Sunday_evening': '星期日晚上',
        'Sunday_morning_note': '星期日早上，服用之藥物',
        'Sunday_noon_note': '星期日中午，服用之藥物',
        'Sunday_evening_note': '星期日晚上，服用之藥物'
    }

    if result:
        # 返回結果為字典形式，便於後續處理
        columns = [column[0] for column in cursor.description]
        print(columns)
        reminders = dict(zip(columns, result))
        filtered_reminders = {}
        for k, v in reminders.items():
            if str(v) != '0' and v not in (None, '','1') and k in field_translation:
                filtered_reminders[field_translation[k]] = v

        return filtered_reminders
    return None

def get_current_day():
    return datetime.datetime.today().strftime('%A')

def check_medication_reminder(user_id):
    current_day = get_current_day()
    
    connection = f'DRIVER={{SQL SERVER}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(connection)
    cursor = conn.cursor()

    query = f"SELECT {current_day}_morning_note, {current_day}_noon_note, {current_day}_evening_note FROM medication_reminders WHERE icare_id = ?"
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()

    reminder_message = None 

    if result:
        morning_note, noon_note, evening_note = result
        messages = []

        if morning_note not in (0, None, '0', ''):
            messages.append(f"早上: {morning_note}")
        if noon_note not in (0, None, '0', ''):
            messages.append(f"中午: {noon_note}")
        if evening_note not in (0, None, '0', ''):
            messages.append(f"晚上: {evening_note}")

        if messages:
            reminder_message = "提醒：今天有用藥記錄。" + "，".join(messages)
    conn.close()
    return reminder_message

def save_to_database_clinic(user_id):
        data = request.json  
        print(data)
        connection = f'DRIVER={{SQL SERVER}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
        conn = pyodbc.connect(connection)
        cursor = conn.cursor()

        query_check = "SELECT COUNT(*) FROM clinic_reminders WHERE user_id = ?"
        cursor.execute(query_check, (user_id,))
        exists = cursor.fetchone()[0] > 0

        if exists:
            query_update = """
            UPDATE clinic_reminders SET 
            Monday_department = ?, Monday_time = ?,
            Tuesday_department = ?, Tuesday_time = ?,
            Wednesday_department = ?, Wednesday_time = ?,
            Thursday_department = ?, Thursday_time = ?,
            Friday_department = ?, Friday_time = ?,
            Saturday_department = ?, Saturday_time = ?,
            Sunday_department = ?, Sunday_time = ?
            WHERE user_id = ?
            """
            values = []
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
                department = data.get(day, {}).get('department', "")
                time = data.get(day, {}).get('time', "")
                values.extend([department, time])
            values.append(user_id)
            cursor.execute(query_update, tuple(values))
        else:
            query_insert = """
            INSERT INTO clinic_reminders (
                user_id, 
                Monday_department, Monday_time,
                Tuesday_department, Tuesday_time,
                Wednesday_department, Wednesday_time,
                Thursday_department, Thursday_time,
                Friday_department, Friday_time,
                Saturday_department, Saturday_time,
                Sunday_department, Sunday_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            values = [user_id]
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
                department = data.get(day, {}).get('department', "")
                time = data.get(day, {}).get('time', "")
                values.extend([department, time])
            cursor.execute(query_insert, tuple(values))

        conn.commit()
        conn.close()

def get_user_clinic_reminders(user_id):
    connection = f'DRIVER={{SQL SERVER}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(connection)
    cursor = conn.cursor()
    query = "SELECT * FROM clinic_reminders WHERE user_id = ?"
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()

    conn.close()
    field_translation = {
        'Monday_department': '星期一回診科别',
        'Monday_time': '星期一回診時間',
        'Tuesday_department': '星期二回診科别',
        'Tuesday_time': '星期二回診時間',
        'Wednesday_department': '星期三回診科别',
        'Wednesday_time': '星期三回診時間',
        'Thursday_department': '星期四回診科别',
        'Thursday_time': '星期四回診時間',
        'Friday_department': '星期五回診科别',
        'Friday_time': '星期五回診時間',
        'Saturday_department': '星期六回診科别',
        'Saturday_time': '星期六回診時間',
        'Sunday_department': '星期日回診科别',
        'Sunday_time': '星期日回診時間'
    }

    if result:
        columns = [column[0] for column in cursor.description]
        print(columns)
        reminders = dict(zip(columns, result))
        filtered_reminders = {}
        for k, v in reminders.items():
            if str(v) != '0' and v not in (None, '', '1') and k in field_translation:
                filtered_reminders[field_translation[k]] = v

        return filtered_reminders
    return None

def check_clinic_reminder(user_id):
    current_day = get_current_day()
    
    connection = f'DRIVER={{SQL SERVER}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(connection)
    cursor = conn.cursor()

    query = f"SELECT {current_day}_department, {current_day}_time FROM clinic_reminders WHERE user_id = ?"
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()

    reminder_message = None

    if result:
        department, time = result
        messages = []

        if department not in (0, None, '0', ''):
            messages.append(f"科別: {department}")
        if time not in (0, None, '0', ''):
            messages.append(f"時間: {time}")

        if messages:
            reminder_message = "提醒：今天有回診提醒。" + "，".join(messages)
    
    conn.close()
    return reminder_message


@app.route('/')
def index():
    return render_template('register.html')

@app.route("/start_recording", methods=["POST"])
def recognize_from_microphone():
    data = request.json
    role = data.get('role')
    user_id = session.get('user_id') 
    image_path = data.get('imgUrl')
    print(f'image: {image_path}')
    print(f"User ID2 from session: {user_id}")
    session['user_id'] = user_id
    recognizer = ContinuousRecognizer(role,image_path, user_id)
    recognizer.start()

    return jsonify({"status": "recognition started"})


@app.route('/get_result_url', methods=['GET'])
def get_result_url():
    try:
        result_url = result_queue.get_nowait() 
    except queue.Empty:
        return jsonify({"error": "影片尚未準備好"})
    else:
        return jsonify({"result_url": result_url})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') + file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    return jsonify({"filepath": file_path}), 200



@app.route('/register_recording', methods=['GET','POST'])
def register_recording():
    try:
        chunk = 1024                    
        sample_format = pyaudio.paInt16 
        channels = 2                    
        fs = 44100                      
        seconds = 5                     
        filename = "temp.wav"   
        p = pyaudio.PyAudio()         
        stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True, input_device_index=0)

        frames = []                     

        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)          
        stream.stop_stream()             
        stream.close()                   
        p.terminate()

        wf = wave.open(filename, 'wb')   
        wf.setnchannels(channels)        
        wf.setsampwidth(p.get_sample_size(sample_format)) 
        wf.setframerate(fs)              
        wf.writeframes(b''.join(frames)) 
        wf.close()
        result = store_tracks("temp.wav")
        return result

        
    except Exception as e:
        return jsonify({"Error": f"錯誤:{str(e)}"}), 500

@app.route('/login_recording', methods=['GET','POST'])
def login_recording():
    try:
        chunk = 1024                    
        sample_format = pyaudio.paInt16 
        channels = 2                    
        fs = 44100                      
        seconds = 5                     
        filename = "temp.wav"   
        p = pyaudio.PyAudio()         
        stream = p.open(format=sample_format, channels=channels,input_device_index=0, rate=fs, frames_per_buffer=chunk, input=True)

        frames = []                     

        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)          
        stream.stop_stream()             
        stream.close()                   
        p.terminate()

        wf = wave.open(filename, 'wb')   
        wf.setnchannels(channels)        
        wf.setsampwidth(p.get_sample_size(sample_format)) 
        wf.setframerate(fs)              
        wf.writeframes(b''.join(frames)) 
        wf.close()
        result, ids = calculate_similarity(filename)

        threshold = 0.4
        #登入
        for idx, dist in enumerate(result):
            if dist < threshold:
                session["user_id"] = ids[idx]
                user_id = session.get('user_id')
                check_medication_reminder(user_id)
                reminder_message = check_medication_reminder(user_id)
                reminders = get_user_medication_reminders(user_id)
                clinic_reminders = get_user_clinic_reminders(user_id)
                clinic_reminder_message = check_clinic_reminder(user_id)
                if data:
                    return render_template('index.html', reminders=reminders, reminder_message=reminder_message,clinic_reminders=clinic_reminders,clinic_reminder_message=clinic_reminder_message)
                else:
                    return render_template('index.html', reminders=None, reminder_message=None,clinic_reminders=None,clinic_reminder_message=None)
            
        return render_template('register.html', message="尚未註冊")
    except Exception as e:
        return jsonify({"Error": f"錯誤:{str(e)}"}), 500
    
@app.route('/submit', methods=['POST'])
def submit():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User ID not found in session'}), 400
    
    save_to_database(user_id)  
    return jsonify({'status': 'success'}), 200

@app.route('/submit_clinic', methods=['POST'])
def submit_clinic():
    user_id = session.get('user_id')
    try:
        data = request.json  
        save_to_database_clinic(user_id)
        return jsonify({"message": "資料已成功提交"}), 200
    except Exception as e:
        print(f"Error: {e}") 
        return jsonify({"message": "提交失敗", "error": str(e)}), 500

@app.route('/upload_voice', methods=['POST'])
def upload_voice():
    file = request.files['file']
    print(file)
    if file.filename == "":
        return jsonify({'message': '未選擇檔案'}), 400
    if file:
        # 取得原始檔案的擴展名
        extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        # 檢查是否為音檔格式
        if extension in ['wav', 'mp3', 'aac', 'ogg', 'flac']:
            os.makedirs("./upload", exist_ok=True)
            user_id = session.get('user_id')
            if not user_id:
                return jsonify({'message': '使用者未登入'}), 400
            # 將檔案以 user_id 和原始擴展名儲存
            file_path = os.path.join("./upload", f"{user_id}.{extension}")
            file.save(file_path)
            make_prompt(name=user_id, audio_prompt_path=file_path)
            return jsonify({'message': '音檔上傳與處理成功', 'filename': f"{user_id}.{extension}"}), 200
        else:
            return jsonify({'message': '請選擇正確的音檔格式'}), 400
    else:
        return jsonify({'message': '檔案上傳失敗'}), 400

@app.route('/familyregister', methods=['GET', 'POST'])
def family_register():
    connection = f'DRIVER={{SQL SERVER}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(connection)
    cursor = conn.cursor()
    if request.method == 'POST':
        name = request.form['name']
        phone = request.form['phone']
        email = request.form['email']
        family_id = request.form['family-id']
        user_password = request.form['password']
        try:
            cursor.execute('''
                INSERT INTO family (family_name, phone, email, family_id, f_password)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, phone, email, family_id, user_password))
            conn.commit()
            flash('註冊成功！', 'success')
        except pyodbc.IntegrityError:
            flash('該電子郵件已被註冊。請使用不同的電子郵件。', 'error')
        finally:
            pass
    cursor.execute("SELECT DISTINCT ID FROM icares")
    user_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return render_template('familyregister.html', user_ids=user_ids)

@app.route('/familylogin', methods=['GET', 'POST'])
def family_login():
    connection = f'DRIVER={{SQL SERVER}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(connection)
    cursor = conn.cursor()
    if request.method == 'POST':
        email = request.form['email']
        user_password = request.form['password']

        cursor.execute('SELECT * FROM family WHERE email = ? AND f_password = ?', (email, user_password))
        user = cursor.fetchone()
        user_id = user[4]
        family_name = user[1]
        session['family_name'] = family_name
        conn.close()
        if user:
            return redirect(url_for('family_page', user_id=user_id))
        else:
            flash('無效的電子郵件或密碼。', 'error')
            return render_template('familylogin.html')
    conn.close()
    return render_template('familylogin.html')

@app.route('/family_page/<int:user_id>')
def family_page(user_id):
    connection = f'DRIVER={{SQL SERVER}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(connection)
    cursor = conn.cursor()

    # 查詢該家屬的對話紀錄
    cursor.execute('SELECT timestamp, message, respond FROM conversation_record WHERE user_id = ?', (user_id,))
    conversations = cursor.fetchall()
    family_name = session.get('family_name', 'Unknown')
    filtered_conversations = [conv for conv in conversations if '痛' in conv[1] or '不舒服' in conv[1]]
    conn.close()

    return render_template('family.html', family_name=family_name, conversations=filtered_conversations)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', use_reloader=False)



