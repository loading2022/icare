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
from flask import Flask, render_template, request, jsonify, session
from deepface import DeepFace
from dotenv import load_dotenv
from datetime import datetime
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cdist

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

global img_url 
result_queue = queue.Queue()
distances = []

def store_tracks(audio_path):
    connection = f'DRIVER={{SQL SERVER}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(connection)
    cursor = conn.cursor()
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    inference = Inference(model, window="whole")
    track_embedding = inference(audio_path).reshape(1, -1)
    result = calculate_simliarity(audio_path)
    threshold = 0.4
    for dist in result:
        if dist < threshold:
            return render_template('login.html', message = "使用者已建立")
        
    query = "INSERT INTO tracks (Embedding) VALUES (?)"
    cursor.execute(query, track_embedding.tobytes())
    conn.commit()
    cursor.close()
    conn.close()
    return render_template('login.html', message="帳號註冊成功")

def retrieve_tracks():
    connection = f'DRIVER={{SQL SERVER}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(connection)
    cursor = conn.cursor()
    query = "SELECT * FROM tracks"
    cursor.execute(query)
    results = cursor.fetchall()
    tracks = [np.frombuffer(row[1], dtype=np.float32).reshape(1, -1) for row in results]
    return tracks

def calculate_simliarity(audio_path):
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    inference = Inference(model, window="whole")
    old_tracks = retrieve_tracks()
    new_track = inference(audio_path).reshape(1, -1)
    for old_track in old_tracks:
        old_track = old_track.reshape(1, -1)
        distance = cdist(old_track , new_track, metric="cosine")[0,0]
        distances.append(distance)
    return distances

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

def create_did(text, img_url):
    url = "https://api.d-id.com/talks"
    print(f"D-ID: {img_url}")
    payload = {
    "script": {
        "type": "text",
        "subtitles": "false",
        "provider": {
            "type": "microsoft",
            "voice_id": "zh-CN-XiaoxiaoNeural"
        },
        "ssml": "false",
        "input": text
    },
    "config": {
        "fluent": "false",
        "pad_audio": "0.0"
    },
    "source_url": img_url
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": os.getenv("DID_KEY")
    }

    response = requests.post(url, json=payload, headers=headers)
    response_json = json.loads(response.text)
    print(response_json)

    talk_id = response_json['id']
    url = "https://api.d-id.com/talks/"+talk_id
    print(url)
    headers = {
        "accept": "application/json",
        "authorization": os.getenv("DID_KEY")
    }
    while True:
        response = requests.get(url, headers=headers)
        response_json = json.loads(response.text)
        status = response_json['status']
        if status == 'done':
            break
    
    result_url = response_json['result_url']
    
    print(result_url)
    return result_url

def send_role():
    data = request.json
    role = data.get('role')
    print(f"User role selected: {role}")
    return jsonify({'status': 'success', 'role': role})
class ContinuousRecognizer:
    def __init__(self, role, imgurl, user_id):
        self.speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
        self.speech_config.speech_recognition_language="zh-TW"
        self.audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        self.recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=self.audio_config)
        self.last_speech_time = time.time()
        self.timer = None
        self.audio_path = None
        self.current_emotion = "未知"
        self.role = role
        self.imgurl = imgurl
        self.user_id = user_id

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
                currentDateAndTime = datetime.now()
                emotion = self.current_emotion
                cursor = conn.cursor()
                print(self.user_id)
                cursor.execute('''
                INSERT INTO conversation_record (timestamp, user_id, message, respond, user_emotion) VALUES (?, ?, ?, ?, ?)
                ''', (currentDateAndTime, self.user_id, text, response_text, emotion)
                )
                conn.commit()
                cursor.close()
                conn.close()
                result_url = create_did(response_text, self.imgurl)
                print(self.imgurl)
                print("Result URL: {}".format(result_url)) 
                result_queue.put(result_url)  # 將 URL 放入佇列中
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


@app.route('/')
def index():
    return render_template('register.html')

@app.route("/start_recording", methods=["POST"])
def recognize_from_microphone():
    data = request.json
    role = data.get('role')
    avater_img = request.get_json() 
    image_url = avater_img['imageUrl'] 
    user_id = session.get('user_id') 
    print(f'image:{image_url}')
    recognizer = ContinuousRecognizer(role, image_url, user_id)
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
        stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)

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
        stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)

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
        result = calculate_simliarity(filename)

    
        threshold = 0.4
        for dist in result:
            if dist < threshold:
                idx = idx + 1
                return render_template('index.html')
            
        return render_template('register.html', message="尚未註冊")
    except Exception as e:
        return jsonify({"Error": f"錯誤:{str(e)}"}), 500
    
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', use_reloader=False)




