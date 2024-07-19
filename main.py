import os
import time
import threading
import queue
import azure.cognitiveservices.speech as speechsdk
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory, session
import openai
import uuid
import requests
import json
import cv2
import numpy as np
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image

# 變數定義
subscription_key = '295f2324100845f3ada64f6cb0f598e5'
region = 'eastus2'
openai_api_key = "sk-proj-5zkksvA3JQIJojvGJ05UT3BlbkFJ7A4stZjfxMC5fM0uFv8o"

app = Flask(__name__)
app.secret_key = 'JQDxzgL0hjIHBO-xcCwGMw' 

global img_url 
# 線程安全的佇列用來存儲結果 URL
result_queue = queue.Queue()
#result_queue.put("https://studio.d-id.com/share?id=401404189a5049f2d9546e8a48e32fa1&utm_source=copy")
# 設定 GPT API
def call_gpt(text, role, emotion):
    openai.api_key = openai_api_key
    print(emotion)
    # 根據角色生成不同的 prompt
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

# 建立 D-ID URL
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
        "authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik53ek53TmV1R3ptcFZTQjNVZ0J4ZyJ9.eyJodHRwczovL2QtaWQuY29tL2ZlYXR1cmVzIjoiIiwiaHR0cHM6Ly9kLWlkLmNvbS9zdHJpcGVfcHJvZHVjdF9pZCI6InByb2RfTXRkY2RrM29nR3hkNlkiLCJodHRwczovL2QtaWQuY29tL3N0cmlwZV9jdXN0b21lcl9pZCI6ImN1c19ROXF5a0E4UThTWnR4SyIsImh0dHBzOi8vZC1pZC5jb20vc3RyaXBlX3Byb2R1Y3RfbmFtZSI6ImxpdGUiLCJodHRwczovL2QtaWQuY29tL3N0cmlwZV9zdWJzY3JpcHRpb25faWQiOiJzdWJfMVBKWEdESnhFS1oyekF5blVMRldtVjF0IiwiaHR0cHM6Ly9kLWlkLmNvbS9zdHJpcGVfYmlsbGluZ19pbnRlcnZhbCI6Im1vbnRoIiwiaHR0cHM6Ly9kLWlkLmNvbS9zdHJpcGVfcGxhbl9ncm91cCI6ImRlaWQtbGl0ZSIsImh0dHBzOi8vZC1pZC5jb20vc3RyaXBlX3ByaWNlX2lkIjoicHJpY2VfMU5TYmR6SnhFS1oyekF5bmNYN1dRVWoxIiwiaHR0cHM6Ly9kLWlkLmNvbS9zdHJpcGVfcHJpY2VfY3JlZGl0cyI6IjY0IiwiaHR0cHM6Ly9kLWlkLmNvbS9jaGF0X3N0cmlwZV9zdWJzY3JpcHRpb25faWQiOiIiLCJodHRwczovL2QtaWQuY29tL2NoYXRfc3RyaXBlX3ByaWNlX2NyZWRpdHMiOiIiLCJodHRwczovL2QtaWQuY29tL2NoYXRfc3RyaXBlX3ByaWNlX2lkIjoiIiwiaHR0cHM6Ly9kLWlkLmNvbS9wcm92aWRlciI6Imdvb2dsZS1vYXV0aDIiLCJodHRwczovL2QtaWQuY29tL2lzX25ldyI6ZmFsc2UsImh0dHBzOi8vZC1pZC5jb20vYXBpX2tleV9tb2RpZmllZF9hdCI6IjIwMjQtMDUtMjNUMDg6NTI6NDUuMTc0WiIsImh0dHBzOi8vZC1pZC5jb20vb3JnX2lkIjoiIiwiaHR0cHM6Ly9kLWlkLmNvbS9hcHBzX3Zpc2l0ZWQiOlsiU3R1ZGlvIl0sImh0dHBzOi8vZC1pZC5jb20vY3hfbG9naWNfaWQiOiIiLCJodHRwczovL2QtaWQuY29tL2NyZWF0aW9uX3RpbWVzdGFtcCI6IjIwMjMtMTEtMjNUMDg6MTM6MjEuMTQ4WiIsImh0dHBzOi8vZC1pZC5jb20vYXBpX2dhdGV3YXlfa2V5X2lkIjoiYmo4eHpxODdkMiIsImh0dHBzOi8vZC1pZC5jb20vdXNhZ2VfaWRlbnRpZmllcl9rZXkiOiJ1c2dfQUU3QXhGWERucTdzOWtFYnJHbjR4IiwiaHR0cHM6Ly9kLWlkLmNvbS9oYXNoX2tleSI6InBlZTlNOUd0OE82eFAtRi1teXNjdSIsImh0dHBzOi8vZC1pZC5jb20vcHJpbWFyeSI6dHJ1ZSwiaHR0cHM6Ly9kLWlkLmNvbS9lbWFpbCI6IjExMXl6dWltYmlsYWJAZ21haWwuY29tIiwiaHR0cHM6Ly9kLWlkLmNvbS9wYXltZW50X3Byb3ZpZGVyIjoic3RyaXBlIiwiaXNzIjoiaHR0cHM6Ly9hdXRoLmQtaWQuY29tLyIsInN1YiI6Imdvb2dsZS1vYXV0aDJ8MTA1OTI5NDcwMjAyMTkxMjc4ODk1IiwiYXVkIjpbImh0dHBzOi8vZC1pZC51cy5hdXRoMC5jb20vYXBpL3YyLyIsImh0dHBzOi8vZC1pZC51cy5hdXRoMC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNzIxMTMzNTU4LCJleHAiOjE3MjEyMTk5NTgsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgcmVhZDpjdXJyZW50X3VzZXIgdXBkYXRlOmN1cnJlbnRfdXNlcl9tZXRhZGF0YSBvZmZsaW5lX2FjY2VzcyIsImF6cCI6Ikd6ck5JMU9yZTlGTTNFZURSZjNtM3ozVFN3MEpsUllxIn0.VOWyKhXnejmtNywbGLVBAmVPsc2np3aQY5PkTMz1YKSbr2nXzH8ig6lJxzbuH4rtxYJ_UxasCnvRr95sIEqvxBHNhI-TroD-POm1cCCnZhFr1Ggg5OrvESBXUqyTnACzIoLyQG8AEyc544ivaTJk-z4cbn78u5aKD3b1zn1B_EkaLJ_BJdQZ9J55I2J4_JwHJe_FAHSB8Jy5lvhiNbUiDBMqisVxJd-i399U1qG6alAGqxvjiAJcq8AWJyWrCAL6kcpNDE7nQCkEMCPbSVx5jXIhZXnLbSH9JWYZsZXeQUh8PgRmUpTitRszPqg0B7Guf9mFVtpaWyXciY2BsDsgAQ"
    }

    response = requests.post(url, json=payload, headers=headers)
    response_json = json.loads(response.text)
    print(response_json)

    talk_id = response_json['id']
    url = "https://api.d-id.com/talks/"+talk_id
    print(url)
    headers = {
        "accept": "application/json",
        "authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik53ek53TmV1R3ptcFZTQjNVZ0J4ZyJ9.eyJodHRwczovL2QtaWQuY29tL2ZlYXR1cmVzIjoiIiwiaHR0cHM6Ly9kLWlkLmNvbS9zdHJpcGVfcHJvZHVjdF9pZCI6InByb2RfTXRkY2RrM29nR3hkNlkiLCJodHRwczovL2QtaWQuY29tL3N0cmlwZV9jdXN0b21lcl9pZCI6ImN1c19ROXF5a0E4UThTWnR4SyIsImh0dHBzOi8vZC1pZC5jb20vc3RyaXBlX3Byb2R1Y3RfbmFtZSI6ImxpdGUiLCJodHRwczovL2QtaWQuY29tL3N0cmlwZV9zdWJzY3JpcHRpb25faWQiOiJzdWJfMVBKWEdESnhFS1oyekF5blVMRldtVjF0IiwiaHR0cHM6Ly9kLWlkLmNvbS9zdHJpcGVfYmlsbGluZ19pbnRlcnZhbCI6Im1vbnRoIiwiaHR0cHM6Ly9kLWlkLmNvbS9zdHJpcGVfcGxhbl9ncm91cCI6ImRlaWQtbGl0ZSIsImh0dHBzOi8vZC1pZC5jb20vc3RyaXBlX3ByaWNlX2lkIjoicHJpY2VfMU5TYmR6SnhFS1oyekF5bmNYN1dRVWoxIiwiaHR0cHM6Ly9kLWlkLmNvbS9zdHJpcGVfcHJpY2VfY3JlZGl0cyI6IjY0IiwiaHR0cHM6Ly9kLWlkLmNvbS9jaGF0X3N0cmlwZV9zdWJzY3JpcHRpb25faWQiOiIiLCJodHRwczovL2QtaWQuY29tL2NoYXRfc3RyaXBlX3ByaWNlX2NyZWRpdHMiOiIiLCJodHRwczovL2QtaWQuY29tL2NoYXRfc3RyaXBlX3ByaWNlX2lkIjoiIiwiaHR0cHM6Ly9kLWlkLmNvbS9wcm92aWRlciI6Imdvb2dsZS1vYXV0aDIiLCJodHRwczovL2QtaWQuY29tL2lzX25ldyI6ZmFsc2UsImh0dHBzOi8vZC1pZC5jb20vYXBpX2tleV9tb2RpZmllZF9hdCI6IjIwMjQtMDUtMjNUMDg6NTI6NDUuMTc0WiIsImh0dHBzOi8vZC1pZC5jb20vb3JnX2lkIjoiIiwiaHR0cHM6Ly9kLWlkLmNvbS9hcHBzX3Zpc2l0ZWQiOlsiU3R1ZGlvIl0sImh0dHBzOi8vZC1pZC5jb20vY3hfbG9naWNfaWQiOiIiLCJodHRwczovL2QtaWQuY29tL2NyZWF0aW9uX3RpbWVzdGFtcCI6IjIwMjMtMTEtMjNUMDg6MTM6MjEuMTQ4WiIsImh0dHBzOi8vZC1pZC5jb20vYXBpX2dhdGV3YXlfa2V5X2lkIjoiYmo4eHpxODdkMiIsImh0dHBzOi8vZC1pZC5jb20vdXNhZ2VfaWRlbnRpZmllcl9rZXkiOiJ1c2dfQUU3QXhGWERucTdzOWtFYnJHbjR4IiwiaHR0cHM6Ly9kLWlkLmNvbS9oYXNoX2tleSI6InBlZTlNOUd0OE82eFAtRi1teXNjdSIsImh0dHBzOi8vZC1pZC5jb20vcHJpbWFyeSI6dHJ1ZSwiaHR0cHM6Ly9kLWlkLmNvbS9lbWFpbCI6IjExMXl6dWltYmlsYWJAZ21haWwuY29tIiwiaHR0cHM6Ly9kLWlkLmNvbS9wYXltZW50X3Byb3ZpZGVyIjoic3RyaXBlIiwiaXNzIjoiaHR0cHM6Ly9hdXRoLmQtaWQuY29tLyIsInN1YiI6Imdvb2dsZS1vYXV0aDJ8MTA1OTI5NDcwMjAyMTkxMjc4ODk1IiwiYXVkIjpbImh0dHBzOi8vZC1pZC51cy5hdXRoMC5jb20vYXBpL3YyLyIsImh0dHBzOi8vZC1pZC51cy5hdXRoMC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNzIxMTMzNTU4LCJleHAiOjE3MjEyMTk5NTgsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgcmVhZDpjdXJyZW50X3VzZXIgdXBkYXRlOmN1cnJlbnRfdXNlcl9tZXRhZGF0YSBvZmZsaW5lX2FjY2VzcyIsImF6cCI6Ikd6ck5JMU9yZTlGTTNFZURSZjNtM3ozVFN3MEpsUllxIn0.VOWyKhXnejmtNywbGLVBAmVPsc2np3aQY5PkTMz1YKSbr2nXzH8ig6lJxzbuH4rtxYJ_UxasCnvRr95sIEqvxBHNhI-TroD-POm1cCCnZhFr1Ggg5OrvESBXUqyTnACzIoLyQG8AEyc544ivaTJk-z4cbn78u5aKD3b1zn1B_EkaLJ_BJdQZ9J55I2J4_JwHJe_FAHSB8Jy5lvhiNbUiDBMqisVxJd-i399U1qG6alAGqxvjiAJcq8AWJyWrCAL6kcpNDE7nQCkEMCPbSVx5jXIhZXnLbSH9JWYZsZXeQUh8PgRmUpTitRszPqg0B7Guf9mFVtpaWyXciY2BsDsgAQ"
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
# 語音辨識類別
class ContinuousRecognizer:
    def __init__(self, role, imgurl):
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
            if text.strip():  # 確認文本不為空
                print(f"Responding to text: {text}")  # 確認此函數被調用
                response_text = call_gpt(text, self.role, self.current_emotion)
                print("GPT Response: {}".format(response_text))  # 檢查 GPT 回應
                result_url = create_did(response_text, self.imgurl)
                print(self.imgurl)
                print("Result URL: {}".format(result_url))  # 確認 URL
                result_queue.put(result_url)  # 將 URL 放入佇列中
        except Exception as e:
            print(f"Error in respond: {e}")  # 打印任何錯誤
    
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
                    analyze = analyze[0]  # 確保返回值為字典而不是列表
                self.current_emotion = text_obj.get(analyze['dominant_emotion'], '未知')  # 取得情緒文字
                #print(f"Detected Emotion: {self.current_emotion}")
            except Exception as e:
                print(f"Error: {e}")
                pass
            cv2.imshow('Emotion Recognition', img)
            if cv2.waitKey(10) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()



# Flask 路由
@app.route('/')
def index():
    return render_template('index.html')

def send_role():
    data = request.json
    role = data.get('role')
    print(f"User role selected: {role}")
    return jsonify({'status': 'success', 'role': role})

@app.route("/start_recording", methods=["POST"])
def recognize_from_microphone():
    data = request.json
    role = data.get('role')
    avater_img = request.get_json() 
    image_url = avater_img['imageUrl'] 
    print(f'image:{image_url}')
    recognizer = ContinuousRecognizer(role, image_url)
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

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', use_reloader=False)




