# iCare 日照機器人
## OverView
iCare: A System for Conversing with the Elderly to Alleviate Dementia
1. Utilizes **Deepface** and **OpenCV** for emotion recognition, and **Azure Speech to Text** to convert speech into text.
2. The recognized emotions and converted text are used as inputs to generate responses through the **OpenAI API** (GPT).
3. The generated response is then passed to **MuseTalk** to create a virtual avatar.

### 待開發功能
> 吃藥回診提醒:alert <br>
> 回報問題:report <br>
> 語音 style:style <br>

一律推到 branch 確定再 push 回 main

### Technologies Used
1. Azure Speech to text
2. OpenAI API
3. MuseTalk
4. Deepface & OpenCV
5. Whisper
## Installation Guide

1. Download [MuseTalk](https://github.com/TMElyralab/MuseTalk) and place it in your folder.
2. Place both in the same folder with the following structure:
    ```
    icare/
    ├── MuseTalk/
    ├── main.py
    ├── requirements.txt
    ├── templates/
    └── static/
    ```
3. Create a new environment:
    ```
    conda create --name myenv python=3.8
    ```
4. Switch to the environment and run the following command:
    ```
    pip install -r requirements.txt
    ```
5. Get API keys of Azure, OpenAI. Then put them in .env file.
6. Run the program:
    ```
    python main.py
    ```
