# iCare 日照機器人
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
2. Download [Whisper](https://github.com/openai/whisper) and place it in your folder.
3. Place both in the same folder with the following structure:
    ```
    icare/
    ├── MuseTalk/
    ├── Whisper/
    ├── main.py
    ├── requirements.txt
    ├── templates/
    └── static/
    ```
4. Create a new environment:
    ```
    conda create --name myenv python=3.8
    ```
5. Switch to the environment and run the following command:
    ```
    pip install -r requirements.txt
    ```
6. Get API keys of Azure, OpenAI and D-ID. Then put them in .env file.
7. Run the program:
    ```
    python main.py
    ```
