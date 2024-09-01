var start_button = document.querySelector(".start");
var loader = document.querySelector(".loader");
const upload_image = document.querySelector("#upload-image");


window.onload = function() {
    if (message){
        alert(message);
    }
};
function startRecording() {
    axios.post('/start_recording')
        .then(function(response) {
            console.log('錄音已開始，正在等待視頻...');
            //img.classList.add("active");
            start_button.classList.add("hidden");
            pollForResultUrl();
        })
        .catch(function(error) {
            console.log('錄音開始失敗:', error);
            document.querySelector('.result').textContent = '錄音開始失敗';
        });
}

function pollForResultUrl() {
    function checkForVideo() {
        axios.get('/get_result_url')
            .then(function(response) {
                if (response.data.result_url) {
                    console.log('獲取到的視頻 URL:', response.data.result_url);
                    loader.classList.add("hidden");
                    //document.querySelector('.placeholder-image').style.display = 'none';
                    var result_video = document.querySelector(".result");
                    result_video.innerHTML = `<video width="640" height="480" controls autoplay>
                        <source src="${response.data.result_url}" type="video/mp4">
                        您的瀏覽器不支援 video 標籤。
                    </video>`;

                    var videoElement = result_video.querySelector("video");
                    videoElement.addEventListener('canplaythrough', event => {
                        videoElement.play(); // 確保影片能夠自動播放
                    });

                    videoElement.addEventListener('ended', () => {
                        console.log('當前影片播放結束，正在請求下一段影片...');
                        fetch('/restart_recognition', {method: 'POST'})  // 發送 POST 請求到後端以重新啟動語音辨識
                        .then(response => response.json())
                        .then(data => console.log(data));
                        loader.classList.add("active");
                        setTimeout(checkForVideo, 3000); // 當前影片播完後稍等一秒再請求新影片
                    });

                } else if (response.data.error) {
                    console.log('錯誤訊息:', response.data.error);
                    //document.querySelector('.result').innerHTML = '<p>小妤休息中...</p>';
                    loader.classList.add("active");
                    setTimeout(checkForVideo, 3000); 
                } else {
                    console.log('小妤腦力激盪中...');
                    loader.classList.add("active");
                    //document.querySelector('.placeholder-image').style.display = 'block'; 
                    setTimeout(checkForVideo, 3000); // 沒有錯誤但也沒有 URL，繼續檢查
                }
            })
            .catch(function(error) {
                console.log('在獲取結果 URL 時發生錯誤:', error);
                //document.querySelector('.result').innerHTML = '<p>小妤休息中...</p>';
                //document.querySelector('.placeholder-image').style.display = 'block'; 
                loader.classList.add("active");
                setTimeout(checkForVideo, 3000); // 確保即使發生錯誤也能重新嘗試
            });
    }
    checkForVideo(); // 開始第一次檢查
}

var start_button = document.querySelector(".start");
var img = document.querySelector(".image");

function pollForResultUrl() {
    function checkForVideo() {
        axios.get('/get_result_url')
            .then(function(response) {
                console.log(response.data.result_url);
                if (response.data.result_url) {
                    console.log('獲取到的視頻 URL:', response.data.result_url);
                    var result_video = document.querySelector(".result");
                    result_video.innerHTML = `<video width="640" height="480" controls autoplay>
                        <source src="${response.data.result_url}" type="video/mp4">
                        您的瀏覽器不支援 video 標籤。
                    </video>`;

                    var videoElement = result_video.querySelector("video");
                    videoElement.addEventListener('canplaythrough', event => {
                        videoElement.play(); // 確保影片能夠自動播放
                    });

                    videoElement.addEventListener('ended', () => {
                        console.log('當前影片播放結束，正在請求下一段影片...');
                        setTimeout(checkForVideo, 5000); // 當前影片播完後稍等一秒再請求新影片
                    });

                } else if (response.data.error) {
                    console.log('錯誤訊息:', response.data.error);
                    document.querySelector('.result').innerHTML = '<p>' + response.data.error + '</p>';
                    setTimeout(checkForVideo, 5000); // 如果有錯誤，仍然繼續檢查
                } else {
                    console.log('視頻尚未準備好，再次檢查...');
                    setTimeout(checkForVideo, 5000); // 沒有錯誤但也沒有 URL，繼續檢查
                }
            })
            .catch(function(error) {
                console.log('在獲取結果 URL 時發生錯誤:', error);
                document.querySelector('.result').innerHTML = '<p>獲取視頻失敗</p>';
                setTimeout(checkForVideo, 5000); // 確保即使發生錯誤也能重新嘗試
            });
    }
    checkForVideo(); // 開始第一次檢查
}

let selectedRole = '';
let imgUrl = '';
function sendUserRole(role) {
    selectedRole = role;
    document.getElementById('role-name').innerText = role;
    document.getElementById('selected-role').style.display = 'block';
}

function startRecording() {
    if (selectedRole === '') {
        alert('請先選擇角色');
        return;
    }

    fetch('/start_recording', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ role: selectedRole, imgUrl:imgUrl }),
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
        pollForResultUrl();
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function uploadImage() {
    const imageInput = document.getElementById('imageInput');
    if (imageInput.files.length > 0) {
        const file = imageInput.files[0];
        const formData = new FormData();
        formData.append('image', file);

        fetch('/upload_image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            imgUrl = data.filepath;  // 保存圖片路徑到全局變數
            console.log('Image saved at:', imgUrl);
            // 不再需要額外發送 imgUrl 到後端，因為會在 startRecording 時傳遞
        })
        .catch((error) => {
            alert("Error:", error);
        });
    } else {
        alert('請選擇一個圖片文件。');
    }
}




function register(){
    
}

function toggleTime(day) {
    const timeCheckboxes = document.getElementById(`${day}-times`);
    if (timeCheckboxes.style.display === "none" || timeCheckboxes.style.display === "") {
        timeCheckboxes.style.display = "block";
    } else {
        timeCheckboxes.style.display = "none";
    }
}

function toggleInput(timeOfDay) {
    const inputField = document.getElementById(`${timeOfDay}-time`);
    if (inputField.style.display === "none" || inputField.style.display === "") {
        inputField.style.display = "inline";
    } else {
        inputField.style.display = "none";
    }
}


function submitForm() {
    const days = ['monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    const data = {};

    days.forEach(day => {
        data[day] = {
            morning: document.querySelector(`#${day}-times input[value='morning']`).checked ? 1 : 0,
            noon: document.querySelector(`#${day}-times input[value='noon']`).checked ? 1 : 0,
            evening: document.querySelector(`#${day}-times input[value='evening']`).checked ? 1 : 0,
            morning_note: document.querySelector(`#${day}-times input[name='morning-time']`).value || "",
            noon_note: document.querySelector(`#${day}-times input[name='noon-time']`).value || "",
            evening_note: document.querySelector(`#${day}-times input[name='evening-time']`).value || "",
        };
    });

    console.log('Sending data:', JSON.stringify(data));  // 调试输出

    fetch('/submit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(result => {
        alert('資料已成功提交！');
    })
    .catch(error => {
        console.error('Error:', error);
    });
}


function resetForm() {
    document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => checkbox.checked = false);
    document.querySelectorAll('.text-input').forEach(input => input.value = '');
}


function resetForm() {
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(checkbox => checkbox.checked = false);
    const textboxes = document.querySelectorAll('.text-input');
    textboxes.forEach(textbox => {
        textbox.value = '';
    });
    const timeCheckboxes = document.querySelectorAll('.time-checkboxes');
    timeCheckboxes.forEach(timeCheckbox => timeCheckbox.style.display = 'none');
}

function toggleInputs(dayId) {
    const inputsDiv = document.getElementById(dayId + '-inputs');
    const checkbox = document.getElementById(dayId);

    if (checkbox.checked) {
        inputsDiv.style.display = 'block';
    } else {
        inputsDiv.style.display = 'none';
    }
}

function submitForm_comeback() {
    const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    const data = {};

    days.forEach(day => {
        const departmentInput = document.querySelector(`#${day}-inputs input[type='text']`);
        const timeInput = document.querySelector(`#${day}-inputs input[type='time']`);

        if (departmentInput && timeInput) {
            data[day] = {
                department: departmentInput.value || "",
                time: timeInput.value || "",
            };
        } else {
            data[day] = {
                department: "",
                time: ""
            };
        }
    });

    console.log('Sending data:', JSON.stringify(data));  // 調試輸出

    fetch('/submit_clinic', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(result => {
        if (result.message) {
            alert('資料已成功提交！');
        } else {
            alert('提交失敗，請重試。');
        }
    })
    .catch(error => {
        console.error('提交失敗：', error);
        alert('提交失敗，請重試。');
    });
}


function resetForm_comeback() {
    document.getElementById('comebackForm').reset();
    const inputs = document.querySelectorAll('.inputs');
    inputs.forEach(inputDiv => inputDiv.style.display = 'none');
}


function uploadVoice() {
    const fileInput = document.getElementById('voiceInput');
    const file = fileInput.files[0]; 
    console.log(file);
    if (!file) {
        alert('請選擇檔案');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // 使用 Fetch API 來發送檔案
    fetch('/upload_voice', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
        alert('上傳成功!');
    })
    .catch((error) => {
        console.error('Error:', error);
        alert('上傳失敗，請重新上傳');
    });
}



