var start_button = document.querySelector(".start");
var loader = document.querySelector(".loader");
const upload_image = document.querySelector("#upload-image");
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
/*
function startRecording() {
    axios.post('/start_recording')
        .then(function(response) {
            console.log('錄音已開始，正在等待視頻...');
            img.classList.add("active");
            start_button.classList.add("hidden");
            pollForResultUrl();
        })
        .catch(function(error) {
            console.log('錄音開始失敗:', error);
            document.querySelector('.result').textContent = '錄音開始失敗';
        });
}
*/
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
        body: JSON.stringify({ role: selectedRole, imageUrl:imgUrl }),
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

function uploadImage(){
    const imageInput = document.getElementById('imageInput');
    if (imageInput.files.length > 0) {
        const file = imageInput.files[0];
        const formData = new FormData();
        formData.append('image', file);
        formData.append('type', 'image');
        formData.append('title', 'Simple upload');
        formData.append('description', 'This is a simple image upload in Imgur');

        fetch('https://api.imgur.com/3/image', {
            method: 'POST',
            headers: {
                Authorization: 'Client-ID fd0c9e6694ddbfd'
            },
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            imgUrl = data.data.link;
            console.log(imgUrl);
            //showImage(imgUrl);
        })
        .catch((error) => {
            alert("Error:", error);
        });
    } else {
        alert('請選擇一個圖片文件。');
    }
}

/*
function showImage(imgUrl){
    const img = document.querySelector("#avater-img");
    img.src = imgUrl;
    img.style = "display:block;";
}
*/
