var start_button = document.querySelector(".start");
var loader = document.querySelector(".loader");
//var img = document.querySelector(".image");
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