document.addEventListener("DOMContentLoaded", () => {
    const recordButton = document.getElementById("recordButton");
    const statusDiv = document.getElementById("status");
    const logDiv = document.getElementById("log");
    const audioPlayer = document.getElementById("audioPlayer");

    let websocket;
    let mediaRecorder;
    let audioChunks = [];

    const connectWebSocket = () => {
        const wsUrl = `ws://${window.location.host}/ws`;
        websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            console.log("Connected to WebSocket server.");
            statusDiv.textContent = "Ready to speak";
        };

        websocket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            
            if (message.type === 'user_text') {
                logDiv.innerHTML += `<p class="user-text"><strong>You:</strong> ${message.data}</p>`;
            } else if (message.type === 'llm_text') {
                logDiv.innerHTML += `<p class="llm-text"><strong>Assistant:</strong> ${message.data}</p>`;
            } else if (message.type === 'audio') {
                statusDiv.textContent = "Playing response...";
                const audioData = atob(message.data);
                const audioBytes = new Uint8Array(audioData.length);
                for (let i = 0; i < audioData.length; i++) {
                    audioBytes[i] = audioData.charCodeAt(i);
                }
                const audioBlob = new Blob([audioBytes], { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                audioPlayer.src = audioUrl;
                audioPlayer.play();
                statusDiv.textContent = "Ready to speak";

            } else if (message.type === 'error') {
                statusDiv.textContent = message.message;
            }
            logDiv.scrollTop = logDiv.scrollHeight;
        };

        websocket.onclose = () => {
            console.log("WebSocket disconnected. Retrying in 2 seconds...");
            statusDiv.textContent = "Connection lost. Retrying...";
            setTimeout(connectWebSocket, 2000);
        };

        websocket.onerror = (error) => {
            console.error("WebSocket error:", error);
            websocket.close();
        };
    };

    const startRecording = () => {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start(); // Capture a single continuous recording

                recordButton.textContent = "Release to Stop";
                recordButton.classList.add("recording");
                statusDiv.textContent = "Recording...";
                
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = async () => {
                    if (audioChunks.length && websocket.readyState === WebSocket.OPEN) {
                        const blob = new Blob(audioChunks, { type: audioChunks[0].type || 'audio/webm' });
                        const buffer = await blob.arrayBuffer();
                        websocket.send(buffer);
                        websocket.send("END_OF_STREAM");
                    }
                    statusDiv.textContent = "Processing...";
                    recordButton.textContent = "Hold to Speak";
                    recordButton.classList.remove("recording");
                };
            })
            .catch(err => {
                console.error("Error accessing microphone:", err);
                statusDiv.textContent = "Could not access microphone.";
            });
    };

    const stopRecording = () => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
        }
    };
    
    // Use mousedown and mouseup to simulate "hold to speak"
    recordButton.addEventListener("mousedown", startRecording);
    recordButton.addEventListener("mouseup", stopRecording);
    // For mobile
    recordButton.addEventListener("touchstart", startRecording);
    recordButton.addEventListener("touchend", stopRecording);


    // Initial connection
    connectWebSocket();
});
