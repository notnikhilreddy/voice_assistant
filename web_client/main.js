console.log("voice client script loaded");

document.addEventListener("DOMContentLoaded", () => {
    const recordButton = document.getElementById("recordButton");
    const statusDiv = document.getElementById("status");
    const logDiv = document.getElementById("log");
    const audioPlayer = document.getElementById("audioPlayer");
    const autoVADToggle = document.getElementById("autoVADToggle");

    let assistantAppended = false;
    let mediaStream = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let audioQueue = [];
    let isPlaying = false;
    let websocket;

    // VAD / barge-in settings
    let vadEnabled = false;
    let vadRunning = false;
    let vadAnalyser = null;
    let vadData = null;
    const VAD_THRESHOLD = 0.08; // tweak for sensitivity
    const VAD_SPEECH_MS = 250;
    const VAD_SILENCE_MS = 400;
    let vadSpeechMs = 0;
    let vadSilenceMs = 0;

    const bargeIn = () => {
        if (isPlaying) {
            audioPlayer.pause();
            audioPlayer.src = "";
            isPlaying = false;
        }
        audioQueue = [];
    };

    const ensureStream = async () => {
        if (mediaStream) return mediaStream;
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                noiseSuppression: true,
                echoCancellation: true,
                autoGainControl: true,
            },
        });
        return mediaStream;
    };

    const playNextAudio = () => {
        if (isPlaying) return;
        const next = audioQueue.shift();
        if (!next) return;
        isPlaying = true;
        audioPlayer.src = next;
        audioPlayer.play().catch(err => {
            console.error("Playback error:", err);
        });
    };

    audioPlayer.onended = () => {
        isPlaying = false;
        URL.revokeObjectURL(audioPlayer.src);
        playNextAudio();
    };

    const enqueueAudioBase64 = (b64) => {
        if (!b64) return;
        const binary = atob(b64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const blob = new Blob([bytes], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        audioQueue.push(url);
        playNextAudio();
    };

    const appendAssistant = (text) => {
        const p = document.createElement("p");
        p.className = "llm-text";
        p.innerHTML = `<strong>Assistant:</strong> ${text}`;
        logDiv.appendChild(p);
        assistantAppended = true;
    };

    const appendUser = (text) => {
        const p = document.createElement("p");
        p.className = "user-text";
        p.innerHTML = `<strong>You:</strong> ${text}`;
        logDiv.appendChild(p);
    };

    const sendBufferedAudio = async () => {
        if (!audioChunks.length || websocket.readyState !== WebSocket.OPEN) return;
        const blob = new Blob(audioChunks, { type: audioChunks[0].type || "audio/webm" });
        const buffer = await blob.arrayBuffer();
        websocket.send(buffer);
        websocket.send("END_OF_STREAM");
        statusDiv.textContent = "Processing...";
        audioChunks = [];
    };

    const stopRecording = () => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
        }
    };

    const startRecording = async (reason = "manual") => {
        await ensureStream();
        if (mediaRecorder && mediaRecorder.state === "recording") {
            return;
        }
        audioChunks = [];
        mediaRecorder = new MediaRecorder(mediaStream);
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };
        mediaRecorder.onstop = sendBufferedAudio;
        mediaRecorder.start();
        statusDiv.textContent = reason === "vad" ? "Listening..." : "Recording...";
    };

    const startVADLoop = async () => {
        if (!vadEnabled || vadRunning) return;
        await ensureStream();
        const ctx = new AudioContext();
        const source = ctx.createMediaStreamSource(mediaStream);
        vadAnalyser = ctx.createAnalyser();
        vadAnalyser.fftSize = 2048;
        source.connect(vadAnalyser);
        vadData = new Uint8Array(vadAnalyser.fftSize);
        vadRunning = true;
        const frameMs = (vadAnalyser.fftSize / ctx.sampleRate) * 1000;

        const loop = () => {
            if (!vadEnabled) {
                vadRunning = false;
                return;
            }
            vadAnalyser.getByteTimeDomainData(vadData);
            let sum = 0;
            for (let i = 0; i < vadData.length; i++) {
                const v = (vadData[i] - 128) / 128;
                sum += v * v;
            }
            const rms = Math.sqrt(sum / vadData.length);

            if (rms > VAD_THRESHOLD) {
                vadSpeechMs += frameMs;
                vadSilenceMs = 0;
                if (vadSpeechMs > VAD_SPEECH_MS) {
                    bargeIn();
                    startRecording("vad");
                }
            } else {
                vadSilenceMs += frameMs;
                vadSpeechMs = 0;
                if (vadSilenceMs > VAD_SILENCE_MS) {
                    stopRecording();
                }
            }

            requestAnimationFrame(loop);
        };
        loop();
    };

    const connectWebSocket = () => {
        const wsUrl = `ws://${window.location.host}/ws`;
        websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            statusDiv.textContent = "Ready to speak";
        };

        websocket.onmessage = (event) => {
            let message;
            try {
                message = JSON.parse(event.data);
            } catch (e) {
                console.error("Failed to parse message", e, event.data);
                return;
            }
            
            if (message.type === 'user_text') {
                console.log("user_text received", message.data);
                appendUser(message.data);
                assistantAppended = false;
                logDiv.scrollTop = logDiv.scrollHeight;
            } else if (message.type === 'llm_partial') {
                // optional partial display: update last assistant node or create one
                if (!assistantAppended) {
                    appendAssistant(message.data);
                } else {
                    const last = logDiv.querySelector("p.llm-text:last-of-type");
                    if (last) last.innerHTML = `<strong>Assistant:</strong> ${message.data}`;
                }
            } else if (message.type === 'llm_chunk') {
                // Optional: could log chunk boundaries; already reflected in llm_partial
            } else if (message.type === 'audio_chunk') {
                // Streaming audio chunks; may include a final marker.
                if (message.data) {
                    statusDiv.textContent = "Playing response...";
                    enqueueAudioBase64(message.data);
                }
                if (message.final) {
                    statusDiv.textContent = "Ready to speak";
                }
            } else if (message.type === 'audio_done') {
                statusDiv.textContent = "Ready to speak";
            } else if (message.type === 'stream_done') {
                if (!assistantAppended) {
                    appendAssistant(message.data);
                }
                statusDiv.textContent = "Ready to speak";
            } else if (message.type === 'llm_text') {
                // Backward compatibility with non-streaming
                appendAssistant(message.data);
            } else if (message.type === 'audio') {
                // Backward compatibility with single audio payload
                statusDiv.textContent = "Playing response...";
                enqueueAudioBase64(message.data);
            } else if (message.type === 'error') {
                statusDiv.textContent = message.message;
            }
            logDiv.scrollTop = logDiv.scrollHeight;
        };

        websocket.onclose = () => {
            statusDiv.textContent = "Connection lost. Retrying...";
            setTimeout(connectWebSocket, 2000);
        };

        websocket.onerror = (error) => {
            console.error("WebSocket error:", error);
            websocket.close();
        };
    };

    // Manual hold-to-speak
    recordButton.addEventListener("mousedown", () => {
        bargeIn();
        startRecording("manual");
    });
    recordButton.addEventListener("mouseup", stopRecording);
    // For mobile
    recordButton.addEventListener("touchstart", () => {
        bargeIn();
        startRecording("manual");
    });
    recordButton.addEventListener("touchend", stopRecording);

    // Auto VAD toggle
    autoVADToggle.addEventListener("change", async (e) => {
        vadEnabled = e.target.checked;
        if (vadEnabled) {
            await ensureStream();
            statusDiv.textContent = "Auto listening (will barge-in if you speak)";
            startVADLoop();
        } else {
            stopRecording();
            vadRunning = false;
            vadSpeechMs = 0;
            vadSilenceMs = 0;
        }
    });


    // Initial connection
    connectWebSocket();
});
