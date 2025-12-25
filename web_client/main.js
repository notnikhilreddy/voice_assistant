console.log("voice client script loaded");

document.addEventListener("DOMContentLoaded", () => {
    const recordButton = document.getElementById("recordButton");
    const statusDiv = document.getElementById("status");
    const logDiv = document.getElementById("log");
    const audioPlayer = document.getElementById("audioPlayer");
    const autoVADToggle = document.getElementById("autoVADToggle");

    let assistantAppended = false;
    let mediaStream = null;
    let websocket = null;

    // Playback ordering state (server provides turn_id/sentence_idx/chunk_idx)
    let isPlaying = false;
    let playbackTurnId = null;
    let expectedSentenceIdx = 0;
    let expectedChunkIdx = 0;
    const bufferedAudio = new Map(); // key: "sentence:chunk" -> objectURL
    const sentenceChunkCounts = new Map(); // sentence_idx -> chunk_count

    // PCM streaming state (20ms frames @ 16kHz)
    let audioCtx = null;
    let sourceNode = null;
    let processorNode = null;
    let zeroGainNode = null;
    let streamingEnabled = false;
    let stopTimer = null;
    let seq = 0;
    let inRate = 48000;
    const outRate = 16000;
    let resampleT = 0.0;
    let carry = [];

    const _key = (s, c) => `${s}:${c}`;

    const bargeIn = () => {
        if (isPlaying) {
            audioPlayer.pause();
            audioPlayer.src = "";
            isPlaying = false;
        }
        for (const url of bufferedAudio.values()) {
            try { URL.revokeObjectURL(url); } catch (e) {}
        }
        bufferedAudio.clear();
        sentenceChunkCounts.clear();
        playbackTurnId = null;
        expectedSentenceIdx = 0;
        expectedChunkIdx = 0;
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

    const tryAdvanceSentence = () => {
        const count = sentenceChunkCounts.get(expectedSentenceIdx);
        if (typeof count === "number" && expectedChunkIdx >= count) {
            expectedSentenceIdx += 1;
            expectedChunkIdx = 0;
        }
    };

    const playNextAudio = () => {
        if (isPlaying) return;
        tryAdvanceSentence();
        const key = _key(expectedSentenceIdx, expectedChunkIdx);
        const url = bufferedAudio.get(key);
        if (!url) return;
        bufferedAudio.delete(key);
        isPlaying = true;
        audioPlayer.src = url;
        audioPlayer.play().catch((err) => console.error("Playback error:", err));
    };

    audioPlayer.onended = () => {
        isPlaying = false;
        try { URL.revokeObjectURL(audioPlayer.src); } catch (e) {}
        expectedChunkIdx += 1;
        playNextAudio();
    };

    const enqueueAudioBase64 = (turnId, sentenceIdx, chunkIdx, chunkCount, b64) => {
        if (!b64) return;
        const binary = atob(b64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const blob = new Blob([bytes], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);

        if (turnId !== null && turnId !== undefined) {
            if (playbackTurnId === null) {
                playbackTurnId = turnId;
            } else if (turnId !== playbackTurnId) {
                bargeIn();
                playbackTurnId = turnId;
            }
        }

        if (typeof sentenceIdx === "number" && typeof chunkIdx === "number") {
            bufferedAudio.set(_key(sentenceIdx, chunkIdx), url);
            if (typeof chunkCount === "number") {
                sentenceChunkCounts.set(sentenceIdx, chunkCount);
            }
        } else {
            bufferedAudio.set(_key(expectedSentenceIdx, expectedChunkIdx), url);
        }
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

    const floatToInt16 = (f) => {
        const v = Math.max(-1, Math.min(1, f));
        return v < 0 ? (v * 0x8000) : (v * 0x7fff);
    };

    const resampleTo16k = (input) => {
        const ratio = inRate / outRate;
        const out = [];
        let t = resampleT;
        while (t < input.length - 1) {
            const i = Math.floor(t);
            const frac = t - i;
            const s = input[i] * (1 - frac) + input[i + 1] * frac;
            out.push(s);
            t += ratio;
        }
        resampleT = t - (input.length - 1);
        return out;
    };

    const sendPcmFrame = (pcm16Bytes) => {
        if (!websocket || websocket.readyState !== WebSocket.OPEN) return;
        const out = new Uint8Array(4 + pcm16Bytes.length);
        new DataView(out.buffer).setUint32(0, seq >>> 0, true);
        seq += 1;
        out.set(pcm16Bytes, 4);
        websocket.send(out.buffer);
    };

    const startAudioPipeline = async () => {
        await ensureStream();
        if (audioCtx) return;
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: "interactive" });
        inRate = audioCtx.sampleRate || 48000;
        sourceNode = audioCtx.createMediaStreamSource(mediaStream);
        processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
        zeroGainNode = audioCtx.createGain();
        zeroGainNode.gain.value = 0.0;
        carry = [];
        resampleT = 0.0;

        sourceNode.connect(processorNode);
        processorNode.connect(zeroGainNode);
        zeroGainNode.connect(audioCtx.destination);

        processorNode.onaudioprocess = (e) => {
            if (!streamingEnabled) return;
            const input = e.inputBuffer.getChannelData(0);
            const out = resampleTo16k(input);
            if (!out.length) return;
            carry = carry.concat(out);
            while (carry.length >= 320) {
                const frame = carry.slice(0, 320);
                carry = carry.slice(320);
                const pcm16 = new Int16Array(320);
                for (let i = 0; i < 320; i++) pcm16[i] = floatToInt16(frame[i]);
                sendPcmFrame(new Uint8Array(pcm16.buffer));
            }
        };
    };

    const stopAudioPipeline = async () => {
        streamingEnabled = false;
        if (stopTimer) {
            clearTimeout(stopTimer);
            stopTimer = null;
        }
        try {
            if (processorNode) processorNode.disconnect();
            if (sourceNode) sourceNode.disconnect();
            if (zeroGainNode) zeroGainNode.disconnect();
        } catch (e) {}
        processorNode = null;
        sourceNode = null;
        zeroGainNode = null;
        if (audioCtx) {
            try { await audioCtx.close(); } catch (e) {}
        }
        audioCtx = null;
    };

    const startStreaming = async (mode) => {
        await startAudioPipeline();
        streamingEnabled = true;
        if (stopTimer) {
            clearTimeout(stopTimer);
            stopTimer = null;
        }
        recordButton.classList.add("recording");
        statusDiv.textContent = mode === "manual" ? "Listening..." : "Auto listening...";
    };

    const stopStreamingSoon = (ms = 700) => {
        if (autoVADToggle.checked) return; // auto mode stays on
        if (stopTimer) clearTimeout(stopTimer);
        stopTimer = setTimeout(async () => {
            await stopAudioPipeline();
            recordButton.classList.remove("recording");
            statusDiv.textContent = "Ready to speak";
        }, ms);
    };

    const connectWebSocket = () => {
        const wsUrl = `ws://${window.location.host}/ws`;
        websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            websocket.send(JSON.stringify({ type: "pcm_stream_start", sample_rate: 16000, frame_ms: 20 }));
            statusDiv.textContent = "Connected";
        };

        websocket.onmessage = (event) => {
            let message;
            try {
                message = JSON.parse(event.data);
            } catch (e) {
                console.error("Failed to parse message", e, event.data);
                return;
            }

            if (message.type === "pcm_stream_ready") {
                statusDiv.textContent = "Ready to speak";
                return;
            }

            if (message.type === "barge_in") {
                bargeIn();
                statusDiv.textContent = "Interrupted (you spoke)";
                return;
            }

            if (message.type === "turn_state") {
                if (message.state === "USER_TALKING") statusDiv.textContent = "Listening...";
                if (message.state === "USER_THINKING") statusDiv.textContent = "Listening (pause)...";
                if (message.state === "USER_FINISHED") statusDiv.textContent = "Thinking...";
                return;
            }

            if (message.type === "user_text") {
                appendUser(message.data);
                assistantAppended = false;
            } else if (message.type === "llm_partial") {
                if (!assistantAppended) {
                    appendAssistant(message.data);
                } else {
                    const last = logDiv.querySelector("p.llm-text:last-of-type");
                    if (last) last.innerHTML = `<strong>Assistant:</strong> ${message.data}`;
                }
            } else if (message.type === "audio_chunk") {
                if (message.data) {
                    statusDiv.textContent = "Playing response...";
                    enqueueAudioBase64(
                        message.turn_id,
                        message.sentence_idx,
                        message.chunk_idx,
                        message.chunk_count,
                        message.data
                    );
                }
            } else if (message.type === "sentence_done") {
                if (typeof message.sentence_idx === "number" && typeof message.chunk_count === "number") {
                    sentenceChunkCounts.set(message.sentence_idx, message.chunk_count);
                    playNextAudio();
                }
            } else if (message.type === "stream_done") {
                if (!assistantAppended) appendAssistant(message.data);
                statusDiv.textContent = "Ready to speak";
            } else if (message.type === "error") {
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
    recordButton.addEventListener("mousedown", async () => {
        bargeIn();
        await startStreaming("manual");
    });
    recordButton.addEventListener("mouseup", () => stopStreamingSoon(700));
    recordButton.addEventListener("touchstart", async () => {
        bargeIn();
        await startStreaming("manual");
    });
    recordButton.addEventListener("touchend", () => stopStreamingSoon(700));

    // Auto voice detection (server-side turn-taking)
    autoVADToggle.addEventListener("change", async (e) => {
        const enabled = e.target.checked;
        if (enabled) {
            bargeIn();
            await startStreaming("auto");
        } else {
            await stopAudioPipeline();
            recordButton.classList.remove("recording");
            statusDiv.textContent = "Ready to speak";
        }
    });

    connectWebSocket();
});
