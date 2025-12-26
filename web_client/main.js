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

    // WebAudio playback (gapless)
    let playCtx = null;
    let playHead = 0;
    let scheduledUntil = 0;
    let audioDoneTimer = null;
    let activeTurnForPlayback = null;
    let sentClientStartForTurn = new Set();

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

        // Stop WebAudio scheduled playback
        if (audioDoneTimer) {
            clearTimeout(audioDoneTimer);
            audioDoneTimer = null;
        }
        if (playCtx) {
            try { playCtx.close(); } catch (e) {}
        }
        playCtx = null;
        playHead = 0;
        scheduledUntil = 0;
        activeTurnForPlayback = null;
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
        // Deprecated: old HTMLAudio pipeline. Kept for backward-compat if server sends JSON base64 audio.
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

    const ensurePlayCtx = async () => {
        if (playCtx) return playCtx;
        playCtx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: "interactive" });
        playHead = playCtx.currentTime + 0.05;
        scheduledUntil = playHead;
        return playCtx;
    };

    const scheduleDecodedBuffer = async (turnId, sentenceIdx, chunkIdx, chunkCount, audioBuffer) => {
        await ensurePlayCtx();

        if (activeTurnForPlayback === null) {
            activeTurnForPlayback = turnId;
        } else if (turnId !== activeTurnForPlayback) {
            bargeIn();
            await ensurePlayCtx();
            activeTurnForPlayback = turnId;
        }

        // Store decoded buffer in ordering map (reuse bufferedAudio map but store AudioBuffer)
        bufferedAudio.set(_key(sentenceIdx, chunkIdx), audioBuffer);
        sentenceChunkCounts.set(sentenceIdx, chunkCount);

        const pump = () => {
            tryAdvanceSentence();
            const k = _key(expectedSentenceIdx, expectedChunkIdx);
            const buf = bufferedAudio.get(k);
            if (!buf) return;
            bufferedAudio.delete(k);

            const src = playCtx.createBufferSource();
            src.buffer = buf;
            src.connect(playCtx.destination);

            const startAt = Math.max(playCtx.currentTime + 0.02, playHead);
            src.start(startAt);
            playHead = startAt + buf.duration;

            // first audio started for this turn
            if (!sentClientStartForTurn.has(turnId)) {
                sentClientStartForTurn.add(turnId);
                try {
                    websocket.send(JSON.stringify({ type: "client_audio_started", turn_id: turnId, client_epoch_ms: Date.now() }));
                } catch (e) {}
            }

            expectedChunkIdx += 1;

            // schedule done timer (updated on every scheduled chunk)
            if (audioDoneTimer) clearTimeout(audioDoneTimer);
            const msUntilDone = Math.max(10, (playHead - playCtx.currentTime) * 1000 + 30);
            audioDoneTimer = setTimeout(() => {
                try {
                    websocket.send(JSON.stringify({ type: "client_audio_done", turn_id: turnId, client_epoch_ms: Date.now() }));
                } catch (e) {}
            }, msUntilDone);

            // keep pumping if next is ready
            pump();
        };

        pump();
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

    const upsertUserPartial = (text) => {
        const last = logDiv.querySelector("p.user-text:last-of-type");
        if (!last || !last.dataset || last.dataset.partial !== "1") {
            const p = document.createElement("p");
            p.className = "user-text";
            p.dataset.partial = "1";
            p.innerHTML = `<strong>You:</strong> ${text}`;
            logDiv.appendChild(p);
            return;
        }
        last.innerHTML = `<strong>You:</strong> ${text}`;
    };

    const finalizeUserPartial = (text) => {
        const last = logDiv.querySelector("p.user-text:last-of-type");
        if (last && last.dataset && last.dataset.partial === "1") {
            last.dataset.partial = "0";
            last.innerHTML = `<strong>You:</strong> ${text}`;
            return;
        }
        appendUser(text);
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
        websocket.binaryType = "arraybuffer";

        websocket.onopen = () => {
            websocket.send(JSON.stringify({ type: "pcm_stream_start", sample_rate: 16000, frame_ms: 20 }));
            statusDiv.textContent = "Connected";
        };

        websocket.onmessage = (event) => {
            // Binary audio frame path: [16-byte header][WAV bytes]
            if (event.data instanceof ArrayBuffer) {
                const buf = event.data;
                if (buf.byteLength > 16) {
                    const dv = new DataView(buf);
                    const turnId = dv.getUint32(0, true);
                    const sentenceIdx = dv.getUint32(4, true);
                    const chunkIdx = dv.getUint32(8, true);
                    const chunkCount = dv.getUint32(12, true);
                    const wavBytes = buf.slice(16);
                    ensurePlayCtx().then(() => {
                        playCtx.decodeAudioData(wavBytes.slice(0)).then((audioBuffer) => {
                            scheduleDecodedBuffer(turnId, sentenceIdx, chunkIdx, chunkCount, audioBuffer);
                        }).catch((e) => {
                            console.error("decodeAudioData failed", e);
                        });
                    });
                }
                return;
            }

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

            if (message.type === "turn_end") {
                // Optional: could display or store for client-side latency debugging
                return;
            }

            if (message.type === "user_text") {
                finalizeUserPartial(message.data);
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
            } else if (message.type === "stt_partial") {
                if (typeof message.data === "string" && message.data.trim()) {
                    upsertUserPartial(message.data);
                }
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
        try {
            websocket.send(JSON.stringify({ type: "mode", value: "manual" }));
            websocket.send(JSON.stringify({ type: "manual_start" }));
        } catch (e) {}
        await startStreaming("manual");
    });
    recordButton.addEventListener("mouseup", () => {
        try { websocket.send(JSON.stringify({ type: "manual_end" })); } catch (e) {}
        stopStreamingSoon(0);
    });
    recordButton.addEventListener("touchstart", async () => {
        bargeIn();
        try {
            websocket.send(JSON.stringify({ type: "mode", value: "manual" }));
            websocket.send(JSON.stringify({ type: "manual_start" }));
        } catch (e) {}
        await startStreaming("manual");
    });
    recordButton.addEventListener("touchend", () => {
        try { websocket.send(JSON.stringify({ type: "manual_end" })); } catch (e) {}
        stopStreamingSoon(0);
    });

    // Auto voice detection (server-side turn-taking)
    autoVADToggle.addEventListener("change", async (e) => {
        const enabled = e.target.checked;
        if (enabled) {
            bargeIn();
            try { websocket.send(JSON.stringify({ type: "mode", value: "auto" })); } catch (e) {}
            await startStreaming("auto");
        } else {
            try { websocket.send(JSON.stringify({ type: "mode", value: "idle" })); } catch (e) {}
            await stopAudioPipeline();
            recordButton.classList.remove("recording");
            statusDiv.textContent = "Ready to speak";
        }
    });

    connectWebSocket();
});
