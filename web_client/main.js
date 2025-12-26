console.log("voice client script loaded");

document.addEventListener("DOMContentLoaded", () => {
    const recordButton = document.getElementById("recordButton");
    const statusDiv = document.getElementById("status");
    const logDiv = document.getElementById("log");
    const audioPlayer = document.getElementById("audioPlayer");
    const autoVADToggle = document.getElementById("autoVADToggle");
    const newChatButton = document.getElementById("newChatButton");
    const conversationList = document.getElementById("conversationList");
    const activeConversationTitle = document.getElementById("activeConversationTitle");

    // --- Conversation state (ChatGPT-like) ---
    const STORAGE_KEY = "voice_assistant_conversations_v1";

    const nowIso = () => new Date().toISOString();
    const safeId = () => (crypto && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}_${Math.random()}`);

    const loadState = () => {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (!raw) return null;
            const parsed = JSON.parse(raw);
            if (!parsed || parsed.version !== 1) return null;
            return parsed;
        } catch (e) {
            return null;
        }
    };

    const saveState = (st) => {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(st));
        } catch (e) {}
    };

    let state =
        loadState() ||
        ({
            version: 1,
            currentId: "",
            conversations: {},
        });

    const ensureAtLeastOneConversation = () => {
        const ids = Object.keys(state.conversations || {});
        if (!ids.length) {
            const id = safeId();
            state.conversations[id] = {
                id,
                title: "New chat",
                createdAt: nowIso(),
                updatedAt: nowIso(),
                messages: [],
            };
            state.currentId = id;
            saveState(state);
        } else if (!state.currentId || !state.conversations[state.currentId]) {
            state.currentId = ids[0];
            saveState(state);
        }
    };

    const getConv = (id) => state.conversations[id];
    const currentConv = () => getConv(state.currentId);

    const formatTitleFromFirstUserMessage = (conv) => {
        const firstUser = (conv.messages || []).find((m) => m.role === "user" && m.text && !m.partial);
        if (!firstUser) return "New chat";
        const t = String(firstUser.text).trim().replace(/\s+/g, " ");
        return t.length > 34 ? `${t.slice(0, 34)}…` : t;
    };

    const renderSidebar = () => {
        const convs = Object.values(state.conversations || {});
        convs.sort((a, b) => String(b.updatedAt || "").localeCompare(String(a.updatedAt || "")));
        conversationList.innerHTML = "";
        for (const c of convs) {
            const li = document.createElement("li");
            li.className = `conv-item ${c.id === state.currentId ? "active" : ""}`;
            li.dataset.id = c.id;
            const title = document.createElement("div");
            title.className = "conv-title";
            title.textContent = c.title || "New chat";
            const sub = document.createElement("div");
            sub.className = "conv-sub";
            const last = (c.messages || []).slice(-1)[0];
            sub.textContent = last ? `${last.role === "user" ? "You" : "Assistant"}: ${String(last.text || "").trim().slice(0, 40)}` : "No messages yet";
            li.appendChild(title);
            li.appendChild(sub);
            li.onclick = () => selectConversation(c.id);
            conversationList.appendChild(li);
        }
    };

    const escapeHtml = (s) =>
        String(s)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;");

    const appendMessageDom = (role, text, { partial = false } = {}) => {
        const row = document.createElement("div");
        row.className = `msg-row ${role}`;
        const bubble = document.createElement("div");
        bubble.className = "msg-bubble";
        bubble.dataset.partial = partial ? "1" : "0";
        bubble.innerHTML = escapeHtml(text);
        row.appendChild(bubble);
        logDiv.appendChild(row);
        return bubble;
    };

    const renderMessages = () => {
        const c = currentConv();
        activeConversationTitle.textContent = c?.title || "Chat";
        logDiv.innerHTML = "";
        for (const m of c.messages || []) {
            appendMessageDom(m.role, m.text || "", { partial: !!m.partial });
        }
        logDiv.scrollTop = logDiv.scrollHeight;
    };

    const touchConversation = (conv) => {
        conv.updatedAt = nowIso();
        if (!conv.title || conv.title === "New chat") {
            conv.title = formatTitleFromFirstUserMessage(conv);
        }
        saveState(state);
        renderSidebar();
        activeConversationTitle.textContent = conv.title || "Chat";
    };

    const upsertUserPartial = (text, convId) => {
        const conv = getConv(convId);
        if (!conv) return;
        const msgs = conv.messages || (conv.messages = []);
        const last = msgs[msgs.length - 1];
        if (!last || last.role !== "user" || !last.partial) {
            msgs.push({ role: "user", text, partial: true, ts: nowIso() });
        } else {
            last.text = text;
        }
        touchConversation(conv);
        if (state.currentId === convId) renderMessages();
    };

    const finalizeUserText = (text, convId) => {
        const conv = getConv(convId);
        if (!conv) return;
        const msgs = conv.messages || (conv.messages = []);
        const last = msgs[msgs.length - 1];
        if (last && last.role === "user" && last.partial) {
            last.partial = false;
            last.text = text;
        } else {
            msgs.push({ role: "user", text, partial: false, ts: nowIso() });
        }
        touchConversation(conv);
        if (state.currentId === convId) renderMessages();
    };

    const upsertAssistantPartial = (text, convId) => {
        const conv = getConv(convId);
        if (!conv) return;
        const msgs = conv.messages || (conv.messages = []);
        const last = msgs[msgs.length - 1];
        if (!last || last.role !== "assistant" || !last.partial) {
            msgs.push({ role: "assistant", text, partial: true, ts: nowIso() });
        } else {
            last.text = text;
        }
        touchConversation(conv);
        if (state.currentId === convId) renderMessages();
    };

    const finalizeAssistantText = (text, convId) => {
        const conv = getConv(convId);
        if (!conv) return;
        const msgs = conv.messages || (conv.messages = []);
        const last = msgs[msgs.length - 1];
        if (last && last.role === "assistant" && last.partial) {
            last.partial = false;
            last.text = text;
        } else if (text && String(text).trim()) {
            msgs.push({ role: "assistant", text, partial: false, ts: nowIso() });
        }
        touchConversation(conv);
        if (state.currentId === convId) renderMessages();
    };

    ensureAtLeastOneConversation();
    renderSidebar();
    renderMessages();

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

    const wsSendJson = (obj) => {
        try {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify(obj));
            }
        } catch (e) {}
    };

    const selectConversation = (id) => {
        if (!id || !state.conversations[id]) return;
        state.currentId = id;
        saveState(state);
        renderSidebar();
        renderMessages();
        bargeIn();
        wsSendJson({ type: "select_conversation", conversation_id: id });
    };

    const newConversation = () => {
        const id = safeId();
        state.conversations[id] = {
            id,
            title: "New chat",
            createdAt: nowIso(),
            updatedAt: nowIso(),
            messages: [],
        };
        state.currentId = id;
        saveState(state);
        renderSidebar();
        renderMessages();
        bargeIn();
        wsSendJson({ type: "new_conversation", conversation_id: id });
    };

    newChatButton.onclick = () => newConversation();

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
        audioPlayer
            .play()
            .then(() => {
                // Notify server when client actually starts playback for this turn.
                try {
                    if (websocket && websocket.readyState === WebSocket.OPEN && playbackTurnId !== null) {
                        websocket.send(JSON.stringify({ type: "client_audio_started", turn_id: playbackTurnId }));
                    }
                } catch (e) {}
            })
            .catch((err) => console.error("Playback error:", err));
    };

    audioPlayer.onended = () => {
        isPlaying = false;
        try {
            URL.revokeObjectURL(audioPlayer.src);
        } catch (e) {}
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
            wsSendJson({ type: "select_conversation", conversation_id: state.currentId });
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

            const convId = message.conversation_id || state.currentId;
            if (message.type === "conversation_ack") {
                // Server accepted conversation switch; nothing else required.
                return;
            }
            if (message.type === "user_text") {
                finalizeUserText(message.data, convId);
            } else if (message.type === "llm_partial") {
                upsertAssistantPartial(message.data, convId);
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
                finalizeAssistantText(message.data, convId);
                statusDiv.textContent = "Ready to speak";
            } else if (message.type === "stt_partial") {
                if (typeof message.data === "string" && message.data.trim()) {
                    upsertUserPartial(message.data, convId);
                }
            } else if (message.type === "error") {
                statusDiv.textContent = message.message;
            }
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

    // --- Unified manual hold-to-speak helpers (mouse/touch/keyboard) ---
    let manualHeld = false;
    const startManualHold = async () => {
        if (manualHeld) return;
        manualHeld = true;
        bargeIn();
        try {
            websocket.send(JSON.stringify({ type: "mode", value: "manual" }));
            websocket.send(JSON.stringify({ type: "manual_start" }));
        } catch (e) {}
        await startStreaming("manual");
    };

    const endManualHold = () => {
        if (!manualHeld) return;
        manualHeld = false;
        try {
            websocket.send(JSON.stringify({ type: "manual_end" }));
        } catch (e) {}
        stopStreamingSoon(0);
    };

    // Manual hold-to-speak
    recordButton.addEventListener("mousedown", startManualHold);
    recordButton.addEventListener("mouseup", endManualHold);
    recordButton.addEventListener("mouseleave", endManualHold);
    recordButton.addEventListener("touchstart", (e) => {
        try { e.preventDefault(); } catch (err) {}
        startManualHold();
    }, { passive: false });
    recordButton.addEventListener("touchend", endManualHold);

    // Hold spacebar to speak (manual mode).
    window.addEventListener("keydown", (e) => {
        // Ignore when typing in inputs.
        const tag = (e.target && e.target.tagName) ? String(e.target.tagName).toLowerCase() : "";
        if (tag === "input" || tag === "textarea" || (e.target && e.target.isContentEditable)) return;
        if (e.code !== "Space") return;
        // Prevent page scroll.
        e.preventDefault();
        // Key repeat guard.
        if (e.repeat) return;
        startManualHold();
    });
    window.addEventListener("keyup", (e) => {
        const tag = (e.target && e.target.tagName) ? String(e.target.tagName).toLowerCase() : "";
        if (tag === "input" || tag === "textarea" || (e.target && e.target.isContentEditable)) return;
        if (e.code !== "Space") return;
        e.preventDefault();
        endManualHold();
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
