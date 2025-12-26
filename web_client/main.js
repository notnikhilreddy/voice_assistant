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
    const textInput = document.getElementById("textInput");
    const sendButton = document.getElementById("sendButton");
    const renameChatButton = document.getElementById("renameChatButton");
    const deleteChatButton = document.getElementById("deleteChatButton");
    const chatSearch = document.getElementById("chatSearch");
    const settingsButton = document.getElementById("settingsButton");
    const settingsModal = document.getElementById("settingsModal");
    const closeSettingsButton = document.getElementById("closeSettingsButton");
    const themeToggle = document.getElementById("themeToggle");
    const sidebarToggle = document.getElementById("sidebarToggle");
    const sidebarClose = document.getElementById("sidebarClose");
    const appContainer = document.getElementById("appContainer");
    const sidebarBackdrop = document.getElementById("sidebarBackdrop");

    // --- Theme (Light/Dark) ---
    const THEME_KEY = "voice_assistant_theme_v1"; // "dark" | "light"
    const applyTheme = (t) => {
        const theme = t === "light" ? "light" : "dark";
        document.documentElement.dataset.theme = theme;
        try {
            if (themeToggle) themeToggle.checked = theme === "light";
        } catch (e) {}
        try {
            localStorage.setItem(THEME_KEY, theme);
        } catch (e) {}
    };
    const initTheme = () => {
        let t = null;
        try { t = localStorage.getItem(THEME_KEY); } catch (e) {}
        // Default to light mode unless user explicitly chose otherwise.
        if (!t) t = "light";
        applyTheme(t);
    };
    initTheme();

    const openSettings = () => {
        if (!settingsModal) return;
        settingsModal.classList.add("open");
        settingsModal.setAttribute("aria-hidden", "false");
    };
    const closeSettings = () => {
        if (!settingsModal) return;
        settingsModal.classList.remove("open");
        settingsModal.setAttribute("aria-hidden", "true");
    };
    if (settingsButton) settingsButton.onclick = () => openSettings();
    if (closeSettingsButton) closeSettingsButton.onclick = () => closeSettings();
    if (settingsModal) {
        settingsModal.addEventListener("click", (e) => {
            if (e.target === settingsModal) closeSettings();
        });
    }
    window.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeSettings();
    });
    if (themeToggle) {
        themeToggle.addEventListener("change", () => {
            applyTheme(themeToggle.checked ? "light" : "dark");
        });
    }

    // --- Sidebar Toggle ---
    const SIDEBAR_KEY = "voice_assistant_sidebar_v1"; // "open" | "closed"
    const isMobile = () => window.innerWidth <= 768;
    
    const getSidebarState = () => {
        try {
            const saved = localStorage.getItem(SIDEBAR_KEY);
            // Default to "open" if not set or if value is invalid
            if (!saved || (saved !== "closed" && saved !== "open")) {
                return "open";
            }
            return saved;
        } catch (e) {
            return "open";
        }
    };
    
    const setSidebarState = (state) => {
        try {
            localStorage.setItem(SIDEBAR_KEY, state);
        } catch (e) {}
    };
    
    const updateSidebarUI = () => {
        const mobile = isMobile();
        const isHidden = appContainer && appContainer.classList.contains("sidebar-hidden");
        
        if (sidebarClose) {
            sidebarClose.style.display = mobile && !isHidden ? "block" : "none";
        }
        if (sidebarBackdrop) {
            sidebarBackdrop.classList.toggle("active", mobile && !isHidden);
        }
    };
    
    const toggleSidebar = () => {
        if (!appContainer) return;
        const isCurrentlyHidden = appContainer.classList.contains("sidebar-hidden");
        if (isCurrentlyHidden) {
            appContainer.classList.remove("sidebar-hidden");
            setSidebarState("open");
        } else {
            appContainer.classList.add("sidebar-hidden");
            setSidebarState("closed");
        }
        updateSidebarUI();
    };
    
    const initSidebar = () => {
        if (!appContainer) return;
        const state = getSidebarState();
        const mobile = isMobile();
        
        // On desktop, default to open unless explicitly closed
        // On mobile, start closed
        if (mobile) {
            appContainer.classList.add("sidebar-hidden");
        } else {
            // Only hide if explicitly saved as closed, otherwise show
            if (state === "closed") {
                appContainer.classList.add("sidebar-hidden");
            } else {
                // Ensure sidebar is visible on desktop by default
                appContainer.classList.remove("sidebar-hidden");
                // If state was not set, save as open
                if (state !== "open") {
                    setSidebarState("open");
                }
            }
        }
        updateSidebarUI();
    };
    
    if (sidebarToggle) {
        sidebarToggle.addEventListener("click", toggleSidebar);
    }
    
    if (sidebarClose) {
        sidebarClose.addEventListener("click", toggleSidebar);
    }
    
    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener("click", () => {
            if (isMobile()) {
                toggleSidebar();
            }
        });
    }
    
    // Handle window resize
    let resizeTimeout;
    window.addEventListener("resize", () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            const mobile = isMobile();
            if (!mobile && appContainer) {
                // On desktop, restore sidebar if it was open
                const state = getSidebarState();
                if (state === "open") {
                    appContainer.classList.remove("sidebar-hidden");
                }
            } else if (mobile && appContainer) {
                // On mobile, always start closed
                appContainer.classList.add("sidebar-hidden");
            }
            updateSidebarUI();
        }, 100);
    });
    
    // Sidebar will be initialized after conversations are loaded

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
        if (!conversationList) {
            console.error("conversationList element not found");
            // Try to find it again after a delay (for Edge browser)
            setTimeout(() => {
                const retryList = document.getElementById("conversationList");
                if (retryList) {
                    console.log("Found conversationList on retry");
                    renderSidebar();
                }
            }, 200);
            return;
        }
        try {
            const q = (chatSearch && chatSearch.value ? String(chatSearch.value) : "").trim().toLowerCase();
            const convs = Object.values(state.conversations || {});
            convs.sort((a, b) => String(b.updatedAt || "").localeCompare(String(a.updatedAt || "")));
            
            // Clear and ensure visibility
            conversationList.innerHTML = "";
            conversationList.style.display = "block";
            conversationList.style.visibility = "visible";
            conversationList.style.opacity = "1";
            
            // Ensure sidebar container is visible if we have content
            if (convs.length > 0 && appContainer && !isMobile()) {
                appContainer.classList.remove("sidebar-hidden");
            }
            
            for (const c of convs) {
                if (q) {
                    const hay = `${c.title || ""}\n${(c.messages || []).slice(-3).map((m) => m.text || "").join("\n")}`.toLowerCase();
                    if (!hay.includes(q)) continue;
                }
                const li = document.createElement("li");
                li.className = `conv-item ${c.id === state.currentId ? "active" : ""}`;
                li.dataset.id = c.id;
                const actions = document.createElement("div");
                actions.className = "conv-actions";
                const renameBtn = document.createElement("button");
                renameBtn.className = "icon-btn";
                renameBtn.title = "Rename";
                renameBtn.innerHTML = `
                    <svg class="icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                        <path d="M4 20h4l11-11a2 2 0 0 0 0-3l-1-1a2 2 0 0 0-3 0L4 16v4Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
                        <path d="M13 6l5 5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                `;
                renameBtn.onclick = (e) => {
                    e.stopPropagation();
                    renameConversation(c.id);
                };
                actions.appendChild(renameBtn);

                const delBtn = document.createElement("button");
                delBtn.className = "icon-btn";
                delBtn.title = "Delete";
                delBtn.innerHTML = `
                    <svg class="icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                        <path d="M4 7h16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                        <path d="M10 11v7M14 11v7" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                        <path d="M6 7l1 14h10l1-14" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
                        <path d="M9 7V4h6v3" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
                    </svg>
                `;
                delBtn.onclick = (e) => {
                    e.stopPropagation();
                    deleteConversation(c.id);
                };
                actions.appendChild(delBtn);
                li.appendChild(actions);

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
            
            // If no conversations match filter, show message
            if (conversationList.children.length === 0) {
                const emptyLi = document.createElement("li");
                emptyLi.className = "conv-item";
                emptyLi.style.cssText = "color: var(--muted2); padding: 10px; text-align: center; font-size: 12px;";
                emptyLi.textContent = q ? "No conversations found" : "No conversations";
                conversationList.appendChild(emptyLi);
            }
        } catch (e) {
            console.error("Error rendering sidebar:", e);
            // Ensure sidebar structure remains visible even on error
            if (conversationList && conversationList.innerHTML === "") {
                const errorLi = document.createElement("li");
                errorLi.className = "conv-item";
                errorLi.style.cssText = "color: var(--muted2); padding: 10px; text-align: center; font-size: 12px;";
                errorLi.textContent = "Error loading conversations";
                conversationList.appendChild(errorLi);
            }
            // Ensure sidebar is visible even on error
            if (appContainer && !isMobile()) {
                appContainer.classList.remove("sidebar-hidden");
            }
        }
    };

    const renameConversation = (id) => {
        const c = getConv(id);
        if (!c) return;
        const next = prompt("Rename chat", c.title || "New chat");
        if (next === null) return;
        c.title = String(next).trim() || "New chat";
        c.updatedAt = nowIso();
        saveState(state);
        renderSidebar();
        if (state.currentId === c.id) {
            activeConversationTitle.textContent = c.title || "Chat";
        }
    };

    const deleteConversation = (id) => {
        const c = getConv(id);
        if (!c) return;
        const ok = confirm(`Delete chat "${c.title || "New chat"}"?`);
        if (!ok) return;
        delete state.conversations[id];
        const remaining = Object.keys(state.conversations || {});
        if (!remaining.length) {
            // Always keep at least one chat.
            const nid = safeId();
            state.conversations[nid] = {
                id: nid,
                title: "New chat",
                createdAt: nowIso(),
                updatedAt: nowIso(),
                messages: [],
            };
            state.currentId = nid;
        } else if (state.currentId === id) {
            state.currentId = remaining[0];
        }
        saveState(state);
        renderSidebar();
        renderMessages();
        bargeIn();
        wsSendJson({ type: "select_conversation", conversation_id: state.currentId });
    };

    const escapeHtml = (s) =>
        String(s)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;");

    const scrollLogToEnd = () => {
        // Force chat to always stick to the bottom (requested behavior).
        requestAnimationFrame(() => {
            try { logDiv.scrollTop = logDiv.scrollHeight; } catch (e) {}
        });
    };

    const appendMessageDom = (role, text, { partial = false } = {}) => {
        const row = document.createElement("div");
        row.className = `msg-row ${role}`;
        const avatar = document.createElement("div");
        avatar.className = `avatar ${role}`;
        avatar.textContent = role === "user" ? "Y" : "A";
        const bubble = document.createElement("div");
        bubble.className = "msg-bubble";
        bubble.dataset.partial = partial ? "1" : "0";
        bubble.innerHTML = escapeHtml(text);
        if (role === "assistant") {
            row.appendChild(avatar);
            row.appendChild(bubble);
        } else {
            row.appendChild(bubble);
            row.appendChild(avatar);
        }
        logDiv.appendChild(row);
        scrollLogToEnd();
        return bubble;
    };

    const renderMessages = () => {
        const c = currentConv();
        activeConversationTitle.textContent = c?.title || "Chat";
        logDiv.innerHTML = "";
        const msgs = c?.messages || [];
        if (!msgs.length) {
            const wrap = document.createElement("div");
            wrap.className = "empty";
            wrap.innerHTML = `
                <div class="empty-card">
                    <svg width="100%" height="110" viewBox="0 0 640 140" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:block;">
                        <defs>
                            <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
                                <stop offset="0" stop-color="#7c3aed" stop-opacity="0.9"/>
                                <stop offset="1" stop-color="#06b6d4" stop-opacity="0.85"/>
                            </linearGradient>
                            <linearGradient id="g2" x1="1" y1="0" x2="0" y2="1">
                                <stop offset="0" stop-color="#22c55e" stop-opacity="0.9"/>
                                <stop offset="1" stop-color="#06b6d4" stop-opacity="0.75"/>
                            </linearGradient>
                        </defs>
                        <circle cx="90" cy="70" r="46" fill="url(#g1)" opacity="0.85"/>
                        <circle cx="170" cy="70" r="28" fill="url(#g2)" opacity="0.85"/>
                        <rect x="250" y="44" width="340" height="18" rx="9" fill="rgba(255,255,255,0.10)"/>
                        <rect x="250" y="72" width="280" height="18" rx="9" fill="rgba(255,255,255,0.07)"/>
                        <rect x="250" y="100" width="220" height="18" rx="9" fill="rgba(255,255,255,0.06)"/>
                    </svg>
                    <div class="empty-title">Start a chat</div>
                    <div class="empty-sub">Type a message, or hold Space / Hold to Speak for voice.</div>
                </div>
            `;
            logDiv.appendChild(wrap);
        } else {
            for (const m of msgs) {
                appendMessageDom(m.role, m.text || "", { partial: !!m.partial });
            }
        }
        scrollLogToEnd();
    };

    const touchConversation = (conv) => {
        conv.updatedAt = nowIso();
        if (!conv.title || conv.title === "New chat") {
            conv.title = formatTitleFromFirstUserMessage(conv);
        }
        saveState(state);
        renderSidebar();
        activeConversationTitle.textContent = conv.title || "Chat";
        if (state.currentId === conv.id) {
            requestAnimationFrame(() => {
                logDiv.scrollTop = logDiv.scrollHeight;
            });
        }
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
    
    // Initialize sidebar state (but don't hide if we have content)
    const initSidebarState = () => {
        if (!appContainer) return;
        const state = getSidebarState();
        const mobile = isMobile();
        
        // On desktop, default to open unless explicitly closed
        // On mobile, start closed
        if (mobile) {
            appContainer.classList.add("sidebar-hidden");
        } else {
            // Only hide if explicitly saved as closed, otherwise show
            if (state === "closed") {
                appContainer.classList.add("sidebar-hidden");
            } else {
                // Ensure sidebar is visible on desktop by default
                appContainer.classList.remove("sidebar-hidden");
                // If state was not set, save as open
                if (state !== "open") {
                    setSidebarState("open");
                }
            }
        }
        updateSidebarUI();
    };
    
    // Initialize sidebar state first
    initSidebarState();
    
    // Render sidebar content - this will also ensure sidebar stays visible if content exists
    renderSidebar();
    
    // Re-render after a short delay for Edge browser compatibility and ensure visibility
    setTimeout(() => {
        renderSidebar();
        // Force sidebar visible on desktop if content exists (override any hidden state)
        if (appContainer && !isMobile()) {
            const hasContent = Object.keys(state.conversations || {}).length > 0;
            if (hasContent) {
                appContainer.classList.remove("sidebar-hidden");
                updateSidebarUI();
            }
        }
    }, 300);
    
    renderMessages();

    if (chatSearch) {
        chatSearch.addEventListener("input", () => renderSidebar());
    }

    if (renameChatButton) {
        renameChatButton.onclick = () => renameConversation(state.currentId);
    }
    if (deleteChatButton) {
        deleteChatButton.onclick = () => deleteConversation(state.currentId);
    }

    let mediaStream = null;
    let websocket = null;

    // STT token streaming (FunASR MLX): throttle UI updates to avoid re-rendering the full chat per token.
    const sttDraftByConv = new Map(); // convId -> latest accumulated text
    let sttDraftRaf = 0;

    const updateUserPartialDom = (text) => {
        // Update/insert the last user partial bubble in-place (no full re-render).
        const bubbles = logDiv.querySelectorAll('.msg-row.user .msg-bubble[data-partial="1"]');
        const last = bubbles && bubbles.length ? bubbles[bubbles.length - 1] : null;
        if (last) {
            last.innerHTML = escapeHtml(text);
        } else {
            appendMessageDom("user", text, { partial: true });
        }
        requestAnimationFrame(() => {
            try { logDiv.scrollTop = logDiv.scrollHeight; } catch (e) {}
        });
    };

    const upsertUserPartialTransient = (text, convId) => {
        const conv = getConv(convId);
        if (!conv) return;
        const msgs = conv.messages || (conv.messages = []);
        const last = msgs[msgs.length - 1];
        if (!last || last.role !== "user" || !last.partial) {
            msgs.push({ role: "user", text, partial: true, ts: nowIso() });
        } else {
            last.text = text;
        }
        // Don't touchConversation()/saveState()/renderSidebar() on every token.
        if (state.currentId === convId) updateUserPartialDom(text);
    };

    const scheduleApplySttDrafts = () => {
        if (sttDraftRaf) return;
        sttDraftRaf = requestAnimationFrame(() => {
            sttDraftRaf = 0;
            for (const [convId, txt] of sttDraftByConv.entries()) {
                upsertUserPartialTransient(txt, convId);
            }
            sttDraftByConv.clear();
        });
    };

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

    const sendTextMessage = () => {
        const raw = textInput ? String(textInput.value || "") : "";
        const text = raw.trim();
        if (!text) return;
        // Ignore accidental/low-signal inputs.
        if (text === "/sil" || text === "you" || text === "Thank you.") {
            if (textInput) {
                textInput.value = "";
                try { textInput.style.height = ""; } catch (e) {}
            }
            return;
        }
        const convId = state.currentId;
        finalizeUserText(text, convId);
        wsSendJson({ type: "text_input", text, conversation_id: convId });
        if (textInput) {
            textInput.value = "";
            try {
                textInput.style.height = "";
            } catch (e) {}
        }
    };

    if (sendButton) {
        sendButton.onclick = () => sendTextMessage();
    }
    if (textInput) {
        // Enter to send; Shift+Enter for newline (ChatGPT-like).
        textInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendTextMessage();
            }
        });
        // Auto-grow textarea.
        textInput.addEventListener("input", () => {
            try {
                textInput.style.height = "auto";
                textInput.style.height = Math.min(textInput.scrollHeight, 140) + "px";
            } catch (e) {}
        });
    }

    // Convert local messages to LLM history format for server sync
    const messagesToLLMHistory = (messages) => {
        if (!messages || !messages.length) return [];
        return messages
            .filter(m => !m.partial && m.text && m.text.trim())
            .map(m => ({ role: m.role, content: m.text.trim() }));
    };

    const selectConversation = (id) => {
        if (!id || !state.conversations[id]) return;
        state.currentId = id;
        saveState(state);
        renderSidebar();
        renderMessages();
        bargeIn();
        const conv = state.conversations[id];
        // Send history so server can continue the conversation after page refresh
        wsSendJson({
            type: "select_conversation",
            conversation_id: id,
            history: messagesToLLMHistory(conv.messages || []),
        });
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
            // Send current conversation with history so server can continue after page refresh
            const conv = state.conversations[state.currentId];
            wsSendJson({
                type: "select_conversation",
                conversation_id: state.currentId,
                history: messagesToLLMHistory(conv ? conv.messages : []),
            });
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
            } else if (message.type === "stt_token") {
                // Token-level (or small-delta) updates from FunASR MLX.
                // Prefer `accumulated` if present (server may batch deltas).
                const acc = typeof message.accumulated === "string" ? message.accumulated : "";
                const delta = typeof message.data === "string" ? message.data : "";
                const nextText = (acc || delta || "").trim();
                if (!nextText || nextText === "/sil") return;
                sttDraftByConv.set(convId, nextText);
                scheduleApplySttDrafts();
            } else if (message.type === "stt_partial") {
                if (typeof message.data === "string" && message.data.trim()) {
                    if (message.data.trim() === "/sil") return;
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
