/**
 * TTS Studio - Frontend Application
 * Connects to RunPod serverless endpoints for TTS generation
 */

// State
const state = {
    endpointUrl: localStorage.getItem('tts_endpoint_url') || '',
    apiKey: localStorage.getItem('tts_api_key') || '',
    currentAudioBlob: null,
    currentAudioUrl: null,
    isGenerating: false,
    history: JSON.parse(localStorage.getItem('tts_history') || '[]'),
    loadingStartTime: null,
    loadingInterval: null,
};

// DOM Elements
const elements = {
    endpointUrl: document.getElementById('endpoint-url'),
    apiKey: document.getElementById('api-key'),
    toggleKey: document.getElementById('toggle-key'),
    testConnection: document.getElementById('test-connection'),
    connectionStatus: document.getElementById('connection-status'),
    
    voiceSelect: document.getElementById('voice-select'),
    voiceInfo: document.getElementById('voice-info'),
    voiceDesignSection: document.getElementById('voice-design-section'),
    voiceDesignPrompt: document.getElementById('voice-design-prompt'),
    
    instruction: document.getElementById('instruction'),
    speed: document.getElementById('speed'),
    speedValue: document.getElementById('speed-value'),
    format: document.getElementById('format'),
    
    textInput: document.getElementById('text-input'),
    charCount: document.getElementById('char-count'),
    clearText: document.getElementById('clear-text'),
    
    generateBtn: document.getElementById('generate-btn'),
    loading: document.getElementById('loading'),
    loadingTime: document.getElementById('loading-time'),
    
    audioContainer: document.getElementById('audio-container'),
    audioPlayer: document.getElementById('audio-player'),
    downloadBtn: document.getElementById('download-btn'),
    regenerateBtn: document.getElementById('regenerate-btn'),
    generationInfo: document.getElementById('generation-info'),
    
    historyList: document.getElementById('history-list'),
    clearHistory: document.getElementById('clear-history'),
    
    toastContainer: document.getElementById('toast-container'),
};

// Initialize
function init() {
    // Load saved settings
    elements.endpointUrl.value = state.endpointUrl;
    elements.apiKey.value = state.apiKey;
    
    // Event listeners
    setupEventListeners();
    
    // Update UI
    updateVoiceInfo();
    updateCharCount();
    renderHistory();
}

function setupEventListeners() {
    // Settings
    elements.endpointUrl.addEventListener('input', (e) => {
        state.endpointUrl = e.target.value;
        localStorage.setItem('tts_endpoint_url', state.endpointUrl);
    });
    
    elements.apiKey.addEventListener('input', (e) => {
        state.apiKey = e.target.value;
        localStorage.setItem('tts_api_key', state.apiKey);
    });
    
    elements.toggleKey.addEventListener('click', () => {
        const type = elements.apiKey.type === 'password' ? 'text' : 'password';
        elements.apiKey.type = type;
    });
    
    elements.testConnection.addEventListener('click', testConnection);
    
    // Voice selection
    elements.voiceSelect.addEventListener('change', () => {
        updateVoiceInfo();
        toggleVoiceDesignSection();
    });
    
    // Voice design presets
    document.querySelectorAll('[data-preset]').forEach(btn => {
        btn.addEventListener('click', () => {
            elements.voiceDesignPrompt.value = btn.dataset.preset;
        });
    });
    
    // Instruction presets
    document.querySelectorAll('[data-instruction]').forEach(btn => {
        btn.addEventListener('click', () => {
            elements.instruction.value = btn.dataset.instruction;
        });
    });
    
    // Speed slider
    elements.speed.addEventListener('input', () => {
        elements.speedValue.textContent = `${elements.speed.value}x`;
    });
    
    // Text input
    elements.textInput.addEventListener('input', updateCharCount);
    elements.clearText.addEventListener('click', () => {
        elements.textInput.value = '';
        updateCharCount();
    });
    
    // Sample texts
    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            elements.textInput.value = btn.dataset.text;
            updateCharCount();
        });
    });
    
    // Generate
    elements.generateBtn.addEventListener('click', generateSpeech);
    elements.regenerateBtn.addEventListener('click', generateSpeech);
    
    // Download
    elements.downloadBtn.addEventListener('click', downloadAudio);
    
    // History
    elements.clearHistory.addEventListener('click', clearHistory);
}

function updateVoiceInfo() {
    const selected = elements.voiceSelect.selectedOptions[0];
    const gender = selected.dataset.gender;
    const lang = selected.dataset.lang;
    
    // Get language name from code
    const langNames = {
        'zh-CN': 'Chinese',
        'en-US': 'English',
        'ja-JP': 'Japanese',
        'ko-KR': 'Korean',
        'mul': 'Multilingual',
    };
    
    const genderClass = gender || 'neutral';
    const langName = langNames[lang] || lang;
    
    elements.voiceInfo.innerHTML = `
        <div class="voice-badge ${genderClass}">${capitalize(gender || 'Neutral')}</div>
        <div class="voice-badge lang">${langName}</div>
    `;
}

function toggleVoiceDesignSection() {
    const isVoiceDesign = elements.voiceSelect.value === 'voice_design';
    elements.voiceDesignSection.style.display = isVoiceDesign ? 'block' : 'none';
}

function updateCharCount() {
    const count = elements.textInput.value.length;
    elements.charCount.textContent = count.toLocaleString();
}

async function testConnection() {
    if (!state.endpointUrl) {
        showToast('Please enter an endpoint URL', 'error');
        return;
    }
    
    elements.testConnection.disabled = true;
    elements.connectionStatus.className = 'status-badge';
    elements.connectionStatus.textContent = 'Testing...';
    
    try {
        // For RunPod, we'll test by making a health check or a simple request
        const response = await fetch(state.endpointUrl.replace('/runsync', '/health'), {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${state.apiKey}`,
            },
        });
        
        if (response.ok) {
            elements.connectionStatus.className = 'status-badge success';
            elements.connectionStatus.textContent = 'Connected';
            showToast('Connection successful!', 'success');
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (error) {
        // Try alternative - just check if the endpoint responds
        try {
            const runpodResponse = await fetch(state.endpointUrl, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${state.apiKey}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    input: {
                        openai_route: '/health',
                        openai_input: {}
                    }
                }),
            });
            
            if (runpodResponse.ok || runpodResponse.status === 400) {
                elements.connectionStatus.className = 'status-badge success';
                elements.connectionStatus.textContent = 'Endpoint reachable';
                showToast('Endpoint is reachable', 'success');
            } else {
                throw new Error(`HTTP ${runpodResponse.status}`);
            }
        } catch (e) {
            elements.connectionStatus.className = 'status-badge error';
            elements.connectionStatus.textContent = 'Connection failed';
            showToast(`Connection failed: ${error.message}`, 'error');
        }
    } finally {
        elements.testConnection.disabled = false;
    }
}

async function generateSpeech() {
    // Validate
    if (!state.endpointUrl) {
        showToast('Please configure the endpoint URL', 'error');
        return;
    }
    
    if (!state.apiKey) {
        showToast('Please enter your API key', 'error');
        return;
    }
    
    const text = elements.textInput.value.trim();
    if (!text) {
        showToast('Please enter some text to synthesize', 'error');
        return;
    }
    
    // Update UI for loading state
    state.isGenerating = true;
    elements.generateBtn.disabled = true;
    elements.loading.classList.remove('hidden');
    elements.audioContainer.classList.add('hidden');
    
    // Start loading timer
    state.loadingStartTime = Date.now();
    elements.loadingTime.textContent = '0s';
    state.loadingInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - state.loadingStartTime) / 1000);
        elements.loadingTime.textContent = `${elapsed}s`;
    }, 1000);
    
    try {
        const voice = elements.voiceSelect.value;
        const format = elements.format.value;
        const speed = parseFloat(elements.speed.value);
        
        // Build request payload for RunPod handler
        // The handler expects: { input: { input: "text", voice: "...", ... } }
        const ttsParams = {
            input: text,  // The text to synthesize
            voice: voice,
            response_format: format,
            speed: speed,
        };
        
        // Add instruction if voice_design or if instruction provided
        if (voice === 'voice_design') {
            ttsParams.instruction = elements.voiceDesignPrompt.value || 'A clear and natural voice';
        } else if (elements.instruction.value.trim()) {
            ttsParams.instruction = elements.instruction.value.trim();
        }
        
        // RunPod request wrapper
        const runpodRequest = {
            input: ttsParams
        };
        
        console.log('Sending request:', runpodRequest);
        
        const response = await fetch(state.endpointUrl, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${state.apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(runpodRequest),
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        
        const result = await response.json();
        console.log('Response:', result);
        
        // Handle RunPod response
        // RunPod wraps the handler response in { output: {...} } or { status: "FAILED", error: "..." }
        if (result.status === 'FAILED') {
            throw new Error(result.error || 'Generation failed');
        }
        
        // Check for error in output
        const output = result.output || result;
        if (output.error) {
            throw new Error(output.error);
        }
        
        // Extract audio data from RunPod response
        // Handler returns: { audio: "base64...", format: "mp3", voice: "...", duration_estimate: ... }
        let audioData;
        if (output.audio) {
            audioData = output.audio;
        } else if (typeof output === 'string') {
            // Base64 string directly
            audioData = output;
        } else {
            console.error('Unexpected response format:', result);
            throw new Error('No audio data in response');
        }
        
        // Convert base64 to blob
        const binaryString = atob(audioData);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        
        const mimeTypes = {
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'opus': 'audio/opus',
            'flac': 'audio/flac',
            'aac': 'audio/aac',
        };
        
        state.currentAudioBlob = new Blob([bytes], { type: mimeTypes[format] || 'audio/mpeg' });
        
        // Revoke old URL if exists
        if (state.currentAudioUrl) {
            URL.revokeObjectURL(state.currentAudioUrl);
        }
        
        state.currentAudioUrl = URL.createObjectURL(state.currentAudioBlob);
        elements.audioPlayer.src = state.currentAudioUrl;
        
        // Show audio container
        elements.audioContainer.classList.remove('hidden');
        
        // Update generation info
        const elapsed = ((Date.now() - state.loadingStartTime) / 1000).toFixed(1);
        elements.generationInfo.innerHTML = `
            <strong>Voice:</strong> ${elements.voiceSelect.selectedOptions[0].text} | 
            <strong>Format:</strong> ${format.toUpperCase()} | 
            <strong>Speed:</strong> ${speed}x | 
            <strong>Time:</strong> ${elapsed}s
        `;
        
        // Add to history
        addToHistory({
            text: text.substring(0, 100) + (text.length > 100 ? '...' : ''),
            fullText: text,
            voice: voice,
            voiceName: elements.voiceSelect.selectedOptions[0].text,
            format: format,
            speed: speed,
            instruction: speechRequest.instruction || null,
            timestamp: Date.now(),
            audioBlob: state.currentAudioBlob,
        });
        
        showToast('Audio generated successfully!', 'success');
        
        // Auto-play
        elements.audioPlayer.play().catch(() => {});
        
    } catch (error) {
        console.error('Generation error:', error);
        showToast(`Generation failed: ${error.message}`, 'error');
    } finally {
        state.isGenerating = false;
        elements.generateBtn.disabled = false;
        elements.loading.classList.add('hidden');
        clearInterval(state.loadingInterval);
    }
}

function downloadAudio() {
    if (!state.currentAudioBlob) return;
    
    const format = elements.format.value;
    const link = document.createElement('a');
    link.href = state.currentAudioUrl;
    link.download = `tts_${Date.now()}.${format}`;
    link.click();
    
    showToast('Download started', 'success');
}

function addToHistory(item) {
    // Store audio as base64 for persistence
    const reader = new FileReader();
    reader.onload = () => {
        item.audioBase64 = reader.result.split(',')[1];
        delete item.audioBlob;
        
        state.history.unshift(item);
        
        // Keep only last 20 items
        if (state.history.length > 20) {
            state.history = state.history.slice(0, 20);
        }
        
        localStorage.setItem('tts_history', JSON.stringify(state.history));
        renderHistory();
    };
    reader.readAsDataURL(item.audioBlob);
}

function renderHistory() {
    if (state.history.length === 0) {
        elements.historyList.innerHTML = '<p class="empty-state">No generations yet. Create your first one above!</p>';
        return;
    }
    
    elements.historyList.innerHTML = state.history.map((item, index) => `
        <div class="history-item" data-index="${index}">
            <button class="history-item-play" onclick="playHistoryItem(${index})">
                <svg viewBox="0 0 24 24" fill="currentColor" stroke="none">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
            </button>
            <div class="history-item-info">
                <div class="history-item-text">${escapeHtml(item.text)}</div>
                <div class="history-item-meta">
                    <span>${item.voiceName}</span>
                    <span>${formatTimestamp(item.timestamp)}</span>
                </div>
            </div>
            <div class="history-item-actions">
                <button onclick="downloadHistoryItem(${index})" title="Download">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="7 10 12 15 17 10"/>
                        <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                </button>
                <button onclick="reuseHistoryItem(${index})" title="Reuse settings">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="23 4 23 10 17 10"/>
                        <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
                    </svg>
                </button>
                <button onclick="deleteHistoryItem(${index})" title="Delete">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"/>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                </button>
            </div>
        </div>
    `).join('');
}

// Global functions for history item actions
window.playHistoryItem = function(index) {
    const item = state.history[index];
    if (!item.audioBase64) return;
    
    const binaryString = atob(item.audioBase64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    
    const mimeTypes = {
        'mp3': 'audio/mpeg',
        'wav': 'audio/wav',
        'opus': 'audio/opus',
        'flac': 'audio/flac',
        'aac': 'audio/aac',
    };
    
    const blob = new Blob([bytes], { type: mimeTypes[item.format] || 'audio/mpeg' });
    const url = URL.createObjectURL(blob);
    
    elements.audioPlayer.src = url;
    elements.audioContainer.classList.remove('hidden');
    elements.audioPlayer.play().catch(() => {});
};

window.downloadHistoryItem = function(index) {
    const item = state.history[index];
    if (!item.audioBase64) return;
    
    const link = document.createElement('a');
    link.href = `data:audio/${item.format};base64,${item.audioBase64}`;
    link.download = `tts_${item.timestamp}.${item.format}`;
    link.click();
};

window.reuseHistoryItem = function(index) {
    const item = state.history[index];
    
    elements.textInput.value = item.fullText || item.text;
    elements.voiceSelect.value = item.voice;
    elements.format.value = item.format;
    elements.speed.value = item.speed;
    elements.speedValue.textContent = `${item.speed}x`;
    
    if (item.instruction) {
        if (item.voice === 'voice_design') {
            elements.voiceDesignPrompt.value = item.instruction;
        } else {
            elements.instruction.value = item.instruction;
        }
    }
    
    updateCharCount();
    updateVoiceInfo();
    toggleVoiceDesignSection();
    
    showToast('Settings restored from history', 'success');
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
};

window.deleteHistoryItem = function(index) {
    state.history.splice(index, 1);
    localStorage.setItem('tts_history', JSON.stringify(state.history));
    renderHistory();
    showToast('Item deleted', 'success');
};

function clearHistory() {
    if (!confirm('Are you sure you want to clear all history?')) return;
    
    state.history = [];
    localStorage.setItem('tts_history', JSON.stringify(state.history));
    renderHistory();
    showToast('History cleared', 'success');
}

// Utility functions
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'success' 
        ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>'
        : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
    
    toast.innerHTML = `${icon}<span>${escapeHtml(message)}</span>`;
    elements.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTimestamp(ts) {
    const date = new Date(ts);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    
    return date.toLocaleDateString();
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);
