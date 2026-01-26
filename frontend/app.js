/**
 * TTS Studio - Frontend Application
 * Connects to RunPod serverless endpoints for TTS generation
 */

// State
const state = {
    endpointUrl: localStorage.getItem('tts_endpoint_url') || '',
    apiKey: localStorage.getItem('tts_api_key') || '',
    model: localStorage.getItem('tts_model') || '1.7b-customvoice',
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
    
    modelSelect: document.getElementById('model-select'),
    modelDesc: document.getElementById('model-desc'),
    
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
    activeRequests: document.getElementById('active-requests'),
    activeCount: document.getElementById('active-count'),
    
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
    elements.modelSelect.value = state.model;
    
    // Event listeners
    setupEventListeners();
    
    // Update UI
    updateModelInfo();
    updateVoiceInfo();
    updateCharCount();
    renderHistory();
    updateActiveRequestsUI();
    
    // Resume any pending requests on page load
    resumePendingRequests();
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
    
    // Model selection
    elements.modelSelect.addEventListener('change', () => {
        state.model = elements.modelSelect.value;
        localStorage.setItem('tts_model', state.model);
        updateModelInfo();
        toggleVoiceDesignSection();
    });
    
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

function updateModelInfo() {
    const modelDescriptions = {
        '1.7b-customvoice': 'Best quality model with instruction control for 9 premium voices.',
        '0.6b-customvoice': 'Faster, lighter model. Same voices, quicker generation.',
        '1.7b-voicedesign': 'Create custom voices from text descriptions. Use "Voice Design" voice option.',
    };
    elements.modelDesc.textContent = modelDescriptions[state.model] || '';
}

function updateVoiceInfo() {
    const selected = elements.voiceSelect.selectedOptions[0];
    const gender = selected.dataset.gender;
    const lang = selected.dataset.lang;
    
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
    const isVoiceDesignModel = state.model === '1.7b-voicedesign';
    
    // Show voice design section if voice_design is selected OR using voicedesign model
    elements.voiceDesignSection.style.display = (isVoiceDesign || isVoiceDesignModel) ? 'block' : 'none';
    
    // Auto-select voice_design voice when using voicedesign model
    if (isVoiceDesignModel && elements.voiceSelect.value !== 'voice_design') {
        elements.voiceSelect.value = 'voice_design';
        updateVoiceInfo();
    }
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
        try {
            const runpodResponse = await fetch(state.endpointUrl, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${state.apiKey}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    input: { input: 'test', voice: 'Serena' }
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
    
    const voice = elements.voiceSelect.value;
    const format = elements.format.value;
    const speed = parseFloat(elements.speed.value);
    const voiceName = elements.voiceSelect.selectedOptions[0].text;
    
    // Build request payload
    const ttsParams = {
        input: text,
        voice: voice,
        response_format: format,
        speed: speed,
        model: state.model,
    };
    
    if (voice === 'voice_design') {
        ttsParams.instruction = elements.voiceDesignPrompt.value || 'A clear and natural voice';
    } else if (elements.instruction.value.trim()) {
        ttsParams.instruction = elements.instruction.value.trim();
    }
    
    // Create pending history item immediately
    const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const startTime = Date.now();
    const pendingItem = {
        id: requestId,
        text: text.substring(0, 100) + (text.length > 100 ? '...' : ''),
        fullText: text,
        voice: voice,
        voiceName: voiceName,
        model: state.model,
        format: format,
        speed: speed,
        instruction: ttsParams.instruction || null,
        timestamp: startTime,
        status: 'pending',
        elapsedSeconds: 0,
        audioBase64: null,
        error: null,
    };
    
    // Add to history immediately
    state.history.unshift(pendingItem);
    if (state.history.length > 30) {  // Keep more items for parallel requests
        state.history = state.history.slice(0, 30);
    }
    saveHistory();
    renderHistory();
    updateActiveRequestsUI();
    
    // Start timer for this specific request
    const timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        updateHistoryItemElapsed(requestId, elapsed);
    }, 1000);
    
    // Mark as processing
    updateHistoryItemStatus(requestId, 'processing');
    
    // Show toast that request is queued
    showToast(`Request queued: "${text.substring(0, 30)}..."`, 'success');
    
    try {
        const runpodRequest = { input: ttsParams };
        console.log('Sending request:', runpodRequest);
        
        // Use async /run endpoint instead of /runsync to avoid 90s timeout
        const runUrl = state.endpointUrl.replace('/runsync', '/run');
        
        const response = await fetch(runUrl, {
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
        
        const submitResult = await response.json();
        console.log('Job submitted:', submitResult);
        
        if (!submitResult.id) {
            throw new Error('No job ID returned from RunPod');
        }
        
        const jobId = submitResult.id;
        const statusUrl = state.endpointUrl.replace('/runsync', `/status/${jobId}`);
        
        // Poll for completion (up to 15 minutes)
        const maxAttempts = 180; // 15 minutes with 5s intervals
        let attempts = 0;
        let result;
        
        while (attempts < maxAttempts) {
            await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5s between polls
            attempts++;
            
            const statusResponse = await fetch(statusUrl, {
                headers: {
                    'Authorization': `Bearer ${state.apiKey}`,
                },
            });
            
            if (!statusResponse.ok) {
                console.warn('Status check failed, retrying...');
                continue;
            }
            
            result = await statusResponse.json();
            console.log(`Poll ${attempts}: status=${result.status}`);
            
            if (result.status === 'COMPLETED') {
                break;
            } else if (result.status === 'FAILED') {
                throw new Error(result.error || 'Generation failed on RunPod');
            } else if (result.status === 'CANCELLED') {
                throw new Error('Job was cancelled');
            }
            // IN_QUEUE or IN_PROGRESS - keep polling
        }
        
        clearInterval(timerInterval);
        
        if (!result || result.status !== 'COMPLETED') {
            throw new Error('Job timed out after 15 minutes');
        }
        
        console.log('Final result:', result);
        
        const output = result.output || result;
        if (output.error) {
            throw new Error(output.error);
        }
        
        let audioData;
        if (output.audio) {
            audioData = output.audio;
        } else if (typeof output === 'string') {
            audioData = output;
        } else {
            console.error('Unexpected response format:', result);
            throw new Error('No audio data in response. Full result: ' + JSON.stringify(result).substring(0, 500));
        }
        
        // Convert to blob for playback
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
        
        const blob = new Blob([bytes], { type: mimeTypes[format] || 'audio/mpeg' });
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        
        // Update history item to ready
        updateHistoryItemComplete(requestId, audioData, parseFloat(elapsed));
        
        // Update player with this audio (latest completed)
        state.currentAudioBlob = blob;
        if (state.currentAudioUrl) {
            URL.revokeObjectURL(state.currentAudioUrl);
        }
        state.currentAudioUrl = URL.createObjectURL(blob);
        elements.audioPlayer.src = state.currentAudioUrl;
        elements.audioContainer.classList.remove('hidden');
        
        elements.generationInfo.innerHTML = `
            <strong>Voice:</strong> ${voiceName} | 
            <strong>Format:</strong> ${format.toUpperCase()} | 
            <strong>Speed:</strong> ${speed}x | 
            <strong>Time:</strong> ${elapsed}s
        `;
        
        showToast(`Audio ready: "${text.substring(0, 30)}..."`, 'success');
        
    } catch (error) {
        clearInterval(timerInterval);
        console.error('Generation error:', error);
        const errorMsg = error.name === 'AbortError' ? 'Request timed out (15 min)' : error.message;
        updateHistoryItemStatus(requestId, 'error', errorMsg);
        showToast(`Failed: ${errorMsg}`, 'error');
    }
}

// Count and display active requests
function updateActiveRequestsUI() {
    const count = state.history.filter(h => h.status === 'pending' || h.status === 'processing').length;
    if (count > 0) {
        elements.activeRequests.style.display = 'flex';
        elements.activeCount.textContent = count;
    } else {
        elements.activeRequests.style.display = 'none';
    }
}

function updateHistoryItemStatus(id, status, error = null) {
    const item = state.history.find(h => h.id === id);
    if (item) {
        item.status = status;
        if (error) item.error = error;
        saveHistory();
        renderHistory();
        updateActiveRequestsUI();
    }
}

function updateHistoryItemElapsed(id, seconds) {
    const item = state.history.find(h => h.id === id);
    if (item && (item.status === 'processing' || item.status === 'pending')) {
        item.elapsedSeconds = seconds;
        // Don't save to localStorage on every tick, just update UI
        renderHistory();
    }
}

function updateHistoryItemComplete(id, audioBase64, elapsedSeconds) {
    const item = state.history.find(h => h.id === id);
    if (item) {
        item.status = 'ready';
        item.audioBase64 = audioBase64;
        item.elapsedSeconds = elapsedSeconds;
        item.error = null;
        saveHistory();
        renderHistory();
        updateActiveRequestsUI();
    }
}

function saveHistory() {
    localStorage.setItem('tts_history', JSON.stringify(state.history));
}

function resumePendingRequests() {
    // Mark any old pending/processing items as failed (they won't complete after page reload)
    let changed = false;
    state.history.forEach(item => {
        if (item.status === 'pending' || item.status === 'processing') {
            item.status = 'error';
            item.error = 'Request interrupted (page reload)';
            changed = true;
        }
    });
    if (changed) {
        saveHistory();
        renderHistory();
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

function renderHistory() {
    if (state.history.length === 0) {
        elements.historyList.innerHTML = '<p class="empty-state">No generations yet. Create your first one above!</p>';
        return;
    }
    
    elements.historyList.innerHTML = state.history.map((item, index) => {
        const statusBadge = getStatusBadge(item);
        const isPlayable = item.status === 'ready' && item.audioBase64;
        const isProcessing = item.status === 'pending' || item.status === 'processing';
        
        return `
        <div class="history-item ${item.status}" data-index="${index}">
            <div class="history-item-status-icon">
                ${isProcessing ? `
                    <div class="spinner-small"></div>
                ` : isPlayable ? `
                    <button class="history-item-play" onclick="playHistoryItem(${index})">
                        <svg viewBox="0 0 24 24" fill="currentColor" stroke="none">
                            <polygon points="5 3 19 12 5 21 5 3"/>
                        </svg>
                    </button>
                ` : `
                    <div class="history-item-error-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="15" y1="9" x2="9" y2="15"/>
                            <line x1="9" y1="9" x2="15" y2="15"/>
                        </svg>
                    </div>
                `}
            </div>
            <div class="history-item-info">
                <div class="history-item-text">${escapeHtml(item.text)}</div>
                <div class="history-item-meta">
                    <span>${item.voiceName}</span>
                    <span>${item.format.toUpperCase()}</span>
                    <span>${item.speed}x</span>
                    ${statusBadge}
                    ${item.elapsedSeconds ? `<span>${formatElapsed(item.elapsedSeconds)}</span>` : ''}
                </div>
                ${item.error ? `<div class="history-item-error">${escapeHtml(item.error)}</div>` : ''}
            </div>
            <div class="history-item-actions">
                ${isPlayable ? `
                    <button onclick="downloadHistoryItem(${index})" title="Download">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="7 10 12 15 17 10"/>
                            <line x1="12" y1="15" x2="12" y2="3"/>
                        </svg>
                    </button>
                ` : ''}
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
    `}).join('');
}

function getStatusBadge(item) {
    switch (item.status) {
        case 'pending':
            return '<span class="status-tag pending">Queued</span>';
        case 'processing':
            return '<span class="status-tag processing">Processing...</span>';
        case 'ready':
            return '<span class="status-tag ready">Ready</span>';
        case 'error':
            return '<span class="status-tag error">Failed</span>';
        default:
            return '';
    }
}

function formatElapsed(seconds) {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
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
    window.scrollTo({ top: 0, behavior: 'smooth' });
};

window.deleteHistoryItem = function(index) {
    state.history.splice(index, 1);
    saveHistory();
    renderHistory();
    showToast('Item deleted', 'success');
};

function clearHistory() {
    if (!confirm('Are you sure you want to clear all history?')) return;
    
    state.history = [];
    saveHistory();
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
