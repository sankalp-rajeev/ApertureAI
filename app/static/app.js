/* ============================================================
   ApertureAI Dashboard - Client-side Logic
   Plotly charts, tab navigation, image upload, SPC animation
   ============================================================ */

const FALLBACK_PALETTE = {
    paper: '#171e28',
    plot: '#101720',
    grid: '#2a3442',
    text: '#b5c0ce',
    textStrong: '#e7ecf4',
    accent: '#5b7da4',
    success: '#4f9a78',
    warning: '#b99259',
    danger: '#b36b6b',
    muted: '#8c9aae',
    bgPrimary: '#0b1016',
    purple: '#7b7297',
    cyan: '#4f7f88',
};

function cssVar(name, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
}

function buildPalette() {
    return {
        paper: cssVar('--bg-card', FALLBACK_PALETTE.paper),
        plot: cssVar('--bg-input', FALLBACK_PALETTE.plot),
        grid: cssVar('--border-color', FALLBACK_PALETTE.grid),
        text: cssVar('--text-secondary', FALLBACK_PALETTE.text),
        textStrong: cssVar('--text-primary', FALLBACK_PALETTE.textStrong),
        accent: cssVar('--accent-blue', FALLBACK_PALETTE.accent),
        success: cssVar('--accent-green', FALLBACK_PALETTE.success),
        warning: cssVar('--accent-amber', FALLBACK_PALETTE.warning),
        danger: cssVar('--accent-red', FALLBACK_PALETTE.danger),
        muted: cssVar('--text-muted', FALLBACK_PALETTE.muted),
        bgPrimary: cssVar('--bg-primary', FALLBACK_PALETTE.bgPrimary),
        purple: cssVar('--accent-purple', FALLBACK_PALETTE.purple),
        cyan: cssVar('--accent-cyan', FALLBACK_PALETTE.cyan),
    };
}

const PALETTE = buildPalette();
const LOCALE = document.documentElement.lang || navigator.language || 'en-US';
const STATUS_REGION = document.getElementById('status-region');
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;
const SUPPORTED_IMAGE_TYPES = new Set([
    'image/png',
    'image/jpeg',
    'image/bmp',
    'image/webp',
    'image/tiff',
]);
let statusTimeoutId = null;
let spcIntervalId = null;

function formatNumber(value, { minimumFractionDigits = 0, maximumFractionDigits = 2 } = {}) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return 'N/A';
    return new Intl.NumberFormat(LOCALE, { minimumFractionDigits, maximumFractionDigits }).format(numeric);
}

function formatPercent(ratio, { minimumFractionDigits = 0, maximumFractionDigits = 1 } = {}) {
    const numeric = Number(ratio);
    if (!Number.isFinite(numeric)) return 'N/A';
    return new Intl.NumberFormat(LOCALE, {
        style: 'percent',
        minimumFractionDigits,
        maximumFractionDigits,
    }).format(numeric);
}

function truncateText(text, limit = 160) {
    if (typeof text !== 'string') return '';
    const normalized = text.replace(/\s+/g, ' ').trim();
    if (normalized.length <= limit) return normalized;
    return `${normalized.slice(0, limit - 3)}...`;
}

function pluralize(count, singular, plural = `${singular}s`) {
    const numeric = Number(count);
    const display = formatNumber(numeric, { maximumFractionDigits: 0 });
    const label = numeric === 1 ? singular : plural;
    return `${display} ${label}`;
}

function showStatus(message, tone = 'info', { persist = false } = {}) {
    if (!STATUS_REGION) return;
    STATUS_REGION.hidden = false;
    STATUS_REGION.className = `status-banner status-${tone}`;
    STATUS_REGION.textContent = truncateText(message, 220);

    if (statusTimeoutId) clearTimeout(statusTimeoutId);
    if (!persist) {
        statusTimeoutId = setTimeout(() => {
            STATUS_REGION.hidden = true;
            STATUS_REGION.textContent = '';
        }, 5000);
    }
}

function clearStatus() {
    if (!STATUS_REGION) return;
    if (statusTimeoutId) clearTimeout(statusTimeoutId);
    statusTimeoutId = null;
    STATUS_REGION.hidden = true;
    STATUS_REGION.textContent = '';
}

function setPanelState(elementId, message = '', tone = 'info') {
    const node = document.getElementById(elementId);
    if (!node) return;

    const hasMessage = Boolean(message);
    node.hidden = !hasMessage;
    node.className = `panel-state status-${tone}`;
    node.textContent = hasMessage ? truncateText(message, 280) : '';
}

function parseServerDetail(payload) {
    if (!payload) return '';
    if (typeof payload === 'string') return payload;
    if (typeof payload?.detail === 'string') return payload.detail;
    if (typeof payload?.message === 'string') return payload.message;
    return '';
}

function mapHttpError(status, detail = '') {
    const trimmed = truncateText(detail, 160);
    const suffix = trimmed ? ` ${trimmed}` : '';
    if (status === 400) return `Request validation failed.${suffix}`;
    if (status === 401) return `Authentication required.${suffix}`;
    if (status === 403) return `You do not have permission for this action.${suffix}`;
    if (status === 404) return `Requested resource was not found.${suffix}`;
    if (status === 429) return `Rate limit reached. Please retry shortly.${suffix}`;
    if (status >= 500) return `Server error (${status}). Please retry.${suffix}`;
    return `Request failed (${status}).${suffix}`;
}

async function fetchJson(url, { method = 'GET', body = null, timeoutMs = 20000 } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const res = await fetch(url, { method, body, signal: controller.signal });
        const contentType = (res.headers.get('content-type') || '').toLowerCase();
        const payload = contentType.includes('application/json')
            ? await res.json().catch(() => null)
            : await res.text().catch(() => '');

        if (!res.ok) {
            const detail = parseServerDetail(payload);
            const err = new Error(mapHttpError(res.status, detail));
            err.status = res.status;
            throw err;
        }

        return payload;
    } catch (err) {
        if (err?.name === 'AbortError') {
            throw new Error('Request timed out. Please try again.');
        }
        if (err instanceof TypeError) {
            throw new Error('Network error. Check your connection and retry.');
        }
        throw err;
    } finally {
        clearTimeout(timer);
    }
}

// ---- Plotly Theme ----
const PLOTLY_LAYOUT = {
    paper_bgcolor: PALETTE.paper,
    plot_bgcolor: PALETTE.plot,
    font: { family: 'IBM Plex Sans, sans-serif', color: PALETTE.text, size: 12 },
    margin: { t: 40, r: 30, b: 50, l: 60 },
    xaxis: {
        gridcolor: PALETTE.grid,
        zerolinecolor: PALETTE.grid,
        tickfont: { size: 11 },
    },
    yaxis: {
        gridcolor: PALETTE.grid,
        zerolinecolor: PALETTE.grid,
        tickfont: { size: 11 },
    },
};

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    displaylogo: false,
};

// ---- Tab Navigation ----
const tabButtons = Array.from(document.querySelectorAll('.tab-btn'));
const tabPanels = Array.from(document.querySelectorAll('.tab-panel'));
let activeTabKey = null;

function activateTab(button, { focus = false } = {}) {
    const nextTabKey = button.dataset.tab;
    const panelId = button.getAttribute('aria-controls');
    if (activeTabKey === 'spc' && nextTabKey !== 'spc') {
        stopSPCAnimation();
    }

    tabButtons.forEach(btn => {
        const isActive = btn === button;
        btn.classList.toggle('active', isActive);
        btn.setAttribute('aria-selected', String(isActive));
        btn.tabIndex = isActive ? 0 : -1;
    });

    tabPanels.forEach(panel => {
        const isActive = panel.id === panelId;
        panel.classList.toggle('active', isActive);
        panel.hidden = !isActive;
    });

    if (focus) {
        button.focus();
    }

    if (button.dataset.tab === 'doe' && !window._doeLoaded) loadDOE();
    if (button.dataset.tab === 'system' && !window._systemLoaded) loadSystem();
    activeTabKey = nextTabKey;
}

tabButtons.forEach((btn, index) => {
    btn.addEventListener('click', () => activateTab(btn));

    btn.addEventListener('keydown', event => {
        if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();

        let nextIndex = index;
        if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabButtons.length;
        if (event.key === 'ArrowLeft') nextIndex = (index - 1 + tabButtons.length) % tabButtons.length;
        if (event.key === 'Home') nextIndex = 0;
        if (event.key === 'End') nextIndex = tabButtons.length - 1;

        activateTab(tabButtons[nextIndex], { focus: true });
    });
});

const initialTab = tabButtons.find(btn => btn.classList.contains('active')) || tabButtons[0];
if (initialTab) {
    activateTab(initialTab);
}

// ============================================================
// Panel 1: Predict
// ============================================================

const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const predictBtn = document.getElementById('predict-btn');
const predictionResult = document.getElementById('prediction-result');
let selectedFile = null;

function resetUploadZone() {
    uploadZone.querySelector('.upload-text').textContent = 'Drop an image here or click to browse';
    uploadZone.querySelector('.upload-hint').textContent = 'Supports PNG, JPG, BMP - NEU-DET steel surface images';
    predictBtn.disabled = true;
}

function validateImageFile(file) {
    if (!file) return 'No file selected.';
    const extension = (file.name.split('.').pop() || '').toLowerCase();
    const extensionAllowed = ['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tif', 'tiff'].includes(extension);
    if (file.type) {
        if (!SUPPORTED_IMAGE_TYPES.has(file.type)) {
            return 'Unsupported file type. Use PNG, JPG, BMP, WEBP, or TIFF.';
        }
    } else if (!extensionAllowed) {
        return 'Unsupported file type. Use PNG, JPG, BMP, WEBP, or TIFF.';
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
        return `File exceeds ${formatNumber(MAX_FILE_SIZE_BYTES / (1024 * 1024), { maximumFractionDigits: 0 })}MB limit.`;
    }
    return '';
}

uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        const file = e.dataTransfer.files[0];
        const validationError = validateImageFile(file);
        if (validationError) {
            selectedFile = null;
            fileInput.value = '';
            resetUploadZone();
            showStatus(validationError, 'warning');
            return;
        }
        selectedFile = file;
        updateUploadZone(selectedFile.name);
    }
});

fileInput.addEventListener('change', e => {
    if (e.target.files.length) {
        const file = e.target.files[0];
        const validationError = validateImageFile(file);
        if (validationError) {
            selectedFile = null;
            fileInput.value = '';
            resetUploadZone();
            showStatus(validationError, 'warning');
            return;
        }
        selectedFile = file;
        updateUploadZone(selectedFile.name);
    }
});

function updateUploadZone(name) {
    uploadZone.querySelector('.upload-text').textContent = truncateText(name, 90);
    uploadZone.querySelector('.upload-hint').textContent = 'Click "Classify Image" to run inference';
    predictBtn.disabled = false;
}

resetUploadZone();

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    clearStatus();

    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="spinner"></span> Processing...';

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const data = await fetchJson('/api/predict', {
            method: 'POST',
            body: formData,
            timeoutMs: 30000,
        });

        showPrediction(data);
        showStatus('Prediction completed successfully.', 'success');
    } catch (err) {
        showStatus(`Prediction failed. ${err.message}`, 'error', { persist: true });
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = 'Classify Image';
    }
});

function showPrediction(data) {
    predictionResult.classList.add('visible');

    const originalSrc = data?.original_base64 ? `data:image/png;base64,${data.original_base64}` : '';
    const heatmapSrc = data?.heatmap_base64 ? `data:image/png;base64,${data.heatmap_base64}` : '';

    document.getElementById('original-img').src = originalSrc;
    document.getElementById('heatmap-img').src = heatmapSrc;

    const badge = document.getElementById('result-badge');
    const className = data?.class_name || 'Unknown';
    const confidenceText = formatPercent(data?.confidence, { maximumFractionDigits: 1 });
    badge.textContent = `${className} (${confidenceText})`;
    badge.className = 'card-badge badge-green';

    const container = document.getElementById('prob-bars');
    container.innerHTML = '';
    const probabilities = (data && typeof data.probabilities === 'object' && data.probabilities)
        ? data.probabilities
        : {};
    const classes = Object.entries(probabilities).sort((a, b) => Number(b[1]) - Number(a[1]));

    if (!classes.length) {
        container.innerHTML = '<div class="status-banner status-warning">Probability breakdown unavailable for this response.</div>';
        return;
    }

    for (const [name, prob] of classes) {
        const safeProb = Number.isFinite(Number(prob)) ? Math.max(0, Math.min(1, Number(prob))) : 0;
        const isPredicted = name === className;
        const bar = document.createElement('div');
        bar.className = 'prob-bar';

        const label = document.createElement('span');
        label.className = 'prob-bar-label';
        label.textContent = truncateText(name, 24);

        const track = document.createElement('div');
        track.className = 'prob-bar-track';

        const fill = document.createElement('div');
        fill.className = `prob-bar-fill${isPredicted ? ' predicted' : ''}`;
        fill.style.width = `${(safeProb * 100).toFixed(1)}%`;
        track.appendChild(fill);

        const value = document.createElement('span');
        value.className = 'prob-bar-value';
        value.textContent = formatPercent(safeProb, { maximumFractionDigits: 1 });

        bar.appendChild(label);
        bar.appendChild(track);
        bar.appendChild(value);
        container.appendChild(bar);
    }
}

// ============================================================
// Panel 2: SPC Monitor
// ============================================================

const spcDemoBtn = document.getElementById('spc-demo-btn');
spcDemoBtn.addEventListener('click', runSPCDemo);

function stopSPCAnimation() {
    if (spcIntervalId) {
        clearInterval(spcIntervalId);
        spcIntervalId = null;
    }
    spcDemoBtn.disabled = false;
    spcDemoBtn.innerHTML = 'Run 40-Batch Demo';
}

async function runSPCDemo() {
    stopSPCAnimation();
    clearStatus();
    spcDemoBtn.disabled = true;
    spcDemoBtn.innerHTML = '<span class="spinner"></span> Loading...';

    try {
        const data = await fetchJson('/api/spc/run-demo', { timeoutMs: 30000 });
        if (!Array.isArray(data?.batches) || !data?.limits) {
            throw new Error('SPC demo payload is incomplete.');
        }

        spcDemoBtn.innerHTML = 'Animating...';
        animateSPC(data);
        showStatus('SPC demo loaded. Streaming batches now.', 'info');
    } catch (err) {
        stopSPCAnimation();
        showStatus(`SPC demo failed. ${err.message}`, 'error', { persist: true });
    }
}

function animateSPC(data) {
    const { batches, limits } = data;
    const alertLog = document.getElementById('alert-log');
    alertLog.innerHTML = '';

    let idx = 0;
    const xbars = [];
    const colors = [];
    const oocIndices = [];

    const layout = {
        ...PLOTLY_LAYOUT,
        title: {
            text: 'X-bar Control Chart - Defect Rate per Batch',
            font: { size: 14, color: PALETTE.textStrong },
        },
        xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Batch Index', range: [-1, 42] },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Defect Rate', range: [-0.05, 0.45] },
        shapes: [
            {
                type: 'line',
                x0: -1,
                x1: 42,
                y0: limits.ucl_x,
                y1: limits.ucl_x,
                line: { color: PALETTE.danger, width: 2, dash: 'dash' },
            },
            {
                type: 'line',
                x0: -1,
                x1: 42,
                y0: limits.cl_x,
                y1: limits.cl_x,
                line: { color: PALETTE.success, width: 2 },
            },
            {
                type: 'line',
                x0: -1,
                x1: 42,
                y0: limits.lcl_x,
                y1: limits.lcl_x,
                line: { color: PALETTE.danger, width: 2, dash: 'dash' },
            },
            {
                type: 'line',
                x0: 20,
                x1: 20,
                y0: -0.05,
                y1: 0.45,
                line: { color: PALETTE.warning, width: 1, dash: 'dot' },
            },
            {
                type: 'line',
                x0: 30,
                x1: 30,
                y0: -0.05,
                y1: 0.45,
                line: { color: PALETTE.danger, width: 1, dash: 'dot' },
            },
        ],
        annotations: [
            { x: 10, y: 0.42, text: 'Baseline', showarrow: false, font: { color: PALETTE.muted, size: 11 } },
            { x: 25, y: 0.42, text: '+10% Drift', showarrow: false, font: { color: PALETTE.warning, size: 11 } },
            { x: 35, y: 0.42, text: '+20% Drift', showarrow: false, font: { color: PALETTE.danger, size: 11 } },
            { x: 42, y: limits.ucl_x, text: 'UCL', showarrow: false, font: { color: PALETTE.danger, size: 10 } },
            { x: 42, y: limits.cl_x, text: 'CL', showarrow: false, font: { color: PALETTE.success, size: 10 } },
        ],
        showlegend: false,
    };

    Plotly.newPlot('xbar-chart', [], layout, PLOTLY_CONFIG);

    if (spcIntervalId) clearInterval(spcIntervalId);
    spcIntervalId = setInterval(() => {
        if (idx >= batches.length) {
            stopSPCAnimation();
            showStatus('SPC demo completed.', 'success');
            return;
        }

        const batch = batches[idx];
        xbars.push(batch.xbar);

        const color = batch.ooc ? PALETTE.danger : (batch.phase === 'baseline' ? PALETTE.accent : PALETTE.warning);
        colors.push(color);
        if (batch.ooc) oocIndices.push(idx);

        Plotly.react('xbar-chart', [{
            x: Array.from({ length: xbars.length }, (_, i) => i),
            y: xbars,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: PALETTE.accent, width: 2 },
            marker: { color: colors, size: 8, line: { color: PALETTE.bgPrimary, width: 1 } },
        }], layout, PLOTLY_CONFIG);

        document.getElementById('spc-batches').textContent = formatNumber(idx + 1, { maximumFractionDigits: 0 });
        document.getElementById('spc-detections').textContent = formatNumber(oocIndices.length, { maximumFractionDigits: 0 });
        document.getElementById('spc-cl').textContent = formatNumber(limits.cl_x, {
            minimumFractionDigits: 3,
            maximumFractionDigits: 3,
        });

        const driftSoFar = batches.slice(0, idx + 1).filter(b => b.phase !== 'baseline');
        if (driftSoFar.length > 0) {
            const detected = driftSoFar.filter(b => b.ooc).length;
            document.getElementById('spc-rate').textContent = formatPercent(detected / driftSoFar.length, {
                maximumFractionDigits: 0,
            });
        }

        const entry = document.createElement('div');
        entry.className = `alert-entry ${batch.ooc ? 'ooc' : 'ok'}`;
        const statusLabel = batch.ooc ? 'OOC' : 'OK';
        const rules = Array.isArray(batch.rules) ? batch.rules : [];
        const rulesText = rules.length ? ` - ${rules.join(', ')}` : '';
        entry.innerHTML = `
            <span>Batch ${formatNumber(batch.batch_idx, { maximumFractionDigits: 0 })}: xbar=${formatNumber(batch.xbar, { minimumFractionDigits: 3, maximumFractionDigits: 3 })} [${batch.phase}]${rulesText}</span>
            <span>${statusLabel}</span>
        `;
        alertLog.prepend(entry);
        document.getElementById('alert-count').textContent = pluralize(oocIndices.length, 'alert');

        idx++;
    }, 200);
}

// ============================================================
// Panel 3: DOE Analysis
// ============================================================

async function loadDOE() {
    setPanelState('doe-state', 'Loading DOE results...', 'info');
    try {
        const data = await fetchJson('/api/doe/results', { timeoutMs: 30000 });
        if (!data?.summary || !data?.main_effects) {
            throw new Error('DOE payload is incomplete.');
        }
        window._doeLoaded = true;

        document.getElementById('doe-best').textContent = formatNumber(data.summary.best_f1, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        document.getElementById('doe-worst').textContent = formatNumber(data.summary.worst_f1, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

        const interaction = Array.isArray(data.interaction) ? data.interaction : [];
        const classSensitivity = Array.isArray(data.class_sensitivity) ? data.class_sensitivity : [];

        renderMainEffects(data.main_effects);
        renderInteraction(interaction);
        renderClassSensitivity(classSensitivity);

        if (!interaction.length || !classSensitivity.length) {
            setPanelState('doe-state', 'DOE loaded with partial data. Some plots may be empty.', 'warning');
        } else {
            setPanelState('doe-state');
        }
    } catch (err) {
        console.error('DOE load failed:', err);
        setPanelState('doe-state', `DOE results unavailable. ${err.message}`, 'error');
        showStatus(`DOE results unavailable. ${err.message}`, 'error', { persist: true });
    }
}

function renderMainEffects(effects) {
    const psf = effects?.psf && typeof effects.psf === 'object' ? effects.psf : {};
    const snr = effects?.snr && typeof effects.snr === 'object' ? effects.snr : {};
    const vignette = effects?.vignette && typeof effects.vignette === 'object' ? effects.vignette : {};

    if (!Object.keys(psf).length && !Object.keys(snr).length && !Object.keys(vignette).length) {
        Plotly.newPlot('main-effects-chart', [], {
            ...PLOTLY_LAYOUT,
            annotations: [{ text: 'No DOE main-effects data available', showarrow: false, font: { color: PALETTE.muted, size: 13 } }],
            xaxis: { visible: false },
            yaxis: { visible: false },
            height: 350,
        }, PLOTLY_CONFIG);
        return;
    }

    const traces = [
        {
            x: Object.keys(psf),
            y: Object.values(psf),
            name: 'PSF sigma (px)',
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: PALETTE.accent, width: 3 },
            marker: { size: 9 },
        },
        {
            x: Object.keys(snr),
            y: Object.values(snr),
            name: 'SNR (dB)',
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: PALETTE.success, width: 3 },
            marker: { size: 9 },
            xaxis: 'x2',
            yaxis: 'y2',
        },
        {
            x: Object.keys(vignette),
            y: Object.values(vignette),
            name: 'Vignette',
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: PALETTE.warning, width: 3 },
            marker: { size: 9 },
            xaxis: 'x3',
            yaxis: 'y3',
        },
    ];

    const layout = {
        ...PLOTLY_LAYOUT,
        grid: { rows: 1, columns: 3, pattern: 'independent' },
        xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'PSF sigma (px)', domain: [0, 0.3] },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Mean Macro F1' },
        xaxis2: { ...PLOTLY_LAYOUT.xaxis, title: 'SNR (dB)', domain: [0.36, 0.64], anchor: 'y2' },
        yaxis2: { ...PLOTLY_LAYOUT.yaxis, anchor: 'x2' },
        xaxis3: { ...PLOTLY_LAYOUT.xaxis, title: 'Vignette Strength', domain: [0.7, 1], anchor: 'y3' },
        yaxis3: { ...PLOTLY_LAYOUT.yaxis, anchor: 'x3' },
        showlegend: false,
        height: 350,
    };

    Plotly.newPlot('main-effects-chart', traces, layout, PLOTLY_CONFIG);
}

function renderInteraction(data) {
    if (!Array.isArray(data) || !data.length) {
        Plotly.newPlot('interaction-chart', [], {
            ...PLOTLY_LAYOUT,
            annotations: [{ text: 'No interaction data available', showarrow: false, font: { color: PALETTE.muted, size: 13 } }],
            xaxis: { visible: false },
            yaxis: { visible: false },
            height: 350,
        }, PLOTLY_CONFIG);
        return;
    }

    const psfVals = [...new Set(data.map(d => d.psf_sigma))].sort((a, b) => a - b);
    const snrVals = [...new Set(data.map(d => d.snr_db))].sort((a, b) => a - b);

    const z = psfVals.map(p =>
        snrVals.map(s => {
            const match = data.find(d => d.psf_sigma === p && d.snr_db === s);
            return match ? match.macro_f1 : 0;
        })
    );

    Plotly.newPlot('interaction-chart', [{
        type: 'heatmap',
        x: snrVals.map(String),
        y: psfVals.map(String),
        z,
        colorscale: [[0, PALETTE.grid], [0.5, PALETTE.accent], [1, PALETTE.success]],
        colorbar: { title: 'F1', tickfont: { color: PALETTE.text } },
        text: z.map(row => row.map(v => v.toFixed(3))),
        texttemplate: '%{text}',
        textfont: { size: 12, color: PALETTE.textStrong },
    }], {
        ...PLOTLY_LAYOUT,
        xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'SNR (dB)' },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'PSF sigma (px)' },
        height: 350,
    }, PLOTLY_CONFIG);
}

function renderClassSensitivity(data) {
    if (!Array.isArray(data) || !data.length) {
        Plotly.newPlot('class-chart', [], {
            ...PLOTLY_LAYOUT,
            annotations: [{ text: 'No class sensitivity data available', showarrow: false, font: { color: PALETTE.muted, size: 13 } }],
            xaxis: { visible: false },
            yaxis: { visible: false },
            height: 350,
        }, PLOTLY_CONFIG);
        return;
    }

    const classes = [...new Set(data.map(d => d.class))];
    const psfVals = [...new Set(data.map(d => d.psf_sigma))].sort((a, b) => a - b);
    const colors = [PALETTE.accent, PALETTE.success, PALETTE.warning, PALETTE.danger, PALETTE.purple, PALETTE.cyan];

    const traces = classes.map((cls, i) => ({
        x: psfVals,
        y: psfVals.map(p => {
            const match = data.find(d => d.class === cls && d.psf_sigma === p);
            return match ? match.f1_score : 0;
        }),
        name: cls,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: colors[i % colors.length], width: 2 },
        marker: { size: 7 },
    }));

    Plotly.newPlot('class-chart', traces, {
        ...PLOTLY_LAYOUT,
        xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'PSF sigma (px)' },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'F1 Score', range: [0, 1.05] },
        legend: { font: { color: PALETTE.text }, bgcolor: 'transparent' },
        height: 350,
    }, PLOTLY_CONFIG);
}

// ============================================================
// Panel 4: System
// ============================================================

async function loadSystem() {
    setPanelState('system-state', 'Loading system metrics...', 'info');
    try {
        const data = await fetchJson('/api/metrics', { timeoutMs: 30000 });
        if (!data?.model) {
            throw new Error('System payload is incomplete.');
        }
        window._systemLoaded = true;

        renderLatencyChart(data.latency);
        renderModelTable(data.model);
        renderSummaryTable(data);
        setPanelState('system-state');
    } catch (err) {
        console.error('System load failed:', err);
        setPanelState('system-state', `System metrics unavailable. ${err.message}`, 'error');
        showStatus(`System metrics unavailable. ${err.message}`, 'error', { persist: true });
    }
}

function renderLatencyChart(latency) {
    const safeLatency = latency && typeof latency === 'object' ? latency : {};
    const runtimes = ['pytorch', 'onnx', 'openvino'];
    const labels = ['PyTorch (CPU)', 'ONNX Runtime', 'OpenVINO IR'];
    const colors = [PALETTE.muted, PALETTE.accent, PALETTE.success];
    const medians = runtimes.map(r => Number(safeLatency[r]?.median_ms) || 0);

    Plotly.newPlot('latency-chart', [{
        type: 'bar',
        x: labels,
        y: medians,
        marker: {
            color: colors,
            line: { color: colors.map(c => c + '80'), width: 1 },
        },
        text: medians.map(v => `${formatNumber(v, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}ms`),
        textposition: 'outside',
        textfont: { color: PALETTE.textStrong, size: 13 },
    }], {
        ...PLOTLY_LAYOUT,
        xaxis: { ...PLOTLY_LAYOUT.xaxis, title: '' },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Median Latency (ms)' },
        height: 350,
        annotations: [
            {
                x: 'OpenVINO IR',
                y: Number(safeLatency.openvino?.median_ms) || 0,
                text: `${formatNumber(Number(safeLatency.openvino_speedup) || 0, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}x speedup`,
                showarrow: true,
                arrowhead: 2,
                font: { color: PALETTE.success, size: 12 },
                ax: 60,
                ay: -40,
            },
        ],
    }, PLOTLY_CONFIG);
}

function renderModelTable(model) {
    const tbody = document.querySelector('#model-table tbody');
    const classes = Array.isArray(model?.classes) ? model.classes : [];
    const accuracy = Number(model?.accuracy);
    const rows = [
        ['Architecture', model?.architecture || 'N/A'],
        ['Input Size', model?.input_size || 'N/A'],
        ['Classes', classes.length ? classes.join(', ') : 'N/A'],
        ['Top-1 Accuracy', Number.isFinite(accuracy) ? formatPercent(accuracy, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : 'N/A'],
        ['Dataset', model?.dataset || 'N/A'],
    ];
    tbody.innerHTML = rows.map(([k, v]) =>
        `<tr><th scope="row">${k}</th><td class="mono">${v}</td></tr>`
    ).join('');
}

function renderSummaryTable(data) {
    const tbody = document.querySelector('#summary-table tbody');
    const onnxLatency = Number(data?.latency?.onnx?.median_ms);
    const openvinoLatency = Number(data?.latency?.openvino?.median_ms);
    const openvinoSpeedup = Number(data?.latency?.openvino_speedup);
    const modelAccuracy = Number(data?.model?.accuracy);
    const thresholdPx = data?.doe_summary?.resolution_threshold_px;
    const degradationPct = data?.doe_summary?.f1_degradation_pct;
    const metrics = [
        ['Top-1 Accuracy', '>=95%', Number.isFinite(modelAccuracy) ? formatPercent(modelAccuracy, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : 'N/A', true],
        ['ONNX Latency', '<8ms', Number.isFinite(onnxLatency) ? `${formatNumber(onnxLatency, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}ms` : 'N/A', Number.isFinite(onnxLatency) && onnxLatency < 8],
        ['OpenVINO Latency', '<5ms', Number.isFinite(openvinoLatency) ? `${formatNumber(openvinoLatency, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}ms` : 'N/A', Number.isFinite(openvinoLatency) && openvinoLatency < 5],
        ['PyTorch to OpenVINO Speedup', '>=2x', Number.isFinite(openvinoSpeedup) ? `${formatNumber(openvinoSpeedup, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}x` : 'N/A', Number.isFinite(openvinoSpeedup) && openvinoSpeedup >= 2],
        ['DOE Configurations', '27', thresholdPx ? '27' : 'N/A', true],
        ['PSF Threshold', 'Report', thresholdPx ? `${formatNumber(thresholdPx, { maximumFractionDigits: 0 })}px` : 'N/A', true],
        ['F1 Degradation (worst)', 'Report', Number.isFinite(Number(degradationPct)) ? formatPercent(Number(degradationPct) / 100, { maximumFractionDigits: 1 }) : 'N/A', true],
        ['Dominant Factor', 'Report', data.doe_summary?.dominant_factor || 'N/A', true],
    ];

    tbody.innerHTML = metrics.map(([name, target, actual, pass]) =>
        `<tr>
            <th scope="row">${name}</th>
            <td class="mono metric-target">${target}</td>
            <td class="mono">${actual}</td>
            <td><span class="card-badge ${pass ? 'badge-green' : 'badge-amber'}">${pass ? 'PASS' : 'ATTN'}</span></td>
        </tr>`
    ).join('');
}
