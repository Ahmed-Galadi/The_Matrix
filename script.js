// Neural Network Architecture
const INPUT = 784;
const HIDDEN = 128;
const OUTPUT = 10;

// Model State
let W1, B1, W2, B2;
let modelLoaded = false;

// DOM Elements
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const statusEl = document.getElementById('status');
const predictionEl = document.getElementById('prediction');
const clearBtn = document.getElementById('clearBtn');
const probContainer = document.getElementById('probContainer');

// Initialize UI
function initUI() {
    for (let i = 0; i < 10; i++) {
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <span class="prob-label">${i}</span>
            <div class="prob-bg"><div class="prob-fill" id="fill-${i}"></div></div>
            <span class="prob-text" id="val-${i}">0.0%</span>
        `;
        probContainer.appendChild(row);
    }
    initCanvas();
}

function initCanvas() {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 30;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}
initUI();

// Drawing Logic
let isDrawing = false;
let lastX = 0, lastY = 0;

function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return {
        x: (cx - rect.left) * (canvas.width / rect.width),
        y: (cy - rect.top) * (canvas.height / rect.height)
    };
}

canvas.addEventListener('mousedown', (e) => { isDrawing = true; const p = getPos(e); lastX = p.x; lastY = p.y; canvas.parentElement.classList.add('active'); });
canvas.addEventListener('mousemove', (e) => { if (!isDrawing) return; const p = getPos(e); ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(p.x, p.y); ctx.stroke(); lastX = p.x; lastY = p.y; });
canvas.addEventListener('mouseup', () => { if (isDrawing) { isDrawing = false; canvas.parentElement.classList.remove('active'); predict(); } });
canvas.addEventListener('mouseleave', () => { if (isDrawing) { isDrawing = false; canvas.parentElement.classList.remove('active'); predict(); } });

canvas.addEventListener('touchstart', (e) => { e.preventDefault(); isDrawing = true; const p = getPos(e); lastX = p.x; lastY = p.y; }, { passive: false });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); if (!isDrawing) return; const p = getPos(e); ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(p.x, p.y); ctx.stroke(); lastX = p.x; lastY = p.y; }, { passive: false });
canvas.addEventListener('touchend', () => { if (isDrawing) { isDrawing = false; predict(); } });

clearBtn.addEventListener('click', () => {
    initCanvas();
    predictionEl.textContent = 'Predicted: -';
    for(let i=0; i<10; i++) {
        document.getElementById(`fill-${i}`).style.width = '0%';
        document.getElementById(`val-${i}`).textContent = '0.0%';
        document.querySelector(`#fill-${i}`).parentElement.parentElement.classList.remove('active');
    }
});

// ================= AUTOMATIC MODEL LOADING =================
async function loadModel() {
    try {
        statusEl.textContent = 'Loading weights...';
        const [fW1, fB1, fW2, fB2] = await Promise.all([
            fetch('Datasets/hidden_weights.txt'),
            fetch('Datasets/hidden_biases.txt'),
            fetch('Datasets/output_weights.txt'),
            fetch('Datasets/output_biases.txt')
        ]);
        if (!fW1.ok || !fB1.ok || !fW2.ok || !fB2.ok) throw new Error('Missing model files');

        W1 = await loadMatrix(await fW1.text(), HIDDEN, INPUT);
        B1 = await loadVector(await fB1.text(), HIDDEN);
        W2 = await loadMatrix(await fW2.text(), OUTPUT, HIDDEN);
        B2 = await loadVector(await fB2.text(), OUTPUT);

        modelLoaded = true;
        clearBtn.disabled = false;
        statusEl.textContent = 'Model Loaded. Draw a digit!';
        statusEl.className = 'status-label success';
    } catch (err) {
        console.error(err);
        statusEl.textContent = 'Error: Could not load model. Run a local server.';
        statusEl.className = 'status-label error';
    }
}

async function loadMatrix(text, expRows, expCols) {
    const tokens = text.trim().split(/\s+/);
    let idx = 0;
    const rows = parseInt(tokens[idx++]);
    const cols = parseInt(tokens[idx++]);
    let matrix = [];
    for (let r = 0; r < rows; r++) {
        let row = [];
        for (let c = 0; c < cols; c++) row.push(parseFloat(tokens[idx++]));
        matrix.push(row);
    }
    if (rows !== expRows || cols !== expCols) {
        const transposed = new Array(cols).fill(null).map(() => new Array(rows));
        for (let r = 0; r < rows; r++)
            for (let c = 0; c < cols; c++)
                transposed[c][r] = matrix[r][c];
        return transposed;
    }
    return matrix;
}

async function loadVector(text, size) {
    const tokens = text.trim().split(/\s+/);
    let idx = 2; 
    const vec = [];
    for (let i = 0; i < size; i++) vec.push(parseFloat(tokens[idx++]));
    return vec;
}

// ================= PREDICTION ENGINE =================
function preprocessCanvas() {
    const imageData = ctx.getImageData(0, 0, 280, 280);
    const data = imageData.data;
    let minX = 280, minY = 280, maxX = 0, maxY = 0, hasInk = false;
    for (let y = 0; y < 280; y++) {
        for (let x = 0; x < 280; x++) {
            if (data[(y * 280 + x) * 4] < 200) {
                hasInk = true;
                if (x < minX) minX = x; if (y < minY) minY = y;
                if (x > maxX) maxX = x; if (y > maxY) maxY = y;
            }
        }
    }
    if (!hasInk) return null;

    let width = maxX - minX + 1, height = maxY - minY + 1;
    minX = Math.max(0, minX - Math.floor(width * 0.2));
    minY = Math.max(0, minY - Math.floor(height * 0.2));
    maxX = Math.min(279, maxX + Math.floor(width * 0.2));
    maxY = Math.min(279, maxY + Math.floor(height * 0.2));

    let cropSize = Math.max(maxX - minX + 1, maxY - minY + 1);
    let cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    let half = Math.floor(cropSize / 2);
    let nx = Math.max(0, cx - half), ny = Math.max(0, cy - half);
    if (nx + cropSize > 280) nx = 280 - cropSize;
    if (ny + cropSize > 280) ny = 280 - cropSize;

    const temp = document.createElement('canvas'); temp.width = temp.height = 20;
    temp.getContext('2d').drawImage(canvas, nx, ny, cropSize, cropSize, 0, 0, 20, 20);

    const final = document.createElement('canvas'); final.width = final.height = 28;
    const fCtx = final.getContext('2d');
    fCtx.fillStyle = '#ffffff'; fCtx.fillRect(0, 0, 28, 28);
    fCtx.drawImage(temp, 4, 4);

    const fData = fCtx.getImageData(0, 0, 28, 28).data;
    const input = new Array(784);
    for (let i = 0; i < 784; i++) input[i] = 1.0 - (fData[i * 4] / 255.0);
    return input;
}

function predict() {
    if (!modelLoaded) return;
    const input = preprocessCanvas();
    if (!input) {
        predictionEl.textContent = 'Predicted: ?';
        return;
    }

    // Hidden Layer (ReLU)
    const hidden = new Array(HIDDEN);
    for (let h = 0; h < HIDDEN; h++) {
        let sum = B1[h];
        for (let i = 0; i < INPUT; i++) sum += input[i] * W1[h][i];
        hidden[h] = Math.max(0, sum);
    }

    // Output Layer (Softmax)
    const logits = new Array(OUTPUT);
    let maxL = -Infinity;
    for (let o = 0; o < OUTPUT; o++) {
        let sum = B2[o];
        for (let h = 0; h < HIDDEN; h++) sum += hidden[h] * W2[o][h];
        logits[o] = sum;
        if (sum > maxL) maxL = sum;
    }

    // Stable Softmax
    let sumExp = 0;
    const probs = new Array(OUTPUT);
    for (let o = 0; o < OUTPUT; o++) {
        logits[o] = Math.exp(logits[o] - maxL);
        sumExp += logits[o];
    }
    for (let o = 0; o < OUTPUT; o++) {
        probs[o] = (logits[o] / sumExp) * 100;
    }

    // Find best prediction
    let pred = 0, maxProb = 0;
    for (let o = 0; o < OUTPUT; o++) {
        if (probs[o] > maxProb) { maxProb = probs[o]; pred = o; }
    }

    // Update UI
    predictionEl.textContent = `Predicted: ${pred} (${maxProb.toFixed(1)}%)`;
    for (let o = 0; o < OUTPUT; o++) {
        const fill = document.getElementById(`fill-${o}`);
        const val = document.getElementById(`val-${o}`);
        const row = fill.parentElement.parentElement;
        
        fill.style.width = `${probs[o]}%`;
        val.textContent = `${probs[o].toFixed(1)}%`;
        
        row.classList.toggle('active', o === pred);
    }
}

// Start loading
loadModel();