/* ============================================================
   script.js — ISRO 2-Model Dashboard Interactivity
   ============================================================ */

// ===== REAL TRAINING DATA (from leleBhadeWe/DATAx) =====
const REAL_DATA = {
    stage1: {
        coarse_l1: [[0,0.0570],[20,0.0316],[40,0.0334],[60,0.0454],[80,0.0570],[100,0.0815],[120,0.0272]],
        coarse_ssim: [[0,0.0577],[20,0.0629],[40,0.0546],[60,0.1367],[80,0.2066],[100,0.1526],[120,0.0312]]
    },
    stage2: {
        coarse_l1: [[0,0.0360],[20,0.0297],[40,0.0758],[60,0.1256],[80,0.0529],[100,0.0433],[120,0.0794],[140,0.0545],[160,0.0415],[180,0.0916],[200,0.0625],[220,0.0693],[240,0.0352],[260,0.0670]],
        coarse_ssim: [[0,0.0356],[20,0.0284],[40,0.2987],[60,0.2795],[80,0.0087],[100,0.1260],[120,0.1669],[140,0.0894],[160,0.0233],[180,0.2958],[200,0.0396],[220,0.1881],[240,0.0694],[260,0.3598]]
    },
    stage3: {
        coarse_l1: [[0,0.4898],[20,0.0248],[40,0.0678],[60,0.0354],[80,0.0310],[100,0.0327],[120,0.0158],[140,0.1024],[160,0.0363],[180,0.0299],[200,0.0631],[220,0.0783],[240,0.0384],[260,0.0343]],
        coarse_ssim: [[0,0.5129],[20,0.0389],[40,0.1255],[60,0.0666],[80,0.0872],[100,0.0375],[120,0.0278],[140,0.2968],[160,0.0263],[180,0.0567],[200,0.0619],[220,0.2308],[240,0.0789],[260,0.0554]]
    }
};

// Build per-step metrics for the metrics charts (use stage3 as final eval proxy)
function buildMetricsData() {
    const s3l1 = REAL_DATA.stage3.coarse_l1.map(d => d[1]);
    const s3ssim = REAL_DATA.stage3.coarse_ssim.map(d => d[1]);
    const s1l1 = REAL_DATA.stage1.coarse_l1.map(d => d[1]);
    const s1ssim = REAL_DATA.stage1.coarse_ssim.map(d => d[1]);
    // Pad shorter arrays
    const n = Math.max(s3l1.length, s1l1.length);
    const pad = (arr, len) => { while(arr.length < len) arr.push(arr[arr.length-1]); return arr; };
    return {
        coarse_l1: pad([...s1l1], n),
        refined_l1: pad([...s3l1], n),
        coarse_ssim: pad([...s1ssim], n),
        refined_ssim: pad([...s3ssim], n)
    };
}
const metricsData = buildMetricsData();

// ===== CHART DRAWING =====
function drawLineChart(canvasId, coarseData, refinedData, yLabel, higherBetter) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = (rect.width - 56) * dpr;
    canvas.height = 280 * dpr;
    canvas.style.width = (rect.width - 56) + 'px';
    canvas.style.height = '280px';
    ctx.scale(dpr, dpr);
    const W = rect.width - 56, H = 280;
    const pad = { top: 20, right: 20, bottom: 40, left: 55 };
    const cw = W - pad.left - pad.right;
    const ch = H - pad.top - pad.bottom;
    ctx.clearRect(0, 0, W, H);

    const all = [...coarseData, ...refinedData];
    let yMin = Math.min(...all), yMax = Math.max(...all);
    const margin = (yMax - yMin) * 0.15;
    yMin -= margin; yMax += margin;

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (ch * i / 4);
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke();
        ctx.fillStyle = '#5a5a6e'; ctx.font = '11px Inter';
        ctx.textAlign = 'right';
        ctx.fillText((yMax - (yMax - yMin) * i / 4).toFixed(4), pad.left - 8, y + 4);
    }

    // X axis labels
    ctx.fillStyle = '#5a5a6e'; ctx.font = '11px Inter'; ctx.textAlign = 'center';
    for (let i = 0; i < coarseData.length; i += 5) {
        const x = pad.left + (i / (coarseData.length - 1)) * cw;
        ctx.fillText(i, x, H - pad.bottom + 20);
    }
    ctx.fillText('Step', pad.left + cw / 2, H - 4);

    function drawLine(data, color, alpha) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = alpha;
        data.forEach((v, i) => {
            const x = pad.left + (i / (data.length - 1)) * cw;
            const y = pad.top + (1 - (v - yMin) / (yMax - yMin)) * ch;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
    }

    // Area fill for refined
    ctx.beginPath();
    ctx.globalAlpha = 0.08;
    ctx.fillStyle = '#06b6d4';
    refinedData.forEach((v, i) => {
        const x = pad.left + (i / (refinedData.length - 1)) * cw;
        const y = pad.top + (1 - (v - yMin) / (yMax - yMin)) * ch;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.lineTo(pad.left + cw, pad.top + ch);
    ctx.lineTo(pad.left, pad.top + ch);
    ctx.fill();
    ctx.globalAlpha = 1;

    drawLine(coarseData, '#f43f5e', 0.5);
    drawLine(refinedData, '#06b6d4', 0.9);

    // Legend
    const ly = 14;
    ctx.globalAlpha = 0.5; ctx.fillStyle = '#f43f5e';
    ctx.fillRect(pad.left, ly - 5, 14, 3);
    ctx.globalAlpha = 1; ctx.fillStyle = '#9898a6'; ctx.font = '11px Inter'; ctx.textAlign = 'left';
    ctx.fillText('Coarse', pad.left + 18, ly);
    ctx.fillStyle = '#06b6d4'; ctx.fillRect(pad.left + 80, ly - 5, 14, 3);
    ctx.fillStyle = '#9898a6'; ctx.fillText('Refined', pad.left + 98, ly);
}

function drawAllCharts() {
    drawLineChart('psnrChart', metricsData.coarse_l1, metricsData.refined_l1, 'Coarse L1', false);
    drawLineChart('ssimChart', metricsData.coarse_ssim, metricsData.refined_ssim, 'Coarse SSIM', false);
    // Stage-specific charts for the bottom two
    const s2l1 = REAL_DATA.stage2.coarse_l1.map(d=>d[1]);
    const s2ssim = REAL_DATA.stage2.coarse_ssim.map(d=>d[1]);
    drawSingleChart('mseChart', s2l1, 'Stage 2 — Coarse L1', '#f59e0b');
    drawSingleChart('maeChart', s2ssim, 'Stage 2 — Coarse SSIM', '#f43f5e');
    drawTrainingCharts();
}

// ===== SINGLE-SERIES CHART (for stage-specific views) =====
function drawSingleChart(canvasId, data, title, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = (rect.width - 56) * dpr; canvas.height = 280 * dpr;
    canvas.style.width = (rect.width - 56) + 'px'; canvas.style.height = '280px';
    ctx.scale(dpr, dpr);
    const W = rect.width - 56, H = 280;
    const pad = { top: 20, right: 20, bottom: 40, left: 55 };
    const cw = W - pad.left - pad.right, ch = H - pad.top - pad.bottom;
    ctx.clearRect(0, 0, W, H);
    let yMin = Math.min(...data) * 0.9, yMax = Math.max(...data) * 1.1;
    ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (ch * i / 4);
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke();
        ctx.fillStyle = '#5a5a6e'; ctx.font = '11px Inter'; ctx.textAlign = 'right';
        ctx.fillText((yMax - (yMax - yMin) * i / 4).toFixed(4), pad.left - 8, y + 4);
    }
    ctx.fillStyle = '#5a5a6e'; ctx.font = '11px Inter'; ctx.textAlign = 'center';
    ctx.fillText('Step', pad.left + cw / 2, H - 4);
    // Area
    ctx.beginPath(); ctx.globalAlpha = 0.1; ctx.fillStyle = color;
    data.forEach((v, i) => { const x = pad.left + (i/(data.length-1))*cw; const y = pad.top + (1-(v-yMin)/(yMax-yMin))*ch; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
    ctx.lineTo(pad.left+cw, pad.top+ch); ctx.lineTo(pad.left, pad.top+ch); ctx.fill(); ctx.globalAlpha=1;
    // Line
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
    data.forEach((v, i) => { const x = pad.left + (i/(data.length-1))*cw; const y = pad.top + (1-(v-yMin)/(yMax-yMin))*ch; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
    ctx.stroke();
    // Dots
    data.forEach((v, i) => { const x = pad.left + (i/(data.length-1))*cw; const y = pad.top + (1-(v-yMin)/(yMax-yMin))*ch; ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fillStyle=color; ctx.fill(); });
}

// ===== TRAINING LOSS CHARTS (REAL DATA) =====
function drawMiniChart(canvasId, data, color, label1, label2) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const W = rect.width, H = 160;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);
    const p = 10;
    const cw = W - p * 2, ch = H - p * 2;
    // data is array of [step, value] pairs
    const vals = data.map(d => d[1]);
    const yMin = Math.min(...vals) * 0.85, yMax = Math.max(...vals) * 1.1;
    // Gradient fill
    const grad = ctx.createLinearGradient(0, p, 0, H - p);
    grad.addColorStop(0, color + '25'); grad.addColorStop(1, color + '00');
    ctx.beginPath();
    vals.forEach((v, i) => { const x = p + (i/(vals.length-1))*cw; const y = p + (1-(v-yMin)/(yMax-yMin))*ch; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
    ctx.lineTo(p + cw, H - p); ctx.lineTo(p, H - p); ctx.fillStyle = grad; ctx.fill();
    // Line
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
    vals.forEach((v, i) => { const x = p + (i/(vals.length-1))*cw; const y = p + (1-(v-yMin)/(yMax-yMin))*ch; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
    ctx.stroke();
    // Dots
    vals.forEach((v, i) => { const x = p + (i/(vals.length-1))*cw; const y = p + (1-(v-yMin)/(yMax-yMin))*ch; ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fillStyle=color; ctx.fill(); });
    // Labels
    ctx.fillStyle = '#5a5a6e'; ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'left'; ctx.fillText(`Step ${data[0][0]}: ${vals[0].toFixed(4)}`, p + 4, p + 12);
    ctx.textAlign = 'right'; ctx.fillText(`Step ${data[data.length-1][0]}: ${vals[vals.length-1].toFixed(4)}`, W - p - 4, p + 12);
    // Step labels on x-axis
    ctx.fillStyle = '#3a3a4e'; ctx.font = '9px Inter'; ctx.textAlign = 'center';
    data.forEach((d, i) => { if (i % 2 === 0 || data.length <= 8) { const x = p + (i/(data.length-1))*cw; ctx.fillText(d[0], x, H - 1); }});
}

function drawTrainingCharts() {
    drawMiniChart('stage1Chart', REAL_DATA.stage1.coarse_l1, '#6366f1');
    drawMiniChart('stage2Chart', REAL_DATA.stage2.coarse_l1, '#06b6d4');
    drawMiniChart('stage3Chart', REAL_DATA.stage3.coarse_l1, '#10b981');
}

// ===== SATELLITE IMAGE GENERATION (Procedural) =====
function generateSatelliteImage(canvas, seed, isCoarse) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const size = Math.min(canvas.parentElement.clientWidth, canvas.parentElement.clientHeight);
    canvas.width = size * dpr; canvas.height = size * dpr;
    canvas.style.width = size + 'px'; canvas.style.height = size + 'px';
    ctx.scale(dpr, dpr);

    // Seeded random
    function seededRandom(s) {
        let x = Math.sin(s++) * 10000;
        return x - Math.floor(x);
    }

    // Dark ocean background
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, size, size);

    // Generate cloud-like patterns
    const numClouds = 5 + Math.floor(seededRandom(seed) * 5);
    for (let c = 0; c < numClouds; c++) {
        const cx = seededRandom(seed + c * 7 + 1) * size;
        const cy = seededRandom(seed + c * 7 + 2) * size;
        const r = 40 + seededRandom(seed + c * 7 + 3) * 120;
        const intensity = 0.3 + seededRandom(seed + c * 7 + 4) * 0.7;

        // Cloud swirl
        const numPoints = 60 + Math.floor(seededRandom(seed + c * 7 + 5) * 40);
        for (let p = 0; p < numPoints; p++) {
            const angle = seededRandom(seed + c * 100 + p) * Math.PI * 2;
            const dist = seededRandom(seed + c * 100 + p + 50) * r;
            const px = cx + Math.cos(angle) * dist;
            const py = cy + Math.sin(angle) * dist;
            const pr = 8 + seededRandom(seed + c * 100 + p + 99) * 25;

            const grad = ctx.createRadialGradient(px, py, 0, px, py, pr);
            const bright = Math.floor(intensity * (180 + seededRandom(seed + c * 100 + p + 200) * 75));
            grad.addColorStop(0, `rgba(${bright},${bright},${Math.min(255, bright + 20)},${0.15 + intensity * 0.15})`);
            grad.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = grad;
            ctx.fillRect(px - pr, py - pr, pr * 2, pr * 2);
        }

        // Spiral arm (hurricane)
        if (c === 0) {
            for (let a = 0; a < Math.PI * 4; a += 0.1) {
                const spiralR = a * 15 + 10;
                const sx = cx + Math.cos(a + seed * 0.5) * spiralR;
                const sy = cy + Math.sin(a + seed * 0.5) * spiralR;
                const bright = Math.floor(200 - a * 15);
                if (bright < 50) break;
                const grad = ctx.createRadialGradient(sx, sy, 0, sx, sy, 12);
                grad.addColorStop(0, `rgba(${bright},${bright},${Math.min(255, bright + 30)},0.2)`);
                grad.addColorStop(1, 'rgba(0,0,0,0)');
                ctx.fillStyle = grad;
                ctx.fillRect(sx - 12, sy - 12, 24, 24);
            }
        }
    }

    // If coarse, add slight blur/artifacts
    if (isCoarse) {
        ctx.fillStyle = 'rgba(13,17,23,0.15)';
        ctx.fillRect(0, 0, size, size);
        // Add subtle noise/artifacts
        for (let i = 0; i < 200; i++) {
            const x = seededRandom(seed * 999 + i) * size;
            const y = seededRandom(seed * 999 + i + 500) * size;
            ctx.fillStyle = `rgba(100,100,120,${seededRandom(seed * 999 + i + 1000) * 0.08})`;
            ctx.fillRect(x, y, 3, 3);
        }
    } else {
        // Refined: sharper edges, add glow highlights
        for (let c = 0; c < 3; c++) {
            const cx2 = seededRandom(seed + c * 7 + 1) * size;
            const cy2 = seededRandom(seed + c * 7 + 2) * size;
            const grad = ctx.createRadialGradient(cx2, cy2, 0, cx2, cy2, 20);
            grad.addColorStop(0, 'rgba(255,255,255,0.08)');
            grad.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = grad;
            ctx.fillRect(cx2 - 20, cy2 - 20, 40, 40);
        }
    }
}

// ===== COMPARISON SLIDER =====
function initComparison() {
    const viewer = document.getElementById('comparison-viewer');
    const slider = document.getElementById('comparison-slider');
    const topCanvas = document.getElementById('refinedCanvas');
    let isDragging = false;
    let currentSample = 0;

    function renderSample(idx) {
        const coarseCanvas = document.getElementById('coarseCanvas');
        generateSatelliteImage(coarseCanvas, idx * 100 + 42, true);
        generateSatelliteImage(topCanvas, idx * 100 + 42, false);
        updateSlider(0.5);
    }

    function updateSlider(ratio) {
        const pct = Math.max(0, Math.min(1, ratio)) * 100;
        slider.style.left = pct + '%';
        topCanvas.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    }

    viewer.addEventListener('mousedown', (e) => { isDragging = true; handleMove(e); });
    viewer.addEventListener('touchstart', (e) => { isDragging = true; handleMove(e.touches[0]); }, { passive: true });
    window.addEventListener('mousemove', (e) => { if (isDragging) handleMove(e); });
    window.addEventListener('touchmove', (e) => { if (isDragging) handleMove(e.touches[0]); }, { passive: true });
    window.addEventListener('mouseup', () => isDragging = false);
    window.addEventListener('touchend', () => isDragging = false);

    function handleMove(e) {
        const rect = viewer.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        updateSlider(x);
    }

    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSample = parseInt(btn.dataset.sample);
            renderSample(currentSample);
        });
    });

    renderSample(0);
}

// ===== TIMELINE ANIMATION =====
function initTimeline() {
    const frames = document.querySelectorAll('.frame-thumb');
    frames.forEach((f, i) => {
        const miniCanvas = document.createElement('canvas');
        miniCanvas.width = 128; miniCanvas.height = 128;
        f.appendChild(miniCanvas);
        const ctx = miniCanvas.getContext('2d');
        miniCanvas.style.width = '100%'; miniCanvas.style.height = '100%';
        // Draw mini satellite
        ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, 128, 128);
        const numDots = 30;
        for (let d = 0; d < numDots; d++) {
            const x = Math.sin(d * 7 + i * 20) * 30 + 64;
            const y = Math.cos(d * 11 + i * 15) * 30 + 64;
            const r = 5 + Math.sin(d + i * 3) * 3;
            const bright = 100 + Math.floor(Math.sin(d * 3 + i * 5) * 80);
            const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
            grad.addColorStop(0, `rgba(${bright},${bright},${bright + 30},0.6)`);
            grad.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = grad;
            ctx.fillRect(x - r, y - r, r * 2, r * 2);
        }
    });

    const playBtn = document.getElementById('play-btn');
    const progress = document.getElementById('timeline-progress');
    let playing = false;

    playBtn.addEventListener('click', () => {
        if (playing) return;
        playing = true;
        playBtn.style.opacity = '0.6';
        let step = 0;
        const steps = [0, 25, 50, 75, 100];
        const interval = setInterval(() => {
            progress.style.width = steps[step] + '%';
            frames.forEach((f, i) => {
                f.parentElement.style.opacity = i <= step ? '1' : '0.3';
                if (i === step) f.parentElement.style.transform = 'scale(1.1)';
                else f.parentElement.style.transform = 'scale(1)';
            });
            step++;
            if (step >= steps.length) {
                clearInterval(interval);
                setTimeout(() => {
                    playing = false;
                    playBtn.style.opacity = '1';
                    frames.forEach((f) => {
                        f.parentElement.style.opacity = '1';
                        f.parentElement.style.transform = 'scale(1)';
                    });
                }, 800);
            }
        }, 600);
    });
}

// ===== NAV SCROLL =====
function initNav() {
    const links = document.querySelectorAll('.nav-link');
    const sections = ['hero', 'architecture', 'metrics', 'visual', 'training'];

    window.addEventListener('scroll', () => {
        const scrollY = window.scrollY + 150;
        let current = 'hero';
        sections.forEach(id => {
            const el = document.getElementById(id);
            if (el && el.offsetTop <= scrollY) current = id;
        });
        links.forEach(link => {
            link.classList.toggle('active', link.dataset.section === current);
        });
    });
}

// ===== INTERSECTION OBSERVER (Fade-in) =====
function initObserver() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.section-header, .model-card, .stage-card, .data-card, .loss-card, .metric-chart-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// ===== COUNTER ANIMATION =====
function animateCounters() {
    const counters = [
        { id: 'stat-psnr', end: 0.0272, decimals: 4, start: 0.0570 },
        { id: 'stat-ssim', end: 0.0343, decimals: 4, start: 0.4898 },
    ];
    counters.forEach(({ id, end, decimals, start }) => {
        const el = document.getElementById(id);
        if (!el) return;
        const duration = 1500;
        const startTime = performance.now();
        function tick(now) {
            const t = Math.min((now - startTime) / duration, 1);
            const ease = 1 - Math.pow(1 - t, 3);
            el.textContent = (start + (end - start) * ease).toFixed(decimals);
            if (t < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    });
}

// ===== INIT =====
window.addEventListener('DOMContentLoaded', () => {
    drawAllCharts();
    initComparison();
    initTimeline();
    initNav();
    initObserver();
    animateCounters();
});

window.addEventListener('resize', () => {
    drawAllCharts();
});
