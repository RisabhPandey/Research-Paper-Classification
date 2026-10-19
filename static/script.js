/* ============================================================
   Research Paper Classification · SimpleGCN
   ============================================================ */

const API = location.protocol === 'file:' ? 'http://127.0.0.1:8000' : '';
const MAX_NODE = 2707;
const CLASSES = [
    "Case-Based",
    "Genetic Algorithms",
    "Neural Networks",
    "Probabilistic Methods",
    "Reinforcement Learning",
    "Rule Learning",
    "Theory"
];
const COLORS = ["#fb923c", "#14b8a6", "#6366f1", "#a855f7", "#ec4899", "#60a5fa", "#334155"];
const TINTS  = ["#fff7ed", "#f0fdfa", "#eef2ff", "#faf5ff", "#fdf2f8", "#eff6ff", "#f1f5f9"];
const LOW_CONF = 50;

const PICTO = [
    '<rect x="3" y="8" width="13" height="12" rx="2"/><path d="M8 8V6a2 2 0 012-2h9a2 2 0 012 2v9a2 2 0 01-2 2h-3"/>',
    '<path d="M7 3c0 6 10 6 10 9s-10 3-10 9"/><path d="M17 3c0 6-10 6-10 9s10 3 10 9"/><path d="M9.5 6h5M8 12h8M9.5 18h5"/>',
    '<circle cx="5" cy="6" r="2"/><circle cx="5" cy="18" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="19" cy="7" r="2"/><circle cx="19" cy="17" r="2"/><path d="M7 7l3.2 4M7 17l3.2-4M14 11l3.2-3M14 13l3.2 3"/>',
    '<path d="M2 20h20"/><path d="M3 20c3 0 4-15 9-15s6 15 9 15"/><path d="M12 5v15" stroke-dasharray="2 2"/>',
    '<path d="M21 12a9 9 0 11-3-6.7"/><path d="M21 4v5h-5"/>',
    '<path d="M4 6l2 2 3-3M4 14l2 2 3-3"/><path d="M12 7h8M12 15h8M4 20h16"/>',
    '<path d="M18 4H6l7 8-7 8h12"/>'
];

const topicArt = (i, size) =>
    `<span class="inline-flex items-center justify-center rounded-xl shrink-0 shadow-sm" style="width:${size}px;height:${size}px;background:${COLORS[i]}" title="${CLASSES[i]}">` +
    `<svg viewBox="0 0 24 24" width="${Math.round(size * 0.55)}" height="${Math.round(size * 0.55)}" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${PICTO[i]}</svg></span>`;

const EMPTY_ART = `
    <svg class="mx-auto mb-4" width="140" height="100" viewBox="0 0 140 100" fill="none" aria-hidden="true">
        <ellipse cx="70" cy="92" rx="46" ry="5" fill="#e2e8f0"/>
        <rect x="32" y="16" width="54" height="68" rx="8" fill="#e0e7ff" transform="rotate(-7 59 50)"/>
        <rect x="42" y="12" width="54" height="68" rx="8" fill="#fff" stroke="#cbd5e1" stroke-width="2"/>
        <rect x="50" y="24" width="32" height="5" rx="2.5" fill="#93c5fd"/>
        <rect x="50" y="35" width="38" height="4" rx="2" fill="#e2e8f0"/>
        <rect x="50" y="44" width="38" height="4" rx="2" fill="#e2e8f0"/>
        <rect x="50" y="53" width="24" height="4" rx="2" fill="#e2e8f0"/>
        <circle cx="100" cy="64" r="16" fill="#fff" stroke="#2563eb" stroke-width="3.5"/>
        <path d="M112 76l13 13" stroke="#2563eb" stroke-width="4.5" stroke-linecap="round"/>
        <circle cx="93" cy="59" r="3" fill="#a855f7"/>
        <circle cx="105" cy="68" r="3" fill="#14b8a6"/>
        <path d="M93 59l12 9" stroke="#94a3b8" stroke-width="1.5"/>
    </svg>`;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

let history = [];
let filter = null;
const expanded = new Set();

try {
    history = JSON.parse(localStorage.getItem('cora_history') || '[]');
} catch {
    history = [];
}

const save = () => {
    try {
        localStorage.setItem('cora_history', JSON.stringify(history.slice(0, 100)));
    } catch { /* ignore */ }
};

/* ---------- Landing (always shown by default) ---------- */
function startApp() {
    const landing = $('landing-page');
    landing.classList.add('leaving');
    setTimeout(() => {
        landing.style.display = 'none';
        landing.classList.remove('leaving');
    }, 450);
}

function showLanding() {
    const landing = $('landing-page');
    landing.style.display = 'flex';
    // force reflow then fade in
    void landing.offsetWidth;
    landing.classList.remove('leaving');
    landing.style.opacity = '1';
    landing.style.visibility = 'visible';
}

/* ---------- Math ---------- */
function softmax(logits) {
    const m = Math.max(...logits);
    const e = logits.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
}

const conf = (r) => r.probs[r.classId] * 100;

/* ---------- Rendering ---------- */
function renderLegend() {
    const counts = CLASSES.map((_, i) => history.filter((h) => h.classId === i).length);
    $('legend').innerHTML = CLASSES.map((name, i) => `
        <button onclick="toggleFilter(${i})"
                class="chip ${filter === i ? 'active' : ''} rounded-xl p-3 border flex items-center gap-3 text-left"
                style="background:${TINTS[i]};border-color:${COLORS[i]}40">
            ${topicArt(i, 28)}
            <span class="text-sm font-medium text-slate-700 flex-1 truncate">${name}</span>
            ${counts[i]
                ? `<span class="text-[11px] font-bold text-white rounded-full px-2 py-0.5 min-w-[1.4rem] text-center" style="background:${COLORS[i]}">${counts[i]}</span>`
                : ''}
        </button>`).join('');
}

function renderStats() {
    $('stat-total').textContent = history.length;
    if (!history.length) {
        $('stat-conf').textContent = '–';
        $('stat-top').textContent = '–';
        return;
    }
    const avg = history.reduce((a, r) => a + conf(r), 0) / history.length;
    $('stat-conf').textContent = avg.toFixed(1) + '%';
    const counts = CLASSES.map((_, i) => history.filter((h) => h.classId === i).length);
    $('stat-top').textContent = CLASSES[counts.indexOf(Math.max(...counts))];
}

function renderResults(fresh = []) {
    const list = $('results-list');
    const shown = history.filter((r) => filter === null || r.classId === filter);
    $('toolbar').classList.toggle('hide', history.length === 0);

    if (!shown.length) {
        list.innerHTML = `
            <div class="empty-state text-center p-10 text-slate-400 text-sm">
                ${EMPTY_ART}
                ${history.length
                    ? 'No results for this topic yet.'
                    : 'No predictions yet. Run a prediction from the right panel.'}
            </div>`;
        return;
    }

    list.innerHTML = shown.map((r) => {
        const c = conf(r);
        const color = COLORS[r.classId];
        const open = expanded.has(r.node);
        const detail = r.probs.map((p, i) => `
            <div class="flex items-center gap-3 text-xs">
                <span class="w-36 shrink-0 truncate ${i === r.classId ? 'font-semibold text-slate-800' : 'text-slate-500'}">${CLASSES[i]}</span>
                <div class="flex-1 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                    <div class="h-full rounded-full" style="width:${(p * 100).toFixed(1)}%;background:${COLORS[i]}"></div>
                </div>
                <span class="w-12 text-right text-slate-500 tabular-nums">${(p * 100).toFixed(1)}%</span>
            </div>`).join('');

        return `
        <div class="${fresh.includes(r.node) ? 'pop' : ''} border-b border-slate-50 last:border-0 hover:bg-slate-50/80 transition" data-node="${r.node}">
            <div class="row flex items-center justify-between gap-3 p-4 cursor-pointer">
                <div class="flex items-center gap-3.5 sm:w-1/3 min-w-0">
                    ${topicArt(r.classId, 38)}
                    <p class="font-semibold text-sm text-slate-900 truncate">Paper #${r.node}</p>
                </div>
                <div class="hidden sm:flex w-1/3 text-sm text-slate-600 font-medium items-center gap-2 min-w-0">
                    <span class="truncate">${esc(r.className)}</span>
                    ${c < LOW_CONF
                        ? '<span class="text-[10px] font-semibold text-amber-600 bg-amber-50 border border-amber-100 rounded-full px-2 py-0.5 shrink-0">Low conf.</span>'
                        : ''}
                </div>
                <div class="flex items-center justify-end flex-1 sm:w-1/3 sm:flex-none">
                    <div class="w-16 md:w-20 bg-slate-100 rounded-full h-1.5 mr-3 overflow-hidden">
                        <div class="bar h-1.5 rounded-full" style="width:0;background:${color}" data-w="${c.toFixed(1)}"></div>
                    </div>
                    <span class="text-xs text-slate-500 font-semibold w-11 text-right tabular-nums">${c.toFixed(1)}%</span>
                    <i class="fas fa-chevron-down text-[10px] text-slate-400 ml-3 transition ${open ? 'rotate-180' : ''}"></i>
                    <button class="remove text-slate-300 hover:text-red-500 ml-2.5 p-1 transition" title="Remove">
                        <i class="fas fa-xmark text-sm"></i>
                    </button>
                </div>
            </div>
            <div class="details ${open ? '' : 'hide'} px-4 pb-4 pt-0.5 space-y-2 sm:pl-[4.25rem]">
                <p class="sm:hidden text-sm font-medium text-slate-600 mb-1.5">${esc(r.className)}</p>
                ${detail}
            </div>
        </div>`;
    }).join('');

    requestAnimationFrame(() =>
        list.querySelectorAll('.bar').forEach((b) => (b.style.width = b.dataset.w + '%'))
    );
}

function renderAll(fresh = []) {
    renderLegend();
    renderStats();
    renderResults(fresh);
}

/* ---------- Interactions ---------- */
$('results-list').addEventListener('click', (e) => {
    const card = e.target.closest('[data-node]');
    if (!card) return;
    const node = Number(card.dataset.node);

    if (e.target.closest('.remove')) {
        history = history.filter((h) => h.node !== node);
        expanded.delete(node);
        save();
        renderAll();
    } else if (e.target.closest('.row')) {
        if (expanded.has(node)) expanded.delete(node);
        else expanded.add(node);
        renderResults();
    }
});

function toggleFilter(i) {
    filter = filter === i ? null : i;
    renderLegend();
    renderResults();
}

function clearAll() {
    history = [];
    expanded.clear();
    filter = null;
    save();
    renderAll();
}

function exportCSV() {
    const rows = ['node_id,class_id,class_name,confidence_percent']
        .concat(history.map((r) => `${r.node},${r.classId},"${r.className}",${conf(r).toFixed(2)}`));
    const url = URL.createObjectURL(new Blob([rows.join('\n')], { type: 'text/csv' }));
    const a = Object.assign(document.createElement('a'), {
        href: url,
        download: 'cora_predictions.csv'
    });
    a.click();
    URL.revokeObjectURL(url);
}

function showError(msg) {
    const el = $('error-msg');
    el.textContent = msg || '';
    el.classList.toggle('hide', !msg);
}

function randomPapers() {
    const ids = new Set();
    while (ids.size < 5) ids.add(Math.floor(Math.random() * (MAX_NODE + 1)));
    $('node-input').value = [...ids].join(', ');
    makePrediction();
}

async function makePrediction() {
    const inputEl = $('node-input');
    const btn = $('predict-btn');
    const ids = [...new Set((inputEl.value.match(/\d+/g) || []).map(Number))];

    if (!ids.length) return showError('Please enter at least one Node ID.');
    if (ids.some((n) => n > MAX_NODE)) return showError(`Node IDs must be between 0 and ${MAX_NODE}.`);
    if (ids.length > 100) return showError('Please classify at most 100 papers at a time.');
    showError('');

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-1.5"></i> Predicting…';
    btn.classList.add('opacity-70', 'cursor-not-allowed');

    try {
        const response = await fetch(`${API}/predict/cora_node`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ node_indices: ids })
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            const d = data.detail;
            throw new Error(
                typeof d === 'string'
                    ? d
                    : Array.isArray(d)
                        ? d.map((x) => x.msg).join('; ')
                        : `Prediction failed (${response.status})`
            );
        }

        const fresh = data.prediction.map((p) => ({
            node: p.node_index,
            classId: p.predicted_class_id,
            className: p.predicted_class_name,
            probs: softmax(p.logits)
        }));
        const freshIds = fresh.map((r) => r.node);
        history = fresh.reverse().concat(history.filter((h) => !freshIds.includes(h.node)));
        save();
        filter = null;
        renderAll(freshIds);
        inputEl.value = '';
    } catch (err) {
        showError(
            err.message.includes('Failed to fetch')
                ? 'Cannot reach the API. Is the FastAPI server running?'
                : err.message
        );
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Predict';
        btn.classList.remove('opacity-70', 'cursor-not-allowed');
    }
}

$('node-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') makePrediction();
});

/* ---------- Health / model info ---------- */
function setHealth(ok, text) {
    const b = $('health-badge');
    b.className =
        'text-xs font-medium px-3 py-1.5 rounded-full border flex items-center gap-1.5 ' +
        (ok
            ? 'text-emerald-700 bg-emerald-50 border-emerald-100'
            : 'text-red-600 bg-red-50 border-red-100');
    b.innerHTML = `<i class="fas ${ok ? 'fa-check-circle' : 'fa-circle-exclamation'} text-[10px]"></i><span>${text}</span>`;
}

async function loadStatus() {
    try {
        const h = await (await fetch(`${API}/health`)).json();
        setHealth(true, 'System Healthy');
        if (h.providers && h.providers[0]) {
            $('m-run').textContent = 'ONNX · ' + h.providers[0].replace('ExecutionProvider', '');
        }
    } catch {
        setHealth(false, 'API Offline');
        return;
    }
    try {
        const m = await (await fetch(`${API}/model-info`)).json();
        $('m-name').textContent = m['Model Name'] || 'SimpleGCN';
        $('m-feat').textContent = m.feature_dimension;
        $('m-cls').textContent = m.num_classes;
    } catch { /* keep defaults */ }
}

/* ---------- Init ---------- */
// Always start on Home (landing). No localStorage skip.
renderAll();
loadStatus();
setInterval(loadStatus, 30000);