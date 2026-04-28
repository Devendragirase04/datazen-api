const API = '';  // use relative paths to call the local Flask server
let sessionId = null;
let summaryData = null;
let currentFilename = 'dataset';
let reportUrl = null;

// ── DRAG & DROP ──
const zone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');

zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
zone.addEventListener('drop', e => {
  e.preventDefault();
  zone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
zone.addEventListener('click', (e) => {
  if (e.target.tagName !== 'BUTTON') fileInput.click();
});
fileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

function handleFile(file) {
  if (!file.name.endsWith('.csv')) { showToast('Only CSV files are supported', 'error'); return; }
  currentFilename = file.name.replace('.csv', '');
  zone.classList.add('hidden');
  document.getElementById('uploadProgress').classList.remove('hidden');
  uploadFile(file);
}

async function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  setRing(20);
  try {
    const res = await fetch(`${API}/upload`, { method: 'POST', body: form });
    setRing(80);
    const data = await res.json();
    if (data.error) { showToast(data.error, 'error'); resetUploadUI(); return; }
    setRing(100);
    sessionId = data.session_id;
    summaryData = data.summary;
    setTimeout(() => { goToPanel(2); buildQualityPanel(); }, 400);
  } catch (e) {
    showToast('Upload failed: ' + e.message, 'error');
    resetUploadUI();
  }
}

function setRing(pct) {
  const circ = 213.6;
  const offset = circ - (pct / 100) * circ;
  document.getElementById('ringFill').style.strokeDashoffset = offset;
  document.getElementById('ringLabel').textContent = pct + '%';
}

function resetUploadUI() {
  document.getElementById('uploadProgress').classList.add('hidden');
  document.getElementById('uploadZone').classList.remove('hidden');
  document.getElementById('ringFill').style.strokeDashoffset = 213.6;
  document.getElementById('ringLabel').textContent = '0%';
}

// ── PANEL NAVIGATION ──
function goToPanel(n) {
  document.querySelectorAll('.panel').forEach(p => { p.classList.add('hidden'); p.classList.remove('active'); });
  const panel = document.getElementById('panel' + n);
  panel.classList.remove('hidden');
  panel.classList.add('active');
  document.querySelectorAll('.step').forEach((s, i) => {
    s.classList.remove('active', 'done');
    if (i + 1 < n) s.classList.add('done');
    if (i + 1 === n) s.classList.add('active');
  });
}

function goToDashboard() {
  goToPanel(3);
  loadDashboard();
}

function goToReport() {
  goToPanel(4);
}

// ── QUALITY PANEL ──
function buildQualityPanel() {
  if (!summaryData) return;
  const s = summaryData;

  document.getElementById('datasetInfo').textContent =
    `${currentFilename}.csv | ${s.shape[0].toLocaleString()} rows x ${s.shape[1]} columns | ${s.memory_usage}`;

  // KPI
  const totalNulls = s.total_nulls;
  const completeness = ((1 - totalNulls / (s.shape[0] * s.shape[1])) * 100).toFixed(1);
  const kpiRow = document.getElementById('kpiRow');
  kpiRow.innerHTML = `
    <div class="kpi-card"><div class="kpi-label">Rows</div><div class="kpi-value accent">${s.shape[0].toLocaleString()}</div></div>
    <div class="kpi-card"><div class="kpi-label">Columns</div><div class="kpi-value">${s.shape[1]}</div></div>
    <div class="kpi-card"><div class="kpi-label">Numerical</div><div class="kpi-value accent">${s.numerical_cols.length}</div></div>
    <div class="kpi-card"><div class="kpi-label">Categorical</div><div class="kpi-value">${s.categorical_cols.length}</div></div>
    <div class="kpi-card"><div class="kpi-label">Missing Values</div><div class="kpi-value ${totalNulls > 0 ? 'danger' : 'success'}">${totalNulls.toLocaleString()}</div></div>
    <div class="kpi-card"><div class="kpi-label">Completeness</div><div class="kpi-value ${parseFloat(completeness) >= 95 ? 'success' : 'danger'}">${completeness}%</div></div>
  `;

  // Column Table
  const tbody = document.getElementById('colTableBody');
  tbody.innerHTML = '';
  s.null_info.forEach(col => {
    const badgeClass = col.type === 'numerical' ? 'badge-num' : 'badge-cat';
    const nullBadge = col.null_count > 0
      ? `<span class="badge badge-warn">${col.null_count}</span>`
      : `<span class="badge badge-ok">0</span>`;
    const opts = col.fill_options.map(o => `<option value="${o}" ${o === col.recommended ? 'selected' : ''}>${o}</option>`).join('');
    const pct = parseFloat(col.null_pct);
    const barColor = pct === 0 ? 'var(--accent2)' : pct > 30 ? 'var(--danger)' : 'var(--warn)';
    tbody.innerHTML += `
      <tr>
        <td><code style="font-family:var(--font-mono);font-size:12px;color:var(--accent)">${col.column}</code></td>
        <td><span class="badge ${badgeClass}">${col.type}</span></td>
        <td style="color:var(--muted)">${col.unique.toLocaleString()}</td>
        <td>${nullBadge}</td>
        <td>
          <div class="null-bar-wrap">
            <div class="null-bar-bg"><div class="null-bar-fill" style="width:${col.null_pct}%;background:${barColor}"></div></div>
            <span style="font-family:var(--font-mono);font-size:10px;color:var(--muted);min-width:38px">${col.null_pct}%</span>
          </div>
        </td>
        <td>
          ${col.null_count > 0
        ? `<select class="strategy-select" id="strategy_${col.column}">${opts}<option value="none">skip</option></select>`
        : `<span style="color:var(--muted);font-size:11px">—</span>`}
        </td>
      </tr>`;
  });

  // Sample table
  if (s.sample && s.sample.length > 0) {
    const cols = Object.keys(s.sample[0]);
    const showCols = cols.slice(0, 6);
    let tbl = `<table class="data-table"><thead><tr>${showCols.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>`;
    s.sample.forEach(row => {
      tbl += `<tr>${showCols.map(c => `<td>${row[c] === '' ? '<span style="color:var(--danger)">NaN</span>' : String(row[c]).substring(0, 18)}</td>`).join('')}</tr>`;
    });
    tbl += '</tbody></table>';
    document.getElementById('sampleTableWrap').innerHTML = tbl;
  }
}

function setAllStrategy(method) {
  if (!summaryData) return;
  summaryData.null_info.forEach(col => {
    if (col.null_count === 0) return;
    const sel = document.getElementById('strategy_' + col.column);
    if (!sel) return;
    if (method === 'mode') {
      sel.value = 'mode';
    } else if ((method === 'mean' || method === 'median') && col.type === 'numerical') {
      sel.value = method;
    } else if (method === 'drop') {
      sel.value = 'drop';
    }
  });
  showToast(`Strategy set: ${method}`, 'success');
}

async function applyFillNulls() {
  if (!sessionId || !summaryData) return;
  const strategy = {};
  summaryData.null_info.forEach(col => {
    if (col.null_count === 0) return;
    const sel = document.getElementById('strategy_' + col.column);
    if (sel && sel.value !== 'none') strategy[col.column] = sel.value;
  });
  if (Object.keys(strategy).length === 0) { showToast('No strategies selected', 'error'); return; }

  document.getElementById('applyFillBtn').textContent = '⏳ Applying...';
  document.getElementById('applyFillBtn').disabled = true;

  try {
    const res = await fetch(`${API}/fill_nulls`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, strategy })
    });
    const data = await res.json();
    summaryData = data.summary;
    buildQualityPanel();
    showToast(`Imputation applied to ${data.applied.length} columns`, 'success');
  } catch (e) {
    showToast('Error: ' + e.message, 'error');
  } finally {
    document.getElementById('applyFillBtn').textContent = '✦ Apply Imputation';
    document.getElementById('applyFillBtn').disabled = false;
  }
}

// ── DASHBOARD ──
async function loadDashboard() {
  if (!sessionId) return;
  const grid = document.getElementById('chartGrid');
  const loading = document.getElementById('loadingCharts');
  grid.innerHTML = '';
  loading.classList.remove('hidden');

  try {
    const res = await fetch(`${API}/dashboard`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    });
    const data = await res.json();
    loading.classList.add('hidden');

    if (!data.charts || data.charts.length === 0) {
      grid.innerHTML = '<p style="color:var(--muted);padding:40px">No charts could be generated for this dataset.</p>';
      return;
    }

    if (data.charts && data.charts.length > 0) {
      data.charts.forEach((chart, i) => {
        if (!chart || !chart.json) return;
        
        const card = document.createElement('div');
        card.className = 'chart-card';
        card.innerHTML = `
          <div class="chart-card-header">
            <span>${chart.title || 'Visualization'}</span>
            <span style="font-size:10px;color:var(--bg3)">Plotly</span>
          </div>
          <div class="chart-card-body" id="chart_${i}"></div>`;
        grid.appendChild(card);
        
        try {
          const chartData = JSON.parse(chart.json);
          Plotly.newPlot(`chart_${i}`, chartData.data || [], chartData.layout || {}, {responsive: true, displayModeBar: false});
        } catch (err) {
          console.error("Plotly Error:", err);
          document.getElementById(`chart_${i}`).innerHTML = '<p class="danger">Chart parse error</p>';
        }
      });
      showToast(`${data.charts.length} charts generated`, 'success');
    } else {
      grid.innerHTML = '<p style="color:var(--muted);padding:40px">No charts could be generated for this dataset.</p>';
    }
  } catch (e) {
    loading.classList.add('hidden');
    showToast('Dashboard error: ' + e.message, 'error');
  }
}

// ── REPORT ──
async function generateReport() {
  if (!sessionId) return;
  document.getElementById('reportPreview').classList.add('hidden');
  document.getElementById('reportReady').classList.add('hidden');
  document.getElementById('genLoading').classList.remove('hidden');
  document.getElementById('genReportBtn').disabled = true;
  document.getElementById('genReportBtn').textContent = '⏳ Building...';

  try {
    const res = await fetch(`${API}/generate_report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, filename: currentFilename })
    });
    const data = await res.json();
    document.getElementById('genLoading').classList.add('hidden');
    if (data.error) { showToast(data.error, 'error'); document.getElementById('reportPreview').classList.remove('hidden'); return; }
    reportUrl = data.report_url;
    document.getElementById('reportReady').classList.remove('hidden');
    showToast('Report generated successfully!', 'success');
  } catch (e) {
    document.getElementById('genLoading').classList.add('hidden');
    document.getElementById('reportPreview').classList.remove('hidden');
    showToast('Report error: ' + e.message, 'error');
  } finally {
    document.getElementById('genReportBtn').disabled = false;
    document.getElementById('genReportBtn').textContent = '⬡ Generate Report';
  }
}

function downloadReport() {
  if (!sessionId) return;
  window.location.href = `${API}/download_report/${sessionId}`;
}

function downloadClean() {
  if (!sessionId) return;
  window.location.href = `${API}/download_clean/${sessionId}`;
}

// ── RESET ──
function resetAll() {
  sessionId = null;
  summaryData = null;
  reportUrl = null;
  currentFilename = 'dataset';
  document.getElementById('fileInput').value = '';
  document.getElementById('chartGrid').innerHTML = '';
  document.getElementById('reportReady').classList.add('hidden');
  document.getElementById('reportPreview').classList.remove('hidden');
  resetUploadUI();
  goToPanel(1);
  showToast('Reset complete', 'success');
}

// ── TOAST ──
function showToast(msg, type = 'info') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `toast show ${type}`;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.classList.remove('show'); }, 3500);
}
