const state = {
  image: null,
  payload: null,
  showTruth: true,
  threshold: 0.25,
};

const $ = (id) => document.getElementById(id);
const canvas = $('imageCanvas');
const context = canvas.getContext('2d');

function setLoading(loading) {
  $('stageLoading').hidden = !loading;
  if (loading) $('imagePlaceholder').hidden = true;
}

function showPlaceholder(message, detail = '') {
  canvas.hidden = true;
  $('imagePlaceholder').hidden = false;
  $('imagePlaceholder').querySelector('strong').textContent = message;
  $('imagePlaceholder').querySelector('span').textContent = detail;
}

function formatNumber(value, digits = 1) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
}

function fitCanvas() {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = state.image.naturalWidth * ratio;
  canvas.height = state.image.naturalHeight * ratio;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.drawImage(state.image, 0, 0);
  drawBoxes();
}

function boxLabel(text, x, y, color) {
  context.font = '600 12px DM Mono, monospace';
  const width = context.measureText(text).width + 12;
  const top = Math.max(0, y - 21);
  context.fillStyle = color;
  context.fillRect(x, top, width, 19);
  context.fillStyle = '#101719';
  context.fillText(text, x + 6, top + 13);
}

function drawBox(box, color, label) {
  const [x1, y1, x2, y2] = box.map(Number);
  context.save();
  context.strokeStyle = color;
  context.lineWidth = Math.max(2, state.image.naturalWidth / 500);
  context.setLineDash([]);
  context.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
  boxLabel(label, x1, y1, color);
  context.restore();
}

function visiblePredictions() {
  return (state.payload?.predictions || []).filter((item) => item.score >= state.threshold);
}

function drawBoxes() {
  if (!state.image || !state.payload) return;
  const width = state.image.naturalWidth;
  const height = state.image.naturalHeight;
  context.drawImage(state.image, 0, 0, width, height);
  const predictions = visiblePredictions();
  if (state.showTruth) {
    (state.payload.ground_truth || []).forEach((box, index) => {
      drawBox(box, '#77dbbb', `GT ${index + 1}`);
    });
  }
  predictions.forEach((item, index) => {
    drawBox(item.box, '#ffa276', `${Math.round(item.score * 100)}% · ${index + 1}`);
  });
  renderPredictionList(predictions);
  $('predictionCount').textContent = state.payload.model_ready ? String(predictions.length).padStart(2, '0') : '—';
  $('truthCount').textContent = String((state.payload.ground_truth || []).length).padStart(2, '0');
  $('resultCount').textContent = `${String(predictions.length).padStart(2, '0')} BOX`;
}

function renderPredictionList(predictions) {
  const list = $('detectionList');
  list.replaceChildren();
  if (!state.payload?.model_ready) {
    const empty = document.createElement('div');
    empty.className = 'empty-list';
    empty.textContent = '尚未找到訓練完成的模型 checkpoint';
    list.append(empty);
    return;
  }
  if (!predictions.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-list';
    empty.textContent = '目前門檻下沒有模型預測框';
    list.append(empty);
    return;
  }
  predictions.forEach((item, index) => {
    const row = document.createElement('div');
    row.className = 'detection-row';
    const number = document.createElement('span');
    number.className = 'detection-index';
    number.textContent = String(index + 1).padStart(2, '0');
    const info = document.createElement('span');
    const name = document.createElement('span');
    name.className = 'detection-name';
    name.textContent = 'Lung opacity';
    const coords = document.createElement('span');
    coords.className = 'detection-coords';
    const [x1, y1, x2, y2] = item.box;
    coords.textContent = `x ${Math.round(x1)} · y ${Math.round(y1)} · ${Math.round(x2 - x1)} × ${Math.round(y2 - y1)}`;
    info.append(name, coords);
    const confidence = document.createElement('span');
    confidence.className = 'confidence';
    confidence.textContent = `${(item.score * 100).toFixed(1)}%`;
    row.append(number, info, confidence);
    list.append(row);
  });
}

function showImage(payload) {
  state.payload = payload;
  state.image = new Image();
  state.image.onload = () => {
    canvas.hidden = false;
    $('imagePlaceholder').hidden = true;
    $('patientLabel').textContent = payload.patient_id || '上傳的 DICOM 影像';
    $('studyTag').textContent = payload.source === 'uploaded DICOM' ? 'UPLOADED' : 'VALIDATION';
    $('imageDimensions').textContent = `${payload.width} × ${payload.height} px`;
    $('sourceLabel').textContent = payload.source === 'uploaded DICOM' ? '檔案僅供本次本機推論' : '資料留存在本機';
    fitCanvas();
  };
  state.image.onerror = () => showPlaceholder('影像載入失敗', '請重新選擇 DICOM 樣本');
  state.image.src = payload.image;
}

async function loadSample(patientId) {
  if (!patientId) return;
  setLoading(true);
  try {
    const response = await fetch(`/api/sample/${encodeURIComponent(patientId)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '無法載入樣本');
    showImage(payload);
  } catch (error) {
    showPlaceholder('無法載入影像', error.message);
  } finally {
    setLoading(false);
  }
}

function renderHistory(history) {
  const body = $('historyBody');
  body.replaceChildren();
  if (!history?.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 5;
    cell.className = 'table-empty';
    cell.textContent = '尚無訓練紀錄';
    row.append(cell);
    body.append(row);
    $('historyNote').textContent = '開始訓練後會記錄每個 epoch';
    return;
  }
  history.forEach((record) => {
    const row = document.createElement('tr');
    const values = [
      `E${String(record.epoch).padStart(2, '0')}`,
      formatNumber(record.train_loss, 4),
      `${(Number(record.map_iou_0_4) * 100).toFixed(2)}%`,
      Number(record.learning_rate).toExponential(1),
      `${Math.round(record.seconds)}s`,
    ];
    values.forEach((value) => {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    });
    body.append(row);
  });
  $('historyNote').textContent = `${history.length} 個 epoch 已記錄`;
}

async function loadDashboard() {
  try {
    const response = await fetch('/api/status');
    const status = await response.json();
    if (!response.ok) throw new Error('無法讀取模型狀態');
    $('mapValue').textContent = status.best_map_percent == null ? '—' : Number(status.best_map_percent).toFixed(2);
    $('mapUnit').textContent = status.best_map_percent == null ? '' : '%';
    $('metricName').textContent = status.metric_name || 'mAP @ IoU 0.4';
    $('valCount').textContent = status.validation_images ? Number(status.validation_images).toLocaleString() : '—';
    $('positiveCount').textContent = status.validation_positive_images ? Number(status.validation_positive_images).toLocaleString() : '—';
    $('epochValue').textContent = status.epoch ? String(status.epoch).padStart(2, '0') : '—';
    $('epochTotal').textContent = status.epochs_requested ? `/ ${status.epochs_requested}` : '';
    $('trainCount').textContent = status.train_images ? `${Number(status.train_images).toLocaleString()} 張訓練影像` : '等待訓練資料';
    $('deviceValue').textContent = status.device === 'cuda:0' ? 'NVIDIA CUDA 加速' : status.device;
    $('environmentLabel').textContent = status.device === 'cuda:0' ? 'LOCAL · CUDA ENABLED' : 'LOCAL ENVIRONMENT';
    $('modelStatus').textContent = status.model_ready ? '模型就緒' : '等待訓練';
    $('statusPulse').classList.toggle('ready', status.model_ready);
    $('modelNote').querySelector('p').textContent = status.model_ready
      ? '模型以留出驗證集評估。預測框僅供研究與課程展示，不作為醫療診斷。'
      : '執行 train.py 完成訓練後，這裡會載入 checkpoint 並顯示預測框。';
    renderHistory(status.history);
  } catch (error) {
    $('modelStatus').textContent = '服務連線中';
  }

  try {
    const response = await fetch('/api/samples');
    const samples = await response.json();
    const select = $('sampleSelect');
    select.replaceChildren();
    if (!response.ok || !samples.length) {
      const option = document.createElement('option');
      option.textContent = '找不到驗證影像';
      select.append(option);
      showPlaceholder('找不到驗證影像', '請確認資料路徑及訓練標註');
      return;
    }
    samples.forEach((sample) => {
      const option = document.createElement('option');
      option.value = sample.patient_id;
      option.textContent = `${sample.patient_id.slice(0, 8)}… · ${sample.positive ? '有框標註' : '無框標註'}`;
      select.append(option);
    });
    const preferred = samples.find((sample) => sample.positive) || samples[0];
    select.value = preferred.patient_id;
    loadSample(preferred.patient_id);
  } catch (_error) {
    showPlaceholder('無法連線至資料服務', '請重新整理頁面');
  }
}

$('sampleSelect').addEventListener('change', (event) => loadSample(event.target.value));
$('truthToggle').addEventListener('click', (event) => {
  state.showTruth = !state.showTruth;
  event.currentTarget.classList.toggle('active', state.showTruth);
  event.currentTarget.setAttribute('aria-pressed', String(state.showTruth));
  if (state.image) fitCanvas();
});
$('thresholdSlider').addEventListener('input', (event) => {
  state.threshold = Number(event.target.value);
  $('thresholdValue').textContent = state.threshold.toFixed(2);
  const percent = ((state.threshold - 0.05) / 0.85) * 100;
  event.target.style.background = `linear-gradient(90deg, var(--mint-strong) ${percent}%, #344344 ${percent}%)`;
  if (state.image) fitCanvas();
});
$('uploadButton').addEventListener('click', () => $('fileInput').click());
$('fileInput').addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  setLoading(true);
  try {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('/api/predict', { method: 'POST', body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '推論失敗');
    showImage(payload);
  } catch (error) {
    showPlaceholder('無法分析此檔案', error.message);
  } finally {
    setLoading(false);
    event.target.value = '';
  }
});
window.addEventListener('resize', () => { if (state.image) fitCanvas(); });

loadDashboard();
