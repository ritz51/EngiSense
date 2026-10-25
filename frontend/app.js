/**
 * EngiSense — Frontend Application Controller
 * Handles tab navigation, API interactions, real-time predictions, Chart.js visualizations,
 * and statistical inference rendering.
 */

const API_BASE = window.location.origin;
let activeSubset = "FD001";
let activeModelData = null;
let testEnginesList = [];
let trajectoryChartInstance = null;
let dashboardCompChartInstance = null;
let dashboardHealthChartInstance = null;
let evalMetricsChartInstance = null;
let evalNasaChartInstance = null;
let currentTrajectoryData = null;
let activeSensorFilter = "sensor_2";

// ---------------------------------------------------------------------------
// Initialization & Event Binding
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initPredictionControls();
  initPlotTabs();
  
  // Initial API loads
  loadHealthStatus();
  loadModelMetrics(activeSubset);
  loadTestEngines(activeSubset);
  loadStatisticalInference();

  // Subset selector on dashboard
  const subsetSelect = document.getElementById("dashboard-subset-select");
  if (subsetSelect) {
    subsetSelect.addEventListener("change", (e) => {
      activeSubset = e.target.value;
      document.getElementById("dataset-badge").textContent = `NASA C-MAPSS • ${activeSubset}`;
      loadModelMetrics(activeSubset);
      loadTestEngines(activeSubset);
    });
  }
});

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
function initNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const targetId = `view-${tab.dataset.tab}`;
      document.querySelectorAll(".tab-pane").forEach((pane) => {
        pane.classList.remove("active");
      });

      const activePane = document.getElementById(targetId);
      if (activePane) activePane.classList.add("active");
    });
  });
}

// ---------------------------------------------------------------------------
// System Health
// ---------------------------------------------------------------------------
async function loadHealthStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();
    document.getElementById("status-label").textContent = "API Online";
    document.getElementById("status-dot").className = "status-dot pulsing";
    document.getElementById("header-best-model").textContent = data.recommended_model || "LSTM";
  } catch (err) {
    document.getElementById("status-label").textContent = "Offline / Loading";
    document.getElementById("status-dot").className = "status-dot";
    document.getElementById("status-dot").style.backgroundColor = "var(--accent-amber)";
  }
}

// ---------------------------------------------------------------------------
// Model Metrics & Dashboard Data
// ---------------------------------------------------------------------------
async function loadModelMetrics(subset) {
  try {
    const res = await fetch(`${API_BASE}/api/models?subset=${subset}`);
    if (!res.ok) throw new Error("Model metrics unavailable");
    const data = await res.json();
    activeModelData = data;

    const models = data.models || {};
    const bestModel = data.recommended_model || "LSTM";

    // Header badge
    document.getElementById("header-best-model").textContent = bestModel;
    document.getElementById("dash-rec-badge").textContent = `Recommended Architecture: ${bestModel}`;

    // KPI Cards
    const bestStats = models[bestModel]?.test || {};
    document.getElementById("kpi-best-rmse").textContent = bestStats.RMSE ? `${bestStats.RMSE.toFixed(2)}` : "--";
    document.getElementById("kpi-best-rmse-model").textContent = `${bestModel} on official test set`;
    document.getElementById("kpi-best-mae").textContent = bestStats.MAE ? `${bestStats.MAE.toFixed(2)}` : "--";
    document.getElementById("kpi-nasa-score").textContent = bestStats.NASA_Score ? `${Math.round(bestStats.NASA_Score)}` : "--";

    // Populate Dashboard Table
    const tbody = document.getElementById("dashboard-models-tbody");
    tbody.innerHTML = "";

    const modelOrder = ["XGBoost", "LSTM", "GRU"];
    modelOrder.forEach((name) => {
      const m = models[name];
      if (!m) return;
      const isBest = name === bestModel;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${name}</strong> ${isBest ? '<span class="badge badge-success" style="font-size:0.65rem; margin-left:4px;">Best</span>' : ''}</td>
        <td><small class="text-muted">${m.architecture}</small></td>
        <td><span class="font-mono ${isBest ? 'text-accent font-bold' : ''}">${m.test.RMSE.toFixed(2)}</span></td>
        <td><span class="font-mono">${m.test.MAE.toFixed(2)}</span></td>
        <td><span class="font-mono">${m.test.R2 ? m.test.R2.toFixed(3) : '--'}</span></td>
        <td><span class="font-mono">${Math.round(m.test.NASA_Score)}</span></td>
        <td><span class="font-mono">${m.training_seconds}s</span></td>
        <td><span class="badge badge-${isBest ? 'success' : 'secondary'}">${isBest ? 'Active' : 'Alternative'}</span></td>
      `;
      tbody.appendChild(tr);
    });

    // Populate Evaluation Tab Table
    const evalTbody = document.getElementById("evaluation-table-body");
    evalTbody.innerHTML = "";
    modelOrder.forEach((name) => {
      const m = models[name];
      if (!m) return;
      const isBest = name === bestModel;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${name}</strong></td>
        <td><span class="font-mono">${m.validation.RMSE.toFixed(2)}</span></td>
        <td><span class="font-mono">${m.validation.MAE.toFixed(2)}</span></td>
        <td><span class="font-mono ${isBest ? 'text-accent font-bold' : ''}">${m.test.RMSE.toFixed(2)}</span></td>
        <td><span class="font-mono">${m.test.MAE.toFixed(2)}</span></td>
        <td><span class="font-mono">${m.test.R2 ? m.test.R2.toFixed(3) : '--'}</span></td>
        <td><span class="font-mono">${Math.round(m.test.NASA_Score)}</span></td>
        <td><span class="font-mono">${m.training_seconds}s</span></td>
        <td><span class="badge badge-${isBest ? 'success' : 'secondary'}">${isBest ? 'Recommended' : 'Evaluated'}</span></td>
      `;
      evalTbody.appendChild(tr);
    });

    // Render Charts
    renderDashboardCharts(models);
    renderEvaluationCharts(models);

    // Initial artifact image
    updateEvaluationArtifact("actual_vs_predicted.png");
  } catch (err) {
    console.error("Error loading model metrics:", err);
  }
}

// ---------------------------------------------------------------------------
// Chart.js Visualizations
// ---------------------------------------------------------------------------
function renderDashboardCharts(models) {
  const modelNames = ["XGBoost", "LSTM", "GRU"];
  const rmseValues = modelNames.map((m) => models[m]?.test?.RMSE || 0);
  const maeValues = modelNames.map((m) => models[m]?.test?.MAE || 0);

  // 1. Dashboard Model Comparison Bar Chart
  const ctxComp = document.getElementById("dashboardComparisonChart");
  if (ctxComp) {
    if (dashboardCompChartInstance) dashboardCompChartInstance.destroy();
    dashboardCompChartInstance = new Chart(ctxComp, {
      type: "bar",
      data: {
        labels: modelNames,
        datasets: [
          {
            label: "Test RMSE (Cycles)",
            data: rmseValues,
            backgroundColor: "rgba(59, 130, 246, 0.8)",
            borderRadius: 6,
          },
          {
            label: "Test MAE (Cycles)",
            data: maeValues,
            backgroundColor: "rgba(16, 185, 129, 0.8)",
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#94a3b8", font: { family: "Outfit" } } },
        },
        scales: {
          x: { ticks: { color: "#cbd5e1" }, grid: { color: "rgba(255,255,255,0.05)" } },
          y: { ticks: { color: "#cbd5e1" }, grid: { color: "rgba(255,255,255,0.05)" }, title: { display: true, text: "Error (Cycles)", color: "#94a3b8" } },
        },
      },
    });
  }

  // 2. Dashboard Health Donut Chart
  const ctxHealth = document.getElementById("dashboardHealthDonutChart");
  if (ctxHealth) {
    if (dashboardHealthChartInstance) dashboardHealthChartInstance.destroy();
    dashboardHealthChartInstance = new Chart(ctxHealth, {
      type: "doughnut",
      data: {
        labels: ["Healthy (RUL ≥ 60)", "Maintenance Warning (25-59)", "Critical Urgent (< 25)"],
        datasets: [
          {
            data: [64, 26, 10], // Sample representative distribution of 100 test engines
            backgroundColor: ["#10b981", "#f59e0b", "#ef4444"],
            borderColor: "#111827",
            borderWidth: 3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom", labels: { color: "#cbd5e1", font: { family: "Outfit", size: 11 } } },
        },
        cutout: "70%",
      },
    });
  }
}

function renderEvaluationCharts(models) {
  const modelNames = ["XGBoost", "LSTM", "GRU"];
  const rmseValues = modelNames.map((m) => models[m]?.test?.RMSE || 0);
  const maeValues = modelNames.map((m) => models[m]?.test?.MAE || 0);
  const nasaValues = modelNames.map((m) => models[m]?.test?.NASA_Score || 0);

  // 1. Eval Accuracy Bar Chart
  const ctxEval = document.getElementById("evalMetricsBarChart");
  if (ctxEval) {
    if (evalMetricsChartInstance) evalMetricsChartInstance.destroy();
    evalMetricsChartInstance = new Chart(ctxEval, {
      type: "bar",
      data: {
        labels: modelNames,
        datasets: [
          { label: "RMSE", data: rmseValues, backgroundColor: "#3b82f6", borderRadius: 4 },
          { label: "MAE", data: maeValues, backgroundColor: "#10b981", borderRadius: 4 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#cbd5e1" } } },
        scales: {
          x: { ticks: { color: "#cbd5e1" }, grid: { color: "rgba(255,255,255,0.05)" } },
          y: { ticks: { color: "#cbd5e1" }, grid: { color: "rgba(255,255,255,0.05)" } },
        },
      },
    });
  }

  // 2. NASA Score Bar Chart
  const ctxNasa = document.getElementById("evalNasaScoreBarChart");
  if (ctxNasa) {
    if (evalNasaChartInstance) evalNasaChartInstance.destroy();
    evalNasaChartInstance = new Chart(ctxNasa, {
      type: "bar",
      data: {
        labels: modelNames,
        datasets: [
          {
            label: "NASA Asymmetric Penalty",
            data: nasaValues,
            backgroundColor: ["#60a5fa", "#34d399", "#a78bfa"],
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#cbd5e1" }, grid: { color: "rgba(255,255,255,0.05)" } },
          y: { ticks: { color: "#cbd5e1" }, grid: { color: "rgba(255,255,255,0.05)" }, title: { display: true, text: "Penalty Score (Lower is Better)", color: "#94a3b8" } },
        },
      },
    });
  }
}

// ---------------------------------------------------------------------------
// Test Engines & Prediction Flow
// ---------------------------------------------------------------------------
async function loadTestEngines(subset) {
  try {
    const res = await fetch(`${API_BASE}/api/engines?subset=${subset}`);
    if (!res.ok) throw new Error("Failed loading engines");
    const data = await res.json();
    testEnginesList = data.engines || [];

    const select = document.getElementById("predict-engine-select");
    select.innerHTML = '<option value="">-- Choose Test Engine --</option>';

    testEnginesList.forEach((eng) => {
      const opt = document.createElement("option");
      opt.value = eng.engine_id;
      opt.textContent = `Engine #${eng.engine_id} (${eng.observed_cycles} cycles observed • Ref RUL: ${eng.official_reference_rul})`;
      select.appendChild(opt);
    });

    document.getElementById("engine-hint-text").textContent = `${testEnginesList.length} test trajectories available in ${subset}`;

    // Select Engine #1 by default for demonstration
    if (testEnginesList.length > 0) {
      select.value = testEnginesList[0].engine_id;
    }
  } catch (err) {
    console.error("Error loading test engines:", err);
  }
}

function initPredictionControls() {
  const btnSelectMode = document.getElementById("btn-mode-select");
  const btnUploadMode = document.getElementById("btn-mode-upload");
  const panelSelect = document.getElementById("panel-select-engine");
  const panelUpload = document.getElementById("panel-upload-file");

  btnSelectMode.addEventListener("click", () => {
    btnSelectMode.classList.add("active");
    btnUploadMode.classList.remove("active");
    panelSelect.classList.remove("hidden");
    panelUpload.classList.add("hidden");
  });

  btnUploadMode.addEventListener("click", () => {
    btnUploadMode.classList.add("active");
    btnSelectMode.classList.remove("active");
    panelUpload.classList.remove("hidden");
    panelSelect.classList.add("hidden");
  });

  // File dropzone click & drag
  const dropzone = document.getElementById("file-dropzone");
  const fileInput = document.getElementById("predict-file-input");
  const fileTag = document.getElementById("selected-file-name");

  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      fileTag.textContent = `Selected: ${e.target.files[0].name} (${(e.target.files[0].size / 1024).toFixed(1)} KB)`;
      fileTag.classList.remove("hidden");
    }
  });

  // Run Prediction Button
  const btnPredict = document.getElementById("btn-run-prediction");
  btnPredict.addEventListener("click", handleRunPrediction);
}

async function handleRunPrediction() {
  const btnPredict = document.getElementById("btn-run-prediction");
  const spinner = document.getElementById("predict-spinner");
  const btnText = document.getElementById("btn-predict-text");
  const modelSelect = document.getElementById("predict-model-select");
  const selectedModel = modelSelect.value;

  const isUploadMode = document.getElementById("btn-mode-upload").classList.contains("active");

  btnPredict.disabled = true;
  spinner.classList.remove("hidden");
  btnText.textContent = "Analyzing Sensor Dynamics...";

  try {
    let result;
    if (isUploadMode) {
      const fileInput = document.getElementById("predict-file-input");
      if (!fileInput.files || fileInput.files.length === 0) {
        throw new Error("Please select a sensor log file first.");
      }
      const formData = new FormData();
      formData.append("file", fileInput.files[0]);
      formData.append("subset", activeSubset);
      formData.append("model_name", selectedModel);

      const res = await fetch(`${API_BASE}/api/upload_predict`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || "Upload prediction failed");
      }
      result = await res.json();
    } else {
      const engineSelect = document.getElementById("predict-engine-select");
      const engId = parseInt(engineSelect.value);
      if (!engId) throw new Error("Please select an engine from the dropdown.");

      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          subset: activeSubset,
          engine_id: engId,
          model_name: selectedModel,
        }),
      });
      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || "Prediction request failed");
      }
      result = await res.json();
    }

    displayPredictionResult(result);
  } catch (err) {
    showAlert(err.message, "danger");
  } finally {
    btnPredict.disabled = false;
    spinner.classList.add("hidden");
    btnText.textContent = "Analyze & Predict RUL";
  }
}

function displayPredictionResult(res) {
  document.getElementById("predict-empty-state").classList.add("hidden");
  document.getElementById("predict-output-content").classList.remove("hidden");

  // Gauge & Number
  const rulVal = res.predicted_rul_cycles;
  document.getElementById("res-rul-value").textContent = rulVal.toFixed(0);
  document.getElementById("res-eol-text").textContent = `Estimated failure at cycle ${res.estimated_failure_cycle}`;

  const gauge = document.getElementById("rul-gauge-indicator");
  gauge.className = "rul-gauge-circle";
  const badge = document.getElementById("result-status-badge");
  const maintBox = document.getElementById("res-maintenance-box");

  if (res.health_status === "CRITICAL") {
    gauge.classList.add("gauge-critical");
    badge.className = "badge badge-danger";
    badge.textContent = "CRITICAL ALERT";
    maintBox.className = "maintenance-box box-critical";
  } else if (res.health_status === "WARNING") {
    gauge.classList.add("gauge-warning");
    badge.className = "badge badge-warning";
    badge.textContent = "MAINTENANCE DUE";
    maintBox.className = "maintenance-box box-warning";
  } else {
    badge.className = "badge badge-success";
    badge.textContent = "NOMINAL HEALTH";
    maintBox.className = "maintenance-box";
  }

  // Numerical metrics
  document.getElementById("res-observed-cycles").textContent = `${res.total_observed_cycles} cycles`;
  const officialRulSpan = document.getElementById("res-official-rul");
  const errorSpan = document.getElementById("res-error-val");

  if (res.official_reference_rul !== null && res.official_reference_rul !== undefined) {
    officialRulSpan.textContent = `${res.official_reference_rul} cycles`;
    const err = res.prediction_error_cycles;
    const sign = err >= 0 ? "+" : "";
    errorSpan.textContent = `${sign}${err} cycles`;
    errorSpan.className = `metric-row-val ${Math.abs(err) <= 15 ? 'text-success' : 'text-danger'}`;
  } else {
    officialRulSpan.textContent = "N/A (Uploaded File)";
    errorSpan.textContent = "--";
  }

  document.getElementById("res-model-used").textContent = res.model_used;

  // Risk bar
  const risk = res.risk_percentage;
  document.getElementById("res-risk-bar").style.width = `${risk}%`;
  document.getElementById("res-risk-bar").style.backgroundColor = risk > 70 ? "var(--accent-rose)" : risk > 40 ? "var(--accent-amber)" : "var(--accent-emerald)";
  document.getElementById("res-risk-text").textContent = `${risk}%`;

  // Guidance text
  document.getElementById("res-guidance-text").textContent = res.maintenance_guidance;

  // Sensor trajectory chart
  if (res.trajectory) {
    currentTrajectoryData = res.trajectory;
    buildSensorChips(Object.keys(res.trajectory.sensors));
    renderTrajectoryChart(res.trajectory, activeSensorFilter);
  }
}

function buildSensorChips(sensorNames) {
  const container = document.getElementById("sensor-toggle-chips");
  container.innerHTML = "";

  sensorNames.forEach((name) => {
    const btn = document.createElement("button");
    btn.className = `sensor-chip ${name === activeSensorFilter ? 'active' : ''}`;
    btn.textContent = name;
    btn.addEventListener("click", () => {
      activeSensorFilter = name;
      document.querySelectorAll(".sensor-chip").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      if (currentTrajectoryData) {
        renderTrajectoryChart(currentTrajectoryData, activeSensorFilter);
      }
    });
    container.appendChild(btn);
  });
}

function renderTrajectoryChart(traj, sensorKey) {
  const ctx = document.getElementById("engineTrajectoryChart");
  if (!ctx) return;

  const cycles = traj.cycles;
  const sensorVals = traj.sensors[sensorKey] || [];

  if (trajectoryChartInstance) trajectoryChartInstance.destroy();

  trajectoryChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: cycles,
      datasets: [
        {
          label: `${sensorKey} Measurement Trajectory`,
          data: sensorVals,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          borderWidth: 2,
          pointRadius: cycles.length > 80 ? 0 : 2,
          pointHoverRadius: 4,
          fill: true,
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cbd5e1", font: { family: "Outfit" } } },
        tooltip: {
          callbacks: {
            title: (items) => `Operating Cycle: ${items[0].label}`,
            label: (item) => `Value: ${item.formattedValue}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255,255,255,0.05)" },
          title: { display: true, text: "Operational Cycle", color: "#94a3b8" },
        },
        y: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255,255,255,0.05)" },
          title: { display: true, text: "Sensor Reading", color: "#94a3b8" },
        },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// Model Diagnostic Artifact Tabs
// ---------------------------------------------------------------------------
function initPlotTabs() {
  const pills = document.querySelectorAll("#eval-plot-tabs .pill");
  pills.forEach((pill) => {
    pill.addEventListener("click", () => {
      pills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      updateEvaluationArtifact(pill.dataset.img);
    });
  });
}

function updateEvaluationArtifact(filename) {
  const img = document.getElementById("eval-artifact-img");
  if (img) {
    img.src = `${API_BASE}/api/plots/${activeSubset}/${filename}`;
  }
}

// ---------------------------------------------------------------------------
// Statistical Inference & EDA Rendering
// ---------------------------------------------------------------------------
async function loadStatisticalInference() {
  try {
    const res = await fetch(`${API_BASE}/api/stats`);
    if (!res.ok) return;
    const data = await res.json();

    // Lifetime Comparison
    const lifeComp = data.lifetime_comparison || {};
    document.getElementById("stat-chosen-test").textContent = lifeComp.chosen_test || "--";
    document.getElementById("stat-test-stat").textContent = lifeComp.test_statistic ? lifeComp.test_statistic.toFixed(2) : "--";
    document.getElementById("stat-p-value").textContent = lifeComp.p_value ? lifeComp.p_value.toExponential(3) : "--";
    document.getElementById("stat-h0-decision").textContent = lifeComp.reject_null ? "Reject H₀ (p < 0.05)" : "Fail to Reject";
    document.getElementById("stat-h0-decision").className = `stat-item-val ${lifeComp.reject_null ? 'text-accent' : ''}`;

    const pwList = document.getElementById("stat-pairwise-list");
    pwList.innerHTML = "";
    const cis = lifeComp.pairwise_median_difference_bootstrap_ci || {};
    for (const [pair, bounds] of Object.entries(cis)) {
      const li = document.createElement("li");
      li.textContent = `${pair.replace("_vs_", " vs ")}: [${bounds[0].toFixed(1)}, ${bounds[1].toFixed(1)}] cycles`;
      pwList.appendChild(li);
    }

    // Sensor-RUL Correlations Table
    const corrList = data.sensor_rul_association?.sensor_rul_corr || [];
    const corrTbody = document.getElementById("sensor-corr-tbody");
    corrTbody.innerHTML = "";

    corrList.forEach((c) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${c.sensor}</strong></td>
        <td><span class="font-mono ${Math.abs(c.pearson_r || 0) > 0.5 ? 'text-accent font-bold' : ''}">${c.pearson_r ? c.pearson_r.toFixed(3) : '--'}</span></td>
        <td><span class="font-mono">${c.spearman_rho ? c.spearman_rho.toFixed(3) : '--'}</span></td>
        <td><span class="font-mono">${c.pearson_p ? c.pearson_p.toExponential(2) : '--'}</span></td>
        <td><span class="font-mono text-success">${c.pearson_p_adjusted_bh ? c.pearson_p_adjusted_bh.toExponential(2) : '--'}</span></td>
      `;
      corrTbody.appendChild(tr);
    });

    // Mixed Effects Table
    const mixedList = data.sensor_cycle_trends?.sensor_cycle_mixed || [];
    const mixedTbody = document.getElementById("mixed-effects-tbody");
    mixedTbody.innerHTML = "";

    mixedList.forEach((m) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${m.sensor}</strong></td>
        <td><span class="font-mono font-bold">${m.slope ? m.slope.toFixed(5) : '--'}</span></td>
        <td><span class="font-mono">${m.std_error ? m.std_error.toFixed(5) : '--'}</span></td>
        <td><span class="font-mono">${m.p_value ? m.p_value.toExponential(2) : '--'}</span></td>
        <td><span class="font-mono">${m.icc ? m.icc.toFixed(3) : '--'}</span></td>
        <td><span class="badge badge-${m.converged ? 'success' : 'warning'}">${m.converged ? 'Converged' : 'Warning'}</span></td>
      `;
      mixedTbody.appendChild(tr);
    });

  } catch (err) {
    console.error("Error loading statistical inference:", err);
  }
}

// ---------------------------------------------------------------------------
// Alerts
// ---------------------------------------------------------------------------
function showAlert(message, type = "info") {
  const banner = document.getElementById("alert-banner");
  const msgSpan = document.getElementById("alert-message");
  msgSpan.textContent = message;
  banner.className = `alert-banner banner-${type}`;
  banner.classList.remove("hidden");
  setTimeout(() => closeAlert(), 6000);
}

function closeAlert() {
  document.getElementById("alert-banner").classList.add("hidden");
}
