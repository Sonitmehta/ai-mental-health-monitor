/**
 * MindScan AI — Screening Page Logic
 * API Fetching, Chart.js Visualizations, Dynamic Indicators, Crisis Triage
 */

let probChartInstance = null;

const EMOTION_ICONS = {
  positive: "☀️",
  neutral:  "🧘",
  anxious:  "⚡",
  sad:      "🌧️"
};

document.addEventListener("DOMContentLoaded", () => {
  const journalInput = document.getElementById("journalInput");
  const charCount    = document.getElementById("charCount");
  const analyzeBtn   = document.getElementById("analyzeBtn");
  const clearBtn     = document.getElementById("clearBtn");
  const sampleChips  = document.querySelectorAll(".sample-chip");

  if (journalInput && charCount) {
    journalInput.addEventListener("input", () => {
      const len = journalInput.value.length;
      charCount.textContent = `${len} characters`;
    });
  }

  // Sample prompt chips
  sampleChips.forEach(chip => {
    chip.addEventListener("click", () => {
      journalInput.value = chip.getAttribute("data-sample");
      journalInput.dispatchEvent(new Event("input"));
      journalInput.focus();
    });
  });

  // Clear button
  if (clearBtn && journalInput) {
    clearBtn.addEventListener("click", () => {
      journalInput.value = "";
      journalInput.dispatchEvent(new Event("input"));
      const resultSec = document.getElementById("result-section");
      const crisisSec = document.getElementById("crisisAlert");
      if (resultSec) resultSec.classList.remove("visible");
      if (crisisSec) crisisSec.classList.remove("visible");
    });
  }

  // Analyze button
  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", handleScreening);
  }
});

async function handleScreening() {
  const inputEl  = document.getElementById("journalInput");
  const btn      = document.getElementById("analyzeBtn");
  const btnText  = document.getElementById("btnText");
  const spinner  = document.getElementById("btnSpinner");
  const text     = inputEl ? inputEl.value.trim() : "";

  if (!text || text.length < 5) {
    showToast("Please enter at least 5 characters to analyze.", "error");
    return;
  }

  btn.disabled = true;
  btnText.classList.add("hidden");
  spinner.classList.remove("hidden");

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    if (!res.ok) {
      if (res.status === 401) {
        window.location.href = "/login";
        return;
      }
      throw new Error("Prediction API call failed");
    }

    const data = await res.json();
    renderResults(data);
    showToast("Analysis complete!", "success");
  } catch (err) {
    showToast("Unable to reach inference engine. Please check backend connection.", "error");
  } finally {
    btn.disabled = false;
    btnText.classList.remove("hidden");
    spinner.classList.add("hidden");
  }
}

function renderResults(data) {
  const resultSec    = document.getElementById("result-section");
  const crisisAlert  = document.getElementById("crisisAlert");
  const scoreDisplay = document.getElementById("scoreDisplay");
  const riskBadge    = document.getElementById("riskBadge");
  const riskSummary  = document.getElementById("riskSummary");
  const scoreProgress= document.getElementById("scoreProgress");
  const emotionLabel = document.getElementById("emotionLabel");
  const emotionIcon  = document.getElementById("emotionIcon");
  const pillsWrap    = document.getElementById("indicatorPills");

  // Show result section
  if (resultSec) resultSec.classList.add("visible");

  const score = data.risk_score;
  scoreDisplay.textContent = score;

  // Classify score presentation
  scoreDisplay.className = "risk-score-num";
  riskBadge.className    = "badge";
  scoreProgress.style.width = `${score}%`;

  if (data.risk === "high" || score >= 70) {
    scoreDisplay.classList.add("risk-score-high");
    riskBadge.classList.add("badge-high");
    riskBadge.textContent = "High Risk";
    riskSummary.textContent = "Elevated distress levels indicated. Supportive intervention recommended.";
    scoreProgress.style.background = "var(--risk-high)";
    if (crisisAlert) crisisAlert.classList.add("visible");
  } else if (data.risk === "medium" || score >= 40) {
    scoreDisplay.classList.add("risk-score-medium");
    riskBadge.classList.add("badge-medium");
    riskBadge.textContent = "Moderate Risk";
    riskSummary.textContent = "Mild-to-moderate emotional strain detected. Consider proactive mindfulness.";
    scoreProgress.style.background = "var(--risk-med)";
    if (crisisAlert) crisisAlert.classList.remove("visible");
  } else {
    scoreDisplay.classList.add("risk-score-low");
    riskBadge.classList.add("badge-low");
    riskBadge.textContent = "Low Risk";
    riskSummary.textContent = "Balanced emotional state. No prominent clinical distress markers.";
    scoreProgress.style.background = "var(--risk-low)";
    if (crisisAlert) crisisAlert.classList.remove("visible");
  }

  // Emotion Output
  emotionLabel.textContent = data.emotion;
  emotionIcon.textContent  = EMOTION_ICONS[data.emotion.toLowerCase()] || "🧘";

  // Indicators / Biomarkers
  pillsWrap.innerHTML = "";
  let hasIndicators = false;

  if (data.indicators) {
    const highInd = data.indicators.high_risk || [];
    const medInd  = data.indicators.medium_risk || [];
    const posInd  = data.indicators.positive || [];

    highInd.forEach(w => {
      hasIndicators = true;
      const span = document.createElement("span");
      span.className = "indicator-pill indicator-high";
      span.textContent = `🚨 ${w}`;
      pillsWrap.appendChild(span);
    });

    medInd.forEach(w => {
      hasIndicators = true;
      const span = document.createElement("span");
      span.className = "indicator-pill indicator-medium";
      span.textContent = `⚠️ ${w}`;
      pillsWrap.appendChild(span);
    });

    posInd.forEach(w => {
      hasIndicators = true;
      const span = document.createElement("span");
      span.className = "indicator-pill indicator-pos";
      span.textContent = `✨ ${w}`;
      pillsWrap.appendChild(span);
    });
  }

  if (!hasIndicators) {
    const span = document.createElement("span");
    span.className = "indicator-pill indicator-pos";
    span.textContent = "No specific distress markers triggered";
    pillsWrap.appendChild(span);
  }

  // Render probabilities chart
  renderProbChart(data.probs_risk, data.probs_emotion);
}

function renderProbChart(probsRisk, probsEmotion) {
  const canvas = document.getElementById("probChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");

  const riskLabels = Object.keys(probsRisk).map(k => `Risk: ${k.toUpperCase()}`);
  const riskValues = Object.values(probsRisk).map(v => (v * 100).toFixed(1));

  const emoLabels  = Object.keys(probsEmotion).map(k => `Emotion: ${k.toUpperCase()}`);
  const emoValues  = Object.values(probsEmotion).map(v => (v * 100).toFixed(1));

  const labels = [...riskLabels, ...emoLabels];
  const data   = [...riskValues, ...emoValues];

  const colors = [
    "rgba(34,197,94,0.75)",  // Low
    "rgba(245,158,11,0.75)", // Medium
    "rgba(239,68,68,0.75)",  // High
    "rgba(20,184,166,0.75)", // Positive
    "rgba(148,163,184,0.75)",// Neutral
    "rgba(245,158,11,0.65)", // Anxious
    "rgba(129,140,248,0.75)" // Sad
  ];

  if (probChartInstance) {
    probChartInstance.destroy();
  }

  probChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Confidence Probability (%)",
        data: data,
        backgroundColor: colors.slice(0, labels.length),
        borderRadius: 6,
        borderWidth: 0
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` Confidence: ${ctx.raw}%`
          }
        }
      },
      scales: {
        x: {
          min: 0,
          max: 100,
          grid: { color: "rgba(255,255,255,0.06)" },
          ticks: { color: "#94A3B8", font: { family: "Inter" } }
        },
        y: {
          grid: { display: false },
          ticks: { color: "#F1F5F9", font: { family: "Inter", size: 11 } }
        }
      }
    }
  });
}
