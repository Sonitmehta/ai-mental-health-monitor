/**
 * MindScan AI — Screening Page Logic
 * API Fetching, Actionable Clinical Steps, Verified Helplines, Chart.js
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

  sampleChips.forEach(chip => {
    chip.addEventListener("click", () => {
      journalInput.value = chip.getAttribute("data-sample");
      journalInput.dispatchEvent(new Event("input"));
      journalInput.focus();
    });
  });

  if (clearBtn && journalInput) {
    clearBtn.addEventListener("click", () => {
      journalInput.value = "";
      journalInput.dispatchEvent(new Event("input"));
      const resultSec = document.getElementById("result-section");
      if (resultSec) resultSec.classList.add("hidden");
    });
  }

  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", handleScreening);
  }
});

async function handleScreening() {
  const inputEl = document.getElementById("journalInput");
  const btn     = document.getElementById("analyzeBtn");
  const btnText = document.getElementById("btnText");
  const spinner = document.getElementById("btnSpinner");
  const text    = inputEl ? inputEl.value.trim() : "";

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
    showToast("Evaluation complete!", "success");
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
  const scoreDisplay = document.getElementById("scoreDisplay");
  const riskBadge    = document.getElementById("riskBadge");
  const riskSummary  = document.getElementById("riskSummary");
  const scoreProgress= document.getElementById("scoreProgress");
  const emotionLabel = document.getElementById("emotionLabel");
  const emotionIcon  = document.getElementById("emotionIcon");
  const pillsWrap    = document.getElementById("indicatorPills");

  if (resultSec) resultSec.classList.remove("hidden");

  const score = data.risk_score;
  scoreDisplay.textContent = score;
  scoreProgress.style.width = `${score}%`;

  scoreDisplay.className = "risk-score-num";
  riskBadge.className    = "badge";

  if (data.risk === "high" || score >= 70) {
    scoreDisplay.classList.add("risk-score-high");
    riskBadge.classList.add("badge-high");
    riskBadge.textContent = "High Distress";
    riskSummary.textContent = "Elevated emotional strain detected. Prioritize self-care and support.";
    scoreProgress.style.background = "var(--risk-high)";
  } else if (data.risk === "medium" || score >= 40) {
    scoreDisplay.classList.add("risk-score-medium");
    riskBadge.classList.add("badge-medium");
    riskBadge.textContent = "Moderate Strain";
    riskSummary.textContent = "Mild-to-moderate emotional tension detected. Try relaxation steps.";
    scoreProgress.style.background = "var(--risk-med)";
  } else {
    scoreDisplay.classList.add("risk-score-low");
    riskBadge.classList.add("badge-low");
    riskBadge.textContent = "Balanced State";
    riskSummary.textContent = "Healthy psychological balance and positive resilience indicated.";
    scoreProgress.style.background = "var(--risk-low)";
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
    span.textContent = "No adverse distress keywords triggered";
    pillsWrap.appendChild(span);
  }

  // Render Actionable Guidance & Coping steps
  renderGuidance(data.guidance);

  // Render AI Cognitive Reframing Solution
  renderCBTSolution(data.cbt_solution);

  // Render probabilities chart
  renderProbChart(data.probs_risk, data.probs_emotion);
}

function renderCBTSolution(cbt) {
  if (!cbt) return;
  const badgeEl = document.getElementById("cbtPatternBadge");
  const insightEl = document.getElementById("cbtInsight");
  const reframedEl = document.getElementById("cbtReframed");
  const microEl = document.getElementById("cbtMicroAction");

  if (badgeEl) badgeEl.textContent = cbt.pattern || "Cognitive Insight";
  if (insightEl) insightEl.textContent = cbt.insight || "";
  if (reframedEl) reframedEl.textContent = `"${cbt.reframed_thought || ""}"`;
  if (microEl) microEl.textContent = cbt.micro_action || "";
}

function renderGuidance(guidance) {
  if (!guidance) return;
  const box     = document.getElementById("guidanceCard");
  const titleEl = document.getElementById("guidanceTitle");
  const sumEl   = document.getElementById("guidanceSummary");
  const listEl  = document.getElementById("actionStepsList");
  const tipEl   = document.getElementById("lifestyleTip");
  const hlList  = document.getElementById("helplinesList");

  box.className = `guidance-box ${guidance.level}`;
  titleEl.textContent = guidance.title;
  sumEl.textContent   = guidance.summary;
  tipEl.textContent   = guidance.lifestyle_tip ? `💡 Tip: ${guidance.lifestyle_tip}` : "";

  // Action steps list
  listEl.innerHTML = "";
  (guidance.action_steps || []).forEach(step => {
    const div = document.createElement("div");
    div.className = "action-step-item";
    div.textContent = step;
    listEl.appendChild(div);
  });

  // Verified Helplines
  hlList.innerHTML = "";
  (guidance.helplines || []).forEach(hl => {
    const a = document.createElement("a");
    a.className = "helpline-card";
    a.href = hl.tel;
    a.innerHTML = `
      <span class="helpline-name">${hl.name}</span>
      <span class="helpline-num">📞 ${hl.number}</span>
      <span class="helpline-desc">${hl.desc}</span>
    `;
    hlList.appendChild(a);
  });
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
    "rgba(22, 163, 74, 0.75)",  // Low
    "rgba(217, 119, 6, 0.75)",  // Med
    "rgba(220, 38, 38, 0.75)",  // High
    "rgba(13, 148, 136, 0.75)", // Positive
    "rgba(148, 163, 184, 0.75)",// Neutral
    "rgba(217, 119, 6, 0.65)",  // Anxious
    "rgba(99, 102, 241, 0.75)"  // Sad
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
        borderRadius: 5,
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
          grid: { color: "rgba(148, 163, 184, 0.15)" },
          ticks: { color: "#94A3B8", font: { family: "Inter", size: 10 } }
        },
        y: {
          grid: { display: false },
          ticks: { color: "var(--text-1)", font: { family: "Inter", size: 11, weight: 500 } }
        }
      }
    }
  });
}
