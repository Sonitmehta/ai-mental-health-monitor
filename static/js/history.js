/**
 * MindScan AI — History & Trends Logic
 * Session Audit Log, Longitudinal Chart.js Time-Series, CSV Export
 */

let historyChartInstance = null;
let currentHistoryData = [];

document.addEventListener("DOMContentLoaded", () => {
  loadHistory();

  const exportBtn       = document.getElementById("exportBtn");
  const clearHistoryBtn = document.getElementById("clearHistoryBtn");

  if (exportBtn)       exportBtn.addEventListener("click", exportCSV);
  if (clearHistoryBtn) clearHistoryBtn.addEventListener("click", clearHistory);
});

async function loadHistory() {
  try {
    const res = await fetch("/api/history");
    if (!res.ok) {
      if (res.status === 401) {
        window.location.href = "/login";
        return;
      }
      throw new Error("Failed to load history");
    }

    const data = await res.json();
    currentHistoryData = data.history || [];
    renderHistoryUI(currentHistoryData);
  } catch (err) {
    showToast("Unable to load history records.", "error");
  }
}

function renderHistoryUI(history) {
  const tableBody = document.getElementById("historyTableBody");
  const statTotal = document.getElementById("statTotalSessions");
  const statAvg   = document.getElementById("statAvgScore");
  const statPeak  = document.getElementById("statPeakScore");

  if (!tableBody) return;

  statTotal.textContent = history.length;

  if (history.length === 0) {
    statAvg.textContent  = "--";
    statPeak.textContent = "--";
    tableBody.innerHTML = `
      <tr>
        <td colspan="6" style="text-align:center; padding: 2.5rem; color:var(--text-3);">
          No screening sessions recorded yet. Run a check-in on the screening page!
        </td>
      </tr>
    `;
    renderHistoryChart([]);
    return;
  }

  // Calculate metrics
  const scores = history.map(item => item.score);
  const avg = (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1);
  const peak = Math.max(...scores);

  statAvg.textContent  = avg;
  statPeak.textContent = peak;

  // Build table rows (most recent first)
  const reversed = [...history].reverse();
  tableBody.innerHTML = "";

  reversed.forEach(item => {
    const tr = document.createElement("tr");
    
    // Risk badge class
    let badgeClass = "badge-low";
    if (item.risk === "high") badgeClass = "badge-high";
    else if (item.risk === "medium") badgeClass = "badge-medium";

    const dateStr = item.timestamp ? new Date(item.timestamp).toLocaleString() : "Just now";

    tr.innerHTML = `
      <td class="font-mono text-xs">#${item.id}</td>
      <td class="text-xs" style="white-space:nowrap;">${dateStr}</td>
      <td style="max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${escapeHtml(item.text)}">
        "${escapeHtml(item.text)}"
      </td>
      <td style="text-transform:capitalize;">${item.emotion}</td>
      <td><span class="badge ${badgeClass}">${item.risk.toUpperCase()}</span></td>
      <td class="font-mono" style="font-weight:700;">${item.score}/100</td>
    `;
    tableBody.appendChild(tr);
  });

  renderHistoryChart(history);
}

function renderHistoryChart(history) {
  const canvas = document.getElementById("historyChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");

  const labels = history.map(h => `Session #${h.id}`);
  const data   = history.map(h => h.score);

  if (historyChartInstance) {
    historyChartInstance.destroy();
  }

  const gradient = ctx.createLinearGradient(0, 0, 0, 260);
  gradient.addColorStop(0, "rgba(20, 184, 166, 0.35)");
  gradient.addColorStop(1, "rgba(20, 184, 166, 0.0)");

  historyChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels.length ? labels : ["No data"],
      datasets: [{
        label: "Risk Score",
        data: data.length ? data : [0],
        borderColor: "#14B8A6",
        backgroundColor: gradient,
        fill: true,
        tension: 0.35,
        pointBackgroundColor: "#14B8A6",
        pointBorderColor: "#080C18",
        pointBorderWidth: 2,
        pointRadius: 5,
        pointHoverRadius: 7,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` Risk Score: ${ctx.raw} / 100`
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#94A3B8", font: { family: "Inter" } }
        },
        y: {
          min: 0,
          max: 100,
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#94A3B8", font: { family: "Inter" } }
        }
      }
    }
  });
}

function exportCSV() {
  if (!currentHistoryData.length) {
    showToast("No history records available to export.", "error");
    return;
  }

  const headers = ["ID", "Timestamp", "Journal_Excerpt", "Emotion", "Risk_Level", "Risk_Score"];
  const rows = currentHistoryData.map(item => [
    item.id,
    `"${item.timestamp}"`,
    `"${(item.text || "").replace(/"/g, '""')}"`,
    item.emotion,
    item.risk,
    item.score
  ]);

  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `mindscan_history_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  showToast("CSV file exported successfully!", "success");
}

async function clearHistory() {
  if (!confirm("Are you sure you want to clear all screening history records?")) {
    return;
  }

  try {
    const res = await fetch("/api/history", { method: "DELETE" });
    if (res.ok) {
      currentHistoryData = [];
      renderHistoryUI([]);
      showToast("History cleared successfully.", "success");
    }
  } catch (err) {
    showToast("Failed to clear history.", "error");
  }
}

function escapeHtml(text) {
  if (!text) return "";
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
