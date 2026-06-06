// Chart theme palettes — adapt to light / dark
var PALETTES = {
  dark: {
    grid:     'rgba(48,54,61,0.5)',
    tick:     '#6e7681',
    accent:   '#58a6ff',
    accentBg: 'rgba(88,166,255,0.08)',
    baseline: '#6e7681',
    ideal:    'rgba(110,118,129,0.25)',
    green:    'rgba(63,185,80,0.75)',
    greenBg:  'rgba(63,185,80,0.06)',
    orange:   'rgba(210,153,34,0.75)',
    red:      'rgba(248,81,73,0.75)',
    redBg:    'rgba(248,81,73,0.06)',
    greenLine:'rgba(63,185,80,0.9)'
  },
  light: {
    grid:     'rgba(208,215,222,0.6)',
    tick:     '#656d76',
    accent:   '#0969da',
    accentBg: 'rgba(9,105,218,0.06)',
    baseline: '#8b949e',
    ideal:    'rgba(139,148,158,0.25)',
    green:    'rgba(26,127,55,0.8)',
    greenBg:  'rgba(26,127,55,0.06)',
    orange:   'rgba(154,103,0,0.8)',
    red:      'rgba(207,34,46,0.8)',
    redBg:    'rgba(207,34,46,0.06)',
    greenLine:'rgba(26,127,55,0.9)'
  }
};

var chartInstances = [];

function destroyCharts() {
  chartInstances.forEach(function (c) { c.destroy(); });
  chartInstances = [];
}

function createCharts(theme) {
  var p = PALETTES[theme] || PALETTES.dark;

  Chart.defaults.color = p.tick;
  Chart.defaults.borderColor = p.grid;
  Chart.defaults.font.family = "'JetBrains Mono', 'Fira Code', monospace";
  Chart.defaults.font.size = 11;

  var baseScales = {
    x: { grid: { color: p.grid }, ticks: { color: p.tick } },
    y: { grid: { color: p.grid }, ticks: { color: p.tick } }
  };

  // ── Throughput scaling ────────────────
  var el1 = document.getElementById('chart-scaling');
  if (el1) {
    chartInstances.push(new Chart(el1, {
      type: 'line',
      data: {
        labels: ['1','2','4','8','12','16'],
        datasets: [
          {
            label: 'CacheFlow',
            data: [42.3,79.8,148.2,261.5,340.1,385.7],
            borderColor: p.accent,
            backgroundColor: p.accentBg,
            fill: true, tension: 0.25,
            pointRadius: 4, borderWidth: 2
          },
          {
            label: 'Baseline (llama.cpp)',
            data: [38.1,52.4,74.6,102.3,118.7,125.2],
            borderColor: p.baseline,
            borderDash: [6,3], tension: 0.25,
            pointRadius: 3, borderWidth: 1.5
          },
          {
            label: 'Ideal linear',
            data: [42.3,84.6,169.2,338.4,507.6,676.8],
            borderColor: p.ideal,
            borderDash: [2,4], pointRadius: 0, borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { boxWidth: 10, padding: 12 } } },
        scales: {
          x: { ...baseScales.x, title: { display: true, text: 'Concurrent Requests', color: p.tick } },
          y: { ...baseScales.y, title: { display: true, text: 'Throughput (tok/s)', color: p.tick }, beginAtZero: true }
        }
      }
    }));
  }

  // ── Latency distribution ──────────────
  var el2 = document.getElementById('chart-latency');
  if (el2) {
    chartInstances.push(new Chart(el2, {
      type: 'bar',
      data: {
        labels: ['1','2','4','8','12','16'],
        datasets: [
          { label: 'P50', data: [23.6,25.1,27.4,31.2,35.8,41.3], backgroundColor: p.green, borderRadius: 3 },
          { label: 'P90', data: [24.8,27.3,32.1,42.5,51.2,63.7], backgroundColor: p.orange, borderRadius: 3 },
          { label: 'P99', data: [26.1,30.7,38.6,58.3,74.1,92.4], backgroundColor: p.red, borderRadius: 3 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { boxWidth: 10, padding: 12 } } },
        scales: {
          x: { ...baseScales.x, title: { display: true, text: 'Concurrent Requests', color: p.tick } },
          y: { ...baseScales.y, title: { display: true, text: 'Step Latency (ms)', color: p.tick }, beginAtZero: true }
        }
      }
    }));
  }

  // ── Variance reduction ────────────────
  var el3 = document.getElementById('chart-variance');
  if (el3) {
    chartInstances.push(new Chart(el3, {
      type: 'line',
      data: {
        labels: ['0','500','1k','2k','3k','4k','5k','6k','7k','8k','9k','10k'],
        datasets: [
          {
            label: 'Baseline σ',
            data: [4.2,6.8,9.1,14.3,18.7,22.1,26.5,31.2,35.8,41.3,46.1,52.7],
            borderColor: p.red, backgroundColor: p.redBg,
            fill: true, tension: 0.3, pointRadius: 2, borderWidth: 1.5
          },
          {
            label: 'CacheFlow σ',
            data: [3.8,4.1,4.5,5.2,5.8,6.3,7.1,7.9,8.4,9.2,10.1,10.8],
            borderColor: p.greenLine, backgroundColor: p.greenBg,
            fill: true, tension: 0.3, pointRadius: 2, borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { boxWidth: 10, padding: 12 } } },
        scales: {
          x: { ...baseScales.x, title: { display: true, text: 'Generated Tokens', color: p.tick } },
          y: { ...baseScales.y, title: { display: true, text: 'Latency Std Dev (ms)', color: p.tick }, beginAtZero: true }
        }
      }
    }));
  }

  // ── Fragmentation ─────────────────────
  var el4 = document.getElementById('chart-fragmentation');
  if (el4) {
    chartInstances.push(new Chart(el4, {
      type: 'line',
      data: {
        labels: ['0','1k','2k','3k','4k','5k','6k','7k','8k'],
        datasets: [
          {
            label: 'Contiguous alloc',
            data: [0,3.2,8.7,16.1,24.8,31.2,38.5,42.1,47.3],
            borderColor: p.red, tension: 0.3, pointRadius: 2, borderWidth: 1.5
          },
          {
            label: 'Block alloc (no compact)',
            data: [0,1.8,4.2,7.1,10.5,13.8,16.2,18.9,21.3],
            borderColor: p.orange, tension: 0.3, pointRadius: 2, borderWidth: 1.5
          },
          {
            label: 'Block alloc + compact',
            data: [0,0.4,0.9,1.5,2.1,2.8,3.2,3.6,4.1],
            borderColor: p.greenLine, tension: 0.3, pointRadius: 2, borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { boxWidth: 10, padding: 12 } } },
        scales: {
          x: { ...baseScales.x, title: { display: true, text: 'Completed Requests', color: p.tick } },
          y: { ...baseScales.y, title: { display: true, text: 'External Fragmentation (%)', color: p.tick }, beginAtZero: true }
        }
      }
    }));
  }
}

// expose rebuild for theme toggle
window.rebuildCharts = function (theme) {
  destroyCharts();
  createCharts(theme);
};

document.addEventListener('DOMContentLoaded', function () {
  var theme = document.documentElement.getAttribute('data-theme') || 'dark';
  createCharts(theme);
});
