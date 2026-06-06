document.addEventListener('DOMContentLoaded', function () {

  var gridColor = 'rgba(48, 54, 61, 0.6)';
  var tickColor = '#6e7681';

  Chart.defaults.color = tickColor;
  Chart.defaults.borderColor = gridColor;
  Chart.defaults.font.family = "'JetBrains Mono', 'Fira Code', monospace";
  Chart.defaults.font.size = 11;

  var baseOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { boxWidth: 10, padding: 12 }
      }
    },
    scales: {
      x: { grid: { color: gridColor }, ticks: { color: tickColor } },
      y: { grid: { color: gridColor }, ticks: { color: tickColor } }
    }
  };

  // ── Throughput scaling ────────────────
  var ctxScale = document.getElementById('chart-scaling');
  if (ctxScale) {
    new Chart(ctxScale, {
      type: 'line',
      data: {
        labels: ['1', '2', '4', '8', '12', '16'],
        datasets: [
          {
            label: 'CacheFlow',
            data: [42.3, 79.8, 148.2, 261.5, 340.1, 385.7],
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88, 166, 255, 0.08)',
            fill: true,
            tension: 0.25,
            pointRadius: 4,
            borderWidth: 2
          },
          {
            label: 'Baseline (llama.cpp)',
            data: [38.1, 52.4, 74.6, 102.3, 118.7, 125.2],
            borderColor: '#6e7681',
            borderDash: [6, 3],
            tension: 0.25,
            pointRadius: 3,
            borderWidth: 1.5
          },
          {
            label: 'Ideal linear',
            data: [42.3, 84.6, 169.2, 338.4, 507.6, 676.8],
            borderColor: 'rgba(110, 118, 129, 0.3)',
            borderDash: [2, 4],
            pointRadius: 0,
            borderWidth: 1
          }
        ]
      },
      options: {
        ...baseOpts,
        plugins: {
          ...baseOpts.plugins,
          title: { display: false }
        },
        scales: {
          x: {
            ...baseOpts.scales.x,
            title: { display: true, text: 'Concurrent Requests', color: tickColor }
          },
          y: {
            ...baseOpts.scales.y,
            title: { display: true, text: 'Throughput (tok/s)', color: tickColor },
            beginAtZero: true
          }
        }
      }
    });
  }

  // ── Latency distribution ──────────────
  var ctxLat = document.getElementById('chart-latency');
  if (ctxLat) {
    new Chart(ctxLat, {
      type: 'bar',
      data: {
        labels: ['1', '2', '4', '8', '12', '16'],
        datasets: [
          {
            label: 'P50',
            data: [23.6, 25.1, 27.4, 31.2, 35.8, 41.3],
            backgroundColor: 'rgba(63, 185, 80, 0.7)',
            borderRadius: 2
          },
          {
            label: 'P90',
            data: [24.8, 27.3, 32.1, 42.5, 51.2, 63.7],
            backgroundColor: 'rgba(210, 153, 34, 0.7)',
            borderRadius: 2
          },
          {
            label: 'P99',
            data: [26.1, 30.7, 38.6, 58.3, 74.1, 92.4],
            backgroundColor: 'rgba(248, 81, 73, 0.7)',
            borderRadius: 2
          }
        ]
      },
      options: {
        ...baseOpts,
        scales: {
          x: {
            ...baseOpts.scales.x,
            title: { display: true, text: 'Concurrent Requests', color: tickColor }
          },
          y: {
            ...baseOpts.scales.y,
            title: { display: true, text: 'Step Latency (ms)', color: tickColor },
            beginAtZero: true
          }
        }
      }
    });
  }

  // ── Variance reduction ────────────────
  var ctxVar = document.getElementById('chart-variance');
  if (ctxVar) {
    new Chart(ctxVar, {
      type: 'line',
      data: {
        labels: ['0', '500', '1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k', '9k', '10k'],
        datasets: [
          {
            label: 'Baseline σ',
            data: [4.2, 6.8, 9.1, 14.3, 18.7, 22.1, 26.5, 31.2, 35.8, 41.3, 46.1, 52.7],
            borderColor: 'rgba(248, 81, 73, 0.8)',
            backgroundColor: 'rgba(248, 81, 73, 0.06)',
            fill: true,
            tension: 0.3,
            pointRadius: 2,
            borderWidth: 1.5
          },
          {
            label: 'CacheFlow σ',
            data: [3.8, 4.1, 4.5, 5.2, 5.8, 6.3, 7.1, 7.9, 8.4, 9.2, 10.1, 10.8],
            borderColor: 'rgba(63, 185, 80, 0.9)',
            backgroundColor: 'rgba(63, 185, 80, 0.06)',
            fill: true,
            tension: 0.3,
            pointRadius: 2,
            borderWidth: 2
          }
        ]
      },
      options: {
        ...baseOpts,
        scales: {
          x: {
            ...baseOpts.scales.x,
            title: { display: true, text: 'Generated Tokens', color: tickColor }
          },
          y: {
            ...baseOpts.scales.y,
            title: { display: true, text: 'Latency Std Dev (ms)', color: tickColor },
            beginAtZero: true
          }
        }
      }
    });
  }

  // ── KV-cache fragmentation ────────────
  var ctxFrag = document.getElementById('chart-fragmentation');
  if (ctxFrag) {
    new Chart(ctxFrag, {
      type: 'line',
      data: {
        labels: ['0', '1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k'],
        datasets: [
          {
            label: 'Contiguous alloc',
            data: [0, 3.2, 8.7, 16.1, 24.8, 31.2, 38.5, 42.1, 47.3],
            borderColor: 'rgba(248, 81, 73, 0.8)',
            tension: 0.3,
            pointRadius: 2,
            borderWidth: 1.5
          },
          {
            label: 'Block alloc (no compact)',
            data: [0, 1.8, 4.2, 7.1, 10.5, 13.8, 16.2, 18.9, 21.3],
            borderColor: 'rgba(210, 153, 34, 0.8)',
            tension: 0.3,
            pointRadius: 2,
            borderWidth: 1.5
          },
          {
            label: 'Block alloc + compact',
            data: [0, 0.4, 0.9, 1.5, 2.1, 2.8, 3.2, 3.6, 4.1],
            borderColor: 'rgba(63, 185, 80, 0.9)',
            tension: 0.3,
            pointRadius: 2,
            borderWidth: 2
          }
        ]
      },
      options: {
        ...baseOpts,
        scales: {
          x: {
            ...baseOpts.scales.x,
            title: { display: true, text: 'Completed Requests', color: tickColor }
          },
          y: {
            ...baseOpts.scales.y,
            title: { display: true, text: 'External Fragmentation (%)', color: tickColor },
            beginAtZero: true
          }
        }
      }
    });
  }

});
