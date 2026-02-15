(function () {
  var center = [8.0, 30.0];
  var map = L.map('map').setView(center, 6);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap'
  }).addTo(map);

  var predictionLayer = null;
  var REFRESH_INTERVAL_MS = 60 * 60 * 1000; // 1 hour

  function probToColor(p) {
    // More distinct color scale: red (low) -> yellow -> green (high)
    if (p <= 0.2) return '#d73027';      // Dark red (very low)
    if (p <= 0.4) return '#fc8d59';      // Orange (low)
    if (p <= 0.6) return '#fee090';      // Yellow (medium)
    if (p <= 0.8) return '#91cf60';      // Light green (high)
    return '#1a9850';                    // Dark green (very high)
  }

  function updateInfo(data) {
    var ts = data && data.timestamp;
    var next = data && data.next_update;
    var meta = (data && data.meta) || {};
    var lastEl = document.getElementById('last-updated');
    var nextEl = document.getElementById('next-update');
    var trendEl = document.getElementById('trend');
    var confEl = document.getElementById('confidence');
    if (lastEl) lastEl.textContent = 'Last updated: ' + (ts || '—');
    if (nextEl) nextEl.textContent = 'Next update: ' + (next || '—');
    if (trendEl) trendEl.textContent = 'Current trend: ' + (meta.movement_direction || '—');
    if (confEl) confEl.textContent = 'Confidence: ' + (meta.confidence || '—');
  }

  function formatTimestamp(iso) {
    if (!iso) return '—';
    try {
      var d = new Date(iso);
      return d.toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' });
    } catch (e) { return iso; }
  }

  function buildGeoJSONLayer(data) {
    return L.geoJSON(data, {
      style: function (f) {
        var p = (f.properties && f.properties.probability) || 0;
        return { fillColor: probToColor(p), fillOpacity: 0.6, weight: 0.5, color: '#333' };
      },
      onEachFeature: function (f, layer) {
        var p = (f.properties && f.properties.probability) != null ? (f.properties.probability * 100).toFixed(1) : '?';
        var sat = (f.properties && f.properties.last_satellite_update) || '—';
        layer.bindPopup(
          '<strong>' + (f.properties && f.properties.cell_id) + '</strong><br>' +
          'Probability: ' + p + '%<br>' +
          'Last satellite: ' + sat
        );
      }
    });
  }

  function loadPredictions() {
    fetch('/api/predictions')
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        if (!data || !data.features) return;
        updateInfo(data);
        if (predictionLayer) map.removeLayer(predictionLayer);
        predictionLayer = buildGeoJSONLayer(data);
        predictionLayer.addTo(map);
        if (predictionLayer.getBounds().isValid()) map.fitBounds(predictionLayer.getBounds());
      })
      .catch(function () { updateInfo(null); });
  }

  loadPredictions();
  setInterval(loadPredictions, REFRESH_INTERVAL_MS);

  // Mobile: toggle info panel
  var infoPanel = document.getElementById('info-panel');
  if (infoPanel) {
    var toggle = document.createElement('button');
    toggle.setAttribute('type', 'button');
    toggle.setAttribute('aria-label', 'Toggle info');
    toggle.className = 'info-panel-toggle';
    toggle.textContent = '▼ Info';
    var inner = document.createElement('div');
    inner.className = 'info-panel-inner';
    while (infoPanel.firstChild) inner.appendChild(infoPanel.firstChild);
    infoPanel.appendChild(toggle);
    infoPanel.appendChild(inner);
    toggle.addEventListener('click', function () {
      infoPanel.classList.toggle('collapsed');
      toggle.textContent = infoPanel.classList.contains('collapsed') ? '▲ Info' : '▼ Info';
    });
  }
})();
