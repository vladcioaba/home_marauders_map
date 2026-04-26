// View page: render the floor plan and poll /api/state for live markers.

(async function () {
  const svg = document.getElementById("plan");
  const statusEl = document.getElementById("status");
  const fpsEl = document.getElementById("fps");
  const camsEl = document.getElementById("cams");

  const houseRes = await fetch("/api/house");
  const house = await houseRes.json();
  MM.renderHouse(svg, house);

  // Build one MJPEG <img> per configured camera. The browser will keep the
  // multipart connection open; placeholder frame is shown until --live is up.
  (house.cameras || []).forEach((c) => {
    const tile = document.createElement("div");
    tile.className = "cam-tile";
    const h = document.createElement("h3");
    h.textContent = c.name || c.id;
    tile.appendChild(h);
    const img = document.createElement("img");
    img.src = `/stream/${encodeURIComponent(c.id)}.mjpg`;
    img.alt = `${c.id} stream`;
    tile.appendChild(img);
    const meta = document.createElement("p");
    meta.className = "cam-meta";
    meta.textContent = String(c.source);
    tile.appendChild(meta);
    camsEl.appendChild(tile);
  });

  async function tick() {
    try {
      const res = await fetch("/api/state");
      const state = await res.json();
      if (state.error) {
        statusEl.textContent = "live: " + state.error;
        statusEl.className = "status error";
      } else if (state.running) {
        statusEl.textContent = state.markers.length
          ? `tracking ${state.markers.length} ${state.markers.length === 1 ? "person" : "people"}`
          : "live (no one in view)";
        statusEl.className = "status live";
      } else {
        statusEl.textContent = "live loop not running (start with --live)";
        statusEl.className = "status";
      }
      fpsEl.textContent = state.fps ? `${state.fps.toFixed(1)} fps` : "";
      MM.renderHouse(svg, house, {
        markers: state.markers,
        trails: state.trails,
      });
    } catch (e) {
      statusEl.textContent = "disconnected";
      statusEl.className = "status error";
    }
  }

  await tick();
  setInterval(tick, 500);
})();
