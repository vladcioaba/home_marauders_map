// Edit page: drag-draw rooms, click-place cameras, drag to move, save to YAML.

(async function () {
  const svg = document.getElementById("plan");
  const saveBtn = document.getElementById("save");
  const saveStatus = document.getElementById("save-status");
  const deleteBtn = document.getElementById("delete");
  const modeButtons = document.querySelectorAll("[data-mode]");

  let house = await (await fetch("/api/house")).json();
  let mode = "select";
  let selected = null;
  let dragging = null;   // { kind, index, anchorX, anchorY, origRect|origPos }
  let pendingRect = null; // { x0, y0, x1, y1 } in meters

  function rerender() {
    MM.renderHouse(svg, house, { selected });
    if (pendingRect) {
      const s = house.floorplan?.scale || 60;
      const { x0, y0, x1, y1 } = pendingRect;
      const x = Math.min(x0, x1) * s;
      const y = Math.min(y0, y1) * s;
      const w = Math.abs(x1 - x0) * s;
      const h = Math.abs(y1 - y0) * s;
      svg.appendChild(MM.svgEl("rect", {
        x, y, width: w, height: h, class: "pending-rect",
      }));
    }
  }

  function setMode(next) {
    mode = next;
    modeButtons.forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  }
  modeButtons.forEach((b) => b.addEventListener("click", () => {
    setMode(b.dataset.mode);
    selected = null;
    rerender();
  }));

  function nextId(prefix, existing) {
    let i = 1;
    const ids = new Set(existing);
    while (ids.has(`${prefix}${i}`)) i++;
    return `${prefix}${i}`;
  }

  function hitTest(mx, my) {
    const cams = house.cameras || [];
    for (let i = cams.length - 1; i >= 0; i--) {
      const [cx, cy] = cams[i].position || [0, 0];
      const dx = mx - cx, dy = my - cy;
      if (dx * dx + dy * dy < 0.4) return { kind: "camera", index: i }; // ~0.6m radius
    }
    const rooms = house.rooms || [];
    for (let i = rooms.length - 1; i >= 0; i--) {
      const [rx, ry, rw, rh] = rooms[i].rect;
      if (mx >= rx && mx <= rx + rw && my >= ry && my <= ry + rh) {
        return { kind: "room", index: i };
      }
    }
    return null;
  }

  svg.addEventListener("mousedown", (e) => {
    const [mx, my] = MM.mouseToMeters(svg, e, house);
    if (mode === "room") {
      pendingRect = { x0: mx, y0: my, x1: mx, y1: my };
      rerender();
      return;
    }
    if (mode === "camera") {
      const url = window.prompt(
        "Camera stream — RTSP/HTTP URL or local webcam index (0, 1, ...):",
        "rtsp://"
      );
      if (url == null) return;
      let source = url.trim();
      if (/^\d+$/.test(source)) source = parseInt(source, 10);
      house.cameras = house.cameras || [];
      const id = nextId("cam", house.cameras.map((c) => c.id));
      house.cameras.push({
        id, name: id, source,
        position: [mx, my],
        heading: 0.0, fov: 90.0,
      });
      selected = { kind: "camera", index: house.cameras.length - 1 };
      setMode("select");
      rerender();
      return;
    }
    // select mode
    const hit = hitTest(mx, my);
    selected = hit;
    if (hit) {
      if (hit.kind === "camera") {
        const c = house.cameras[hit.index];
        dragging = { ...hit, anchorX: mx, anchorY: my, origPos: [...c.position] };
      } else if (hit.kind === "room") {
        const r = house.rooms[hit.index];
        dragging = { ...hit, anchorX: mx, anchorY: my, origRect: [...r.rect] };
      }
    }
    rerender();
  });

  svg.addEventListener("mousemove", (e) => {
    const [mx, my] = MM.mouseToMeters(svg, e, house);
    if (pendingRect) {
      pendingRect.x1 = mx; pendingRect.y1 = my;
      rerender();
      return;
    }
    if (!dragging) return;
    const dx = mx - dragging.anchorX;
    const dy = my - dragging.anchorY;
    if (dragging.kind === "camera") {
      house.cameras[dragging.index].position = [
        dragging.origPos[0] + dx,
        dragging.origPos[1] + dy,
      ];
    } else if (dragging.kind === "room") {
      const [ox, oy, ow, oh] = dragging.origRect;
      house.rooms[dragging.index].rect = [ox + dx, oy + dy, ow, oh];
    }
    rerender();
  });

  svg.addEventListener("mouseup", () => {
    if (pendingRect) {
      const x = Math.min(pendingRect.x0, pendingRect.x1);
      const y = Math.min(pendingRect.y0, pendingRect.y1);
      const w = Math.abs(pendingRect.x1 - pendingRect.x0);
      const h = Math.abs(pendingRect.y1 - pendingRect.y0);
      if (w > 0.3 && h > 0.3) {
        const name = window.prompt("Room name:", "Room") || "Room";
        const id = nextId(name.toLowerCase().replace(/\s+/g, "_"),
          (house.rooms || []).map((r) => r.id));
        house.rooms = house.rooms || [];
        house.rooms.push({ id, name, rect: [x, y, w, h] });
        selected = { kind: "room", index: house.rooms.length - 1 };
      }
      pendingRect = null;
      setMode("select");
      rerender();
      return;
    }
    dragging = null;
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (!selected) return;
    if (selected.kind === "camera") {
      const c = house.cameras[selected.index];
      if (e.key === "ArrowLeft") { c.heading = (c.heading || 0) - 5; rerender(); e.preventDefault(); }
      else if (e.key === "ArrowRight") { c.heading = (c.heading || 0) + 5; rerender(); e.preventDefault(); }
      else if (e.key === "ArrowUp") { c.fov = Math.min(170, (c.fov || 90) + 5); rerender(); e.preventDefault(); }
      else if (e.key === "ArrowDown") { c.fov = Math.max(20, (c.fov || 90) - 5); rerender(); e.preventDefault(); }
    }
    if (e.key === "Delete" || e.key === "Backspace") {
      deleteSelected(); e.preventDefault();
    }
  });

  function deleteSelected() {
    if (!selected) return;
    if (selected.kind === "camera") house.cameras.splice(selected.index, 1);
    else if (selected.kind === "room") house.rooms.splice(selected.index, 1);
    selected = null;
    rerender();
  }
  deleteBtn.addEventListener("click", deleteSelected);

  saveBtn.addEventListener("click", async () => {
    saveStatus.textContent = "saving…";
    try {
      const res = await fetch("/api/house", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(house),
      });
      const data = await res.json();
      if (data.ok) saveStatus.textContent = `saved → ${data.path}`;
      else saveStatus.textContent = `error: ${data.error || "unknown"}`;
    } catch (e) {
      saveStatus.textContent = `error: ${e.message}`;
    }
  });

  rerender();
})();
