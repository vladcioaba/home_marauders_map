// Calibration page: click matching points on the frame and on the floor plan.

(async function () {
  const shell = document.querySelector(".calibrate-shell");
  const camId = shell.dataset.camId;
  const imgEl = document.getElementById("frame");
  const overlay = document.getElementById("frame-overlay");
  const plan = document.getElementById("plan");
  const nextTargetEl = document.getElementById("next-target");
  const pairCountEl = document.getElementById("pair-count");
  const saveStatus = document.getElementById("save-status");
  const frameStatus = document.getElementById("frame-status");

  const house = await (await fetch("/api/house")).json();

  const pairs = [];        // [{ image: [x,y], floor: [x,y] }]
  let partial = null;       // { image: [x,y] } when we're waiting for floor click
  let nextTarget = "image"; // "image" | "floor"

  function setFrameOverlayBox() {
    if (!imgEl.naturalWidth) return;
    overlay.setAttribute(
      "viewBox", `0 0 ${imgEl.naturalWidth} ${imgEl.naturalHeight}`,
    );
  }

  function redraw() {
    nextTargetEl.textContent = nextTarget;
    pairCountEl.textContent = `${pairs.length} pair${pairs.length === 1 ? "" : "s"}`;

    setFrameOverlayBox();
    while (overlay.firstChild) overlay.removeChild(overlay.firstChild);
    pairs.forEach((p, i) => {
      overlay.appendChild(MM.svgEl("circle", {
        cx: p.image[0], cy: p.image[1], r: 9,
        fill: "rgba(42,74,138,0.65)", stroke: "white", "stroke-width": 2,
      }));
      overlay.appendChild(MM.svgEl("text", {
        x: p.image[0] + 14, y: p.image[1] + 6,
        fill: "white", "font-size": 18,
        "font-family": "monospace", "font-weight": "bold",
        stroke: "black", "stroke-width": 0.6,
      }, [document.createTextNode(String(i + 1))]));
    });
    if (partial && partial.image) {
      overlay.appendChild(MM.svgEl("circle", {
        cx: partial.image[0], cy: partial.image[1], r: 7,
        fill: "yellow", stroke: "black", "stroke-width": 1.5,
      }));
    }

    MM.renderHouse(plan, house);
    const s = house.floorplan?.scale || 60;
    pairs.forEach((p, i) => {
      plan.appendChild(MM.svgEl("circle", {
        cx: p.floor[0] * s, cy: p.floor[1] * s, r: 6,
        fill: "rgba(42,74,138,0.75)", stroke: "white", "stroke-width": 1.5,
      }));
      plan.appendChild(MM.svgEl("text", {
        x: p.floor[0] * s + 10, y: p.floor[1] * s + 5,
        fill: "white", "font-size": 13,
        "font-family": "monospace", "font-weight": "bold",
        stroke: "black", "stroke-width": 0.5,
      }, [document.createTextNode(String(i + 1))]));
    });
  }

  imgEl.addEventListener("click", (e) => {
    if (nextTarget !== "image") return;
    if (!imgEl.naturalWidth) return;
    const rect = imgEl.getBoundingClientRect();
    const sx = imgEl.naturalWidth / rect.width;
    const sy = imgEl.naturalHeight / rect.height;
    const x = (e.clientX - rect.left) * sx;
    const y = (e.clientY - rect.top) * sy;
    partial = { image: [x, y] };
    nextTarget = "floor";
    redraw();
  });

  plan.addEventListener("click", (e) => {
    if (nextTarget !== "floor" || !partial) return;
    const [mx, my] = MM.mouseToMeters(plan, e, house);
    pairs.push({ image: partial.image, floor: [mx, my] });
    partial = null;
    nextTarget = "image";
    redraw();
  });

  document.getElementById("refresh").addEventListener("click", () => {
    frameStatus.textContent = "loading…";
    imgEl.src = `/api/frame/${encodeURIComponent(camId)}.jpg?t=${Date.now()}`;
  });

  document.getElementById("undo").addEventListener("click", () => {
    if (partial) { partial = null; nextTarget = "image"; }
    else if (pairs.length) pairs.pop();
    redraw();
  });

  document.getElementById("clear").addEventListener("click", () => {
    pairs.length = 0; partial = null; nextTarget = "image";
    redraw();
  });

  document.getElementById("save").addEventListener("click", async () => {
    if (pairs.length < 4) {
      saveStatus.textContent = `need ${4 - pairs.length} more pair(s)`;
      return;
    }
    saveStatus.textContent = "saving…";
    try {
      const res = await fetch("/api/calibration", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cam_id: camId,
          image_points: pairs.map((p) => p.image),
          floor_points: pairs.map((p) => p.floor),
        }),
      });
      const data = await res.json();
      if (data.ok) {
        saveStatus.textContent = `saved ${data.pairs} pairs → ${data.path}. Restart serve to apply.`;
      } else {
        saveStatus.textContent = `error: ${data.error || "unknown"}`;
      }
    } catch (e) {
      saveStatus.textContent = `error: ${e.message}`;
    }
  });

  imgEl.addEventListener("load", () => {
    frameStatus.textContent = `${imgEl.naturalWidth} × ${imgEl.naturalHeight}`;
    redraw();
  });
  imgEl.addEventListener("error", () => {
    frameStatus.textContent = "no live frame — start `marauders serve --live`, then refresh";
  });

  if (imgEl.complete) {
    if (imgEl.naturalWidth) {
      frameStatus.textContent = `${imgEl.naturalWidth} × ${imgEl.naturalHeight}`;
    }
    redraw();
  }
})();
