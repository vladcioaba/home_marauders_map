// Shared SVG renderer for the floor plan. Both view.js and edit.js call
// renderHouse() with the current house + (for view) live markers/trails.
//
// Coordinate system: house data is in meters, scale = pixels per meter.

function svgEl(name, attrs = {}, children = []) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  for (const c of children) el.appendChild(c);
  return el;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function planSize(house) {
  const [w, h] = house.floorplan?.size || [10, 6];
  const s = house.floorplan?.scale || 60;
  return { w, h, s, wpx: Math.round(w * s), hpx: Math.round(h * s) };
}

function setPlanViewBox(svg, house) {
  const { wpx, hpx } = planSize(house);
  svg.setAttribute("viewBox", `0 0 ${wpx} ${hpx}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  // Style a sensible max width via attribute; CSS handles the rest.
  svg.style.maxWidth = `${wpx}px`;
}

function renderHouse(svg, house, {
  selected = null,         // { kind: "room"|"camera", index: number } | null
  markers = [],
  trails = {},
  showCalibrationHint = true,
} = {}) {
  clear(svg);
  setPlanViewBox(svg, house);
  const { w, h, s } = planSize(house);

  const layerRooms = svgEl("g", { class: "layer-rooms" });
  const layerDoors = svgEl("g", { class: "layer-doors" });
  const layerCams = svgEl("g", { class: "layer-cams" });
  const layerLive = svgEl("g", { class: "layer-live" });
  svg.appendChild(layerRooms);
  svg.appendChild(layerDoors);
  svg.appendChild(layerCams);
  svg.appendChild(layerLive);

  (house.rooms || []).forEach((r, i) => {
    const [rx, ry, rw, rh] = r.rect;
    const sel = selected && selected.kind === "room" && selected.index === i;
    layerRooms.appendChild(svgEl("rect", {
      x: rx * s, y: ry * s, width: rw * s, height: rh * s,
      class: "room-rect" + (sel ? " selected" : ""),
      "data-kind": "room", "data-index": i,
    }));
    layerRooms.appendChild(svgEl("text", {
      x: rx * s + 8, y: ry * s + 18, class: "room-label",
    }, [document.createTextNode(r.name || r.id)]));
  });

  (house.doors || []).forEach((d) => {
    const [[ax, ay], [bx, by]] = d.segment;
    layerDoors.appendChild(svgEl("line", {
      x1: ax * s, y1: ay * s, x2: bx * s, y2: by * s,
      class: "door-line",
    }));
    layerDoors.appendChild(svgEl("line", {
      x1: ax * s, y1: ay * s, x2: bx * s, y2: by * s,
      class: "door-line-overlay",
    }));
  });

  (house.cameras || []).forEach((c, i) => {
    const [cx, cy] = c.position || [0, 0];
    const heading = (c.heading || 0) * Math.PI / 180;
    const fov = (c.fov || 90) * Math.PI / 180;
    const px = cx * s, py = cy * s;
    const length = 1.4 * s;

    // FOV cone (two rays + arc)
    const a1 = heading - fov / 2;
    const a2 = heading + fov / 2;
    const x1 = px + Math.cos(a1) * length, y1 = py + Math.sin(a1) * length;
    const x2 = px + Math.cos(a2) * length, y2 = py + Math.sin(a2) * length;
    layerCams.appendChild(svgEl("path", {
      d: `M ${px} ${py} L ${x1} ${y1} A ${length} ${length} 0 0 1 ${x2} ${y2} Z`,
      class: "cam-cone",
    }));

    const sel = selected && selected.kind === "camera" && selected.index === i;
    layerCams.appendChild(svgEl("circle", {
      cx: px, cy: py, r: 6,
      class: "cam-dot" + (sel ? " selected" : ""),
      "data-kind": "camera", "data-index": i,
    }));
    layerCams.appendChild(svgEl("text", {
      x: px + 10, y: py - 8, class: "cam-label",
    }, [document.createTextNode(c.id)]));
  });

  // Trails (sepia polylines for now — JS-side footprint stamping is overkill).
  Object.entries(trails).forEach(([gid, pts]) => {
    if (!pts || pts.length < 2) return;
    const points = pts.map(([x, y]) => `${x * s},${y * s}`).join(" ");
    layerLive.appendChild(svgEl("polyline", {
      points, class: "trail-line",
    }));
  });

  // Live markers (current footprint pair + name label).
  markers.forEach((m) => {
    const px = m.x * s, py = m.y * s;
    layerLive.appendChild(svgEl("ellipse", {
      cx: px - 5, cy: py, rx: 5, ry: 3, class: "marker-foot",
    }));
    layerLive.appendChild(svgEl("ellipse", {
      cx: px + 5, cy: py, rx: 5, ry: 3, class: "marker-foot",
    }));
    const label = m.room ? `${m.name} @ ${m.room}` : m.name;
    layerLive.appendChild(svgEl("text", {
      x: px + 14, y: py + 5, class: "marker-label",
    }, [document.createTextNode(label)]));
  });
}

// Coordinate helpers — map mouse events to floor-plan meters.
function mouseToMeters(svg, evt, house) {
  const pt = svg.createSVGPoint();
  pt.x = evt.clientX; pt.y = evt.clientY;
  const ctm = svg.getScreenCTM();
  if (!ctm) return [0, 0];
  const local = pt.matrixTransform(ctm.inverse());
  const s = house.floorplan?.scale || 60;
  return [local.x / s, local.y / s];
}

window.MM = { renderHouse, mouseToMeters, svgEl, planSize };
