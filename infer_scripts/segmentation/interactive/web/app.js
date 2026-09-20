(function () {
  "use strict";

  var palette = ["#d7ff4f", "#4fe5d2", "#ff8a3d", "#8c78ff", "#ff5f8f", "#5f8cff", "#ffc857", "#78e08f"];
  var floorColor = "#8c78ff";
  var state = {
    cases: [], caseId: null, caseInfo: null, review: null, image: null,
    objects: [], floor: null, selectedId: null, tool: "select",
    points: [], labels: [], box: null, boxStart: null, drawing: false,
    opacity: .58, brushSize: 24, candidates: [], candidateIndex: 0,
    undo: [], redo: [], reviewed: new Set(), zoom: 1, panX: 0, panY: 0,
    view: null, panning: false, panStart: null, spacePressed: false,
    cursorPoint: null, busy: false, predictionVersion: 0,
    graphPayload: null, graphOriginal: null, graphDirty: false, graphZoom: 1, graphZoomManual: false, graphPoll: null, graphPending: false, selectedGraphNodeId: null,
    automaticAvailable: true,
    graphResizeStart: null,
    estimateSceneGraph: false,
    canvasDrawFrame: null
  };

  var byId = function (id) { return document.getElementById(id); };
  var canvas = byId("scene-canvas");
  var ctx = canvas.getContext("2d", { willReadFrequently: true });
  var maskCanvas = byId("selected-mask-canvas");
  var maskCtx = maskCanvas.getContext("2d");
  var paintCursor = byId("paint-cursor");
  var selectedMaskCursor = byId("selected-mask-cursor");

  async function api(url, options) {
    var response = await fetch(url, options || {}), data = {};
    try { data = await response.json(); } catch (_) {}
    if (!response.ok) { var error = new Error(data.detail || data.error || ("HTTP " + response.status)); error.status = response.status; throw error; }
    return data;
  }

  function json(method, body) {
    return { method: method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) };
  }

  function toast(message, error) {
    var node = document.createElement("div");
    node.className = "toast" + (error ? " error" : "");
    node.textContent = message;
    byId("toasts").append(node);
    setTimeout(function () { node.remove(); }, 4500);
  }

  function setSystem(label, kind) {
    byId("system-state").textContent = label;
    byId("state-dot").className = "state-dot " + (kind || "");
  }

  function setBusy(busy, title, detail) {
    state.busy = busy;
    byId("run-overlay").classList.toggle("hidden", !busy);
    byId("case-upload-btn").classList.toggle("busy", busy);
    if (title) byId("run-title").textContent = title;
    if (detail) byId("run-detail").textContent = detail;
    byId("auto-segment").disabled = busy || !state.caseId || !state.automaticAvailable;
    byId("save-review").disabled = busy || !state.review;
    byId("toggle-add-object").disabled = busy || !state.review;
    byId("save-scene-graph").disabled = busy || !state.graphDirty;
  }

  function routeCase() { return new URLSearchParams(location.search).get("case"); }

  function updateUrl() {
    var url = state.caseId ? "?case=" + encodeURIComponent(state.caseId) : location.pathname;
    history.pushState({}, "", url);
  }

  function imageFrom(url) {
    return new Promise(function (resolve, reject) {
      var image = new Image();
      image.onload = function () { resolve(image); };
      image.onerror = function () { reject(new Error("Failed to load image")); };
      image.src = url;
    });
  }

  async function maskFrom(base64, width, height) {
    var image = await imageFrom("data:image/png;base64," + base64);
    var source = document.createElement("canvas"), result = document.createElement("canvas");
    source.width = result.width = width; source.height = result.height = height;
    var sourceContext = source.getContext("2d"), resultContext = result.getContext("2d");
    sourceContext.drawImage(image, 0, 0, width, height);
    var pixels = sourceContext.getImageData(0, 0, width, height);
    for (var i = 0; i < pixels.data.length; i += 4) {
      var active = pixels.data[i] > 127 || pixels.data[i + 3] < 255 && pixels.data[i + 3] > 127;
      pixels.data[i] = pixels.data[i + 1] = pixels.data[i + 2] = 255;
      pixels.data[i + 3] = active ? 255 : 0;
    }
    resultContext.putImageData(pixels, 0, 0);
    return result;
  }

  function maskBase64(mask) {
    var output = document.createElement("canvas");
    output.width = mask.width; output.height = mask.height;
    var outputContext = output.getContext("2d");
    outputContext.fillStyle = "#000"; outputContext.fillRect(0, 0, output.width, output.height);
    outputContext.drawImage(mask, 0, 0);
    return output.toDataURL("image/png").split(",")[1];
  }

  function blankMask(width, height) {
    var result = document.createElement("canvas"); result.width = width; result.height = height; return result;
  }

  function cloneCanvas(source) {
    var result = document.createElement("canvas"); result.width = source.width; result.height = source.height;
    result.getContext("2d").drawImage(source, 0, 0); return result;
  }

  function selected() {
    if (state.selectedId === "__floor__") return state.floor;
    return state.objects.find(function (item) { return item.id === state.selectedId; }) || null;
  }

  function selectedLabel() {
    var item = selected(); return item ? (item.isFloor ? "Floor mask" : (item.short_name || item.name || "Unnamed mask")) : "No mask selected";
  }

  function shortDisplayName(value) {
    var text = String(value || "").replace(/\s+/g, " ").trim().split(/[;,.:!?。；：！？]/, 1)[0].trim();
    return (text.split(" ").slice(0, 8).join(" ") || "object").slice(0, 80);
  }

  function scheduleCanvasDraw() {
    if (state.canvasDrawFrame !== null) cancelAnimationFrame(state.canvasDrawFrame);
    state.canvasDrawFrame = requestAnimationFrame(function () {
      state.canvasDrawFrame = null;
      draw();
    });
  }

  function setSelectedMaskPaneVisible(visible) {
    var workspace = byId("segmentation-workspace"), pane = document.querySelector(".selected-mask-pane");
    workspace.classList.toggle("loaded", Boolean(state.image)); workspace.classList.toggle("split", visible); pane.hidden = !visible;
    pane.setAttribute("aria-hidden", visible ? "false" : "true");
    scheduleCanvasDraw();
  }

  function setGraphPaneVisible(visible) {
    var workspace = byId("segmentation-workspace"), pane = document.querySelector(".scene-graph-pane"), divider = byId("scene-graph-divider");
    workspace.classList.toggle("graph-visible", Boolean(visible));
    if (!visible) workspace.style.gridTemplateRows = "";
    if (pane) pane.hidden = !visible;
    if (divider) divider.hidden = !visible;
    scheduleCanvasDraw();
  }

  async function refreshCases() {
    state.cases = await api("/api/cases", { cache: "no-store" });
    renderCases();
  }

  function renderCases() {
    var root = byId("case-list"); root.replaceChildren();
    byId("case-count").textContent = String(state.cases.length);
    if (!state.cases.length) { root.innerHTML = '<div class="empty-small">No preprocessed cases yet.</div>'; return; }
    state.cases.forEach(function (item) {
      var button = document.createElement("button"), image = document.createElement("img"), copy = document.createElement("span"), name = document.createElement("b"), status = document.createElement("i");
      button.type = "button"; button.className = "case-item"; button.dataset.caseId = item.id;
      button.setAttribute("role", "option"); button.setAttribute("aria-selected", item.id === state.caseId ? "true" : "false");
      image.loading = "lazy"; image.alt = item.id + " input preview"; image.src = item.image_url || ("/api/cases/" + encodeURIComponent(item.id) + "/image");
      image.onerror = function () { image.classList.add("load-error"); image.alt = "Preview unavailable"; };
      copy.className = "case-item-copy"; name.textContent = item.id; status.className = "case-ready" + (item.ready ? " ready" : ""); status.title = item.ready ? "Masks ready" : "Needs segmentation";
      copy.append(name, status); button.append(image, copy); button.onclick = function () { openCase(item.id, false); }; root.append(button);
    });
  }

  function resetEditor() {
    state.review = null; state.image = null; state.objects = []; state.floor = null; state.selectedId = null;
    state.reviewed = new Set(); state.undo = []; state.redo = []; state.view = null; state.graphPayload = null; state.graphOriginal = null; state.graphDirty = false; state.graphZoom = 1; state.graphZoomManual = false; state.graphPending = false; state.selectedGraphNodeId = null; state.estimateSceneGraph = false; byId("estimate-scene-graph").checked = false; if (state.graphPoll) { clearTimeout(state.graphPoll); state.graphPoll = null; } resetPrompts(); resetCanvasView();
    canvas.style.display = "none"; byId("empty-stage").classList.remove("hidden"); setSelectedMaskPaneVisible(false); setGraphPaneVisible(false);
    byId("object-list").innerHTML = '<div class="empty-small">Objects will appear here after segmentation</div>';
    byId("object-count").textContent = "0"; byId("review-progress").textContent = "0 / 0 reviewed";
    byId("canvas-meta").textContent = state.caseId || "No scene loaded"; byId("result-summary").textContent = state.caseId ? "Run segmentation to create editable masks." : "Select a case to begin.";
    syncHistory(); syncControls();
  }

  async function openCase(caseId, keepUrl) {
    state.predictionVersion += 1; state.caseId = caseId;
    state.caseInfo = state.cases.find(function (item) { return item.id === caseId; }) || { id: caseId, ready: false };
    resetEditor(); renderCases(); setSystem("Loading", "running");
    try {
      state.image = await imageFrom("/api/cases/" + encodeURIComponent(caseId) + "/image?v=" + Date.now());
      canvas.style.display = "block"; byId("empty-stage").classList.add("hidden"); setSelectedMaskPaneVisible(false); setGraphPaneVisible(false);
      byId("canvas-meta").textContent = caseId + " · " + state.image.naturalWidth + "×" + state.image.naturalHeight;
      // Every prepared image is editable.  The review endpoint lazily creates
      // an empty annotation for a case that has not been segmented yet.
      await loadReview();
      if (!keepUrl) updateUrl(); syncControls(); setSystem("Ready", "ready");
    } catch (error) { setSystem("Failed", "failed"); toast(error.message, true); }
  }

  async function loadReview() {
    var review = await api("/api/cases/" + encodeURIComponent(state.caseId) + "/review?v=" + Date.now(), { cache: "no-store" });
    var width = review.image && review.image.width || state.image.naturalWidth || 518;
    var height = review.image && review.image.height || state.image.naturalHeight || 518;
    var objects = [];
    for (var i = 0; i < review.objects.length; i++) {
      var object = Object.assign({}, review.objects[i]);
      object.short_name = shortDisplayName(object.short_name || object.name);
      object._canvas = await maskFrom(object.mask_b64, width, height); object._visible = true; object.isFloor = false; objects.push(object);
    }
    state.review = review; state.objects = objects;
    state.floor = { id: "__floor__", name: "Floor mask", isFloor: true, _visible: true, _canvas: await maskFrom(review.floor_mask_b64, width, height) };
    state.selectedId = objects.length ? objects[0].id : "__floor__"; state.reviewed = new Set(); state.undo = []; state.redo = [];
    setSelectedMaskPaneVisible(objects.length > 0); setGraphPaneVisible(false); resetCanvasView(); resetPrompts(); renderObjects(); syncHistory(); syncControls(); draw(); loadSceneGraph();
    byId("result-summary").textContent = objects.length ? (objects.length + " object masks + floor ready · revision " + review.revision) : "No masks yet · add an object interactively or run automatic segmentation.";
  }

  function syncControls() {
    byId("auto-segment").disabled = state.busy || !state.caseId || !state.automaticAvailable;
    byId("save-review").disabled = state.busy || !state.review;
    byId("toggle-add-object").disabled = state.busy || !state.review;
    if (byId("run-text-prompt")) byId("run-text-prompt").disabled = state.busy || !selected() || selected().isFloor;
    byId("save-scene-graph").disabled = state.busy || !state.graphDirty;
  }

  function selectLayer(id, preserveList) {
    state.selectedId = id; resetPrompts();
    if (preserveList) document.querySelectorAll(".object-item").forEach(function (row) { row.classList.toggle("selected", row.dataset.objectId === id); });
    else renderObjects();
    syncGraphSelectionFromLayer(); updateEditPanel(); draw();
  }

  function updateEditPanel() {
    var item = selected(), label = selectedLabel();
    byId("selected-mask-name").textContent = label; byId("edit-target").textContent = item ? "Editing: " + label : "No mask selected";
    byId("edit-hint").textContent = item ? (item.isFloor ? "The floor is saved separately from object masks." : "Selected mask is highlighted. Refine it with SAM3 or paint tools.") : "Select a mask, then refine it with SAM3 or paint tools.";
    byId("caption-editor").hidden = !item || item.isFloor; byId("caption-input").value = item && !item.isFloor ? (item.caption || "") : "";
    if (byId("text-prompt-editor")) byId("text-prompt-editor").hidden = !item || item.isFloor || state.tool !== "text";
    syncControls();
  }

  function renderObjects() {
    var root = byId("object-list"); root.replaceChildren(); byId("object-count").textContent = String(state.objects.length);
    byId("review-progress").textContent = state.reviewed.size + " / " + state.objects.length + " reviewed";
    if (!state.review) { root.innerHTML = '<div class="empty-small">Run segmentation to create editable masks.</div>'; updateEditPanel(); return; }
    state.objects.forEach(function (object, index) { root.append(layerRow(object, palette[index % palette.length])); });
    root.append(layerRow(state.floor, floorColor)); updateEditPanel();
  }

  function layerRow(item, color) {
    var row = document.createElement("div"), swatch = document.createElement("span"), copy = document.createElement("div"), input = document.createElement("input"), status = document.createElement("span"), actions = document.createElement("div"), visibility = document.createElement("button");
    row.className = "object-item" + (item.id === state.selectedId ? " selected" : "") + (item._visible === false ? " mask-hidden" : "") + (item.isFloor ? " floor-item" : ""); row.dataset.objectId = item.id;
    swatch.className = "swatch"; swatch.style.background = color; copy.className = "object-copy";
    input.className = "object-name"; input.value = item.isFloor ? "Floor mask" : (item.short_name || item.name); input.disabled = item.isFloor; input.setAttribute("aria-label", item.isFloor ? "Floor mask" : "Object short name");
    input.oninput = function () { item.short_name = input.value; item.name = input.value; updateEditPanel(); }; input.onclick = function (event) { event.stopPropagation(); }; input.onfocus = function () { if (state.selectedId !== item.id) selectLayer(item.id, true); };
    status.className = "object-status" + (item.isFloor ? " floor" : (state.reviewed.has(item.id) ? " done" : "")); status.textContent = item.isFloor ? "independent layer" : (state.reviewed.has(item.id) ? "reviewed" : (item.source || "mask")); copy.append(input, status);
    actions.className = "object-actions"; visibility.type = "button"; visibility.textContent = item._visible === false ? "○" : "◉"; visibility.title = item._visible === false ? "Show mask" : "Hide mask";
    visibility.onclick = function (event) { event.stopPropagation(); item._visible = item._visible === false; renderObjects(); draw(); }; actions.append(visibility);
    if (!item.isFloor) { var remove = document.createElement("button"); remove.type = "button"; remove.className = "delete"; remove.textContent = "×"; remove.title = "Delete mask"; remove.onclick = function (event) { event.stopPropagation(); state.objects = state.objects.filter(function (candidate) { return candidate !== item; }); state.reviewed.delete(item.id); if (state.selectedId === item.id) state.selectedId = state.objects[0] ? state.objects[0].id : "__floor__"; renderObjects(); draw(); }; actions.append(remove); }
    row.append(swatch, copy, actions); row.onclick = function (event) { if (event.target !== input) selectLayer(item.id); }; return row;
  }

  function resetPrompts() {
    state.predictionVersion += 1; state.points = []; state.labels = []; state.box = null; state.boxStart = null; state.drawing = false; state.candidates = []; state.candidateIndex = 0; renderCandidates(); updatePromptCounts();
  }

  function updatePromptCounts() {
    byId("positive-count").textContent = String(state.labels.filter(function (value) { return value === 1; }).length);
    byId("negative-count").textContent = String(state.labels.filter(function (value) { return value === 0; }).length);
    byId("box-count").textContent = state.box ? "1" : "0";
  }

  function renderCandidates() {
    var strip = byId("candidate-strip"), root = byId("candidate-buttons"); root.replaceChildren(); strip.classList.toggle("hidden", !state.candidates.length);
    byId("candidate-count").textContent = String(state.candidates.length); byId("candidate-active").textContent = state.candidates.length ? "selected " + (state.candidateIndex + 1) : "";
    state.candidates.forEach(function (candidate, index) { var button = document.createElement("button"); button.type = "button"; button.className = index === state.candidateIndex ? "active" : ""; button.textContent = String(index + 1); button.onclick = function () { var item = selected(); if (!item) return; state.candidateIndex = index; item._canvas = cloneCanvas(candidate); renderCandidates(); draw(); }; root.append(button); });
  }

  function fitView() {
    if (!state.image) return null;
    var rect = canvas.getBoundingClientRect(), width = state.review && state.review.image && state.review.image.width || state.image.naturalWidth, height = state.review && state.review.image && state.review.image.height || state.image.naturalHeight, margin = 24;
    var scale = Math.min(Math.max(1, rect.width - margin * 2) / width, Math.max(1, rect.height - margin * 2) / height) * state.zoom;
    return { x: (rect.width - width * scale) / 2 + state.panX, y: (rect.height - height * scale) / 2 + state.panY, width: width * scale, height: height * scale, scale: scale, sourceWidth: width, sourceHeight: height };
  }

  function resetCanvasView() { state.zoom = 1; state.panX = 0; state.panY = 0; state.view = null; byId("reset-zoom").textContent = "100%"; draw(); }
  function setCanvasZoom(next, anchorX, anchorY) {
    if (!state.image) return; var rect = canvas.getBoundingClientRect(), current = state.view || fitView(), x = anchorX === undefined ? rect.width / 2 : anchorX, y = anchorY === undefined ? rect.height / 2 : anchorY;
    var imageX = (x - current.x) / current.scale, imageY = (y - current.y) / current.scale; state.zoom = Math.max(.5, Math.min(8, next)); var future = fitView();
    state.panX += x - (future.x + imageX * future.scale); state.panY += y - (future.y + imageY * future.scale); byId("reset-zoom").textContent = Math.round(state.zoom * 100) + "%"; draw();
  }

  function point(event) {
    if (!state.view) return null; var rect = canvas.getBoundingClientRect(), x = (event.clientX - rect.left - state.view.x) / state.view.scale, y = (event.clientY - rect.top - state.view.y) / state.view.scale;
    return x < 0 || y < 0 || x >= state.view.sourceWidth || y >= state.view.sourceHeight ? null : [x, y];
  }

  function tint(mask, color) { var result = document.createElement("canvas"); result.width = mask.width; result.height = mask.height; var context = result.getContext("2d"); context.fillStyle = color; context.fillRect(0, 0, result.width, result.height); context.globalCompositeOperation = "destination-in"; context.drawImage(mask, 0, 0); return result; }

  function sizeCanvas(target, context) {
    var rect = target.getBoundingClientRect(), ratio = window.devicePixelRatio || 1, width = Math.max(1, Math.round(rect.width * ratio)), height = Math.max(1, Math.round(rect.height * ratio));
    if (target.width !== width || target.height !== height) { target.width = width; target.height = height; } context.setTransform(ratio, 0, 0, ratio, 0, 0); return rect;
  }

  function draw() {
    drawSelectedMask(); if (!state.image || canvas.style.display === "none") return;
    var rect = sizeCanvas(canvas, ctx); ctx.clearRect(0, 0, rect.width, rect.height); state.view = fitView(); var view = state.view; if (!view) return;
    ctx.drawImage(state.image, view.x, view.y, view.width, view.height);
    (state.floor ? [state.floor].concat(state.objects) : state.objects).forEach(function (item) { if (item._visible === false) return; var objectIndex = state.objects.indexOf(item), color = item.isFloor ? floorColor : palette[objectIndex % palette.length]; ctx.save(); ctx.globalAlpha = item.id === state.selectedId ? Math.min(1, state.opacity + .17) : state.opacity; ctx.drawImage(tint(item._canvas, color), view.x, view.y, view.width, view.height); ctx.restore(); });
    state.points.forEach(function (position, index) { var x = view.x + position[0] * view.scale, y = view.y + position[1] * view.scale; ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fillStyle = state.labels[index] ? "#4fe5d2" : "#ff5f68"; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = "#0b0d10"; ctx.stroke(); ctx.fillStyle = "#0b0d10"; ctx.font = "bold 10px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(state.labels[index] ? "+" : "−", x, y + .5); });
    if (state.box) { ctx.save(); ctx.strokeStyle = "#d7ff4f"; ctx.lineWidth = 2; ctx.setLineDash([7, 5]); ctx.strokeRect(view.x + state.box[0] * view.scale, view.y + state.box[1] * view.scale, (state.box[2] - state.box[0]) * view.scale, (state.box[3] - state.box[1]) * view.scale); ctx.restore(); }
    updatePaintCursor();
  }

  function selectedMaskView() {
    var item = selected(); if (!state.review || !item) return null; var rect = maskCanvas.getBoundingClientRect(), margin = 24, width = item._canvas.width, height = item._canvas.height, scale = Math.min(Math.max(1, rect.width - margin * 2) / width, Math.max(1, rect.height - margin * 2) / height);
    return { x: (rect.width - width * scale) / 2, y: (rect.height - height * scale) / 2, width: width * scale, height: height * scale, scale: scale };
  }

  function drawSelectedMask() {
    var rect = sizeCanvas(maskCanvas, maskCtx); maskCtx.fillStyle = "#000"; maskCtx.fillRect(0, 0, rect.width, rect.height); var item = selected(), empty = byId("selected-mask-empty");
    byId("selected-mask-name").textContent = selectedLabel(); empty.classList.toggle("hidden", Boolean(item)); if (item) { var view = selectedMaskView(); maskCtx.drawImage(tint(item._canvas, "#fff"), view.x, view.y, view.width, view.height); } updateSelectedMaskCursor();
  }

  function updatePaintCursor() {
    var visible = Boolean(state.cursorPoint && state.view && !state.panning && (state.tool === "brush" || state.tool === "erase")); paintCursor.hidden = !visible; canvas.classList.toggle("has-paint-cursor", visible);
    if (visible) { paintCursor.style.left = (state.view.x + state.cursorPoint[0] * state.view.scale) + "px"; paintCursor.style.top = (state.view.y + state.cursorPoint[1] * state.view.scale) + "px"; paintCursor.style.width = state.brushSize + "px"; paintCursor.style.height = state.brushSize + "px"; paintCursor.classList.toggle("erase", state.tool === "erase"); } updateSelectedMaskCursor();
  }

  function updateSelectedMaskCursor() {
    var active = ["positive", "negative", "box", "brush", "erase"].includes(state.tool), visible = Boolean(active && state.cursorPoint && state.review && selected() && !state.panning), view = visible && selectedMaskView(); selectedMaskCursor.hidden = !visible;
    if (!visible || !view) return; selectedMaskCursor.className = "selected-mask-cursor cursor-" + state.tool; selectedMaskCursor.style.left = (view.x + state.cursorPoint[0] * view.scale) + "px"; selectedMaskCursor.style.top = (view.y + state.cursorPoint[1] * view.scale) + "px";
    var diameter = state.tool === "brush" || state.tool === "erase" ? Math.max(4, state.brushSize / Math.max(.01, state.view ? state.view.scale : view.scale) * view.scale) : 18; selectedMaskCursor.style.width = diameter + "px"; selectedMaskCursor.style.height = diameter + "px";
  }

  function pushUndo() { var item = selected(); if (!item) return; state.undo.push({ id: item.id, canvas: cloneCanvas(item._canvas) }); if (state.undo.length > 50) state.undo.shift(); state.redo = []; syncHistory(); }
  function syncHistory() { byId("undo-paint").disabled = !state.undo.length; byId("redo-paint").disabled = !state.redo.length; }
  function restoreHistory(from, to) { var snapshot = from.pop(); if (!snapshot) return; var item = snapshot.id === "__floor__" ? state.floor : state.objects.find(function (object) { return object.id === snapshot.id; }); if (!item) { syncHistory(); return; } to.push({ id: item.id, canvas: cloneCanvas(item._canvas) }); item._canvas = snapshot.canvas; if (state.selectedId !== item.id) selectLayer(item.id); syncHistory(); draw(); }
  function paint(position) { var item = selected(); if (!item || !position) return; var context = item._canvas.getContext("2d"), radius = state.brushSize / (2 * Math.max(.01, state.view.scale)); context.save(); context.globalCompositeOperation = state.tool === "erase" ? "destination-out" : "source-over"; context.fillStyle = "#fff"; context.beginPath(); context.arc(position[0], position[1], radius, 0, Math.PI * 2); context.fill(); context.restore(); draw(); }

  async function predict() {
    var item = selected(); if (!item || !state.review || state.busy) return; var version = ++state.predictionVersion, selectedId = item.id; setSystem("SAM3 running", "running");
    try {
      var payload = { points: state.points, labels: state.labels, box: state.box, mask_b64: maskBase64(item._canvas) };
      var response = await api("/api/cases/" + encodeURIComponent(state.caseId) + "/predict", json("POST", payload));
      if (version !== state.predictionVersion || state.selectedId !== selectedId) return; var candidates = [];
      for (var i = 0; i < (response.masks || []).length; i++) candidates.push(await maskFrom(response.masks[i], item._canvas.width, item._canvas.height));
      if (version !== state.predictionVersion || state.selectedId !== selectedId) return; state.candidates = candidates; state.candidateIndex = response.best_idx || 0;
      if (candidates.length) { pushUndo(); item._canvas = cloneCanvas(candidates[state.candidateIndex]); } renderCandidates(); draw(); setSystem("Ready", "ready");
    } catch (error) { if (version === state.predictionVersion) { setSystem("Runtime unavailable", "failed"); toast(error.message, true); } }
  }

  async function runTextPrompt() {
    var item = selected(), prompt = byId("layer-text-prompt").value.trim();
    if (!item || item.isFloor || !prompt || state.busy) return;
    var version = ++state.predictionVersion, selectedId = item.id; setSystem("SAM3 running", "running");
    try {
      var response = await api("/api/cases/" + encodeURIComponent(state.caseId) + "/predict", json("POST", { prompt: prompt, mask_b64: maskBase64(item._canvas) }));
      if (version !== state.predictionVersion || state.selectedId !== selectedId) return;
      state.candidates = []; for (var i = 0; i < (response.masks || []).length; i++) state.candidates.push(await maskFrom(response.masks[i], item._canvas.width, item._canvas.height));
      state.candidateIndex = response.best_idx || 0; if (state.candidates.length) { pushUndo(); item._canvas = cloneCanvas(state.candidates[state.candidateIndex]); }
      renderCandidates(); draw(); setSystem("Ready", "ready");
    } catch (error) { if (version === state.predictionVersion) { setSystem("Runtime unavailable", "failed"); toast(error.message, true); } }
  }

  async function addText() {
    var prompt = byId("text-prompt").value.trim(); if (!prompt || !state.review || state.busy) return; setSystem("SAM3 running", "running");
    try { var response = await api("/api/cases/" + encodeURIComponent(state.caseId) + "/predict", json("POST", { prompt: prompt })); if (!response.masks || !response.masks.length) throw new Error("SAM3 returned no masks"); var id = "object-" + Date.now(), width = state.floor._canvas.width, height = state.floor._canvas.height; state.objects.push({ id: id, name: prompt, caption: prompt, source: "sam3-text", _visible: true, isFloor: false, _canvas: await maskFrom(response.masks[response.best_idx || 0], width, height) }); state.selectedId = id; byId("text-prompt").value = ""; renderObjects(); draw(); setSystem("Ready", "ready"); } catch (error) { setSystem("Runtime unavailable", "failed"); toast(error.message, true); }
  }

  function addEmpty() { if (!state.review) return; var id = "object-" + Date.now(); state.objects.push({ id: id, name: "new object", caption: "", source: "sam3-prompt", _visible: true, isFloor: false, _canvas: blankMask(state.floor._canvas.width, state.floor._canvas.height) }); state.selectedId = id; renderObjects(); setTool("positive"); }

  async function saveReview() {
    if (!state.review || state.busy) return; var objects = state.objects.map(function (item) { return { id: item.id, name: (item.short_name || item.name).trim(), short_name: (item.short_name || item.name).trim(), caption: (item.caption || item.name).trim(), name_locked: true, caption_locked: true, source: "human-review", mask_b64: maskBase64(item._canvas) }; });
    setBusy(true, "Saving edited masks", "Publishing a new review revision…"); setSystem("Saving", "running");
    try { state.review = await api("/api/cases/" + encodeURIComponent(state.caseId) + "/review", json("PUT", { objects: objects, floor_mask_b64: maskBase64(state.floor._canvas), estimate_scene_graph: byId("estimate-scene-graph").checked })); state.graphPending = byId("estimate-scene-graph").checked; toast("Saved review revision " + state.review.revision + (state.graphPending ? " · estimating scene graph" : "")); await refreshCases(); state.caseInfo = state.cases.find(function (item) { return item.id === state.caseId; }); await loadReview(); setSystem("Ready", "ready"); } catch (error) { setSystem("Save failed", "failed"); toast(error.message, true); } finally { setBusy(false); }
  }

  async function autoSegment() {
    if (!state.caseId || state.busy) return; if (state.review && !window.confirm("Rerun automatic segmentation? The current review will be archived as a revision.")) return;
    setBusy(true, "Running segmentation", "This can take several minutes. Please keep this page open…"); setSystem("Segmenting", "running");
    try { await api("/api/cases/" + encodeURIComponent(state.caseId) + "/automatic", { method: "POST" }); state.graphPending = true; await refreshCases(); state.caseInfo = state.cases.find(function (item) { return item.id === state.caseId; }); await loadReview(); toast("Automatic segmentation completed"); setSystem("Ready", "ready"); } catch (error) { setSystem("Segmentation failed", "failed"); toast(error.message, true); } finally { setBusy(false); }
  }

  function cloneGraph(value) { return JSON.parse(JSON.stringify(value)); }
  function graphDirty(value) { state.graphDirty = value; byId("save-scene-graph").disabled = !value; byId("reset-scene-graph").disabled = !value; }
  function markHumanEdge(edge) { if (edge) edge.confidence = 1; return edge; }
  function graphLayout(nodes, edges) {
    var children = new Map(), depth = new Map(); nodes.forEach(function (n) { children.set(n.id, []); }); edges.forEach(function (e) { if (e.parent && children.has(e.parent)) children.get(e.parent).push(e.child); });
    function visit(id, seen) { if (depth.has(id)) return depth.get(id); if (seen.has(id)) return 0; seen.add(id); var edge = edges.find(function (e) { return e.child === id && e.parent; }); var d = edge ? visit(edge.parent, seen) + 1 : 0; depth.set(id, d); return d; }
    nodes.forEach(function (n) { visit(n.id, new Set()); }); var groups = new Map(); depth.forEach(function (d, id) { if (!groups.has(d)) groups.set(d, []); groups.get(d).push(nodes.find(function (n) { return n.id === id; })); }); return { groups: groups, maxDepth: Math.max.apply(null, Array.from(groups.keys()).concat([0])) };
  }
  function drawGraph(host, payload) {
    var nodes = payload.nodes || [], byNode = new Map(nodes.map(function (n) { return [n.id, n]; })), edges = (payload.edges || []).filter(function (e) { return byNode.has(e.child) && byNode.has(e.parent); });
    if (!nodes.length) { host.innerHTML = '<div class="viewer-empty"><strong>No graph nodes available</strong><span>scene_graph.json contains no nodes.</span></div>'; return; }
    var layout = graphLayout(nodes, edges), nodeW = 104, nodeH = 36, hGap = 24, vGap = 52, padX = 28, padY = 20, max = Math.max.apply(null, Array.from(layout.groups.values()).map(function (g) { return g.length; })), width = Math.max(360, padX * 2 + max * nodeW + Math.max(0, max - 1) * hGap), height = Math.max(190, padY * 2 + (layout.maxDepth + 1) * nodeH + layout.maxDepth * vGap), pos = new Map();
    layout.groups.forEach(function (group, d) { var total = group.length * nodeW + Math.max(0, group.length - 1) * hGap, start = (width - total) / 2, y = padY + (layout.maxDepth - d) * (nodeH + vGap) + nodeH / 2; group.forEach(function (n, i) { pos.set(n.id, { x: start + i * (nodeW + hGap) + nodeW / 2, y: y }); }); });
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); svg.setAttribute("viewBox", "0 0 " + width + " " + height); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Scene support graph. Supporters are below the objects they support."); svg.classList.add("scene-graph-svg"); var defs = document.createElementNS(svg.namespaceURI, "defs"), marker = document.createElementNS(svg.namespaceURI, "marker"), arrow = document.createElementNS(svg.namespaceURI, "path"); marker.id = "support-arrow"; marker.setAttribute("viewBox", "0 0 10 10"); marker.setAttribute("refX", "9"); marker.setAttribute("refY", "5"); marker.setAttribute("markerWidth", "5"); marker.setAttribute("orient", "auto-start-reverse"); arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 z"); arrow.setAttribute("fill", "#8d98a5"); arrow.setAttribute("stroke", "#8d98a5"); marker.append(arrow); defs.append(marker); svg.append(defs);
    var edgeGroup = document.createElementNS(svg.namespaceURI, "g"); edgeGroup.classList.add("scene-graph-edges"); edges.forEach(function (e) { var from = pos.get(e.child), to = pos.get(e.parent), x1 = from.x, y1 = from.y + nodeH / 2, x2 = to.x, y2 = to.y - nodeH / 2, curve = Math.max(22, Math.abs(y2 - y1) * .42), path = document.createElementNS(svg.namespaceURI, "path"); path.setAttribute("d", "M " + x1 + " " + y1 + " C " + x1 + " " + (y1 + curve) + ", " + x2 + " " + (y2 - curve) + ", " + x2 + " " + y2); path.setAttribute("marker-end", "url(#support-arrow)"); path.classList.toggle("non-operational", e.operational === false); var title = document.createElementNS(svg.namespaceURI, "title"); title.textContent = (byNode.get(e.child).name || e.child) + " → " + (byNode.get(e.parent).name || e.parent) + " · " + (e.relation || "supported by") + (Number.isFinite(e.confidence) ? " · " + Math.round(e.confidence * 100) + "%" : ""); path.append(title); edgeGroup.append(path); }); svg.append(edgeGroup);
    var group = document.createElementNS(svg.namespaceURI, "g"); nodes.forEach(function (n) { var p = pos.get(n.id), g = document.createElementNS(svg.namespaceURI, "g"), rect = document.createElementNS(svg.namespaceURI, "rect"), label = document.createElementNS(svg.namespaceURI, "text"), meta = document.createElementNS(svg.namespaceURI, "text"), color = n.kind === "object" ? palette[(n.mask_index || 0) % palette.length] : floorColor, name = shortDisplayName(n.short_name || n.name || n.id).replace(/_/g, " "); g.setAttribute("transform", "translate(" + (p.x - nodeW / 2) + " " + (p.y - nodeH / 2) + ")"); g.classList.add("scene-graph-node"); if (n.kind !== "object") g.classList.add("environment-node"); if (n.id === state.selectedGraphNodeId) g.classList.add("selected"); if (n.kind === "object") { g.setAttribute("role", "button"); g.onclick = function () { state.selectedGraphNodeId = n.id; var object = state.objects.find(function (item) { return item.id !== "__floor__" && state.objects.indexOf(item) === Number(n.mask_index); }); if (object) selectLayer(object.id); else if (n.id === "floor") selectLayer("__floor__"); drawGraph(host, payload); renderGraphEditor(); }; } else if (n.id === "floor") { g.setAttribute("role", "button"); g.onclick = function () { state.selectedGraphNodeId = n.id; selectLayer("__floor__"); drawGraph(host, payload); renderGraphEditor(); }; } rect.setAttribute("width", nodeW); rect.setAttribute("height", nodeH); rect.setAttribute("rx", "7"); rect.setAttribute("fill", n.kind === "object" ? color : "#171c22"); rect.setAttribute("fill-opacity", n.kind === "object" ? ".22" : "1"); rect.setAttribute("stroke", color); label.setAttribute("x", nodeW / 2); label.setAttribute("y", "15"); label.textContent = name.length > 15 ? name.slice(0, 14) + "…" : name; meta.setAttribute("x", nodeW / 2); meta.setAttribute("y", "28"); meta.classList.add("scene-graph-node-meta"); meta.textContent = n.kind === "object" ? "MASK " + String(n.mask_index).padStart(3, "0") : (n.id === "floor" ? "SUPPORT SURFACE" : "STATIC ANCHOR"); g.append(rect, label, meta); group.append(g); }); svg.append(group); var legend = document.createElement("div"); legend.className = "scene-graph-legend"; legend.textContent = "arrows show direct support"; host.replaceChildren(svg, legend); updateGraphZoom(true); }
  function syncGraphSelectionFromLayer() { if (!state.graphPayload) return; if (state.selectedId === "__floor__") state.selectedGraphNodeId = "floor"; else { var index = state.objects.findIndex(function (item) { return item.id === state.selectedId; }); var node = (state.graphPayload.nodes || []).find(function (item) { return item.kind === "object" && Number(item.mask_index) === index; }); state.selectedGraphNodeId = node ? node.id : null; } drawGraph(byId("stage-graph"), state.graphPayload); }
  function updateGraphZoom(autoFit) { var svg = byId("stage-graph").querySelector("svg"); if (svg && autoFit && !state.graphZoomManual) { var host = byId("stage-graph"), viewBox = svg.viewBox.baseVal, available = Math.max(1, host.clientHeight - 12), natural = Math.max(1, viewBox.height); state.graphZoom = Math.max(.5, Math.min(2, available / natural)); } if (svg) { svg.style.transform = "scale(" + state.graphZoom + ")"; svg.style.transformOrigin = "top center"; } byId("graph-zoom-reset").textContent = Math.round(state.graphZoom * 100) + "%"; }
  function renderGraphEditor() { var host = byId("scene-graph-editor"), payload = state.graphPayload, node = payload && (payload.nodes || []).find(function (n) { return n.id === state.selectedGraphNodeId; }); if (!node || node.kind !== "object") { host.innerHTML = "<span>Select an object node to edit its support relationship.</span>"; return; } var edge = (payload.edges || []).find(function (e) { return e.child === node.id; }); host.innerHTML = ""; function field(title, control) { var label = document.createElement("label"); label.innerHTML = "<span>" + title + "</span>"; label.append(control); host.append(label); } var name = document.createElement("input"); name.value = shortDisplayName(node.short_name || node.name || node.id); name.onchange = function () { node.short_name = shortDisplayName(name.value); node.name = node.short_name; var object = state.objects[Number(node.mask_index)]; if (object) { object.short_name = node.short_name; object.name = node.short_name; } graphDirty(true); renderObjects(); drawGraph(byId("stage-graph"), payload); }; field("Name", name); var relation = document.createElement("select"); ["rests_on", "fixed_to", "supported_by", "hangs_from", "unknown"].forEach(function (v) { relation.append(new Option(v.replace(/_/g, " "), v)); }); relation.value = edge && edge.relation || "rests_on"; relation.onchange = function () { if (edge) { edge.relation = relation.value; markHumanEdge(edge); } else { edge = { child: node.id, parent: null, relation: relation.value, confidence: 1, operational: false }; payload.edges.push(edge); } graphDirty(true); }; field("Relation", relation); var support = document.createElement("select"); support.append(new Option("Unknown / none", "")); (payload.nodes || []).filter(function (n) { return n.id !== node.id; }).forEach(function (n) { support.append(new Option(String(n.short_name || n.name || n.id).replace(/_/g, " "), n.id)); }); support.value = edge && edge.parent || ""; support.onchange = function () { if (edge) { edge.parent = support.value || null; edge.operational = Boolean(edge.parent); markHumanEdge(edge); } else if (support.value) { edge = { child: node.id, parent: support.value, relation: relation.value, confidence: 1, operational: true }; payload.edges.push(edge); } graphDirty(true); drawGraph(byId("stage-graph"), payload); }; field("Direct Supporter", support); var orientation = document.createElement("select"); [["auto", "Auto (from support)"], ["force", "Always upright"], ["free", "Allow free orientation"]].forEach(function (x) { orientation.append(new Option(x[1], x[0])); }); orientation.value = node.upright_mode || "auto"; orientation.onchange = function () { node.upright_mode = orientation.value; graphDirty(true); drawGraph(byId("stage-graph"), payload); }; field("Orientation", orientation); }
  async function loadSceneGraph() { var host = byId("stage-graph"), status = byId("graph-status"); if (!state.caseId || !state.review) return; if (state.graphPoll) clearTimeout(state.graphPoll); status.textContent = "Loading…"; try { var payload = await api("/api/cases/" + encodeURIComponent(state.caseId) + "/scene-graph?v=" + Date.now(), { cache: "no-store" }); state.graphPayload = payload; state.graphOriginal = cloneGraph(payload); state.graphZoomManual = false; var graphStatus = payload.generation_status || "ready"; state.graphPending = graphStatus === "estimating"; graphDirty(false); setGraphPaneVisible(true); syncGraphSelectionFromLayer(); renderGraphEditor(); status.textContent = graphStatus === "stale" ? "Outdated · masks changed" : graphStatus === "failed" ? "Estimation failed · previous graph" : graphStatus === "estimating" ? "Estimating…" : "Ready"; if (state.graphPending) state.graphPoll = setTimeout(loadSceneGraph, 3000); } catch (error) { if (error.status === 404) { state.graphPayload = null; state.graphOriginal = null; setGraphPaneVisible(false); host.innerHTML = '<div class="viewer-empty"><strong>Scene graph unavailable</strong><span>Check Estimate Scene Graph when saving masks to create one.</span></div>'; status.textContent = "Not generated"; if (state.graphPending) { api("/api/cases/" + encodeURIComponent(state.caseId) + "/scene-graph/status", { cache: "no-store" }).then(function (graphState) { if (graphState.status === "estimating") { status.textContent = "Estimating…"; state.graphPoll = setTimeout(loadSceneGraph, 3000); } else { state.graphPending = false; status.textContent = graphState.status === "failed" ? "Estimation failed" : "Not generated"; } }).catch(function () { state.graphPending = false; }); } return; } setGraphPaneVisible(true); host.innerHTML = '<div class="viewer-empty"><strong>Scene graph generating…</strong><span>Retrying automatically.</span></div>'; status.textContent = "Waiting…"; state.graphPoll = setTimeout(loadSceneGraph, 3000); } }
  async function saveSceneGraph() { if (!state.graphPayload || !state.graphDirty || state.busy) return; setBusy(true, "Saving scene graph", "Publishing manual graph edits…"); setSystem("Saving graph", "running"); try { var saved = await api("/api/cases/" + encodeURIComponent(state.caseId) + "/scene-graph", json("PUT", state.graphPayload)); state.graphPayload = saved; state.graphOriginal = cloneGraph(saved); graphDirty(false); toast("Scene graph saved"); setSystem("Ready", "ready"); } catch (error) { setSystem("Graph save failed", "failed"); toast(error.message, true); } finally { setBusy(false); } }

  async function upload(file) {
    if (!file || state.busy) return; var form = new FormData(); form.append("file", file); setBusy(true, "Preparing image", "Adding the image to the case list…"); setSystem("Preparing", "running");
    try { var response = await api("/api/cases", { method: "POST", body: form }); await refreshCases(); await openCase(response.id, false); toast("Added " + file.name + ". Choose interactive or automatic segmentation."); } catch (error) { setSystem("Upload failed", "failed"); toast(error.message, true); } finally { byId("case-upload").value = ""; setBusy(false); }
  }

  function setTool(tool) { state.tool = tool; state.cursorPoint = null; document.querySelectorAll(".tool").forEach(function (button) { button.classList.toggle("active", button.dataset.tool === tool); }); byId("brush-size-control").hidden = tool !== "brush" && tool !== "erase"; if (byId("text-prompt-editor")) byId("text-prompt-editor").hidden = tool !== "text" || !selected() || selected().isFloor; ["select", "positive", "negative", "box", "text", "brush", "erase"].forEach(function (name) { canvas.classList.toggle("tool-" + name, name === tool); }); updatePaintCursor(); requestAnimationFrame(draw); }

  canvas.onpointerdown = function (event) {
    if ((state.spacePressed || event.button === 1) && state.image) { event.preventDefault(); state.panning = true; state.panStart = { x: event.clientX, y: event.clientY, panX: state.panX, panY: state.panY }; canvas.classList.add("panning"); canvas.setPointerCapture(event.pointerId); updatePaintCursor(); return; }
    if (!state.review) return; var position = point(event), item = selected(); if (!position) return; state.cursorPoint = position; updatePaintCursor();
    if (state.tool === "positive" || state.tool === "negative") { state.points.push(position); state.labels.push(state.tool === "positive" ? 1 : 0); updatePromptCounts(); draw(); predict(); }
    else if (state.tool === "box") { state.boxStart = position; state.box = [position[0], position[1], position[0], position[1]]; state.drawing = true; canvas.setPointerCapture(event.pointerId); draw(); }
    else if (state.tool === "brush" || state.tool === "erase") { if (!item) return; pushUndo(); state.drawing = true; paint(position); canvas.setPointerCapture(event.pointerId); }
    else if (state.tool === "select") { var layers = state.floor ? [state.floor].concat(state.objects) : state.objects; for (var i = layers.length - 1; i >= 0; i--) { if (layers[i]._visible === false) continue; var pixel = layers[i]._canvas.getContext("2d").getImageData(Math.floor(position[0]), Math.floor(position[1]), 1, 1).data; if (pixel[3] > 20) { selectLayer(layers[i].id); break; } } }
  };
  canvas.onpointermove = function (event) { if (state.panning && state.panStart) { state.panX = state.panStart.panX + event.clientX - state.panStart.x; state.panY = state.panStart.panY + event.clientY - state.panStart.y; draw(); return; } var position = point(event); state.cursorPoint = position; updatePaintCursor(); byId("cursor-meta").textContent = position ? "x " + Math.round(position[0]) + " · y " + Math.round(position[1]) : "x — · y —"; if (!position) return; if (state.drawing && state.tool === "box") { state.box = [Math.min(state.boxStart[0], position[0]), Math.min(state.boxStart[1], position[1]), Math.max(state.boxStart[0], position[0]), Math.max(state.boxStart[1], position[1])]; draw(); } else if (state.drawing && (state.tool === "brush" || state.tool === "erase")) paint(position); };
  canvas.onpointerup = function (event) { if (state.panning) { state.panning = false; state.panStart = null; canvas.classList.remove("panning"); updatePaintCursor(); return; } if (state.tool === "box" && state.boxStart) { var position = point(event); if (position) state.box = [Math.min(state.boxStart[0], position[0]), Math.min(state.boxStart[1], position[1]), Math.max(state.boxStart[0], position[0]), Math.max(state.boxStart[1], position[1])]; if (state.box && (state.box[2] - state.box[0] < 2 || state.box[3] - state.box[1] < 2)) state.box = null; state.boxStart = null; state.drawing = false; updatePromptCounts(); draw(); if (state.box) predict(); } state.drawing = false; };
  canvas.onpointercancel = function () { state.drawing = false; state.boxStart = null; state.panning = false; state.panStart = null; state.cursorPoint = null; canvas.classList.remove("panning"); updatePaintCursor(); };
  canvas.onpointerleave = function () { state.cursorPoint = null; updatePaintCursor(); };
  canvas.onwheel = function (event) { if (!state.image) return; event.preventDefault(); var rect = canvas.getBoundingClientRect(); setCanvasZoom(state.zoom * (event.deltaY < 0 ? 1.16 : 1 / 1.16), event.clientX - rect.left, event.clientY - rect.top); };

  document.querySelectorAll(".tool").forEach(function (button) { button.onclick = function () { setTool(button.dataset.tool); }; });
  byId("opacity").oninput = function () { state.opacity = Number(this.value) / 100; draw(); };
  byId("brush-size").oninput = function () { state.brushSize = Number(this.value); byId("brush-size-value").textContent = this.value + " px"; updatePaintCursor(); };
  byId("undo-paint").onclick = function () { restoreHistory(state.undo, state.redo); }; byId("redo-paint").onclick = function () { restoreHistory(state.redo, state.undo); };
  byId("clear-prompts").onclick = function () { resetPrompts(); draw(); }; byId("save-review").onclick = saveReview; byId("save-scene-graph").onclick = saveSceneGraph; byId("estimate-scene-graph").onchange = function () { state.estimateSceneGraph = this.checked; }; byId("auto-segment").onclick = autoSegment;
  byId("run-text-prompt").onclick = runTextPrompt; byId("layer-text-prompt").onkeydown = function (event) { if (event.key === "Enter") { event.preventDefault(); runTextPrompt(); } };
  byId("toggle-add-object").onclick = function () { var panel = byId("add-object-panel"), show = panel.hidden; panel.hidden = !show; this.setAttribute("aria-expanded", show ? "true" : "false"); };
  byId("add-by-text").onclick = addText; byId("text-prompt").onkeydown = function (event) { if (event.key === "Enter") addText(); }; byId("add-prompt-mask").onclick = addEmpty;
  byId("caption-input").oninput = function () { var item = selected(); if (item && !item.isFloor) item.caption = this.value; };
  byId("zoom-out").onclick = function () { setCanvasZoom(state.zoom / 1.25); }; byId("zoom-in").onclick = function () { setCanvasZoom(state.zoom * 1.25); }; byId("reset-zoom").onclick = resetCanvasView;
  byId("case-upload-btn").onclick = function () { if (!state.busy) byId("case-upload").click(); }; byId("case-upload-btn").onkeydown = function (event) { if ((event.key === "Enter" || event.key === " ") && !state.busy) { event.preventDefault(); byId("case-upload").click(); } }; byId("case-upload").onchange = function (event) { upload(event.target.files[0]); };
  byId("case-upload-btn").ondragover = function (event) { event.preventDefault(); if (!state.busy) this.classList.add("dragging"); }; byId("case-upload-btn").ondragleave = function () { this.classList.remove("dragging"); }; byId("case-upload-btn").ondrop = function (event) { event.preventDefault(); this.classList.remove("dragging"); upload(event.dataTransfer.files[0]); };
  window.addEventListener("keydown", function (event) { var tag = document.activeElement && document.activeElement.tagName; if (event.code === "Space" && !/INPUT|TEXTAREA/.test(tag)) { state.spacePressed = true; event.preventDefault(); } if (event.key === "Enter" && !/INPUT|TEXTAREA|SELECT|BUTTON/.test(tag) && selected()) { if (!selected().isFloor) state.reviewed.add(selected().id); resetPrompts(); renderObjects(); draw(); toast("Layer confirmed"); } }); window.addEventListener("keyup", function (event) { if (event.code === "Space") state.spacePressed = false; }); window.addEventListener("resize", function () { scheduleCanvasDraw(); if (!state.graphZoomManual) updateGraphZoom(true); });
  if (window.ResizeObserver) {
    var canvasResizeObserver = new ResizeObserver(scheduleCanvasDraw);
    canvasResizeObserver.observe(document.querySelector(".segmentation-canvas-surface"));
    canvasResizeObserver.observe(document.querySelector(".segmentation-mask-surface"));
  }
  byId("graph-zoom-out").onclick = function () { state.graphZoomManual = true; state.graphZoom = Math.max(.5, state.graphZoom - .1); updateGraphZoom(); }; byId("graph-zoom-in").onclick = function () { state.graphZoomManual = true; state.graphZoom = Math.min(2, state.graphZoom + .1); updateGraphZoom(); }; byId("graph-zoom-reset").onclick = function () { state.graphZoomManual = false; state.graphZoom = 1; updateGraphZoom(true); }; byId("reset-scene-graph").onclick = function () { state.graphPayload = cloneGraph(state.graphOriginal); state.graphZoomManual = false; graphDirty(false); drawGraph(byId("stage-graph"), state.graphPayload); renderGraphEditor(); };
  var divider = byId("scene-graph-divider");
  function setGraphRows(clientY) { var workspace = byId("segmentation-workspace"), rect = workspace.getBoundingClientRect(), y = Math.max(180, Math.min(rect.height - 180, clientY - rect.top)), dividerHeight = 10, graphHeight = rect.height - y - dividerHeight; workspace.style.gridTemplateRows = y + "px " + dividerHeight + "px " + Math.max(170, graphHeight) + "px"; updateGraphZoom(true); draw(); }
  divider.onpointerdown = function (event) { state.graphResizeStart = true; document.body.classList.add("graph-resizing"); divider.setPointerCapture(event.pointerId); };
  divider.onpointermove = function (event) { if (state.graphResizeStart) setGraphRows(event.clientY); };
  divider.onpointerup = divider.onpointercancel = function () { state.graphResizeStart = false; document.body.classList.remove("graph-resizing"); };
  divider.onkeydown = function (event) { var workspace = byId("segmentation-workspace"), current = parseFloat(getComputedStyle(workspace).gridTemplateRows.split(" ")[0]) || workspace.clientHeight * .65; if (event.key === "ArrowUp") { event.preventDefault(); setGraphRows(workspace.getBoundingClientRect().top + current - 20); } if (event.key === "ArrowDown") { event.preventDefault(); setGraphRows(workspace.getBoundingClientRect().top + current + 20); } };
  window.onpopstate = function () { var requested = routeCase(); if (requested) openCase(requested, true); else { state.caseId = null; resetEditor(); renderCases(); } };

  (async function boot() { try { var capabilities = await api("/api/capabilities", { cache: "no-store" }); state.automaticAvailable = capabilities.automatic_segmentation !== false; byId("auto-segment").title = state.automaticAvailable ? "Run automatic segmentation" : "Automatic segmentation is not installed"; await refreshCases(); var requested = routeCase(); if (requested && state.cases.some(function (item) { return item.id === requested; })) await openCase(requested, true); else { resetEditor(); renderCases(); setSystem("Select a case", ""); } } catch (error) { setSystem("Failed", "failed"); toast(error.message, true); } })();
})();
