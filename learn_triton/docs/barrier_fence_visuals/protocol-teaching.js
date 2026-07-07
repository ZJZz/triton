(function () {
  const page = window.PROTOCOL_PAGE;
  if (!page) return;

  const state = {
    step: 0,
    timer: null,
  };

  const heroEl = document.getElementById("hero");
  const journeyEl = document.getElementById("journey");
  const topologyEl = document.getElementById("topology");
  const scopeEl = document.getElementById("scope");
  const timelineEl = document.getElementById("timeline");
  const currentStepEl = document.getElementById("current-step");
  const stepListEl = document.getElementById("step-list");
  const notesEl = document.getElementById("notes");
  const sliderEl = document.getElementById("step-slider");
  const counterEl = document.getElementById("step-counter");
  const playEl = document.getElementById("play-toggle");
  const prevEl = document.getElementById("step-prev");
  const nextEl = document.getElementById("step-next");

  document.title = page.title;

  renderHero();
  renderNotes();
  bindControls();
  update();

  function bindControls() {
    sliderEl.max = String(page.steps.length - 1);
    sliderEl.addEventListener("input", () => {
      stopPlayback();
      state.step = Number(sliderEl.value);
      update();
    });
    playEl.addEventListener("click", togglePlayback);
    prevEl.addEventListener("click", () => shiftStep(-1));
    nextEl.addEventListener("click", () => shiftStep(1));
  }

  function togglePlayback() {
    if (state.timer) {
      stopPlayback();
      return;
    }
    playEl.textContent = "暂停";
    playEl.classList.add("btn-primary");
    state.timer = window.setInterval(() => {
      if (state.step >= page.steps.length - 1) {
        stopPlayback();
        return;
      }
      state.step += 1;
      update();
    }, page.playIntervalMs || 1700);
  }

  function stopPlayback() {
    if (!state.timer) return;
    window.clearInterval(state.timer);
    state.timer = null;
    playEl.textContent = "播放";
    playEl.classList.remove("btn-primary");
  }

  function shiftStep(delta) {
    stopPlayback();
    state.step = clamp(state.step + delta, 0, page.steps.length - 1);
    update();
  }

  function update() {
    const step = page.steps[state.step];
    sliderEl.value = String(state.step);
    counterEl.textContent = `${state.step + 1} / ${page.steps.length}`;
    renderJourney(step);
    renderTopology(step);
    renderScope(step);
    renderTimeline(step);
    renderCurrentStep(step);
    renderStepList();
  }

  function renderHero() {
    const links = page.links || [
      { href: "./protocol-overview.html", label: "返回协议总览" },
      { href: "./README.md", label: "查看目录说明" },
    ];
    heroEl.innerHTML = `
      <div class="hero-top">
        <div>
          <div class="eyebrow">${escapeHtml(page.eyebrow)}</div>
          <h1>${escapeHtml(page.title)}</h1>
          <p class="hero-copy">${escapeHtml(page.subtitle)}</p>
        </div>
        <div class="hero-links">
          ${links.map((link) => `<a class="hero-link" href="${link.href}">${escapeHtml(link.label)}</a>`).join("")}
        </div>
      </div>
      <div class="chip-row">
        ${page.chips.map((chip) => `<div class="chip">${chip}</div>`).join("")}
      </div>
      <div class="quick-grid">
        ${page.quickFacts.map((fact) => `
          <div class="fact-card">
            <div class="fact-label">${escapeHtml(fact.label)}</div>
            <div class="fact-value">${escapeHtml(fact.value)}</div>
            <div class="fact-note">${escapeHtml(fact.note)}</div>
          </div>
        `).join("")}
      </div>
    `;
  }

  function renderJourney(step) {
    const cfg = page.journey;
    const nodeMap = Object.fromEntries(cfg.nodes.map((node) => [node.id, node]));
    const activeNodes = new Set(step.activeNodes || []);
    const activeArrows = new Set(step.activeArrows || []);
    const svg = [];
    svg.push(`<svg viewBox="0 0 ${cfg.width} ${cfg.height}" role="img" aria-label="${escapeHtml(cfg.ariaLabel || page.title)}">`);
    svg.push(defs());

    for (const region of cfg.regions || []) {
      svg.push(regionSvg(region, step));
    }

    for (const arrow of cfg.arrows) {
      svg.push(arrowSvg(arrow, nodeMap, activeArrows.has(arrow.id)));
    }

    for (const node of cfg.nodes) {
      svg.push(nodeSvg(node, activeNodes.has(node.id)));
    }

    for (const note of cfg.notes || []) {
      svg.push(`
        <text x="${note.x}" y="${note.y}" fill="#5d6d84" font-size="13" font-weight="700">${escapeHtml(note.text)}</text>
      `);
    }

    svg.push(`</svg>`);
    journeyEl.innerHTML = `
      <div class="legend-row">
        ${page.legend.map((item) => `
          <div class="legend-chip">
            <span class="legend-swatch" style="background:${item.color}"></span>
            <span>${escapeHtml(item.label)}</span>
          </div>
        `).join("")}
      </div>
      <div class="journey-wrap">${svg.join("")}</div>
    `;
  }

  function renderScope(step) {
    const activeRows = new Set(step.activeScopeRows || []);
    scopeEl.innerHTML = `
      <div class="scope-summary">${escapeHtml(page.scope.summary)}</div>
      <div class="scope-table">
        ${page.scope.rows.map((row) => `
          <div class="scope-row ${activeRows.has(row.id) ? "active" : ""}">
            <div class="scope-name">${escapeHtml(row.scope)}</div>
            <div><span class="scope-role role-${row.roleClass}">${escapeHtml(row.role)}</span></div>
            <div class="scope-detail">${escapeHtml(row.detail)}</div>
          </div>
        `).join("")}
      </div>
      <div class="scope-callouts">
        ${page.scope.callouts.map((callout) => `
          <div class="scope-callout">
            <strong>${escapeHtml(callout.title)}</strong>
            <p>${escapeHtml(callout.body)}</p>
          </div>
        `).join("")}
      </div>
    `;
  }

  function renderTopology(step) {
    if (!topologyEl || !page.topology) return;
    const cfg = page.topology;
    const nodeMap = Object.fromEntries(cfg.modules.map((node) => [node.id, node]));
    const activeNodes = new Set([...(step.activeNodes || []), ...(step.activeTopologyNodes || [])]);
    const activeArrows = new Set([...(step.activeArrows || []), ...(step.activeTopologyArrows || [])]);
    const activeBands = new Set(step.activeTopologyLayers || []);
    const svg = [];
    svg.push(`<svg viewBox="0 0 ${cfg.width} ${cfg.height}" role="img" aria-label="${escapeHtml(cfg.ariaLabel || `${page.title} topology`)}">`);
    svg.push(defs());

    for (const band of cfg.bands || []) {
      svg.push(topologyBandSvg(band, activeBands.has(band.id), cfg.width));
    }

    for (const frame of cfg.frames || []) {
      svg.push(regionSvg(frame, { activeRegions: activeBands.has(frame.id) ? [frame.id] : [] }));
    }

    for (const arrow of cfg.arrows) {
      svg.push(arrowSvg(arrow, nodeMap, activeArrows.has(arrow.id)));
    }

    for (const module of cfg.modules) {
      svg.push(nodeSvg(module, activeNodes.has(module.id)));
    }

    for (const note of cfg.notes || []) {
      svg.push(`
        <text x="${note.x}" y="${note.y}" fill="#5d6d84" font-size="13" font-weight="700">${escapeHtml(note.text)}</text>
      `);
    }

    svg.push(`</svg>`);
    topologyEl.innerHTML = `
      <div class="legend-row">
        ${(cfg.legend || page.legend).map((item) => `
          <div class="legend-chip">
            <span class="legend-swatch" style="background:${item.color}"></span>
            <span>${escapeHtml(item.label)}</span>
          </div>
        `).join("")}
      </div>
      <div class="topology-wrap">${svg.join("")}</div>
      <div class="topology-caption">${escapeHtml(cfg.caption || "")}</div>
    `;
  }

  function renderTimeline(step) {
    const cfg = page.timeline;
    const tickCount = cfg.ticks.length;
    const left = 140;
    const top = 58;
    const laneGap = 68;
    const rightPad = 48;
    const bottomPad = 42;
    const innerWidth = 120 * (tickCount - 1) + 40;
    const width = left + innerWidth + rightPad;
    const height = top + laneGap * cfg.lanes.length + bottomPad;
    const tickX = (index) => left + (innerWidth / Math.max(tickCount - 1, 1)) * index;

    const svg = [];
    svg.push(`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(cfg.ariaLabel || `${page.title} timeline`)}">`);
    svg.push(defs());
    svg.push(`<rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="#ffffff"/>`);

    cfg.ticks.forEach((tick, index) => {
      const x = tickX(index);
      svg.push(`<line x1="${x}" y1="${top - 16}" x2="${x}" y2="${height - 26}" stroke="#d7dfec" stroke-width="1.5"/>`);
      svg.push(`<text x="${x}" y="26" text-anchor="middle" fill="#334155" font-size="12" font-weight="700">${escapeHtml(tick.label)}</text>`);
      if (tick.note) {
        svg.push(`<text x="${x}" y="42" text-anchor="middle" fill="#8291a7" font-size="10.5">${escapeHtml(tick.note)}</text>`);
      }
    });

    cfg.lanes.forEach((lane, laneIndex) => {
      const y = top + laneIndex * laneGap;
      svg.push(`<rect x="18" y="${y - 26}" width="${width - 36}" height="52" rx="14" fill="${lane.bg || "#f6f9fe"}" stroke="#e2e8f0"/>`);
      svg.push(`<text x="32" y="${y - 4}" fill="#0f172a" font-size="13" font-weight="800">${escapeHtml(lane.label)}</text>`);
      svg.push(`<text x="32" y="${y + 14}" fill="#64748b" font-size="11.5">${escapeHtml(lane.note || "")}</text>`);
    });

    for (const bar of cfg.bars) {
      const laneIndex = cfg.lanes.findIndex((lane) => lane.id === bar.lane);
      const y = top + laneIndex * laneGap - 16 + (bar.offsetY || 0);
      const x = tickX(bar.start);
      const w = tickX(bar.end) - tickX(bar.start);
      const status = step.tick >= bar.end ? "done" : (step.tick >= bar.start ? "active" : "future");
      const style = toneStyle(bar.tone, status);
      svg.push(`
        <rect x="${x}" y="${y}" width="${Math.max(w, 16)}" height="${bar.height || 32}" rx="12"
          fill="${style.fill}" stroke="${style.stroke}" stroke-width="${style.width}"
          ${status === "active" ? `filter="url(#focus-shadow)"` : ""}/>
      `);
      svg.push(`<text x="${x + 12}" y="${y + 14}" fill="${style.text}" font-size="11.5" font-weight="800">${escapeHtml(bar.label)}</text>`);
      if (bar.sub) {
        svg.push(`<text x="${x + 12}" y="${y + 27}" fill="${style.sub}" font-size="10.5">${escapeHtml(bar.sub)}</text>`);
      }
    }

    for (const marker of cfg.markers || []) {
      const laneIndex = cfg.lanes.findIndex((lane) => lane.id === marker.lane);
      const x = tickX(marker.at);
      const y = top + laneIndex * laneGap + 24 + (marker.offsetY || 0);
      const style = markerStyle(marker.tone, step.tick >= marker.at, step.tick === marker.at);
      svg.push(markerSvg(marker.kind, x, y, style));
      svg.push(`<text x="${x + 12}" y="${y + 4}" fill="#475569" font-size="10.5" font-weight="700">${escapeHtml(marker.label)}</text>`);
    }

    const cursorX = tickX(step.tick);
    svg.push(`<line x1="${cursorX}" y1="${top - 18}" x2="${cursorX}" y2="${height - 20}" stroke="#2563eb" stroke-width="3" stroke-linecap="round"/>`);
    svg.push(`<circle cx="${cursorX}" cy="${top - 18}" r="7" fill="#2563eb"/>`);
    svg.push(`</svg>`);

    timelineEl.innerHTML = `
      <div class="timeline-wrap">${svg.join("")}</div>
      <div class="timeline-caption">${escapeHtml(cfg.caption)}</div>
    `;
  }

  function renderCurrentStep(step) {
    currentStepEl.innerHTML = `
      <div class="current-step">
        <div>
          <div class="eyebrow">当前步骤</div>
          <h2>${escapeHtml(step.title)}</h2>
        </div>
        <p>${escapeHtml(step.summary)}</p>
        <div class="step-meta">
          ${step.meta.map((item) => `<div class="step-pill">${escapeHtml(item)}</div>`).join("")}
        </div>
      </div>
    `;
  }

  function renderStepList() {
    stepListEl.innerHTML = page.steps.map((step, index) => `
      <button class="step-item ${index === state.step ? "active" : ""}" data-step="${index}" type="button">
        <strong>${escapeHtml(step.short)}</strong>
        <span>${escapeHtml(step.summary)}</span>
      </button>
    `).join("");
    stepListEl.querySelectorAll("[data-step]").forEach((button) => {
      button.addEventListener("click", () => {
        stopPlayback();
        state.step = Number(button.getAttribute("data-step"));
        update();
      });
    });
  }

  function renderNotes() {
    notesEl.innerHTML = `
      <div class="notes-grid">
        ${page.notes.map((note) => `
          <div class="note-card">
            <h3>${escapeHtml(note.title)}</h3>
            ${note.body ? `<p>${escapeHtml(note.body)}</p>` : ""}
            ${note.items ? `<ul>${note.items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}
          </div>
        `).join("")}
      </div>
      <div class="footer-note">${escapeHtml(page.footerNote)}</div>
    `;
  }

  function nodeSvg(node, active) {
    const palette = nodePalette(node.tone);
    const outerStroke = active ? "#2563eb" : palette.stroke;
    const outerWidth = active ? 3 : 2;
    const glow = active ? `filter="url(#focus-shadow)"` : "";
    const labelY = node.y + 22;
    const subY = labelY + 18;
    const metaY = node.y + node.h - 10;
    return `
      <g ${glow}>
        <rect x="${node.x}" y="${node.y}" width="${node.w}" height="${node.h}" rx="20"
          fill="${palette.fill}" stroke="${outerStroke}" stroke-width="${outerWidth}"/>
        <text x="${node.x + 16}" y="${labelY}" fill="#0f172a" font-size="13.5" font-weight="800">${escapeHtml(node.label)}</text>
        <text x="${node.x + 16}" y="${subY}" fill="#475569" font-size="11.5">${escapeHtml(node.sub)}</text>
        <text x="${node.x + 16}" y="${metaY}" fill="${palette.stroke}" font-size="10.8" font-weight="700">${escapeHtml(node.meta)}</text>
      </g>
    `;
  }

  function arrowSvg(arrow, nodeMap, active) {
    const fromNode = nodeMap[arrow.from];
    const toNode = nodeMap[arrow.to];
    const start = anchor(fromNode, arrow.fromSide || "right");
    const end = anchor(toNode, arrow.toSide || "left");
    const points = [start].concat(arrow.via || [], [end]);
    const d = pathFromPoints(points);
    const palette = arrowPalette(arrow.tone, active);
    const label = arrow.labelAt || midpoint(points);
    return `
      <g>
        <path d="${d}" fill="none" stroke="${palette.stroke}" stroke-width="${palette.width}" stroke-linecap="round" stroke-linejoin="round"
          ${palette.dash ? `stroke-dasharray="${palette.dash}"` : ""} marker-end="url(#arrow-${palette.marker})"/>
        <rect x="${label.x - label.w / 2}" y="${label.y - 12}" width="${label.w}" height="24" rx="12" fill="#ffffff" opacity="0.92"/>
        <text x="${label.x}" y="${label.y + 4}" fill="${palette.stroke}" font-size="11.5" font-weight="800" text-anchor="middle">${escapeHtml(arrow.label)}</text>
      </g>
    `;
  }

  function regionSvg(region, step) {
    const active = (step.activeRegions || []).includes(region.id);
    return `
      <g>
        <rect x="${region.x}" y="${region.y}" width="${region.w}" height="${region.h}" rx="26"
          fill="${region.fill}" stroke="${active ? "#2563eb" : region.stroke}" stroke-width="${active ? 2.5 : 2}"
          stroke-dasharray="${region.dash || "8 7"}"/>
        <text x="${region.x + 16}" y="${region.y + 24}" fill="${active ? "#2563eb" : region.stroke}" font-size="12.5" font-weight="800">${escapeHtml(region.label)}</text>
      </g>
    `;
  }

  function topologyBandSvg(band, active, width) {
    return `
      <g>
        <rect x="${band.x || 0}" y="${band.y}" width="${band.w || width}" height="${band.h}" rx="${band.r || 20}"
          fill="${band.fill}" stroke="${active ? "#2563eb" : band.stroke}" stroke-width="${active ? 2.4 : 1.4}" />
        <text x="${(band.x || 0) + 18}" y="${band.y + 24}" fill="${active ? "#2563eb" : "#334155"}" font-size="13" font-weight="800">${escapeHtml(band.label)}</text>
        ${band.sub ? `<text x="${(band.x || 0) + 18}" y="${band.y + 42}" fill="#64748b" font-size="11.5">${escapeHtml(band.sub)}</text>` : ""}
      </g>
    `;
  }

  function defs() {
    return `
      <defs>
        <marker id="arrow-data" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#4f83db"/>
        </marker>
        <marker id="arrow-sync" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#b7791f"/>
        </marker>
        <marker id="arrow-publish" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#6c43d6"/>
        </marker>
        <marker id="arrow-control" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f8194"/>
        </marker>
        <marker id="arrow-hazard" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#ba3e2f"/>
        </marker>
        <filter id="focus-shadow" x="-30%" y="-30%" width="160%" height="160%">
          <feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#7ca7ff" flood-opacity="0.35"/>
        </filter>
      </defs>
    `;
  }

  function markerSvg(kind, x, y, style) {
    if (kind === "wait") {
      return `<polygon points="${x},${y + 8} ${x - 8},${y - 6} ${x + 8},${y - 6}" fill="${style.fill}" stroke="${style.stroke}" stroke-width="2"/>`;
    }
    if (kind === "fence") {
      return `<rect x="${x - 7}" y="${y - 7}" width="14" height="14" rx="3" fill="${style.fill}" stroke="${style.stroke}" stroke-width="2"/>`;
    }
    if (kind === "hazard") {
      return `<polygon points="${x},${y - 9} ${x + 9},${y + 8} ${x - 9},${y + 8}" fill="${style.fill}" stroke="${style.stroke}" stroke-width="2"/>`;
    }
    if (kind === "commit") {
      return `
        <rect x="${x - 8}" y="${y - 8}" width="7" height="16" rx="3" fill="${style.fill}" stroke="${style.stroke}" stroke-width="2"/>
        <rect x="${x + 1}" y="${y - 8}" width="7" height="16" rx="3" fill="${style.fill}" stroke="${style.stroke}" stroke-width="2"/>
      `;
    }
    return `<circle cx="${x}" cy="${y}" r="7" fill="${style.fill}" stroke="${style.stroke}" stroke-width="2"/>`;
  }

  function toneStyle(tone, status) {
    const base = nodePalette(tone);
    if (status === "active") {
      return { fill: base.fill, stroke: "#2563eb", width: 2.6, text: "#0f172a", sub: "#475569" };
    }
    if (status === "done") {
      return { fill: base.fill, stroke: base.stroke, width: 1.8, text: "#0f172a", sub: "#64748b" };
    }
    return { fill: "#f8fafc", stroke: "#cbd5e1", width: 1.5, text: "#94a3b8", sub: "#a0aec0" };
  }

  function markerStyle(tone, passed, current) {
    const palette = nodePalette(tone);
    if (current) return { fill: "#ffffff", stroke: "#2563eb" };
    if (passed) return { fill: palette.fill, stroke: palette.stroke };
    return { fill: "#ffffff", stroke: "#cbd5e1" };
  }

  function nodePalette(tone) {
    switch (tone) {
      case "source": return { fill: "#dcecff", stroke: "#1d5fd1" };
      case "engine": return { fill: "#daf8ea", stroke: "#0f8a5f" };
      case "output": return { fill: "#ece5ff", stroke: "#6c43d6" };
      case "sync": return { fill: "#fff1d2", stroke: "#b7791f" };
      case "consumer": return { fill: "#e6effa", stroke: "#38506f" };
      case "descriptor": return { fill: "#dff6fa", stroke: "#0f8194" };
      case "hazard": return { fill: "#ffe3de", stroke: "#ba3e2f" };
      case "actor": return { fill: "#edf2f7", stroke: "#4d6078" };
      default: return { fill: "#edf2f7", stroke: "#64748b" };
    }
  }

  function arrowPalette(tone, active) {
    const map = {
      data: { stroke: "#4f83db", marker: "data", dash: "", width: active ? 4 : 3 },
      sync: { stroke: "#b7791f", marker: "sync", dash: "8 7", width: active ? 4 : 3 },
      publish: { stroke: "#6c43d6", marker: "publish", dash: "", width: active ? 4 : 3 },
      control: { stroke: "#0f8194", marker: "control", dash: "5 6", width: active ? 4 : 3 },
      hazard: { stroke: "#ba3e2f", marker: "hazard", dash: "7 6", width: active ? 4 : 3 },
    };
    return map[tone] || map.data;
  }

  function anchor(node, side) {
    const midX = node.x + node.w / 2;
    const midY = node.y + node.h / 2;
    if (side === "left") return { x: node.x, y: midY };
    if (side === "right") return { x: node.x + node.w, y: midY };
    if (side === "top") return { x: midX, y: node.y };
    return { x: midX, y: node.y + node.h };
  }

  function pathFromPoints(points) {
    return points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
  }

  function midpoint(points) {
    if (points.length < 2) return { x: 0, y: 0, w: 0 };
    const midIndex = Math.floor((points.length - 1) / 2);
    const a = points[midIndex];
    const b = points[midIndex + 1] || a;
    return {
      x: (a.x + b.x) / 2,
      y: (a.y + b.y) / 2 - 12,
      w: 18 + 8 * 6,
    };
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function escapeHtml(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }
})();
