/**
 * chart-logic.js
 * Hyperspectral EDA – GitHub Pages static site
 * Handles: Plotly reflectance chart, band highlighting, MathJax hover interaction
 */

"use strict";

// ─────────────────────────────────────────────────────────────────────────────
// Dataset registry — loaded dynamically from docs/data/datasets.json
// ─────────────────────────────────────────────────────────────────────────────
let DATASETS = {};

async function loadDatasetsRegistry() {
  const base = document.querySelector("meta[name='data-base']")
    ? document.querySelector("meta[name='data-base']").content
    : "data/";
  try {
    const resp = await fetch(base + "datasets.json");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const reg = await resp.json();
    DATASETS = reg.datasets || {};
  } catch (e) {
    console.error("Could not load datasets.json:", e);
    DATASETS = {};
  }
}

function buildDatasetCard(key, d) {
  return `
    <div class="col-md-6">
      <div class="card shadow-sm" id="card-${key}">
        <div class="card-header ${d.cardHeaderClass || 'bg-primary text-white'}">${d.cardLabel || 'Dataset'} — <span class="ds-label">${d.label}</span></div>
        <div class="card-body">
          <table class="table table-sm table-borderless mb-2">
            <tbody>
              <tr><th scope="row" class="text-muted">Cube shape</th><td class="ds-shape font-monospace">—</td></tr>
              <tr><th scope="row" class="text-muted">REP mean</th><td class="ds-rep">—</td></tr>
              <tr><th scope="row" class="text-muted">PCA variance</th><td class="ds-pca">—</td></tr>
              <tr><th scope="row" class="text-muted">Anomaly P99</th><td class="ds-anomaly">—</td></tr>
            </tbody>
          </table>
          <a href="images/maps/${key}/02_index_maps.png" target="_blank">
            <img class="overview-img" src="images/maps/${key}/02_index_maps.png"
                 alt="Index Maps Overview ${d.label}" />
          </a>
          <p class="text-muted small mt-2 mb-0">All-index grid map. Click to enlarge.</p>
        </div>
      </div>
    </div>`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Index configuration: formula, description, band highlight regions
// Each band entry: { label, x0, x1, color, alpha }
// ─────────────────────────────────────────────────────────────────────────────
const INDEX_CONFIG = {
  NDVI: {
    title: "Normalized Difference Vegetation Index",
    formula: "\\frac{R_{NIR} - R_{Red}}{R_{NIR} + R_{Red}}",
    description:
      "NDVI measures photosynthetically active vegetation density. Higher values indicate denser, healthier canopy.",
    wikiVars: {
      R_NIR: [790, 810],
      R_Red: [660, 680],
    },
    bands: [
      { label: "R_Red (670 nm)", x0: 660, x1: 680, color: "rgba(220,50,50,0.25)" },
      { label: "R_NIR (800 nm)", x0: 790, x1: 810, color: "rgba(50,160,50,0.25)" },
    ],
  },
  PRI: {
    title: "Photochemical Reflectance Index",
    formula: "\\frac{R_{531} - R_{570}}{R_{531} + R_{570}}",
    description:
      "PRI tracks xanthophyll pigment changes linked to light-use efficiency and photosynthetic stress.",
    wikiVars: {
      R_531: [526, 536],
      R_570: [565, 575],
    },
    bands: [
      { label: "R531", x0: 526, x1: 536, color: "rgba(50,100,220,0.25)" },
      { label: "R570", x0: 565, x1: 575, color: "rgba(230,130,20,0.25)" },
    ],
  },
  WBI: {
    title: "Water Band Index",
    formula: "\\frac{R_{970}}{R_{900}}",
    description:
      "WBI is sensitive to leaf water content. Values below 1 indicate water stress.",
    wikiVars: {
      R_970: [965, 975],
      R_900: [895, 905],
    },
    bands: [
      { label: "R900 (denom)", x0: 895, x1: 905, color: "rgba(0,180,220,0.25)" },
      { label: "R970 (num)", x0: 965, x1: 975, color: "rgba(30,60,200,0.25)" },
    ],
  },
  WI: {
    title: "Water Index",
    formula: "\\frac{R_{900}}{R_{970}}",
    description:
      "WI is the inverse ratio of WBI, also tracking leaf water content.",
    wikiVars: {
      R_900: [895, 905],
      R_970: [965, 975],
    },
    bands: [
      { label: "R900", x0: 895, x1: 905, color: "rgba(0,180,220,0.25)" },
      { label: "R970", x0: 965, x1: 975, color: "rgba(30,60,200,0.25)" },
    ],
  },
  PSRI: {
    title: "Plant Senescence Reflectance Index",
    formula: "\\frac{R_{680} - R_{500}}{R_{750}}",
    description:
      "PSRI quantifies the ratio of carotenoids to chlorophyll, indicating leaf aging and senescence.",
    wikiVars: {
      R_680: [675, 685],
      R_500: [495, 505],
      R_750: [745, 755],
    },
    bands: [
      { label: "R500 (blue)", x0: 495, x1: 505, color: "rgba(30,100,220,0.25)" },
      { label: "R680 (red)", x0: 675, x1: 685, color: "rgba(220,50,50,0.25)" },
      { label: "R750 (denom)", x0: 745, x1: 755, color: "rgba(150,50,200,0.25)" },
    ],
  },
  REP: {
    title: "Red Edge Position",
    formula: "\\lambda_{REP} = 700 + 40 \\cdot \\frac{\\bar{R}_{RE} - R_{700}}{R_{740} - R_{700}}",
    description:
      "REP tracks the inflection point of the red-edge, a proxy for chlorophyll content and canopy stress.",
    wikiVars: {
      R_700: [695, 705],
      R_740: [735, 745],
    },
    bands: [
      { label: "Red Edge Region", x0: 670, x1: 780, color: "rgba(200,80,80,0.12)" },
      { label: "R700", x0: 695, x1: 705, color: "rgba(220,50,50,0.3)" },
      { label: "R740", x0: 735, x1: 745, color: "rgba(80,180,80,0.3)" },
    ],
  },
  ARI: {
    title: "Anthocyanin Reflectance Index",
    formula: "\\frac{1}{R_{550}} - \\frac{1}{R_{700}}",
    description:
      "ARI is sensitive to leaf anthocyanin content. High values indicate stress-induced anthocyanin accumulation.",
    wikiVars: {
      R_550: [545, 555],
      R_700: [695, 705],
    },
    bands: [
      { label: "R550", x0: 545, x1: 555, color: "rgba(50,180,80,0.25)" },
      { label: "R700", x0: 695, x1: 705, color: "rgba(180,40,40,0.25)" },
    ],
  },
  CRI550: {
    title: "Carotenoid Reflectance Index 550",
    formula: "\\frac{1}{R_{510}} - \\frac{1}{R_{550}}",
    description:
      "CRI550 estimates carotenoid content in the visible range.",
    wikiVars: {
      R_510: [505, 515],
      R_550: [545, 555],
    },
    bands: [
      { label: "R510", x0: 505, x1: 515, color: "rgba(130,50,200,0.25)" },
      { label: "R550", x0: 545, x1: 555, color: "rgba(50,180,80,0.25)" },
    ],
  },
  CRI700: {
    title: "Carotenoid Reflectance Index 700",
    formula: "\\frac{1}{R_{510}} - \\frac{1}{R_{700}}",
    description:
      "CRI700 is a stronger carotenoid estimator extending into the red edge.",
    wikiVars: {
      R_510: [505, 515],
      R_700: [695, 705],
    },
    bands: [
      { label: "R510", x0: 505, x1: 515, color: "rgba(130,50,200,0.25)" },
      { label: "R700", x0: 695, x1: 705, color: "rgba(180,40,40,0.25)" },
    ],
  },
  MCARI: {
    title: "Modified Chlorophyll Absorption Ratio Index",
    formula:
      "\\left[(R_{700}-R_{670}) - 0.2(R_{700}-R_{550})\\right] \\cdot \\frac{R_{700}}{R_{670}}",
    description:
      "MCARI is sensitive to chlorophyll absorption while minimising soil and LAI effects.",
    wikiVars: {
      R_700: [695, 705],
      R_670: [665, 675],
      R_550: [545, 555],
    },
    bands: [
      { label: "R550", x0: 545, x1: 555, color: "rgba(50,180,80,0.25)" },
      { label: "R670", x0: 665, x1: 675, color: "rgba(180,40,40,0.25)" },
      { label: "R700", x0: 695, x1: 705, color: "rgba(120,20,20,0.3)" },
    ],
  },
  MCARI_OSAVI: {
    title: "MCARI / OSAVI",
    formula:
      "\\frac{\\text{MCARI}}{\\text{OSAVI}} \\quad \\text{where OSAVI} = \\frac{1.16(R_{800}-R_{670})}{R_{800}+R_{670}+0.16}",
    description:
      "MCARI/OSAVI decouples chlorophyll concentration from canopy structure effects.",
    wikiVars: {
      R_800: [795, 805],
      R_670: [665, 675],
      R_700: [695, 705],
      R_550: [545, 555],
    },
    bands: [
      { label: "R550", x0: 545, x1: 555, color: "rgba(50,180,80,0.25)" },
      { label: "R670", x0: 665, x1: 675, color: "rgba(180,40,40,0.25)" },
      { label: "R700", x0: 695, x1: 705, color: "rgba(120,20,20,0.3)" },
      { label: "R800 (OSAVI)", x0: 795, x1: 805, color: "rgba(80,80,180,0.25)" },
    ],
  },
  NDLI: {
    title: "Normalized Difference Lignin Index",
    formula:
      "\\frac{\\log(1/R_{1754}) - \\log(1/R_{1680})}{\\log(1/R_{1754}) + \\log(1/R_{1680})}",
    description:
      "NDLI estimates cellulose/lignin concentration using SWIR absorption features.",
    wikiVars: {
      R_1680: [1674, 1686],
      R_1754: [1748, 1760],
    },
    bands: [
      { label: "R1680", x0: 1674, x1: 1686, color: "rgba(200,130,20,0.3)" },
      { label: "R1754", x0: 1748, x1: 1760, color: "rgba(120,50,180,0.3)" },
    ],
  },
  NDNI: {
    title: "Normalized Difference Nitrogen Index",
    formula:
      "\\frac{\\log(1/R_{1510}) - \\log(1/R_{1680})}{\\log(1/R_{1510}) + \\log(1/R_{1680})}",
    description:
      "NDNI estimates foliar nitrogen concentration from SWIR log-ratio reflectance.",
    wikiVars: {
      R_1510: [1504, 1516],
      R_1680: [1674, 1686],
    },
    bands: [
      { label: "R1510", x0: 1504, x1: 1516, color: "rgba(0,180,180,0.3)" },
      { label: "R1680", x0: 1674, x1: 1686, color: "rgba(200,130,20,0.3)" },
    ],
  },
  PROTEIN_PROXY_2180_2100: {
    title: "Protein Proxy (2180/2100)",
    formula: "\\frac{R_{2180}}{R_{2100}}",
    description:
      "A SWIR ratio proxy for leaf protein content. Protein-N absorption occurs near 2180 nm.",
    wikiVars: {
      R_2180: [2174, 2186],
      R_2100: [2094, 2106],
    },
    bands: [
      { label: "R2100 (denom)", x0: 2094, x1: 2106, color: "rgba(200,170,0,0.3)" },
      { label: "R2180 (num)", x0: 2174, x1: 2186, color: "rgba(220,80,160,0.3)" },
    ],
  },
  CAI: {
    title: "Cellulose Absorption Index",
    formula: "0.5(R_{2000} + R_{2200}) - R_{2100}",
    description:
      "CAI measures dry carbon (cellulose/lignin) in litter and residues via SWIR.",
    wikiVars: {
      R_2000: [1994, 2006],
      R_2100: [2094, 2106],
      R_2200: [2194, 2206],
    },
    bands: [
      { label: "R2000", x0: 1994, x1: 2006, color: "rgba(30,100,220,0.25)" },
      { label: "R2100 (trough)", x0: 2094, x1: 2106, color: "rgba(220,50,50,0.3)" },
      { label: "R2200", x0: 2194, x1: 2206, color: "rgba(50,180,80,0.25)" },
    ],
  },
  LCAI: {
    title: "Lignin-Cellulose Absorption Index",
    formula: "0.5(R_{1100} + R_{2200}) - R_{2100}",
    description:
      "LCAI extends CAI with a NIR anchor band to improve discrimination of cellulose and lignin.",
    wikiVars: {
      R_1100: [1094, 1106],
      R_2100: [2094, 2106],
      R_2200: [2194, 2206],
    },
    bands: [
      { label: "R1100 (NIR anchor)", x0: 1094, x1: 1106, color: "rgba(0,200,180,0.25)" },
      { label: "R2100 (trough)", x0: 2094, x1: 2106, color: "rgba(220,50,50,0.3)" },
      { label: "R2200", x0: 2194, x1: 2206, color: "rgba(50,180,80,0.25)" },
    ],
  },
  MSI: {
    title: "Moisture Stress Index",
    formula: "\\frac{R_{1599}}{R_{819}}",
    description:
      "MSI is inversely related to leaf water content. Higher values indicate increasing water stress.",
    wikiVars: {
      R_1599: [1593, 1605],
      R_819: [813, 825],
    },
    bands: [
      { label: "R819 (denom)", x0: 813, x1: 825, color: "rgba(50,180,80,0.3)" },
      { label: "R1599 (num)", x0: 1593, x1: 1605, color: "rgba(200,120,20,0.3)" },
    ],
  },
  HDWI: {
    title: "Hyperspectral Difference Water Index",
    formula: "\\frac{R_{572} - R_{420}}{R_{572} + R_{420}}",
    description:
      "HDWI is designed specifically for hyperspectral data to detect water bodies while mitigating effects of shadows and low-albedo surfaces. Uses narrow visible bands (572 nm green, 420 nm violet).",
    wikiVars: {
      R_572: [567, 577],
      R_420: [415, 425],
    },
    bands: [
      { label: "R572 — Green (num)", x0: 567, x1: 577, color: "rgba(50,200,100,0.3)" },
      { label: "R420 — Violet (denom)", x0: 415, x1: 425, color: "rgba(120,60,200,0.3)" },
    ],
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────
let _currentDataset = null; // set after registry loads
let _currentIndex = null;
let _spectraCache = {}; // { datasetKey: { wavelengths, mean_reflectance } }

// ─────────────────────────────────────────────────────────────────────────────
// Fetch + cache spectra JSON
// ─────────────────────────────────────────────────────────────────────────────
async function fetchSpectra(datasetKey) {
  if (_spectraCache[datasetKey]) return _spectraCache[datasetKey];
  const filename = DATASETS[datasetKey]?.jsonFile ?? `mean_spectra_${datasetKey}.json`;
  // Support both root-relative and relative paths
  const basePath = document.querySelector("meta[name='data-base']")
    ? document.querySelector("meta[name='data-base']").content
    : "../data/";
  const resp = await fetch(basePath + filename);
  if (!resp.ok) throw new Error(`Cannot load ${filename}: ${resp.status}`);
  const data = await resp.json();
  _spectraCache[datasetKey] = data;
  return data;
}

// ─────────────────────────────────────────────────────────────────────────────
// Plotly chart
// ─────────────────────────────────────────────────────────────────────────────
async function renderChart(datasetKey, indexName) {
  const el = document.getElementById("reflectance-plot");
  if (!el) return;
  let data;
  try {
    data = await fetchSpectra(datasetKey);
  } catch (e) {
    el.innerHTML = `<div class="alert alert-warning m-3">Could not load spectra data: ${e.message}</div>`;
    return;
  }

  const trace = {
    x: data.wavelengths,
    y: data.mean_reflectance,
    mode: "lines",
    name: "Mean reflectance (SG)",
    line: { color: "#2c7bb6", width: 2 },
    hovertemplate: "%{x:.1f} nm: %{y:.4f}<extra></extra>",
  };

  const layout = {
    title: {
      text: `Mean Reflectance Spectrum — ${DATASETS[datasetKey].label}`,
      font: { size: 14 },
    },
    xaxis: {
      title: "Wavelength (nm)",
      range: [350, 2520],
      gridcolor: "#e8e8e8",
      showgrid: true,
    },
    yaxis: {
      title: "Surface Reflectance",
      gridcolor: "#e8e8e8",
      showgrid: true,
    },
    plot_bgcolor: "#fafafa",
    paper_bgcolor: "#ffffff",
    margin: { l: 60, r: 20, t: 50, b: 50 },
    shapes: indexName ? buildShapes(indexName) : [],
    showlegend: false,
  };

  Plotly.react(el, [trace], layout, { responsive: true, displaylogo: false });
}

// ─────────────────────────────────────────────────────────────────────────────
// Band highlight shapes
// ─────────────────────────────────────────────────────────────────────────────
function buildShapes(indexName, highlightBandLabel) {
  const cfg = INDEX_CONFIG[indexName];
  if (!cfg) return [];
  return cfg.bands.map((b) => ({
    type: "rect",
    xref: "x",
    yref: "paper",
    x0: b.x0,
    x1: b.x1,
    y0: 0,
    y1: 1,
    fillcolor: highlightBandLabel && b.label === highlightBandLabel
      ? b.color.replace(/[\d.]+\)$/, "0.55)")  // stronger on hover
      : b.color,
    line: { width: 0 },
  }));
}

function highlightBands(indexName, focusBandLabel) {
  const el = document.getElementById("reflectance-plot");
  if (!el) return;
  Plotly.relayout(el, { shapes: buildShapes(indexName, focusBandLabel) });
}

function clearHighlight(indexName) {
  const el = document.getElementById("reflectance-plot");
  if (!el) return;
  // Restore all bands at default opacity (no focus)
  Plotly.relayout(el, { shapes: buildShapes(indexName) });
}

// ─────────────────────────────────────────────────────────────────────────────
// Variable chip helpers  (rendered OUTSIDE LaTeX so MathJax stays clean)
// ─────────────────────────────────────────────────────────────────────────────
function varNameToHtml(varName) {
  // R_700 → R<sub>700</sub>  |  R_NIR → R<sub>NIR</sub>
  return varName.replace(/^R_(.+)$/, "R<sub>$1</sub>");
}

// Builds hoverable variable chips into #variable-chips
function buildVariableChips(indexName) {
  const cfg = INDEX_CONFIG[indexName];
  const el = document.getElementById("variable-chips");
  if (!el || !cfg || !cfg.wikiVars) return;

  el.innerHTML = Object.entries(cfg.wikiVars)
    .map(([varName, [x0, x1]]) => {
      const band = cfg.bands.find((b) => b.x0 === x0 && b.x1 === x1);
      const bg = band ? band.color : "rgba(150,150,150,0.15)";
      const border = band ? band.color.replace(/[\d.]+\)$/, "0.75)") : "#bbb";
      return `<span class="var-chip"
                data-var="${varName}"
                style="background:${bg};border:1.5px solid ${border};"
                title="Hover to highlight ${x0}–${x1} nm on chart">
                ${varNameToHtml(varName)}
                <span class="chip-nm">${x0}–${x1} nm</span>
              </span>`;
    })
    .join("");
}

// ─────────────────────────────────────────────────────────────────────────────
// Formula Explorer — binds hover events on .var-chip elements
// ─────────────────────────────────────────────────────────────────────────────
function attachFormulaHover(indexName) {
  const cfg = INDEX_CONFIG[indexName];
  if (!cfg || !cfg.wikiVars) return;

  document.querySelectorAll(".var-chip[data-var]").forEach((chip) => {
    const varName = chip.dataset.var;
    const range = cfg.wikiVars[varName];
    if (!range) return;
    const [x0, x1] = range;
    const band = cfg.bands.find((b) => b.x0 === x0 && b.x1 === x1);
    if (!band) return;

    chip.addEventListener("mouseenter", () =>
      highlightBands(indexName, band.label)
    );
    chip.addEventListener("mouseleave", () => clearHighlight(indexName));
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataset dropdown handler
// ─────────────────────────────────────────────────────────────────────────────
function updateDataset(newDatasetKey) {
  _currentDataset = newDatasetKey;
  // Update map image
  const imgEl = document.getElementById("index-map");
  if (imgEl && _currentIndex) {
    imgEl.src = `../images/maps/${newDatasetKey}/${_currentIndex}.png`;
    imgEl.alt = `${_currentIndex} map — ${newDatasetKey}`;
  }
  // Re-render chart with new dataset
  if (_currentIndex) {
    renderChart(newDatasetKey, _currentIndex);
  }
  // Update dataset info badge
  const infoBadge = document.getElementById("dataset-info");
  if (infoBadge && DATASETS[newDatasetKey]) {
    const d = DATASETS[newDatasetKey];
    infoBadge.innerHTML = `
      <span class="badge bg-secondary me-1">Cube: ${d.cubeShape}</span>
      <span class="badge bg-info text-dark me-1">REP: ${d.repMean}</span>
      <span class="badge bg-success me-1">PCA: ${d.pcaVar}</span>
      <span class="badge bg-warning text-dark">Anomaly P99: ${d.anomalyThr}</span>
    `;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Page initializer — called from each index page
// ─────────────────────────────────────────────────────────────────────────────
async function initIndexPage(indexName) {
  await loadDatasetsRegistry();
  _currentDataset = Object.keys(DATASETS)[0]
    || document.getElementById("dataset-select")?.options[0]?.value
    || null;
  _currentIndex = indexName;
  const cfg = INDEX_CONFIG[indexName];
  if (!cfg) return;

  // Fill title + description
  const titleEl = document.getElementById("index-title");
  const descEl = document.getElementById("index-description");
  if (titleEl) titleEl.textContent = cfg.title;
  if (descEl) descEl.textContent = cfg.description;

  // Set initial map image
  const imgEl = document.getElementById("index-map");
  if (imgEl) {
    imgEl.src = `../images/maps/${_currentDataset}/${indexName}.png`;
    imgEl.alt = `${indexName} map — ${_currentDataset}`;
    imgEl.onerror = function () {
      this.alt = "Map image not available";
      this.src =
        "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='200'%3E%3Crect width='100%25' height='100%25' fill='%23f8f9fa'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%236c757d' font-family='sans-serif'%3EImage not found%3C/text%3E%3C/svg%3E";
    };
  }

  // Set dataset dropdown default
  const sel = document.getElementById("dataset-select");
  if (sel) {
    sel.value = _currentDataset;
    sel.addEventListener("change", (e) => updateDataset(e.target.value));
  }

  // Build band legend table
  const legendEl = document.getElementById("band-legend");
  if (legendEl && cfg.bands) {
    legendEl.innerHTML = cfg.bands
      .map(
        (b) =>
          `<tr>
            <td><span style="display:inline-block;width:14px;height:14px;background:${b.color};border-radius:2px;"></span></td>
            <td class="ps-2 fw-semibold font-monospace">${b.label}</td>
            <td class="text-muted">${b.x0}–${b.x1} nm</td>
          </tr>`
      )
      .join("");
  }

  // Render chart
  renderChart(_currentDataset, indexName);

  // Update dataset info badge
  updateDataset(_currentDataset);

  // Build variable chips and attach hover events
  buildVariableChips(indexName);
  attachFormulaHover(indexName);

  // Re-typeset any MathJax in chip labels (subscripts etc.)
  if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
    const chipsEl = document.getElementById("variable-chips");
    if (chipsEl) MathJax.typesetPromise([chipsEl]);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dashboard page helpers (index.html)
// ─────────────────────────────────────────────────────────────────────────────
async function initDashboard() {
  await loadDatasetsRegistry();

  // Build dataset cards dynamically
  const container = document.getElementById("dataset-cards-container");
  if (container) {
    container.innerHTML = Object.entries(DATASETS)
      .map(([key, d]) => buildDatasetCard(key, d))
      .join("");
  }

  // Fill stats into cards (works whether cards were static or just injected)
  Object.entries(DATASETS).forEach(([key, d]) => {
    const card = document.getElementById(`card-${key}`);
    if (!card) return;
    const qs = (sel) => card.querySelector(sel);
    if (qs(".ds-label")) qs(".ds-label").textContent = d.label;
    if (qs(".ds-shape")) qs(".ds-shape").textContent = d.cubeShape;
    if (qs(".ds-rep")) qs(".ds-rep").textContent = d.repMean;
    if (qs(".ds-pca")) qs(".ds-pca").textContent = d.pcaVar;
    if (qs(".ds-anomaly")) qs(".ds-anomaly").textContent = d.anomalyThr;
  });

  // Update hero date badges
  const badgesEl = document.getElementById("dataset-date-badges");
  if (badgesEl) {
    badgesEl.innerHTML = Object.values(DATASETS)
      .map((d) => `<span class="badge bg-light text-dark fs-6">📅 ${d.label}</span>`)
      .join(" ");
  }
}
