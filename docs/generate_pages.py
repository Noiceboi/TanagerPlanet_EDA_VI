"""
generate_pages.py  —  run once to build all 17 index HTML pages
Run from repo root: python docs/generate_pages.py
"""
import os, pathlib, textwrap

DOCS = pathlib.Path(__file__).parent
PAGES = DOCS / "pages"
PAGES.mkdir(exist_ok=True)

PAGES_DEF = [
    # (filename,  INDEX_NAME,  nav_category,  page_title)
    ("ndvi",              "NDVI",                    "Vegetation",  "NDVI — Normalized Difference Vegetation Index"),
    ("psri",              "PSRI",                    "Vegetation",  "PSRI — Plant Senescence Reflectance Index"),
    ("rep",               "REP",                     "Vegetation",  "REP — Red Edge Position"),
    ("pri",               "PRI",                     "Pigments",    "PRI — Photochemical Reflectance Index"),
    ("ari",               "ARI",                     "Pigments",    "ARI — Anthocyanin Reflectance Index"),
    ("cri550",            "CRI550",                  "Pigments",    "CRI550 — Carotenoid Reflectance Index"),
    ("cri700",            "CRI700",                  "Pigments",    "CRI700 — Carotenoid Reflectance Index 700"),
    ("mcari",             "MCARI",                   "Pigments",    "MCARI — Modified Chlorophyll Absorption Ratio"),
    ("mcari_osavi",       "MCARI_OSAVI",             "Pigments",    "MCARI/OSAVI — Chlorophyll Ratio"),
    ("wbi",               "WBI",                     "Water",       "WBI — Water Band Index"),
    ("wi",                "WI",                      "Water",       "WI — Water Index"),
    ("msi",               "MSI",                     "Water",       "MSI — Moisture Stress Index"),
    ("ndli",              "NDLI",                    "Carbon/N",    "NDLI — Normalized Difference Lignin Index"),
    ("ndni",              "NDNI",                    "Carbon/N",    "NDNI — Normalized Difference Nitrogen Index"),
    ("protein_proxy",     "PROTEIN_PROXY_2180_2100", "Carbon/N",    "Protein Proxy (R2180/R2100)"),
    ("cai",               "CAI",                     "Carbon/N",    "CAI — Cellulose Absorption Index"),
    ("lcai",              "LCAI",                    "Carbon/N",    "LCAI — Lignin-Cellulose Absorption Index"),
]

TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{page_title} | Hyperspectral EDA</title>
  <!-- Bootstrap -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" />
  <!-- MathJax 3 — render LaTeX formulas (pure LaTeX, no HTML inside $$) -->
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$','$'], ['\\\\(','\\\\)']],
        displayMath: [['$$','$$'], ['\\\\[','\\\\]']],
        tags: 'none',
      }},
      svg: {{ fontCache: 'global', scale: 1.1 }},
      options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre'] }},
      startup: {{
        ready() {{
          MathJax.startup.defaultReady();
        }}
      }}
    }};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" id="MathJax-script" async></script>
  <!-- Plotly.js -->
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
  <!-- base path for data fetches -->
  <meta name="data-base" content="../data/" />
  <style>
    body {{ background: #f4f6f9; font-family: 'Segoe UI', sans-serif; }}
    .navbar-brand {{ font-weight: 700; }}
    .formula-box {{
      background: #fff; border: 1px solid #dee2e6; border-radius: 8px;
      padding: 1.5rem; margin-bottom: 1rem;
    }}
    /* Formula rendered by MathJax — no spans inside LaTeX, stays clean */
    .formula-box .formula-main {{
      font-size: 1.5rem; text-align: center;
      margin: 1.2rem 0 0.8rem;
      min-height: 2.5rem;
    }}
    /* Variable chips — interactive, outside the MathJax block */
    #variable-chips {{ display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .6rem; }}
    .var-chip {{
      display: inline-flex; align-items: center; gap: .3rem;
      padding: .3em .7em; border-radius: 20px;
      font-family: 'Courier New', monospace; font-size: .9rem; font-weight: 600;
      cursor: pointer; user-select: none;
      transition: transform .12s, box-shadow .12s;
    }}
    .var-chip:hover {{
      transform: translateY(-2px);
      box-shadow: 0 3px 10px rgba(0,0,0,.18);
    }}
    .chip-nm {{ font-size: .75rem; font-weight: 400; opacity: .75; }}
    .band-legend td, .band-legend th {{ vertical-align: middle; padding: .25rem .5rem; }}
    #reflectance-plot {{ width: 100%; height: 420px; }}
    #index-map {{ width: 100%; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,.12); }}
    .hint-text {{ font-size: .82rem; color: #6c757d; }}
    footer {{ background: #1a3a4a; color: rgba(255,255,255,.65); padding: 1.2rem; text-align: center; font-size: .85rem; }}
  </style>
</head>
<body>

<!-- Navbar -->
<nav class="navbar navbar-expand-lg navbar-dark bg-dark sticky-top">
  <div class="container-fluid">
    <a class="navbar-brand" href="../index.html">🛰 Hyperspectral EDA</a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navmenu">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navmenu">
      <ul class="navbar-nav me-auto">
        <li class="nav-item"><a class="nav-link" href="../index.html">Dashboard</a></li>
        <li class="nav-item dropdown">
          <a class="nav-link dropdown-toggle" href="#" data-bs-toggle="dropdown">Vegetation</a>
          <ul class="dropdown-menu">
            <li><a class="dropdown-item" href="ndvi.html">NDVI</a></li>
            <li><a class="dropdown-item" href="psri.html">PSRI</a></li>
            <li><a class="dropdown-item" href="rep.html">REP</a></li>
          </ul>
        </li>
        <li class="nav-item dropdown">
          <a class="nav-link dropdown-toggle" href="#" data-bs-toggle="dropdown">Pigments</a>
          <ul class="dropdown-menu">
            <li><a class="dropdown-item" href="pri.html">PRI</a></li>
            <li><a class="dropdown-item" href="ari.html">ARI</a></li>
            <li><a class="dropdown-item" href="cri550.html">CRI550</a></li>
            <li><a class="dropdown-item" href="cri700.html">CRI700</a></li>
            <li><a class="dropdown-item" href="mcari.html">MCARI</a></li>
            <li><a class="dropdown-item" href="mcari_osavi.html">MCARI/OSAVI</a></li>
          </ul>
        </li>
        <li class="nav-item dropdown">
          <a class="nav-link dropdown-toggle" href="#" data-bs-toggle="dropdown">Water &amp; Stress</a>
          <ul class="dropdown-menu">
            <li><a class="dropdown-item" href="wbi.html">WBI</a></li>
            <li><a class="dropdown-item" href="wi.html">WI</a></li>
            <li><a class="dropdown-item" href="msi.html">MSI</a></li>
          </ul>
        </li>
        <li class="nav-item dropdown">
          <a class="nav-link dropdown-toggle" href="#" data-bs-toggle="dropdown">Carbon / N</a>
          <ul class="dropdown-menu">
            <li><a class="dropdown-item" href="ndli.html">NDLI</a></li>
            <li><a class="dropdown-item" href="ndni.html">NDNI</a></li>
            <li><a class="dropdown-item" href="protein_proxy.html">Protein Proxy</a></li>
            <li><a class="dropdown-item" href="cai.html">CAI</a></li>
            <li><a class="dropdown-item" href="lcai.html">LCAI</a></li>
          </ul>
        </li>
      </ul>
    </div>
  </div>
</nav>

<!-- Page header -->
<div class="bg-dark text-white py-3 px-4">
  <div class="container">
    <nav aria-label="breadcrumb">
      <ol class="breadcrumb mb-1">
        <li class="breadcrumb-item"><a href="../index.html" class="text-light">Dashboard</a></li>
        <li class="breadcrumb-item text-secondary">{category}</li>
        <li class="breadcrumb-item active text-white">{index_name}</li>
      </ol>
    </nav>
    <h4 class="mb-0" id="index-title">{page_title}</h4>
  </div>
</div>

<div class="container py-4">
  <div class="row g-4">

    <!-- Left column: Map + dataset selector -->
    <div class="col-lg-5">
      <div class="card shadow-sm mb-3">
        <div class="card-header d-flex justify-content-between align-items-center">
          <span class="fw-bold">Spatial Map</span>
          <select class="form-select form-select-sm w-auto" id="dataset-select">
            <option value="20250301_143913_32_4001">March 01 2025</option>
            <option value="20250407_035527_47_4001">April 07 2025</option>
          </select>
        </div>
        <div class="card-body p-2">
          <img id="index-map" src="" alt="{index_name} map" />
        </div>
      </div>
      <!-- Dataset stats badges -->
      <div id="dataset-info" class="d-flex flex-wrap gap-1 mb-3"></div>
    </div>

    <!-- Right column: Formula + Chart -->
    <div class="col-lg-7">

      <!-- Formula Explorer -->
      <div class="formula-box mb-3">
        <div class="d-flex justify-content-between align-items-start mb-1">
          <h6 class="fw-bold mb-0">Formula Explorer</h6>
          <span class="hint-text">💡 Hover over a variable chip to highlight its band on the chart</span>
        </div>
        <!-- Pure LaTeX — MathJax renders this cleanly with no HTML contamination -->
        <div class="formula-main" id="formula-display">
{formula_html}
        </div>
        <p class="text-muted small mb-2" id="index-description"></p>
        <!-- Variable chips — built dynamically by chart-logic.js, hover triggers band highlight -->
        <div class="mt-2">
          <small class="fw-semibold text-muted d-block mb-1">INTERACTIVE VARIABLES</small>
          <div id="variable-chips"></div>
        </div>
        <!-- Band legend -->
        <div class="mt-3">
          <small class="fw-semibold text-muted">BAND LEGEND</small>
          <table class="table table-sm band-legend mt-1 mb-0">
            <tbody id="band-legend"></tbody>
          </table>
        </div>
      </div>

      <!-- Reflectance Chart -->
      <div class="card shadow-sm">
        <div class="card-header fw-bold">Mean Reflectance Spectrum</div>
        <div class="card-body p-2">
          <div id="reflectance-plot"></div>
          <p class="hint-text mt-1 mb-0 px-1">Savitzky-Golay smoothed mean spectrum. Coloured bands correspond to the formula variables above.</p>
        </div>
      </div>

    </div>
  </div>
</div>

<footer>
  Planet Tanager Hyperspectral EDA · Vietnam · 2025 &nbsp;|&nbsp; {index_name}
</footer>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script src="../js/chart-logic.js"></script>
<script>initIndexPage('{index_name}');</script>
</body>
</html>
"""

# Pure LaTeX formulas — NO HTML inside $$...$$  (MathJax cannot mix HTML and LaTeX)
# Variable hover interaction is handled separately via #variable-chips chips in JS
FORMULA_HTML = {
    "NDVI":   r"$$\text{NDVI} = \frac{R_{NIR} - R_{Red}}{R_{NIR} + R_{Red}}$$",
    "PRI":    r"$$\text{PRI} = \frac{R_{531} - R_{570}}{R_{531} + R_{570}}$$",
    "WBI":    r"$$\text{WBI} = \frac{R_{970}}{R_{900}}$$",
    "WI":     r"$$\text{WI} = \frac{R_{900}}{R_{970}}$$",
    "PSRI":   r"$$\text{PSRI} = \frac{R_{680} - R_{500}}{R_{750}}$$",
    "REP":    r"$$\lambda_{REP} = 700 + 40 \cdot \frac{\bar{R}_{RE} - R_{700}}{R_{740} - R_{700}}$$",
    "ARI":    r"$$\text{ARI} = \frac{1}{R_{550}} - \frac{1}{R_{700}}$$",
    "CRI550": r"$$\text{CRI}_{550} = \frac{1}{R_{510}} - \frac{1}{R_{550}}$$",
    "CRI700": r"$$\text{CRI}_{700} = \frac{1}{R_{510}} - \frac{1}{R_{700}}$$",
    "MCARI":  r"$$\text{MCARI} = \left[(R_{700}-R_{670}) - 0.2\,(R_{700}-R_{550})\right] \cdot \frac{R_{700}}{R_{670}}$$",
    "MCARI_OSAVI": r"$$\frac{\text{MCARI}}{\text{OSAVI}}, \quad \text{OSAVI} = \frac{1.16\,(R_{800}-R_{670})}{R_{800}+R_{670}+0.16}$$",
    "NDLI":   r"$$\text{NDLI} = \frac{\log(1/R_{1754}) - \log(1/R_{1680})}{\log(1/R_{1754}) + \log(1/R_{1680})}$$",
    "NDNI":   r"$$\text{NDNI} = \frac{\log(1/R_{1510}) - \log(1/R_{1680})}{\log(1/R_{1510}) + \log(1/R_{1680})}$$",
    "PROTEIN_PROXY_2180_2100": r"$$\text{Protein Proxy} = \frac{R_{2180}}{R_{2100}}$$",
    "CAI":    r"$$\text{CAI} = 0.5\,(R_{2000} + R_{2200}) - R_{2100}$$",
    "LCAI":   r"$$\text{LCAI} = 0.5\,(R_{1100} + R_{2200}) - R_{2100}$$",
    "MSI":    r"$$\text{MSI} = \frac{R_{1599}}{R_{819}}$$",
}

for fname, index_name, category, page_title in PAGES_DEF:
    formula_html = FORMULA_HTML.get(index_name, f"$$\\text{{{index_name}}}$$")
    html = TEMPLATE.format(
        page_title=page_title,
        index_name=index_name,
        category=category,
        formula_html=formula_html,
    )
    out = PAGES / f"{fname}.html"
    out.write_text(html, encoding="utf-8")
    print(f"  written: pages/{fname}.html")

print(f"\nAll {len(PAGES_DEF)} index pages generated.")
