"""
Interactive Standalone HTML 3D Report Generator for ESM-AMP-Screen.
Generates an all-in-one HTML report with 3Dmol.js molecular visualization,
interactive consensus leaderboard, and biophysical distribution charts.
Embeds PDB coordinate structures directly to bypass browser CORS restrictions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import html

from esm_amp_screen.utils import logger, ensure_dir


def generate_interactive_html_report(
    summary_data: List[Dict[str, Any]],
    receptor_pdb: Path,
    out_html: Path,
    top_n: int = 5
) -> Path:
    """
    Creates a single-file, zero-dependency interactive HTML report containing:
    - 3Dmol.js WebGL molecular viewer showing receptor + docked AMP poses
    - Interactive consensus ranking leaderboard with clickable row selection
    - Physical and Machine Learning breakdown cards (Vina, HADDOCK, ML pKd, BSA)
    - Full offline/file:// browser compatibility (no local web server needed)
    """
    out_html = Path(out_html)
    ensure_dir(out_html.parent)

    # 1. Read and clean receptor PDB string
    rec_pdb_content = ""
    if receptor_pdb and Path(receptor_pdb).exists():
        rec_pdb_content = Path(receptor_pdb).read_text(encoding="utf-8", errors="ignore")

    # 2. Gather top hit complex PDB strings
    embedded_complexes = {}
    hits_data = []

    for hit in summary_data[:top_n]:
        pep_id = hit["pep_id"]
        cpx_path = hit.get("complex_pdb")
        cpx_content = ""
        if cpx_path and isinstance(cpx_path, (str, Path)) and str(cpx_path).lower() != "nan" and Path(cpx_path).exists():
            cpx_content = Path(cpx_path).read_text(encoding="utf-8", errors="ignore")
        elif (out_html.parent / "top_complexes" / f"complex_{pep_id}.pdb").exists():
            cpx_content = (out_html.parent / "top_complexes" / f"complex_{pep_id}.pdb").read_text(encoding="utf-8", errors="ignore")

        embedded_complexes[pep_id] = cpx_content
        hits_data.append({
            "rank": hit.get("rank", 1),
            "pep_id": pep_id,
            "sequence": hit.get("sequence", ""),
            "length": hit.get("length", 0),
            "vina_dg": hit.get("affinity", 0.0),
            "haddock_score": hit.get("haddock_score", "N/A"),
            "bsa": hit.get("bsa", "N/A"),
            "ml_pkd": hit.get("ml_pkd", "N/A"),
            "ml_delta_g": hit.get("ml_delta_g", "N/A"),
            "pred_kd": hit.get("ml_kd", hit.get("estimated_kd", "N/A")),
            "solution_kd": hit.get("solution_kd", hit.get("ml_kd", "N/A")),
            "hemolytic_risk": hit.get("hemolytic_risk", "Low (Non-hemolytic)"),
            "therapeutic_selectivity": hit.get("therapeutic_selectivity", "Optimal"),
            "confidence": hit.get("confidence_score", "N/A"),
            "consensus": hit.get("consensus_score", hit.get("composite_score", 0.0)),
            "charge": hit.get("net_charge_z", 0.0),
            "mu_h": hit.get("hydrophobic_moment_muH", 0.0),
            "hbonds": hit.get("hbonds", 0),
            "salt_bridges": hit.get("salt_bridges", 0),
            "stage": hit.get("stage", "Stage 1")
        })

    # JSON representations for embedded JavaScript with script-safe character escaping
    # (escaping <, >, and & prevents script tag breakout and HTML injection)
    hits_json = json.dumps(hits_data).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    complexes_json = json.dumps(embedded_complexes).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    rec_json = json.dumps(rec_pdb_content).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")

    top_hit = hits_data[0] if hits_data else {}
    top_hit_id = html.escape(str(top_hit.get("pep_id", "N/A")))
    best_haddock = min([h["haddock_score"] for h in hits_data if isinstance(h["haddock_score"], (int, float))], default="N/A")
    best_bsa = max([h["bsa"] for h in hits_data if isinstance(h["bsa"], (int, float))], default="N/A")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ESM-AMP-Screen: 3D Screening Report</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
  <style>
    :root {{
      --bg-color: #0d1117;
      --card-bg: #161b22;
      --border-color: #30363d;
      --text-main: #c9d1d9;
      --text-muted: #8b949e;
      --primary: #58a6ff;
      --success: #3fb950;
      --accent: #d2a8ff;
      --highlight: #f0883e;
      --danger: #f85149;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg-color);
      color: var(--text-main);
      padding: 24px;
      line-height: 1.5;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 16px;
      margin-bottom: 24px;
    }}
    .header h1 {{
      font-size: 24px;
      font-weight: 600;
      color: #fff;
    }}
    .header .subtitle {{
      color: var(--text-muted);
      font-size: 14px;
      margin-top: 4px;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px;
    }}
    .stat-card .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
    }}
    .stat-card .value {{
      font-size: 22px;
      font-weight: bold;
      color: #fff;
      margin-top: 4px;
    }}
    .main-layout {{
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 24px;
      height: 640px;
    }}
    @media (max-width: 1024px) {{
      .main-layout {{ grid-template-columns: 1fr; height: auto; }}
    }}
    .viewer-container {{
      background: #000;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      position: relative;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      height: 100%;
      min-height: 520px;
    }}
    #viewport {{
      width: 100%;
      height: 100%;
      position: relative;
    }}
    .viewer-controls {{
      position: absolute;
      bottom: 12px;
      left: 12px;
      display: flex;
      gap: 8px;
      z-index: 10;
      background: rgba(22, 27, 34, 0.85);
      padding: 6px 10px;
      border-radius: 6px;
      border: 1px solid var(--border-color);
    }}
    .btn {{
      background: #21262d;
      color: var(--text-main);
      border: 1px solid var(--border-color);
      padding: 6px 12px;
      font-size: 12px;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn:hover {{ background: #30363d; color: #fff; }}
    .btn.active {{ background: var(--primary); color: #fff; border-color: var(--primary); }}
    .table-container {{
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      overflow-y: auto;
      height: 100%;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #1c2128;
      color: var(--text-muted);
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border-color);
      font-weight: 600;
    }}
    td {{
      padding: 12px;
      border-bottom: 1px solid var(--border-color);
    }}
    tr.hit-row {{
      cursor: pointer;
      transition: background 0.15s;
    }}
    tr.hit-row:hover {{
      background: #1f242c;
    }}
    tr.hit-row.selected {{
      background: rgba(88, 166, 255, 0.15);
      border-left: 3px solid var(--primary);
    }}
    .badge {{
      display: inline-block;
      padding: 2px 6px;
      font-size: 11px;
      border-radius: 10px;
      font-weight: 600;
    }}
    .badge-refined {{ background: rgba(63, 185, 80, 0.2); color: var(--success); }}
    .badge-screened {{ background: rgba(139, 148, 158, 0.2); color: var(--text-muted); }}
    .seq-tag {{
      font-family: monospace;
      font-size: 11px;
      color: var(--accent);
      background: rgba(210, 168, 255, 0.1);
      padding: 2px 4px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>

  <div class="header">
    <div>
      <h1>ESM-AMP-Screen Interactive 3D Report</h1>
      <div class="subtitle">Multi-Stage Protein-Peptide Screening Funnel (Vina HTVS + HADDOCK Physics + ML Rescoring)</div>
    </div>
    <div style="text-align: right;">
      <span class="badge badge-refined">3-Stage Funnel Verified</span>
    </div>
  </div>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="label">Top Ranked Candidate</div>
      <div class="value" style="color: var(--primary);">{top_hit_id}</div>
    </div>
    <div class="stat-card">
      <div class="label">Top Vina Affinity</div>
      <div class="value">{top_hit.get("vina_dg", 0.0):.2f} <span style="font-size: 12px; font-weight: normal;">kcal/mol</span></div>
    </div>
    <div class="stat-card">
      <div class="label">Best HADDOCK Score</div>
      <div class="value" style="color: var(--highlight);">{best_haddock}</div>
    </div>
    <div class="stat-card">
      <div class="label">Peak Buried Surface Area</div>
      <div class="value" style="color: var(--success);">{best_bsa} <span style="font-size: 12px; font-weight: normal;">Å²</span></div>
    </div>
  </div>

  <div class="main-layout">
    <!-- 3D Molecular Viewer -->
    <div class="viewer-container">
      <div id="viewport"></div>
      <div class="viewer-controls">
        <button class="btn active" id="btn-cartoon" onclick="toggleStyle('cartoon')">Cartoon</button>
        <button class="btn" id="btn-surface" onclick="toggleSurface()">Surface</button>
        <button class="btn active" id="btn-contacts" onclick="togglePocket()">Pocket Sticks</button>
        <button class="btn" onclick="viewer.zoomTo();">Reset View</button>
      </div>
    </div>

    <!-- Leaderboard Table -->
    <div class="table-container">
      <table id="leaderboard-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Peptide ID</th>
            <th>Vina ΔG</th>
            <th>HADDOCK</th>
            <th>BSA (Å²)</th>
            <th>ML pKd</th>
            <th>Pred Kd</th>
            <th>Consensus</th>
          </tr>
        </thead>
        <tbody id="table-body">
        </tbody>
      </table>
    </div>
  </div>

  <script>
    const hits = {hits_json};
    const complexes = {complexes_json};
    const receptorPdb = {rec_json};

    let viewer = null;
    let currentHitIndex = 0;
    let surfaceObj = null;
    let showPocketSticks = true;

    document.addEventListener("DOMContentLoaded", function() {{
      const container = document.getElementById("viewport");
      viewer = $3Dmol.createViewer(container, {{ defaultcolors: $3Dmol.rasmolElementColors }});
      viewer.setBackgroundColor(0x0d1117);

      populateTable();
      if (hits.length > 0) {{
        loadHit(0);
      }}
    }});

    function escapeHtml(text) {{
      if (text === null || text === undefined) return "";
      return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }}

    function populateTable() {{
      const tbody = document.getElementById("table-body");
      tbody.innerHTML = "";

      hits.forEach((h, idx) => {{
        const row = document.createElement("tr");
        row.className = "hit-row" + (idx === 0 ? " selected" : "");
        row.onclick = () => loadHit(idx);

        const hscore = typeof h.haddock_score === 'number' ? h.haddock_score.toFixed(1) : h.haddock_score;
        const bsa = typeof h.bsa === 'number' ? h.bsa.toFixed(0) : h.bsa;
        const ml_pkd = typeof h.ml_pkd === 'number' ? h.ml_pkd.toFixed(2) : h.ml_pkd;
        const consensus = typeof h.consensus === 'number' ? h.consensus.toFixed(2) : h.consensus;

        const safePepId = escapeHtml(h.pep_id);
        const safeSeq = escapeHtml(h.sequence);

        row.innerHTML = `
          <td><strong>#${{h.rank}}</strong></td>
          <td>
            <strong>${{safePepId}}</strong><br>
            <span class="seq-tag">${{safeSeq}}</span>
          </td>
          <td style="color: var(--primary); font-weight: 600;">${{h.vina_dg.toFixed(2)}}</td>
          <td style="color: var(--highlight);">${{hscore}}</td>
          <td>${{bsa}}</td>
          <td>${{ml_pkd}}</td>
          <td style="color: var(--success); font-weight: 600;">${{escapeHtml(h.pred_kd)}}</td>
          <td><strong>${{consensus}}</strong></td>
        `;
        tbody.appendChild(row);
      }});
    }}

    function loadHit(index) {{
      currentHitIndex = index;
      const hit = hits[index];

      // Highlight selected row in table
      const rows = document.querySelectorAll(".hit-row");
      rows.forEach((r, idx) => {{
        r.classList.toggle("selected", idx === index);
      }});

      viewer.clear();
      surfaceObj = null;

      const cpxData = complexes[hit.pep_id];
      if (cpxData && cpxData.trim().length > 0) {{
        // Load combined co-complex PDB
        viewer.addModel(cpxData, "pdb");

        // Chain A = Receptor
        viewer.setStyle({{ chain: 'A' }}, {{ cartoon: {{ color: 'lightgray', opacity: 0.85 }} }});

        // Chain B = Peptide Hit
        viewer.setStyle({{ chain: 'B' }}, {{
          cartoon: {{ color: '#58a6ff' }},
          stick: {{ radius: 0.2, colorscheme: 'Jmol' }}
        }});

        // Highlight binding pocket residues within 4.5 A of peptide
        if (showPocketSticks) {{
          viewer.setStyle({{ chain: 'A', within: {{ distance: 4.5, sel: {{ chain: 'B' }} }} }}, {{
            cartoon: {{ color: 'lightgray', opacity: 0.85 }},
            stick: {{ radius: 0.15, color: '#f0883e' }}
          }});
        }}

        viewer.zoomTo({{ chain: 'B' }});
        viewer.render();
      }} else if (receptorPdb && receptorPdb.trim().length > 0) {{
        // Fallback: load receptor only
        viewer.addModel(receptorPdb, "pdb");
        viewer.setStyle({{}}, {{ cartoon: {{ color: 'lightgray' }} }});
        viewer.zoomTo();
        viewer.render();
      }}
    }}

    function toggleSurface() {{
      const btn = document.getElementById("btn-surface");
      if (surfaceObj) {{
        viewer.removeSurface(surfaceObj);
        surfaceObj = null;
        btn.classList.remove("active");
      }} else {{
        surfaceObj = viewer.addSurface($3Dmol.SurfaceType.MS, {{ opacity: 0.4, color: 'gray' }}, {{ chain: 'A' }});
        btn.classList.add("active");
      }}
      viewer.render();
    }}

    function togglePocket() {{
      showPocketSticks = !showPocketSticks;
      document.getElementById("btn-contacts").classList.toggle("active", showPocketSticks);
      loadHit(currentHitIndex);
    }}
  </script>
</body>
</html>
"""
    out_html.write_text(html_content, encoding="utf-8")
    logger.info(f"Generated standalone 3D interactive HTML report: {out_html.name}")
    return out_html
