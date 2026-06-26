"""Phase 7 — HTML Report Generator.

Generates a fully self-contained, interactive, dark-themed HTML report 
without any external CDN dependencies.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any

from neurodiff.engines.context_engine import GlobalContext
from neurodiff.engines.correction_engine import CodeCorrection

@dataclass
class FullReport:
    """Aggregated report of all engines."""
    metadata: dict
    semantic_events: list[Any]
    security: list[Any]
    duplication: list[Any]
    arch: Any
    cognitive: Any
    llm: Any | None
    global_context: GlobalContext | None
    corrections: list[CodeCorrection]


def generate_html(report: FullReport) -> str:
    """Generates a complete standalone HTML string from a FullReport."""
    
    css = """
    :root {
        --bg: #0f172a;
        --surface: #1e293b;
        --surface-hover: #334155;
        --border: #334155;
        --text: #f8fafc;
        --text-dim: #94a3b8;
        --accent: #3b82f6;
        --green: #22c55e;
        --yellow: #eab308;
        --orange: #f97316;
        --red: #ef4444;
        --font-sans: system-ui, -apple-system, sans-serif;
        --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    * { box-sizing: border-box; }
    body {
        margin: 0; padding: 0;
        background-color: var(--bg);
        color: var(--text);
        font-family: var(--font-sans);
        line-height: 1.5;
    }
    header {
        background: var(--surface);
        padding: 1rem 2rem;
        border-bottom: 1px solid var(--border);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    .grid-4 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem;
    }
    .card-title {
        font-size: 0.875rem;
        text-transform: uppercase;
        color: var(--text-dim);
        margin: 0 0 0.5rem 0;
    }
    .card-value {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0;
    }
    h2 { border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; margin-top: 3rem; }
    table {
        width: 100%; border-collapse: collapse; margin-bottom: 1rem;
    }
    th, td {
        padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border);
    }
    th { color: var(--text-dim); font-weight: normal; }
    .badge {
        display: inline-block; padding: 0.25rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: bold;
    }
    .bg-red { background: #ef444433; color: var(--red); border: 1px solid var(--red); }
    .bg-yellow { background: #eab30833; color: var(--yellow); border: 1px solid var(--yellow); }
    .bg-green { background: #22c55e33; color: var(--green); border: 1px solid var(--green); }
    
    .diff-container { display: flex; gap: 1rem; margin-top: 1rem; overflow-x: auto; }
    .diff-pane { flex: 1; min-width: 300px; background: #000; border: 1px solid var(--border); border-radius: 4px; padding: 1rem; }
    .diff-pane h4 { margin-top: 0; color: var(--text-dim); }
    pre { font-family: var(--font-mono); font-size: 0.875rem; margin: 0; white-space: pre-wrap; }
    .line-removed { background-color: #ef444433; display: block; }
    .line-added { background-color: #22c55e33; display: block; }
    
    .btn {
        background: var(--surface-hover); color: var(--text); border: 1px solid var(--border);
        padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; font-size: 0.875rem; display: inline-flex; align-items: center; gap: 0.5rem;
    }
    .btn:hover { background: #475569; }
    .btn-primary { background: var(--accent); color: white; border: none; }
    .btn-primary:hover { background: #2563eb; }
    
    .actions { display: flex; gap: 0.5rem; margin-top: 1rem; }
    
    svg.gauge { width: 100%; max-width: 300px; height: auto; }
    
    .finding-row { cursor: pointer; }
    .finding-row:hover { background: var(--surface); }
    .finding-details { display: none; padding: 1rem; background: var(--surface); border-bottom: 1px solid var(--border); }
    .finding-details.active { display: table-cell; }
    
    @media print {
        body { background: white; color: black; }
        .card, header, .diff-pane { border-color: #ccc; }
    }
    """

    js = """
    function toggleDetails(id) {
        const el = document.getElementById(id);
        if (el.style.display === 'table-cell' || el.classList.contains('active')) {
            el.style.display = 'none';
            el.classList.remove('active');
        } else {
            el.style.display = 'table-cell';
            el.classList.add('active');
        }
    }
    function copyText(id) {
        const text = document.getElementById(id).innerText;
        navigator.clipboard.writeText(text).then(() => alert("Copied to clipboard!"));
    }
    function downloadPatch(contentId, filename) {
        const text = document.getElementById(contentId).innerText;
        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }
    function applyPatch(findingId) {
        alert("To apply this patch, run in your terminal:\\n\\nneurodiff apply --finding-id " + findingId);
    }
    """
    
    # Process Metrics
    num_crit_sec = sum(1 for f in report.security if getattr(f, "severity", "") == "critical")
    num_high_sec = sum(1 for f in report.security if getattr(f, "severity", "") == "high")
    sec_risk = f"🔴 {num_crit_sec} CRIT" if num_crit_sec > 0 else f"🟠 {num_high_sec} HIGH" if num_high_sec > 0 else "🟢 SAFE"
    
    num_high_dup = sum(1 for f in report.duplication if getattr(f, "similarity_score", 0) > 0.9)
    dup_risk = f"🟠 {num_high_dup} HIGH" if num_high_dup > 0 else "🟢 SAFE"
    
    num_arch = len(getattr(report.arch, "layer_violations", [])) + len(getattr(report.arch, "circular_deps", []))
    arch_risk = f"🔴 {len(getattr(report.arch, 'circular_deps', []))} CIRC" if getattr(report.arch, "circular_deps", []) else f"{num_arch} VIOLATIONS"
    
    cfi = report.cognitive.fatigue_index if report.cognitive else None
    cfi_text = f"CFI: {cfi.total_score}/{cfi.grade}" if cfi else "N/A"
    ai_prob = report.cognitive.ai_generated.probability if report.cognitive else 0
    ai_text = f"AI: {ai_prob:.0%}"
    
    overall = getattr(report.cognitive, "overall_verdict", "safe").upper()
    overall_icon = "🔴" if overall == "DANGER" else "⚠️" if overall == "CAUTION" else "✅"

    # Gauge SVG for CFI
    gauge_svg = ""
    if cfi:
        grade_colors = {"A": "#22c55e", "B": "#84cc16", "C": "#eab308", "D": "#f97316", "F": "#ef4444"}
        color = grade_colors.get(cfi.grade, "#ccc")
        # map 0-100 to 0-180 degrees (SVG dasharray math: circumference of semi-circle radius 80 is ~251)
        # full circle = 2*pi*r = 502. 
        # dasharray="251 502" shows half.
        val_pct = cfi.total_score / 100.0
        dash_fill = val_pct * 251.2
        dash_empty = 502.4 - dash_fill
        
        gauge_svg = f"""
        <svg class="gauge" viewBox="0 0 200 110" xmlns="http://www.w3.org/2000/svg">
            <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="var(--border)" stroke-width="16" stroke-linecap="round"/>
            <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round" stroke-dasharray="{dash_fill} {dash_empty}"/>
            <text x="100" y="90" text-anchor="middle" fill="var(--text)" font-size="24" font-weight="bold">{cfi.total_score}</text>
            <text x="100" y="105" text-anchor="middle" fill="var(--text-dim)" font-size="12">Grade {cfi.grade}</text>
        </svg>
        """
        
    # Blast Radius SVG (concentric rings)
    br = report.cognitive.blast_radius if report.cognitive else None
    br_svg = ""
    if br:
        br_color = {"contained": "#22c55e", "moderate": "#eab308", "wide": "#f97316", "critical": "#ef4444"}.get(br.risk_score, "#ccc")
        br_svg = f"""
        <svg class="gauge" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <circle cx="100" cy="100" r="90" fill="none" stroke="var(--border)" stroke-width="1" />
            <circle cx="100" cy="100" r="60" fill="none" stroke="var(--border)" stroke-width="1" />
            <circle cx="100" cy="100" r="30" fill="{br_color}33" stroke="{br_color}" stroke-width="2" />
            <text x="100" y="105" text-anchor="middle" fill="var(--text)" font-size="12" font-weight="bold">Epicenter</text>
            <text x="100" y="60" text-anchor="middle" fill="var(--text-dim)" font-size="10">Ring 1 ({len(br.first_ring)})</text>
            <text x="100" y="25" text-anchor="middle" fill="var(--text-dim)" font-size="10">Ring 2+3 ({len(br.second_ring) + len(br.third_ring)})</text>
        </svg>
        """

    # Active Corrections HTML
    corrections_html = ""
    if report.corrections:
        corrections_html = "<h2>🔧 Active Corrections</h2>"
        for i, corr in enumerate(report.corrections):
            conf_pct = int(corr.confidence * 100)
            conf_bar = "█" * int(conf_pct / 10) + "░" * (10 - int(conf_pct / 10))
            
            orig_lines = ""
            for ln in corr.original_code.splitlines():
                orig_lines += f"<span class='line-removed'>- {ln}</span>\n"
            
            corr_lines = ""
            for ln in corr.corrected_code.splitlines():
                corr_lines += f"<span class='line-added'>+ {ln}</span>\n"
                
            corrections_html += f"""
            <div class="card" style="margin-bottom: 1rem;">
                <h3 style="margin-top:0; color:var(--red);">🔴 {corr.finding_id}</h3>
                <div style="color:var(--text-dim); margin-bottom: 1rem;">{corr.file_path}</div>
                
                <div class="diff-container">
                    <div class="diff-pane">
                        <h4>ORIGINAL CODE</h4>
                        <pre>{orig_lines}</pre>
                    </div>
                    <div class="diff-pane">
                        <h4>CORRECTED CODE</h4>
                        <pre id="code-{i}">{corr_lines}</pre>
                        <pre id="patch-{i}" style="display:none;">{corr.patch}</pre>
                    </div>
                </div>
                
                <div style="margin-top: 1rem; background: var(--bg); padding: 1rem; border-radius: 4px;">
                    <div>💡 <strong>Explanation:</strong> {corr.explanation}</div>
                    <div style="margin-top: 0.5rem;">⚠️ <strong>Breaking changes:</strong> {', '.join(corr.breaking_changes) or 'None'}</div>
                    <div style="margin-top: 0.5rem;">📋 <strong>Follow-up:</strong> {', '.join(corr.follow_up_tasks) or 'None'}</div>
                    <div style="margin-top: 0.5rem;">Confidence: {conf_bar} {conf_pct}%</div>
                </div>
                
                <div class="actions">
                    <button class="btn" onclick="copyText('code-{i}')">📋 Copy corrected code</button>
                    <button class="btn" onclick="downloadPatch('patch-{i}', '{corr.finding_id}.patch')">⬇ Download .patch</button>
                    <button class="btn btn-primary" onclick="applyPatch('{corr.finding_id}')">✅ Apply</button>
                </div>
            </div>
            """

    # Global Context HTML
    context_html = ""
    if report.global_context:
        am = report.global_context.arch_map
        conv = report.global_context.project_conventions
        context_html = f"""
        <h2>🗺️ Project Understanding</h2>
        <div class="card">
            <div class="grid-4" style="margin-bottom:0;">
                <div>
                    <div class="card-title">Framework</div>
                    <div>{am.detected_framework}</div>
                </div>
                <div>
                    <div class="card-title">Pattern</div>
                    <div>{', '.join(am.detected_patterns) or 'N/A'}</div>
                </div>
                <div>
                    <div class="card-title">Conventions</div>
                    <div>{conv.get("naming_style")}, {conv.get("docstring_style")} docstrings</div>
                </div>
                <div>
                    <div class="card-title">Scope</div>
                    <div>{am.total_functions} functions, {am.total_classes} classes</div>
                </div>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>NeuroDiff Report</title>
    <style>{css}</style>
    <script>{js}</script>
</head>
<body>
    <header>
        <div><strong>NeuroDiff Report</strong> | Repo: {report.metadata.get('repo', 'unknown')} | {report.metadata.get('base_ref')} &rarr; {report.metadata.get('head_ref')}</div>
        <div>Overall Risk: <strong>{overall_icon} {overall}</strong> | Generated in {report.metadata.get('duration_s', 0):.1f}s</div>
    </header>
    
    <div class="container">
        <div class="grid-4">
            <div class="card">
                <div class="card-title">SECURITY</div>
                <div class="card-value">{len(report.security)} findings</div>
                <div style="color:var(--text-dim); margin-top:0.5rem;">{sec_risk}</div>
            </div>
            <div class="card">
                <div class="card-title">DUPLICATION</div>
                <div class="card-value">{len(report.duplication)} warnings</div>
                <div style="color:var(--text-dim); margin-top:0.5rem;">{dup_risk}</div>
            </div>
            <div class="card">
                <div class="card-title">ARCHITECTURE</div>
                <div class="card-value">{num_arch} violations</div>
                <div style="color:var(--text-dim); margin-top:0.5rem;">{arch_risk}</div>
            </div>
            <div class="card">
                <div class="card-title">COGNITIVE</div>
                <div class="card-value">{cfi_text}</div>
                <div style="color:var(--text-dim); margin-top:0.5rem;">{ai_text}</div>
            </div>
        </div>
        
        {corrections_html}
        
        <div style="display:flex; gap:2rem;">
            <div style="flex:1;">
                <h2>🧠 Cognitive Load</h2>
                <div class="card" style="text-align:center;">
                    {gauge_svg}
                </div>
            </div>
            <div style="flex:1;">
                <h2>🏗️ Blast Radius</h2>
                <div class="card" style="text-align:center;">
                    {br_svg}
                </div>
            </div>
        </div>
        
        {context_html}
        
        <!-- Add more sections here for detailed tables of security, arch, etc. as needed -->
        
        <footer style="margin-top: 4rem; text-align: center; color: var(--text-dim); font-size: 0.875rem;">
            <p>🤖 NeuroDiff v0.7.0 | Engines: ast, security, duplication, arch, cognitive, context, llm, corrections</p>
            <p>{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>"""
    return html
