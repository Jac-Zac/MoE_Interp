#!/usr/bin/env python3
"""Build a self-contained, searchable HTML explorer for Expert Pursuit results.

Reads a pursuit `results.jsonl` (one record per layer/expert with `tokens` and
cumulative `evr`) and emits a single static HTML file with the data embedded.
The page lets you filter experts by layer, by token substring, and sort by EVR
or activation count, then inspect each expert's full atom list and EVR curve.

Usage:
    python scripts/build_expert_explorer.py \
        data/.../pursuit/pile10k/results.jsonl \
        -o data/.../pursuit/pile10k/expert_explorer.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(jsonl_path: Path) -> list[dict]:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            evr = r.get("evr") or [0.0]
            records.append(
                {
                    "layer": r["layer"],
                    "expert": r["expert"],
                    "n": r.get("n_activations", 0),
                    "tokens": r.get("tokens", []),
                    "evr": evr,
                }
            )
    return records


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OLMoE Expert Explorer &mdash; {title}</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 14px; line-height: 1.5; background: #f9f7f4; color: #1f1f1e;
  height: 100vh; display: flex; flex-direction: column;
}}
header {{
  background: #1f1f1e; color: #f0ece5; padding: 12px 22px;
  display: flex; align-items: baseline; gap: 16px; flex-shrink: 0;
  border-bottom: 2px solid rgba(200,149,106,0.6);
}}
header h1 {{ font-size: 1rem; font-weight: 700; }}
header .meta {{ font-size: .75rem; color: rgba(240,236,229,0.55); margin-left: auto; }}
.controls {{
  background: #fff; border-bottom: 1px solid rgba(0,0,0,0.08); padding: 9px 22px;
  display: flex; gap: 14px; align-items: center; flex-shrink: 0; flex-wrap: wrap;
}}
.controls label {{ font-size: .78rem; color: #555; display: flex; align-items: center; gap: 5px; }}
.controls select, .controls input {{
  font-family: inherit; border: 1px solid #ccc; border-radius: 3px; padding: 5px 8px;
  font-size: .8rem; background: #fafaf8; outline: none; color: #1f1f1e;
}}
.controls input[type=text] {{ width: 230px; }}
.controls .count {{ font-size: .75rem; color: #888; margin-left: auto; }}
.layout {{ display: flex; flex: 1; overflow: hidden; }}
.list {{ flex: 1; overflow-y: auto; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{
  position: sticky; top: 0; background: #efeae3; text-align: left; padding: 7px 12px;
  font-size: .72rem; text-transform: uppercase; letter-spacing: .4px; color: #6b6b68;
  cursor: pointer; user-select: none; border-bottom: 1px solid rgba(0,0,0,0.1);
}}
th.num {{ text-align: right; }}
th:hover {{ background: #e6dfd5; }}
td {{ padding: 6px 12px; border-bottom: 1px solid rgba(0,0,0,0.05); vertical-align: top; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; color: #444; }}
tr.row {{ cursor: pointer; }}
tr.row:hover {{ background: #fff6ec; }}
tr.row.sel {{ background: #f3e7d6; }}
.le {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-weight: 700; white-space: nowrap; }}
.toks {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .8rem; color: #333; }}
.evrbar {{ display: inline-block; height: 9px; background: #c8956a; border-radius: 2px; vertical-align: middle; }}
.evrwrap {{ display: flex; align-items: center; gap: 7px; justify-content: flex-end; }}
.detail {{
  width: 380px; flex-shrink: 0; background: #fff; border-left: 1px solid rgba(0,0,0,0.1);
  overflow-y: auto; padding: 18px 20px;
}}
.detail.empty {{ color: #aaa; display: flex; align-items: center; justify-content: center; text-align: center; }}
.detail h2 {{ font-size: 1.1rem; margin-bottom: 2px; }}
.detail .sub {{ font-size: .78rem; color: #888; margin-bottom: 14px; }}
.detail .stat {{ display: flex; justify-content: space-between; font-size: .82rem; padding: 3px 0; border-bottom: 1px dashed rgba(0,0,0,0.07); }}
.detail h3 {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .4px; color: #6b6b68; margin: 18px 0 8px; }}
.atomlist {{ list-style: none; }}
.atomlist li {{
  display: flex; align-items: center; gap: 8px; padding: 3px 0;
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .82rem;
}}
.atomlist .rank {{ color: #bbb; width: 22px; text-align: right; flex-shrink: 0; }}
.atomlist .tok {{ background: #f3ede4; padding: 1px 6px; border-radius: 3px; }}
svg .curve {{ fill: none; stroke: #c8956a; stroke-width: 2; }}
svg .axis {{ stroke: #ddd; stroke-width: 1; }}
mark {{ background: #ffe08a; padding: 0 1px; }}
</style>
</head>
<body>
<header>
  <h1>OLMoE Expert Explorer</h1>
  <span style="font-size:.8rem;color:rgba(240,236,229,0.7)">{title}</span>
  <span class="meta">{n_experts} experts &middot; {n_layers} layers &middot; {total_acts} activations</span>
</header>
<div class="controls">
  <label>Layer
    <select id="layerSel"><option value="">all</option>{layer_opts}</select>
  </label>
  <label>Topic family
    <select id="clusSel"><option value="">all</option>{cluster_opts}</select>
  </label>
  <label>Token search
    <input type="text" id="tokSearch" placeholder="e.g. amongst, protein, number">
  </label>
  <label><input type="checkbox" id="hideFrag"> hide pure fragments</label>
  <span class="count" id="count"></span>
</div>
<div class="layout">
  <div class="list">
    <table>
      <thead><tr>
        <th data-sort="le">Expert</th>
        <th data-sort="n" class="num">Acts</th>
        <th data-sort="evr25" class="num">EVR@25</th>
        <th data-sort="evr50" class="num">EVR@50</th>
        <th data-sort="toks">Top atoms</th>
      </tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
  <div class="detail empty" id="detail">Select an expert to inspect its atom basis.</div>
</div>
<script>
const DATA = {data_json};
const CLUSTERS = {cluster_json};
const maxEvr = Math.max(...DATA.map(d => d.evr50));
let sortKey = "evr50", sortDir = -1, selKey = null;

const $ = id => document.getElementById(id);
const esc = s => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const fragRe = /^[^a-zA-Z0-9]*$/;
const isFrag = t => {{ const s=t.trim(); return s.length>0 && s.length<=3 && /^[a-z]+$/.test(s)===false && fragRe.test(s)===false ? false : (s.length<=2); }};

function fmt(x) {{ return x.toFixed(3); }}

function filtered() {{
  const L = $("layerSel").value;
  const C = $("clusSel").value;
  const q = $("tokSearch").value.trim().toLowerCase();
  const hf = $("hideFrag").checked;
  let rows = DATA.filter(d => {{
    if (L !== "" && d.layer != L) return false;
    if (C !== "" && String(d.cluster) !== C) return false;
    if (q && !d.tokens.some(t => t.toLowerCase().includes(q))) return false;
    return true;
  }});
  rows.sort((a,b) => {{
    let va = a[sortKey], vb = b[sortKey];
    if (sortKey === "le") {{ va = a.layer*100+a.expert; vb = b.layer*100+b.expert; }}
    if (sortKey === "toks") {{ va = a.tokens[0]||""; vb = b.tokens[0]||""; return sortDir*String(va).localeCompare(String(vb)); }}
    return sortDir*(va-vb);
  }});
  return {{rows, q, hf}};
}}

function renderList() {{
  const {{rows, q, hf}} = filtered();
  const tb = $("tbody");
  const mark = t => {{
    let e = esc(t);
    if (q) e = e.replace(new RegExp("("+q.replace(/[.*+?^${{}}()|[\\]\\\\]/g,"\\\\$&")+")","ig"), "<mark>$1</mark>");
    return e;
  }};
  tb.innerHTML = rows.map(d => {{
    let toks = d.tokens;
    if (hf) toks = toks.filter(t => t.trim().length > 2 && /[a-zA-Z]{{3}}/.test(t));
    const preview = toks.slice(0,9).map(t => `<span class="tok">${{mark(t)}}</span>`).join(" ");
    const key = d.layer+"_"+d.expert;
    const w = Math.round(60*d.evr50/maxEvr);
    return `<tr class="row ${{key===selKey?'sel':''}}" data-key="${{key}}">
      <td class="le">L${{String(d.layer).padStart(2,'0')}} E${{String(d.expert).padStart(2,'0')}}</td>
      <td class="num">${{d.n.toLocaleString()}}</td>
      <td class="num">${{fmt(d.evr25)}}</td>
      <td class="num"><div class="evrwrap"><span class="evrbar" style="width:${{w}}px"></span>${{fmt(d.evr50)}}</div></td>
      <td class="toks">${{preview}}</td>
    </tr>`;
  }}).join("");
  $("count").textContent = rows.length + " / " + DATA.length + " experts";
  tb.querySelectorAll("tr.row").forEach(tr =>
    tr.onclick = () => {{ selKey = tr.dataset.key; renderList(); renderDetail(); }});
}}

function sparkline(evr) {{
  const W=340, H=90, n=evr.length;
  const x = i => 4 + i*(W-8)/(n-1);
  const y = v => H-4 - v*(H-8)/maxEvr;
  const pts = evr.map((v,i)=>`${{x(i).toFixed(1)}},${{y(v).toFixed(1)}}`).join(" ");
  return `<svg width="${{W}}" height="${{H}}" style="margin-top:4px">
    <line class="axis" x1="4" y1="${{H-4}}" x2="${{W-4}}" y2="${{H-4}}"/>
    <polyline class="curve" points="${{pts}}"/></svg>
    <div style="font-size:.7rem;color:#999;display:flex;justify-content:space-between">
      <span>1 atom</span><span>${{n}} atoms</span></div>`;
}}

function renderDetail() {{
  const d = DATA.find(r => r.layer+"_"+r.expert === selKey);
  const el = $("detail");
  if (!d) {{ el.className="detail empty"; el.textContent="Select an expert."; return; }}
  el.className = "detail";
  const cl = CLUSTERS[d.cluster];
  const clHtml = cl ? `<div class="stat"><span>Topic family</span><b title="${{cl.terms.join(', ')}}">${{cl.label}}</b></div>` : "";
  const atoms = d.tokens.map((t,i) =>
    `<li><span class="rank">${{i+1}}</span><span class="tok">${{esc(t)}}</span></li>`).join("");
  el.innerHTML = `
    <h2>L${{String(d.layer).padStart(2,'0')}} &middot; E${{String(d.expert).padStart(2,'0')}}</h2>
    <div class="sub">layer ${{d.layer}}, expert ${{d.expert}}</div>
    <div class="stat"><span>Activations</span><b>${{d.n.toLocaleString()}}</b></div>
    ${{clHtml}}
    <div class="stat"><span>EVR @ 25 atoms</span><b>${{fmt(d.evr25)}}</b></div>
    <div class="stat"><span>EVR @ ${{d.evr.length}} atoms</span><b>${{fmt(d.evr50)}}</b></div>
    <h3>Cumulative EVR curve</h3>
    ${{sparkline(d.evr)}}
    <h3>Atom basis (ranked)</h3>
    <ul class="atomlist">${{atoms}}</ul>`;
}}

document.querySelectorAll("th[data-sort]").forEach(th =>
  th.onclick = () => {{
    const k = th.dataset.sort;
    if (sortKey === k) sortDir *= -1; else {{ sortKey = k; sortDir = (k==="le"||k==="toks")?1:-1; }}
    renderList();
  }});
["layerSel","clusSel","tokSearch","hideFrag"].forEach(id => {{
  $(id).addEventListener("input", renderList);
  $(id).addEventListener("change", renderList);
}});
renderList();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path, help="path to pursuit results.jsonl")
    ap.add_argument("-o", "--output", type=Path, default=None, help="output html path")
    ap.add_argument("--title", default=None, help="title shown in the header")
    ap.add_argument(
        "--clusters",
        type=Path,
        default=None,
        help="optional clusters.json from analyze_pursuit.py (default: alongside results)",
    )
    args = ap.parse_args()

    records = load_records(args.results)
    if not records:
        raise SystemExit(f"no records in {args.results}")

    layers = sorted({r["layer"] for r in records})
    total_acts = sum(r["n"] for r in records)

    # Optional topic clusters: map (layer, expert) -> cluster id, and build a
    # short readable label per cluster from its top TF-IDF terms.
    clus_path = args.clusters or args.results.with_name("clusters.json")
    cl_of: dict[tuple[int, int], int] = {}
    cluster_meta: dict[str, dict] = {}
    if clus_path.exists():
        cdata = json.loads(clus_path.read_text())
        cl_of = {(d["layer"], d["expert"]): d["cluster"] for d in cdata["labels"]}
        for cid, info in cdata["clusters"].items():
            label = ", ".join(info["terms"][:3])
            cluster_meta[cid] = {
                "label": label,
                "terms": info["terms"],
                "size": info["size"],
            }

    # Compact payload: EVR@25, EVR@final, plus the full curve and tokens.
    payload = []
    for r in records:
        evr = r["evr"]
        payload.append(
            {
                "layer": r["layer"],
                "expert": r["expert"],
                "n": r["n"],
                "tokens": r["tokens"],
                "cluster": cl_of.get((r["layer"], r["expert"]), -1),
                "evr25": round(evr[min(24, len(evr) - 1)], 4),
                "evr50": round(evr[-1], 4),
                "evr": [round(v, 4) for v in evr],
            }
        )

    # Cluster dropdown options, ordered by mean EVR (highest-EVR family first).
    cluster_opts = ""
    if cluster_meta:

        def _mean_evr(cid: str) -> float:
            vals = [p["evr50"] for p in payload if str(p["cluster"]) == cid]
            return sum(vals) / len(vals) if vals else 0.0

        order = sorted(cluster_meta.items(), key=lambda kv: -_mean_evr(kv[0]))
        cluster_opts = "".join(
            f"<option value='{cid}'>{m['label']} ({m['size']})</option>"
            for cid, m in order
        )

    title = args.title or args.results.parent.name
    html = HTML_TEMPLATE.format(
        title=title,
        n_experts=len(records),
        n_layers=len(layers),
        total_acts=f"{total_acts:,}",
        layer_opts="".join(f"<option value='{L}'>L{L:02d}</option>" for L in layers),
        cluster_opts=cluster_opts,
        data_json=json.dumps(payload, ensure_ascii=False),
        cluster_json=json.dumps(cluster_meta, ensure_ascii=False),
    )

    out = args.output or args.results.with_name("expert_explorer.html")
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(records)} experts)")


if __name__ == "__main__":
    main()
