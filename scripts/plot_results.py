"""Scatter plots of the model comparison, with the Pareto front marked. No dependencies: writes SVG.

  python scripts/plot_results.py results/models-2026-09-25.json docs/

Two charts: hard-set accuracy against time per decision, and against cost per 1,000 decisions. A model is on the
Pareto front when no other model is both at least as accurate and at least as fast (or cheap), and better on one.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

COLORS = {"OpenAI": "#10a37f", "Anthropic": "#d97757", "local": "#3b6fd8"}
W, H = 820, 470
ML, MR, MT, MB = 64, 24, 40, 56


def pareto(points, x):
    """Non-dominated points for (smaller x, larger accuracy), ordered by x."""
    front, best = [], -1.0
    for p in sorted(points, key=lambda p: (p[x], -p["hard"])):
        if p["hard"] > best + 1e-9:
            front.append(p)
            best = p["hard"]
    return front


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def chart(rows, x, title, xlabel, fmt, out: Path, note: str = ""):
    pts = [r for r in rows if r.get(x) is not None and r[x] > 0]
    lo, hi = min(p[x] for p in pts), max(p[x] for p in pts)
    lx0, lx1 = math.log10(lo) - 0.08, math.log10(hi) + 0.08
    y0, y1 = 0.6, 1.02
    px = lambda v: ML + (math.log10(v) - lx0) / (lx1 - lx0) * (W - ML - MR)
    py = lambda v: MT + (y1 - v) / (y1 - y0) * (H - MT - MB)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{ML}" y="24" font-size="15" font-weight="600" fill="#1f2328">{esc(title)}</text>']
    # grid and axes
    for yv in (0.6, 0.7, 0.8, 0.9, 1.0):
        s.append(f'<line x1="{ML}" x2="{W - MR}" y1="{py(yv):.1f}" y2="{py(yv):.1f}" stroke="#e5e7eb"/>')
        s.append(f'<text x="{ML - 8}" y="{py(yv) + 4:.1f}" text-anchor="end" fill="#57606a">{yv:.1f}</text>')
    decade = 10 ** math.floor(lx0)
    while decade <= 10 ** lx1:
        for m in (1, 2, 5):
            v = decade * m
            if 10 ** lx0 <= v <= 10 ** lx1:
                s.append(f'<line x1="{px(v):.1f}" x2="{px(v):.1f}" y1="{MT}" y2="{H - MB}" stroke="#eef0f2"/>')
                s.append(f'<text x="{px(v):.1f}" y="{H - MB + 18}" text-anchor="middle" fill="#57606a">{esc(fmt(v))}</text>')
        decade *= 10
    s.append(f'<text x="{(ML + W - MR) / 2}" y="{H - 14}" text-anchor="middle" fill="#1f2328">{esc(xlabel)}</text>')
    s.append(f'<text transform="translate(16,{(MT + H - MB) / 2}) rotate(-90)" text-anchor="middle" fill="#1f2328">'
             'accuracy, hard set (30 decisions)</text>')
    # the front: a step line, then the points
    front = pareto(pts, x)
    path = []
    for i, p in enumerate(front):
        path.append(f"{'M' if i == 0 else 'L'}{px(p[x]):.1f},{py(p['hard']):.1f}")
        if i + 1 < len(front):
            path.append(f"L{px(front[i + 1][x]):.1f},{py(p['hard']):.1f}")
    s.append(f'<path d="{" ".join(path)}" fill="none" stroke="#1f2328" stroke-width="1.4" stroke-dasharray="5 4"/>')
    on_front = {id(p) for p in front}
    for p in sorted(pts, key=lambda p: id(p) in on_front):
        cx, cy, c = px(p[x]), py(p["hard"]), COLORS.get(p["provider"], "#888")
        stroke = ' stroke="#1f2328" stroke-width="1.6"' if id(p) in on_front else ' stroke="#ffffff" stroke-width="1"'
        tip = f"<title>{esc(p['name'])}: {p['hard']:.2f}, {esc(fmt(p[x]))}</title>"
        if p["how"].startswith("answer"):
            s.append(f'<rect x="{cx - 5:.1f}" y="{cy - 5:.1f}" width="10" height="10" transform="rotate(45 {cx:.1f} {cy:.1f})" '
                     f'fill="{c}"{stroke}>{tip}</rect>')
        else:
            s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{c}"{stroke}>{tip}</circle>')
    for p in front:
        cx, cy = px(p[x]), py(p["hard"])
        anchor, dx = ("end", -9) if cx > W - 190 else ("start", 9)
        s.append(f'<text x="{cx + dx:.1f}" y="{cy - 8:.1f}" text-anchor="{anchor}" fill="#1f2328" font-weight="600">'
                 f'{esc(p["name"])} · {esc(fmt(p[x]))}</text>')
    # legend
    present = {p["provider"] for p in pts}
    items = [it for it in [("OpenAI", "circle", "OpenAI, logprob readout"),
                           ("Anthropic", "diamond", "Anthropic, answer mode (may reason)"),
                           ("local", "circle", "local 9B, logprob readout")] if it[0] in present]
    bw, bh = 300, 24 + 17 * (len(items) + 1) + (16 if note else 0)
    lx, ly = W - MR - bw + 4, H - MB - bh + 8
    s.append(f'<rect x="{lx - 10}" y="{ly - 16}" width="{bw}" height="{bh}" fill="#ffffff" stroke="#d0d7de" rx="4"/>')
    for i, (prov, shape, label) in enumerate(items):
        yy = ly + i * 17
        if shape == "circle":
            s.append(f'<circle cx="{lx}" cy="{yy - 4}" r="5" fill="{COLORS[prov]}"/>')
        else:
            s.append(f'<rect x="{lx - 4.5}" y="{yy - 8.5}" width="9" height="9" transform="rotate(45 {lx} {yy - 4})" fill="{COLORS[prov]}"/>')
        s.append(f'<text x="{lx + 12}" y="{yy}" fill="#1f2328">{esc(label)}</text>')
    yy = ly + len(items) * 17
    s.append(f'<path d="M{lx - 5},{yy - 4} L{lx + 6},{yy - 4}" stroke="#1f2328" stroke-dasharray="5 4"/>'
             f'<text x="{lx + 12}" y="{yy}" fill="#1f2328">Pareto front (outlined points)</text>')
    if note:
        s.append(f'<text x="{lx - 2}" y="{yy + 18}" fill="#57606a" font-size="11">{esc(note)}</text>')
    s.append("</svg>")
    out.write_text("\n".join(s))
    return front


def main():
    rows = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs")
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r["seconds"] = r["dec"] / 1000 if r.get("dec") else None
    ft = chart(rows, "seconds", "Accuracy against time per decision (median, both option orders)",
               "time per decision, seconds (log scale)", lambda v: f"{v:.2g} s", out / "pareto-time.svg",
               note=next((r["time_note"] for r in rows if r.get("time_note")), ""))
    fc = chart(rows, "cost", "Accuracy against cost (list prices, September 2026)",
               "US dollars per 1,000 decisions (log scale)", lambda v: f"${v:.3g}", out / "pareto-cost.svg",
               note="Not shown: local 9B (free); gpt-4, gpt-4-turbo (no list price)")
    for name, f in (("time", ft), ("cost", fc)):
        print(name, "front:", ", ".join(p["name"] for p in f))


if __name__ == "__main__":
    main()
