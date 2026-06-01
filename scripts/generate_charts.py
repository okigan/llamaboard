#!/usr/bin/env python3
"""Generate benchmark charts from llamaboard result JSON files.

Outputs:
  - artifacts/charts/*.svg  (static charts for README)
  - artifacts/site/index.html (interactive GitHub Pages site)
"""
import argparse
import json
from pathlib import Path
from typing import Any


def load_results(results_root: Path) -> list[dict[str, Any]]:
    """Load all JSON result files into a flat list of run records."""
    records: list[dict[str, Any]] = []

    for file in sorted(results_root.glob("**/*.json")):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            continue

        schema = data.get("schema_version", 1)
        hw = data.get("hardware", {})
        timestamp = data.get("run_timestamp_utc", "")
        commit = data.get("llama_cpp", {}).get("commit", "")[:8]
        host_hash = data.get("host_sha256", "")[:12]

        # Derive a human-friendly host label
        cpu = hw.get("cpu_model") or "unknown"
        cpu_label = cpu.split("@")[0].strip()
        host_gpu = hw.get("gpu_model")
        motherboard = (hw.get("motherboard") or "").strip()
        if motherboard in ("n/a", "n/a ", "None"):
            motherboard = ""
        host_label = cpu_label
        if host_gpu:
            host_label += f" + {host_gpu}"

        runs = data.get("runs", [])
        for run in runs:
            if not isinstance(run, dict):
                continue
            metrics = run.get("metrics", {})
            # Support both schema v3 (model_id in benchmark) and v4 (model_id per run)
            model_id = run.get("model_id") or data.get("benchmark", {}).get("model_id", "")
            target = run.get("target", "?")

            # Device label: actual GPU model for GPU runs, CPU model for CPU runs
            if target.startswith("gpu"):
                run_gpu = (run.get("gpu") or {}).get("model")
                device = run_gpu or host_gpu or target
            else:
                device = cpu_label

            for pp in metrics.get("pp", []):
                records.append({
                    "timestamp": timestamp,
                    "host_hash": host_hash,
                    "host_label": host_label,
                    "motherboard": motherboard,
                    "device": device,
                    "commit": commit,
                    "model_id": model_id,
                    "target": target,
                    "test": f"pp{pp['tokens']}",
                    "tokens_per_sec": pp["tokens_per_sec"],
                })
            for tg in metrics.get("tg", []):
                records.append({
                    "timestamp": timestamp,
                    "host_hash": host_hash,
                    "host_label": host_label,
                    "motherboard": motherboard,
                    "device": device,
                    "commit": commit,
                    "model_id": model_id,
                    "target": target,
                    "test": f"tg{tg['tokens']}",
                    "tokens_per_sec": tg["tokens_per_sec"],
                })

    return records


def generate_svg_charts(records: list[dict[str, Any]], output_dir: Path) -> list[str]:
    """Generate static SVG bar charts. Returns list of generated file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # Group by test type (pp512, tg128)
    tests = sorted(set(r["test"] for r in records))

    for test in tests:
        test_records = [r for r in records if r["test"] == test]
        # Group by (host_label, model_id, target)
        bars: list[dict[str, Any]] = []
        seen: set[str] = set()
        for r in test_records:
            key = f"{r['host_label']}|{r['device']}|{r['model_id']}|{r['target']}"
            if key in seen:
                continue
            seen.add(key)
            model_short = r["model_id"].split("/")[-1].replace("-GGUF", "")
            top = r["motherboard"] or r["host_label"]
            kind = "GPU" if r["target"].startswith("gpu") else "CPU"
            bars.append({
                "label": f"{top}\n{r['device']}\n{kind}",
                "value": r["tokens_per_sec"],
                "target": r["target"],
            })

        bars.sort(key=lambda x: x["value"], reverse=True)

        if not bars:
            continue

        # SVG dimensions
        bar_height = 32
        label_width = 380
        max_val = max(b["value"] for b in bars)
        chart_width = 400
        padding = 20
        svg_width = label_width + chart_width + padding * 2
        svg_height = len(bars) * (bar_height + 8) + 60

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" '
            f'font-family="monospace" font-size="12">',
            f'<rect width="{svg_width}" height="{svg_height}" fill="#0d1117"/>',
            f'<text x="{svg_width // 2}" y="24" text-anchor="middle" fill="#e6edf3" font-size="16" font-weight="bold">'
            f'{test} — tokens/sec (higher is better)</text>',
        ]

        y = 50
        for bar in bars:
            bar_w = int((bar["value"] / max_val) * chart_width) if max_val > 0 else 0
            color = "#58a6ff" if bar["target"].startswith("gpu") else "#3fb950"

            # Label lines: motherboard, device, CPU/GPU kind
            label_parts = bar["label"].split("\n")
            lines.append(
                f'<text x="{label_width - 8}" y="{y + 8}" text-anchor="end" fill="#8b949e" font-size="11">'
                f'{label_parts[0]}</text>'
            )
            if len(label_parts) > 1:
                lines.append(
                    f'<text x="{label_width - 8}" y="{y + 21}" text-anchor="end" fill="#6e7681" font-size="10">'
                    f'{label_parts[1]}</text>'
                )
            if len(label_parts) > 2:
                lines.append(
                    f'<text x="{label_width - 8}" y="{y + 34}" text-anchor="end" fill="{color}" '
                    f'font-size="10" font-weight="bold">{label_parts[2]}</text>'
                )

            # Bar
            lines.append(
                f'<rect x="{label_width}" y="{y + 4}" width="{bar_w}" height="{bar_height - 8}" '
                f'fill="{color}" rx="3"/>'
            )
            # Value
            lines.append(
                f'<text x="{label_width + bar_w + 6}" y="{y + 20}" fill="#e6edf3" font-size="12">'
                f'{bar["value"]:.1f}</text>'
            )

            y += bar_height + 8

        # Legend
        y += 10
        lines.append(f'<rect x="{label_width}" y="{y}" width="12" height="12" fill="#3fb950" rx="2"/>')
        lines.append(f'<text x="{label_width + 18}" y="{y + 10}" fill="#8b949e" font-size="11">CPU</text>')
        lines.append(f'<rect x="{label_width + 60}" y="{y}" width="12" height="12" fill="#58a6ff" rx="2"/>')
        lines.append(f'<text x="{label_width + 78}" y="{y + 10}" fill="#8b949e" font-size="11">GPU</text>')

        lines.append("</svg>")

        svg_path = output_dir / f"{test}.svg"
        svg_path.write_text("\n".join(lines), encoding="utf-8")
        generated.append(str(svg_path))

    return generated


def generate_html_site(records: list[dict[str, Any]], output_dir: Path) -> str:
    """Generate an interactive HTML page with Chart.js charts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for JS
    records_json = json.dumps(records, indent=None)

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llamaboard — llama.cpp benchmark results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 20px; }}
  h1 {{ text-align: center; }}
  .chart-container {{ max-width: 900px; margin: 30px auto; background: #161b22; border-radius: 8px; padding: 20px; }}
  select {{ background: #21262d; color: #e6edf3; border: 1px solid #30363d; padding: 6px 12px; border-radius: 6px; margin: 0 8px; }}
  .controls {{ text-align: center; margin: 20px 0; }}
  footer {{ text-align: center; color: #6e7681; margin-top: 40px; font-size: 12px; }}
</style>
</head>
<body>
<h1>llamaboard</h1>
<p style="text-align:center;color:#8b949e">llama.cpp inference benchmarks across hardware</p>

<div class="controls">
  <label>Test: <select id="testSelect"></select></label>
  <label>Model: <select id="modelSelect"><option value="">All</option></select></label>
  <label>Target: <select id="targetSelect"><option value="">All</option><option value="cpu">CPU</option><option value="gpu">GPU</option></select></label>
</div>

<div class="chart-container">
  <canvas id="chart"></canvas>
</div>

<footer>Generated by llamaboard &middot; data updated on each merged PR</footer>

<script>
const records = {records_json};

const tests = [...new Set(records.map(r => r.test))].sort();
const models = [...new Set(records.map(r => r.model_id))].sort();

const testSelect = document.getElementById('testSelect');
const modelSelect = document.getElementById('modelSelect');
const targetSelect = document.getElementById('targetSelect');

tests.forEach(t => {{ const o = document.createElement('option'); o.value = t; o.text = t; testSelect.appendChild(o); }});
models.forEach(m => {{ const o = document.createElement('option'); o.value = m; o.text = m.split('/').pop().replace('-GGUF',''); modelSelect.appendChild(o); }});

let chart = null;

function render() {{
  const test = testSelect.value;
  const model = modelSelect.value;
  const target = targetSelect.value;
  let filtered = records.filter(r => r.test === test);
  if (model) filtered = filtered.filter(r => r.model_id === model);
  if (target === 'cpu') filtered = filtered.filter(r => r.target === 'cpu');
  else if (target === 'gpu') filtered = filtered.filter(r => r.target.startsWith('gpu'));

  // Deduplicate by host_label + device + model + target, keep latest
  const map = new Map();
  filtered.forEach(r => {{
    const key = r.host_label + '|' + r.device + '|' + r.model_id + '|' + r.target;
    map.set(key, r);
  }});
  const bars = [...map.values()].sort((a, b) => b.tokens_per_sec - a.tokens_per_sec);

  const labels = bars.map(b => {{
    const device = b.device.replace(/\\(R\\)|\\(TM\\)/g, '').replace(/CPU /,'').trim();
    const top = (b.motherboard && b.motherboard.trim()) ? b.motherboard.trim() : b.host_label;
    const kind = b.target.startsWith('gpu') ? 'GPU' : 'CPU';
    return [top, device, kind];
  }});
  const values = bars.map(b => b.tokens_per_sec);
  const colors = bars.map(b => b.target.startsWith('gpu') ? '#58a6ff' : '#3fb950');

  if (chart) chart.destroy();
  chart = new Chart(document.getElementById('chart'), {{
    type: 'bar',
    data: {{
      labels: labels,
      datasets: [{{ data: values, backgroundColor: colors, borderRadius: 4 }}]
    }},
    options: {{
      indexAxis: 'y',
      plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: test + ' — tokens/sec (higher is better)', color: '#e6edf3' }} }},
      scales: {{
        x: {{ grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e' }} }},
        y: {{ grid: {{ display: false }}, ticks: {{ color: '#e6edf3', font: {{ size: 11 }} }} }}
      }}
    }}
  }});
}}

testSelect.addEventListener('change', render);
modelSelect.addEventListener('change', render);
targetSelect.addEventListener('change', render);
render();
</script>
</body>
</html>
"""
    path = output_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate llamaboard charts from result JSON files.")
    parser.add_argument("--results-root", default="results", help="Root directory with benchmark JSONs.")
    parser.add_argument("--output", default="artifacts", help="Output directory for charts and site.")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output = Path(args.output)

    records = load_results(results_root)
    if not records:
        print("No benchmark records found.")
        return

    print(f"Loaded {len(records)} records")

    svgs = generate_svg_charts(records, output / "charts")
    for s in svgs:
        print(f"  SVG: {s}")

    html = generate_html_site(records, output / "site")
    print(f"  HTML: {html}")

    print("Done.")


if __name__ == "__main__":
    main()
