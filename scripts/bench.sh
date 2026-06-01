#!/usr/bin/env bash
# llamaboard — benchmark llama.cpp locally or on remote hosts
# Usage (local):  ./scripts/bench.sh
# Usage (remote): LLAMABOARD_HOSTS=host1,host2 ./scripts/bench.sh
# Usage (curl):   curl -sSL https://raw.githubusercontent.com/.../scripts/bench.sh | bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source .env if present
ENV_FILE="${LLAMABOARD_ENV_FILE:-$ROOT_DIR/.env}"
if [ -f "$ENV_FILE" ]; then
    set -a; . "$ENV_FILE"; set +a
fi

# --- CONFIGURATION ---
REPO_URL="${LLAMABOARD_REPO_URL:-https://github.com/ggml-org/llama.cpp}"
HF_MODEL_IDS="${LLAMABOARD_MODEL_IDS:-ggml-org/gemma-3-1b-it-GGUF ggml-org/Qwen3-4B-GGUF}"
LLAMABOARD_REPO="${LLAMABOARD_REPO:-okigan/llamaboard}"
WORK_DIR="${LLAMABOARD_WORKDIR:-$HOME/.llamaboard}"
RESULTS_ROOT="$ROOT_DIR/results"

# ============================================================
# If LLAMABOARD_HOSTS is set, SSH into each host and run remotely
# ============================================================
if [ -n "${LLAMABOARD_HOSTS:-}" ]; then
    DATE_PATH="$(date -u +%Y/%m/%d)"
    RESULT_DIR="$RESULTS_ROOT/$DATE_PATH"
    mkdir -p "$RESULT_DIR"

    IFS=',' read -r -a HOSTS <<< "$LLAMABOARD_HOSTS"
    for raw_host in "${HOSTS[@]}"; do
        host="$(echo "$raw_host" | tr -d '[:space:]')"
        [ -z "$host" ] && continue

        echo "=== Benchmarking $host ==="
        HOST_HASH="$(printf '%s' "$host" | shasum -a 256 | awk '{print $1}')"
        RAW_LOG_FILE="$RESULT_DIR/$HOST_HASH.raw.log"
        RESULT_JSON="$RESULT_DIR/$HOST_HASH.json"

        ssh "$host" bash -s "$REPO_URL" "${HF_MODEL_IDS// /,}" <<'REMOTE' | tee "$RAW_LOG_FILE"
set -euo pipefail
REPO_URL="$1"; HF_MODEL_IDS="${2//,/ }"; WORK_DIR="$HOME/.llamaboard"

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

REPO_DIR="$WORK_DIR/llama.cpp"
BUILD_CPU="$WORK_DIR/build_cpu"
BUILD_CUDA="$WORK_DIR/build_cuda"
mkdir -p "$WORK_DIR/tmp_build"
export TMPDIR="$WORK_DIR/tmp_build"

HAS_GPU=false
if command -v nvcc >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    HAS_GPU=true
fi

if [ ! -d "$REPO_DIR" ]; then
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# Helper: build llama-bench with given flags into a given dir
build_if_needed() {
    local build_dir="$1"
    local cmake_flags="$2"
    local use_cuda="$3"
    local bench_bin="$build_dir/bin/llama-bench"
    local built_key="$(cat "$build_dir/.build_key" 2>/dev/null || echo '')"
    if [ ! -f "$bench_bin" ] || [ "$built_key" != "$cmake_flags" ]; then
        rm -rf "$build_dir"
        mkdir -p "$build_dir"
        cmake -B "$build_dir" $cmake_flags
        local njobs="$(nproc)"
        if [ "$use_cuda" = true ] && [ "$njobs" -gt 4 ]; then
            njobs=4
        fi
        cmake --build "$build_dir" --config Release --parallel "$njobs" --target llama-bench
        echo "$cmake_flags" > "$build_dir/.build_key"
    fi
}

# Always build CPU-only
CPU_FLAGS="-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON"
build_if_needed "$BUILD_CPU" "$CPU_FLAGS" false
BENCH_CPU="$BUILD_CPU/bin/llama-bench"

# Build CUDA if GPU available
BENCH_CUDA=""
if [ "$HAS_GPU" = true ]; then
    CUDA_FLAGS="-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=ON"
    build_if_needed "$BUILD_CUDA" "$CUDA_FLAGS" true
    BENCH_CUDA="$BUILD_CUDA/bin/llama-bench"
fi

echo "TARGET:cpu"
for HF_MODEL_ID in $HF_MODEL_IDS; do
    echo "MODEL:$HF_MODEL_ID"
    CUDA_VISIBLE_DEVICES="" "$BENCH_CPU" -hf "$HF_MODEL_ID" -ngl 0 2>&1 || true
done

if [ -n "$BENCH_CUDA" ]; then
    echo "TARGET:gpu0"
    echo "GPU_TELEMETRY:$(nvidia-smi --query-gpu=name,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits -i 0 2>/dev/null || echo 'n/a,n/a,n/a,n/a')"
    for HF_MODEL_ID in $HF_MODEL_IDS; do
        echo "MODEL:$HF_MODEL_ID"
        CUDA_VISIBLE_DEVICES=0 "$BENCH_CUDA" -hf "$HF_MODEL_ID" -ngl 99 2>&1 || true
    done
fi

cpu_model="$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^[ \t]+/,"",$2); print $2; exit}')" || true
if [ -z "${cpu_model:-}" ]; then
    cpu_model="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
fi
echo "META:cpu_model=$cpu_model"

cpu_temp="$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | awk '{printf "%.1f", $1/1000}')" || true
echo "META:cpu_temp_c=${cpu_temp:-n/a}"

mb_vendor="$(cat /sys/devices/virtual/dmi/id/board_vendor 2>/dev/null || true)"
mb_name="$(cat /sys/devices/virtual/dmi/id/board_name 2>/dev/null || true)"
echo "META:motherboard=${mb_vendor:-n/a} ${mb_name:-}"

echo "META:commit=$(git rev-parse HEAD)"
echo "META:host=$(hostname)"
echo "META:use_cuda=$HAS_GPU"
REMOTE

        # Parse raw log into JSON
        python3 - "$RESULT_JSON" "$HOST_HASH" "$RAW_LOG_FILE" <<'PY'
import json, re, sys
from datetime import datetime, timezone
from pathlib import Path

result_json, host_hash = sys.argv[1], sys.argv[2]
raw_log = Path(sys.argv[3]).read_text(encoding="utf-8")

def parse_metrics(text):
    pp = re.findall(r"\|\s*pp(\d+)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*(?:±|\+/-)?", text)
    tg = re.findall(r"\|\s*tg(\d+)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*(?:±|\+/-)?", text)
    return {
        "pp": [{"tokens": int(t), "tokens_per_sec": float(v)} for t, v in pp],
        "tg": [{"tokens": int(t), "tokens_per_sec": float(v)} for t, v in tg],
    }

commit = ""
for m in re.finditer(r"META:commit=(.+)", raw_log):
    commit = m.group(1).strip()
cpu_model = ""
for m in re.finditer(r"META:cpu_model=(.+)", raw_log):
    cpu_model = m.group(1).strip()
cpu_temp_c = None
for m in re.finditer(r"META:cpu_temp_c=(.+)", raw_log):
    try: cpu_temp_c = float(m.group(1).strip())
    except ValueError: pass
motherboard = ""
for m in re.finditer(r"META:motherboard=(.+)", raw_log):
    motherboard = m.group(1).strip()

sections = re.split(r"TARGET:(cpu|gpu0)\n?", raw_log)
runs = []
i = 1
while i < len(sections):
    target, body = sections[i], sections[i + 1] if i + 1 < len(sections) else ""
    i += 2
    gpu_info = None
    if target == "gpu0":
        tel = re.search(r"GPU_TELEMETRY:(.+)", body)
        if tel:
            parts = tel.group(1).split(",")
            if len(parts) >= 4:
                def to_float(s):
                    try: return float(s.strip())
                    except: return None
                gpu_info = {"model": parts[0].strip(), "vram_mb": to_float(parts[1]),
                            "power_w": to_float(parts[2]), "temperature_c": to_float(parts[3])}
    model_parts = re.split(r"MODEL:(\S+)\n?", body)
    j = 1
    while j < len(model_parts):
        model_id = model_parts[j]
        model_body = model_parts[j + 1] if j + 1 < len(model_parts) else ""
        j += 2
        metrics = parse_metrics(model_body)
        if metrics["pp"] or metrics["tg"]:
            runs.append({"target": target, "model_id": model_id,
                         "use_cuda": target == "gpu0", "gpu": gpu_info, "metrics": metrics})

model_ids = sorted(set(r["model_id"] for r in runs))
payload = {
    "schema_version": 4,
    "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "host_sha256": host_hash,
    "hardware": {
        "motherboard": motherboard or None, "cpu_model": cpu_model or None, "cpu_temp_c": cpu_temp_c,
        "gpu_model": next((r["gpu"]["model"] for r in runs if r.get("gpu") and r["gpu"].get("model")), None),
        "gpu_temp_c": next((r["gpu"]["temperature_c"] for r in runs if r.get("gpu") and r["gpu"].get("temperature_c")), None),
    },
    "benchmark": {"model_ids": model_ids, "targets": sorted(set(r["target"] for r in runs))},
    "llama_cpp": {"commit": commit},
    "runs": runs,
}
with open(result_json, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
print(f"Saved: {result_json}")
PY

        echo "=== Done: $host ==="
        echo ""
    done
    echo "All remote hosts benchmarked."

# ============================================================
# Regenerate charts after remote benchmarks
# ============================================================
echo ""
echo "Updating charts..."
python3 "$SCRIPT_DIR/generate_charts.py" --results-root "$RESULTS_ROOT" --output "$ROOT_DIR/artifacts"
exit 0
fi

# ============================================================
# Local mode — benchmark this machine
# ============================================================

# --- Preflight ---
echo "=== llamaboard ==="
echo "This script benchmarks llama.cpp on your machine and submits results via PR."
echo ""

if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: GitHub CLI (gh) is required. Install: https://cli.github.com"
    exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
    echo "ERROR: Not authenticated with gh. Run: gh auth login"
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: cmake is required to build llama.cpp."
    exit 1
fi

# --- Setup ---
mkdir -p "$WORK_DIR"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

LLAMA_DIR="$WORK_DIR/llama.cpp"
BUILD_CPU="$WORK_DIR/build_cpu"
BUILD_CUDA="$WORK_DIR/build_cuda"

HAS_GPU=false
if command -v nvcc >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    HAS_GPU=true
fi

# --- Build llama-bench ---
if [ ! -d "$LLAMA_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth 1 "$REPO_URL" "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"

# Only pull if explicitly requested
if [ "${LLAMABOARD_UPDATE:-0}" = "1" ]; then
    echo "Updating llama.cpp..."
    git fetch --depth 1 origin && git reset --hard origin/HEAD
fi

# Helper: build llama-bench with given flags
build_if_needed() {
    local build_dir="$1"
    local cmake_flags="$2"
    local use_cuda="$3"
    local bench_bin="$build_dir/bin/llama-bench"
    local built_key="$(cat "$build_dir/.build_key" 2>/dev/null || echo '')"
    local current_commit="$(git rev-parse HEAD)"
    local built_commit="$(cat "$build_dir/.commit" 2>/dev/null || echo '')"
    if [ ! -f "$bench_bin" ] || [ "$built_key" != "$cmake_flags" ] || [ "$current_commit" != "$built_commit" ]; then
        echo "Building llama-bench ($build_dir)..."
        rm -rf "$build_dir"
        mkdir -p "$build_dir"
        cmake -B "$build_dir" $cmake_flags
        local njobs="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
        if [ "$use_cuda" = true ] && [ "$njobs" -gt 4 ]; then
            njobs=4
        fi
        cmake --build "$build_dir" --config Release --parallel "$njobs" --target llama-bench
        echo "$cmake_flags" > "$build_dir/.build_key"
        git rev-parse HEAD > "$build_dir/.commit"
    fi
}

CPU_FLAGS="-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON"
build_if_needed "$BUILD_CPU" "$CPU_FLAGS" false
BENCH_CPU="$BUILD_CPU/bin/llama-bench"

BENCH_CUDA=""
if [ "$HAS_GPU" = true ]; then
    CUDA_FLAGS="-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=ON"
    build_if_needed "$BUILD_CUDA" "$CUDA_FLAGS" true
    BENCH_CUDA="$BUILD_CUDA/bin/llama-bench"
fi

# --- Run benchmark ---
echo ""
echo "Running benchmark..."
RAW_LOG="$WORK_DIR/bench.log"
: > "$RAW_LOG"

run_bench() {
    echo "TARGET:cpu"
    for HF_MODEL_ID in $HF_MODEL_IDS; do
        echo "MODEL:$HF_MODEL_ID"
        CUDA_VISIBLE_DEVICES="" "$BENCH_CPU" -hf "$HF_MODEL_ID" -ngl 0 2>&1 || true
    done

    if [ -n "$BENCH_CUDA" ]; then
        echo "TARGET:gpu0"
        echo "GPU_TELEMETRY:$(nvidia-smi --query-gpu=name,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits -i 0 2>/dev/null || echo 'n/a,n/a,n/a,n/a')"
        for HF_MODEL_ID in $HF_MODEL_IDS; do
            echo "MODEL:$HF_MODEL_ID"
            CUDA_VISIBLE_DEVICES=0 "$BENCH_CUDA" -hf "$HF_MODEL_ID" -ngl 99 2>&1 || true
        done
    fi

    local cpu_model cpu_temp mb_vendor mb_name
    cpu_model="$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^[ \t]+/,"",$2); print $2; exit}')" || true
    if [ -z "$cpu_model" ]; then
        cpu_model="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
    fi
    echo "META:cpu_model=$cpu_model"

    cpu_temp="$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | awk '{printf "%.1f", $1/1000}')" || true
    echo "META:cpu_temp_c=${cpu_temp:-n/a}"

    mb_vendor="$(cat /sys/devices/virtual/dmi/id/board_vendor 2>/dev/null || true)"
    mb_name="$(cat /sys/devices/virtual/dmi/id/board_name 2>/dev/null || true)"
    echo "META:motherboard=${mb_vendor:-n/a} ${mb_name:-}"

    echo "META:commit=$(git rev-parse HEAD)"
    echo "META:use_cuda=$HAS_GPU"
}

run_bench 2>&1 | tee "$RAW_LOG"

# --- Parse results ---
echo ""
echo "Parsing results..."

HOSTNAME_SLUG="$(hostname | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/^-//;s/-$//')"
DATE_PATH="$(date -u +%Y/%m/%d)"
HOST_HASH="$(printf '%s' "$(hostname)" | shasum -a 256 | awk '{print $1}')"
RESULT_JSON="$WORK_DIR/result.json"

python3 - "$RESULT_JSON" "$HOST_HASH" "$RAW_LOG" <<'PY'
import json, re, sys
from datetime import datetime, timezone
from pathlib import Path

result_json, host_hash = sys.argv[1], sys.argv[2]
raw_log = Path(sys.argv[3]).read_text(encoding="utf-8")


def parse_metrics(text):
    pp = re.findall(r"\|\s*pp(\d+)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*(?:±|\+/-)?", text)
    tg = re.findall(r"\|\s*tg(\d+)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*(?:±|\+/-)?", text)
    return {
        "pp": [{"tokens": int(t), "tokens_per_sec": float(v)} for t, v in pp],
        "tg": [{"tokens": int(t), "tokens_per_sec": float(v)} for t, v in tg],
    }


commit = ""
for m in re.finditer(r"META:commit=(.+)", raw_log):
    commit = m.group(1).strip()

cpu_model = ""
for m in re.finditer(r"META:cpu_model=(.+)", raw_log):
    cpu_model = m.group(1).strip()

cpu_temp_c = None
for m in re.finditer(r"META:cpu_temp_c=(.+)", raw_log):
    val = m.group(1).strip()
    try:
        cpu_temp_c = float(val)
    except ValueError:
        pass

motherboard = ""
for m in re.finditer(r"META:motherboard=(.+)", raw_log):
    motherboard = m.group(1).strip()

# Parse into target sections, then split each by MODEL: markers
sections = re.split(r"TARGET:(cpu|gpu0)\n?", raw_log)
runs = []
i = 1
while i < len(sections):
    target, body = sections[i], sections[i + 1] if i + 1 < len(sections) else ""
    i += 2

    gpu_info = None
    if target == "gpu0":
        tel = re.search(r"GPU_TELEMETRY:(.+)", body)
        if tel:
            parts = tel.group(1).split(",")
            if len(parts) >= 4:
                def to_float(s):
                    try: return float(s.strip())
                    except: return None
                gpu_info = {
                    "model": parts[0].strip(),
                    "vram_mb": to_float(parts[1]),
                    "power_w": to_float(parts[2]),
                    "temperature_c": to_float(parts[3]),
                }

    # Split by MODEL: markers within this target
    model_parts = re.split(r"MODEL:(\S+)\n?", body)
    j = 1
    while j < len(model_parts):
        model_id = model_parts[j]
        model_body = model_parts[j + 1] if j + 1 < len(model_parts) else ""
        j += 2
        metrics = parse_metrics(model_body)
        if metrics["pp"] or metrics["tg"]:
            runs.append({
                "target": target,
                "model_id": model_id,
                "use_cuda": target == "gpu0",
                "gpu": gpu_info,
                "metrics": metrics,
            })

model_ids = sorted(set(r["model_id"] for r in runs))

payload = {
    "schema_version": 4,
    "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "host_sha256": host_hash,
    "hardware": {
        "motherboard": motherboard or None,
        "cpu_model": cpu_model or None,
        "cpu_temp_c": cpu_temp_c,
        "gpu_model": next((r["gpu"]["model"] for r in runs if r.get("gpu") and r["gpu"].get("model")), None),
        "gpu_temp_c": next((r["gpu"]["temperature_c"] for r in runs if r.get("gpu") and r["gpu"].get("temperature_c")), None),
    },
    "benchmark": {"model_ids": model_ids, "targets": sorted(set(r["target"] for r in runs))},
    "llama_cpp": {"commit": commit},
    "runs": runs,
}

with open(result_json, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY

echo "Benchmark complete."
python3 -c "
import json, sys
d = json.load(open('$RESULT_JSON'))
hw = d.get('hardware', {})
print(f\"  CPU: {hw.get('cpu_model', '?')}\")
print(f\"  GPU: {hw.get('gpu_model', 'none') or 'none'}\")
print(f\"  Models: {', '.join(d.get('benchmark', {}).get('model_ids', []))}\")
"

# --- Save results locally ---
LOCAL_RESULTS="$ROOT_DIR/results/$DATE_PATH"
mkdir -p "$LOCAL_RESULTS"
cp "$RESULT_JSON" "$LOCAL_RESULTS/${HOST_HASH}.json"
echo "Saved: $LOCAL_RESULTS/${HOST_HASH}.json"

# --- Fork, branch, commit, PR (optional) ---
if [ "${LLAMABOARD_SUBMIT:-0}" = "1" ]; then
    echo ""
    echo "Submitting results..."

    FORK_DIR="$WORK_DIR/llamaboard_fork"
    if [ ! -d "$FORK_DIR" ]; then
        gh repo fork "$LLAMABOARD_REPO" --clone --remote -- "$FORK_DIR" 2>/dev/null || \
            gh repo clone "$LLAMABOARD_REPO" "$FORK_DIR"
    fi

    cd "$FORK_DIR"
    git fetch upstream main 2>/dev/null || git fetch origin main
    git checkout -B "bench/$HOSTNAME_SLUG/$DATE_PATH" upstream/main 2>/dev/null || \
        git checkout -B "bench/$HOSTNAME_SLUG/$DATE_PATH" origin/main

    DEST_DIR="results/$DATE_PATH"
    mkdir -p "$DEST_DIR"
    cp "$RESULT_JSON" "$DEST_DIR/${HOST_HASH}.json"

    git add "results/"
    git commit -m "bench: $HOSTNAME_SLUG $(date -u +%Y-%m-%d)"

    git push --force origin "bench/$HOSTNAME_SLUG/$DATE_PATH"

    PR_TITLE="bench: $HOSTNAME_SLUG $(date -u +%Y-%m-%d)"
    PR_BODY="Automated benchmark submission from \`$(hostname)\`.

**Hardware:**
$(python3 -c "
import json
d = json.load(open('$RESULT_JSON'))
hw = d.get('hardware', {})
print(f\"- CPU: {hw.get('cpu_model', 'unknown')}\")
print(f\"- GPU: {hw.get('gpu_model', 'none') or 'none'}\")
print(f\"- Motherboard: {hw.get('motherboard', 'unknown')}\")
")

**Models:** $(python3 -c "import json; d=json.load(open('$RESULT_JSON')); print(', '.join(d.get('benchmark',{}).get('model_ids',[])))")
"

    gh pr create --repo "$LLAMABOARD_REPO" \
        --title "$PR_TITLE" \
        --body "$PR_BODY" \
        --head "$(gh api user -q .login):bench/$HOSTNAME_SLUG/$DATE_PATH" \
        --base main 2>/dev/null || echo "PR already exists or was updated."

    echo ""
    echo "=== Done! Your benchmark has been submitted as a PR. ==="
else
    echo ""
    echo "=== Done! To submit results as a PR, run with: LLAMABOARD_SUBMIT=1 ./scripts/bench.sh ==="
fi

# --- Regenerate charts ---
echo ""
echo "Updating charts..."
python3 "$SCRIPT_DIR/generate_charts.py" --results-root "$RESULTS_ROOT" --output "$ROOT_DIR/artifacts"
