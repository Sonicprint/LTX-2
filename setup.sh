#!/usr/bin/env bash
# =============================================================================
#  LTX-2.3 — Automated Setup & Model Download Script
#  Compatible with: Vast.ai PyTorch containers, Ubuntu 22.04+
#
#  Usage:
#    chmod +x setup.sh
#    HF_TOKEN=hf_xxx ./setup.sh          # full setup (default)
#    HF_TOKEN=hf_xxx ./setup.sh minimal  # skip optional models & LoRAs
#
#  Environment variables (all optional except HF_TOKEN for gated models):
#    HF_TOKEN          — HuggingFace token (required for Gemma download)
#    INSTALL_DIR       — where to clone the repo (default: /workspace/LTX-2)
#    SKIP_LORAS        — set to "1" to skip LoRA downloads
#    SKIP_OPTIONAL     — set to "1" to skip optional upscalers
#    DISTILLED_MODEL   — "distilled" (default) or "dev-fp8"
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }
header()  { echo -e "\n${BOLD}━━━  $*  ━━━${RESET}"; }

# ── Configuration ─────────────────────────────────────────────────────────────
INSTALL_DIR="${INSTALL_DIR:-/workspace/LTX-2}"
CKPT_DIR="${CKPT_DIR:-${INSTALL_DIR}/checkpoints}"
LORAS_DIR="${LORAS_DIR:-${CKPT_DIR}/loras}"
SKIP_LORAS="${SKIP_LORAS:-0}"
SKIP_OPTIONAL="${SKIP_OPTIONAL:-0}"
DISTILLED_MODEL="${DISTILLED_MODEL:-distilled}"   # "distilled" | "dev-fp8"
MODE="${1:-full}"   # "full" | "minimal"

[[ "$MODE" == "minimal" ]] && SKIP_LORAS=1 && SKIP_OPTIONAL=1

# ── HuggingFace token check ───────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    warn "HF_TOKEN is not set. Gemma download (gated repo) will likely fail."
    warn "Set it with:  export HF_TOKEN=hf_..."
fi

# =============================================================================
header "Step 1 — System dependencies"
# =============================================================================
info "Updating apt packages..."
apt-get update -qq
apt-get install -y -qq \
    git git-lfs curl wget ffmpeg \
    python3-pip python3-venv \
    libgl1-mesa-glx libglib2.0-0 \
    aria2 \
    2>/dev/null
git lfs install --skip-smudge 2>/dev/null || true
success "System dependencies installed"

# =============================================================================
header "Step 2 — Install uv (fast Python package manager)"
# =============================================================================
if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    info "uv already installed: $(uv --version)"
fi
success "uv ready"

# =============================================================================
header "Step 3 — Clone LTX-2 repository"
# =============================================================================
if [[ -d "$INSTALL_DIR/.git" ]]; then
    info "Repo already exists at ${INSTALL_DIR}, pulling latest changes..."
    git -C "$INSTALL_DIR" pull --ff-only || warn "Git pull failed (local changes?), continuing..."
else
    info "Cloning LTX-2 repository into ${INSTALL_DIR}..."
    git clone https://github.com/Sonicprint/LTX-2.git "$INSTALL_DIR"
fi
success "Repository ready at ${INSTALL_DIR}"

# =============================================================================
header "Step 4 — Install Python dependencies"
# =============================================================================
cd "$INSTALL_DIR"
info "Syncing Python environment with uv..."
uv sync --all-packages --no-progress 2>&1 | tail -5
success "Python environment ready"

# =============================================================================
header "Step 5 — Configure HuggingFace CLI"
# =============================================================================
HF_BIN="$(uv run python -c 'import shutil; print(shutil.which("huggingface-cli") or "")'  2>/dev/null || true)"
if [[ -z "$HF_BIN" ]]; then
    info "Installing huggingface-hub[cli] standalone..."
    uv pip install --quiet "huggingface-hub[cli]"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    info "Logging in to HuggingFace..."
    uv run huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || \
        huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
    success "HuggingFace authenticated"
fi

HF_DL() {
    # HF_DL <repo-id> <filename> <dest-dir> [--no-token]
    local repo="$1" file="$2" dest="$3"
    mkdir -p "$dest"
    if [[ -f "${dest}/${file##*/}" ]]; then
        info "  Already exists: ${file##*/} — skipping"
        return 0
    fi
    info "  Downloading: ${file} from ${repo}"
    uv run huggingface-cli download \
        --repo-type model \
        ${HF_TOKEN:+--token "$HF_TOKEN"} \
        --local-dir "$dest" \
        "$repo" "$file" 2>&1
}

HF_DL_DIR() {
    # HF_DL_DIR <repo-id> <dest-dir>  — clone entire repo (LFS)
    local repo="$1" dest="$2"
    if [[ -d "${dest}" && "$(ls -A "$dest" 2>/dev/null | wc -l)" -gt 3 ]]; then
        info "  Already exists: ${dest##*/} — skipping"
        return 0
    fi
    info "  Cloning: ${repo}"
    mkdir -p "${dest%/*}"
    GIT_LFS_SKIP_SMUDGE=0 \
    uv run huggingface-cli download \
        --repo-type model \
        ${HF_TOKEN:+--token "$HF_TOKEN"} \
        --local-dir "$dest" \
        "$repo" 2>&1 | tail -3
}

mkdir -p "$CKPT_DIR" "$LORAS_DIR"

# =============================================================================
header "Step 6 — Download main transformer checkpoint"
# =============================================================================
case "$DISTILLED_MODEL" in
    dev-fp8)
        info "Downloading dev-fp8 checkpoint (~27 GB)..."
        HF_DL "Lightricks/LTX-2.3" \
               "ltx-2.3-22b-dev-fp8.safetensors" \
               "$CKPT_DIR"
        ;;
    *)
        info "Downloading distilled checkpoint (~43 GB)..."
        HF_DL "Lightricks/LTX-2.3" \
               "ltx-2.3-22b-distilled.safetensors" \
               "$CKPT_DIR"
        ;;
esac
success "Main transformer checkpoint downloaded"

# =============================================================================
header "Step 7 — Download spatial upscaler + distilled LoRA (required)"
# =============================================================================
info "Downloading spatial upscaler x2 (~950 MB)..."
HF_DL "Lightricks/LTX-2.3" \
       "ltx-2.3-spatial-upscaler-x2-1.0.safetensors" \
       "$CKPT_DIR"

# Required by all pipelines EXCEPT DistilledPipeline and ICLoraPipeline
info "Downloading distilled LoRA 384 (~7 GB)..."
HF_DL "Lightricks/LTX-2.3" \
       "ltx-2.3-22b-distilled-lora-384.safetensors" \
       "$CKPT_DIR"
success "Spatial upscaler + distilled LoRA downloaded"

# =============================================================================
header "Step 8 — Download optional upscalers"
# =============================================================================
if [[ "$SKIP_OPTIONAL" != "1" ]]; then
    info "Downloading spatial upscaler x1.5 (~1 GB)..."
    HF_DL "Lightricks/LTX-2.3" \
           "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors" \
           "$CKPT_DIR"

    info "Downloading temporal upscaler x2 (~250 MB)..."
    HF_DL "Lightricks/LTX-2.3" \
           "ltx-2.3-temporal-upscaler-x2-1.0.safetensors" \
           "$CKPT_DIR"
    success "Optional upscalers downloaded"
else
    info "Skipping optional upscalers (SKIP_OPTIONAL=1)"
fi

# =============================================================================
header "Step 9 — Download Gemma 3 text encoder (gated — requires HF_TOKEN)"
# =============================================================================
GEMMA_DEST="${CKPT_DIR}/gemma-3-12b-it-qat-q4_0-unquantized"
if [[ -d "$GEMMA_DEST" && "$(ls -A "$GEMMA_DEST" 2>/dev/null | wc -l)" -gt 3 ]]; then
    info "Gemma already downloaded — skipping"
else
    if [[ -z "${HF_TOKEN:-}" ]]; then
        warn "HF_TOKEN not set — skipping Gemma download. Set HF_TOKEN and re-run."
    else
        info "Downloading Gemma 3 12B quantized (Q4) — ~8 GB, gated repo..."
        info "  If this fails: visit https://huggingface.co/google/gemma-3-12b-it and accept the licence"
        HF_DL_DIR "google/gemma-3-12b-it-qat-q4_0-unquantized" "$GEMMA_DEST"
        success "Gemma text encoder downloaded"
    fi
fi

# =============================================================================
header "Step 10 — Download LoRAs"
# =============================================================================
# Each LoRA lives in its own HuggingFace repository (not a shared monorepo).
if [[ "$SKIP_LORAS" != "1" ]]; then
    info "Downloading LoRAs — each from its own HF repo (~17 GB total)..."

    # 22B IC LoRAs
    HF_DL "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control" \
           "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
           "$LORAS_DIR" || warn "Failed: union-control LoRA"

    HF_DL "Lightricks/LTX-2.3-22b-IC-LoRA-Inpainting" \
           "ltx-2.3-22b-ic-lora-inpainting.safetensors" \
           "$LORAS_DIR" || warn "Failed: inpainting LoRA"

    HF_DL "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control" \
           "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors" \
           "$LORAS_DIR" || warn "Failed: motion-track-control LoRA"

    # 19B IC LoRAs
    HF_DL "Lightricks/LTX-2-19b-IC-LoRA-Detailer" \
           "ltx-2-19b-ic-lora-detailer.safetensors" \
           "$LORAS_DIR" || warn "Failed: detailer LoRA"

    HF_DL "Lightricks/LTX-2-19b-IC-LoRA-Pose-Control" \
           "ltx-2-19b-ic-lora-pose-control.safetensors" \
           "$LORAS_DIR" || warn "Failed: pose-control LoRA"

    # 19B Camera-control LoRAs
    HF_DL "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In" \
           "ltx-2-19b-lora-camera-control-dolly-in.safetensors" \
           "$LORAS_DIR" || warn "Failed: dolly-in LoRA"

    HF_DL "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out" \
           "ltx-2-19b-lora-camera-control-dolly-out.safetensors" \
           "$LORAS_DIR" || warn "Failed: dolly-out LoRA"

    HF_DL "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left" \
           "ltx-2-19b-lora-camera-control-dolly-left.safetensors" \
           "$LORAS_DIR" || warn "Failed: dolly-left LoRA"

    HF_DL "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right" \
           "ltx-2-19b-lora-camera-control-dolly-right.safetensors" \
           "$LORAS_DIR" || warn "Failed: dolly-right LoRA"

    HF_DL "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down" \
           "ltx-2-19b-lora-camera-control-jib-down.safetensors" \
           "$LORAS_DIR" || warn "Failed: jib-down LoRA"

    HF_DL "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up" \
           "ltx-2-19b-lora-camera-control-jib-up.safetensors" \
           "$LORAS_DIR" || warn "Failed: jib-up LoRA"

    HF_DL "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static" \
           "ltx-2-19b-lora-camera-control-static.safetensors" \
           "$LORAS_DIR" || warn "Failed: static LoRA"

    success "LoRAs downloaded"
else
    info "Skipping LoRAs (SKIP_LORAS=1 or minimal mode)"
fi

# =============================================================================
header "Step 11 — Smoke test"
# =============================================================================
cd "$INSTALL_DIR"
info "Running Python import check..."
uv run python - <<'PYEOF'
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("packages/ltx-core/src").resolve()))
sys.path.insert(0, str(pathlib.Path("packages/ltx-pipelines/src").resolve()))
try:
    import ltx_core
    import ltx_pipelines
    print("[OK] ltx_core and ltx_pipelines importable")
except ImportError as e:
    print(f"[WARN] Import check failed: {e}")
PYEOF

info "Checking checkpoint files..."
python3 - <<PYEOF
import os, pathlib
ckpt = pathlib.Path("${CKPT_DIR}")
required = [
    "ltx-2.3-22b-distilled.safetensors" if "${DISTILLED_MODEL}" != "dev-fp8"
        else "ltx-2.3-22b-dev-fp8.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    "gemma-3-12b-it-qat-q4_0-unquantized",
]
missing = []
for f in required:
    p = ckpt / f
    if not p.exists():
        missing.append(str(p))
    else:
        size = sum(fp.stat().st_size for fp in p.rglob("*") if fp.is_file()) if p.is_dir() else p.stat().st_size
        print(f"  ✓  {f}  ({size/1e9:.1f} GB)")
if missing:
    print("\nMissing files:")
    for m in missing:
        print(f"  ✗  {m}")
else:
    print("\nAll required checkpoints present ✓")
PYEOF

# =============================================================================
echo ""
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}${BOLD}  LTX-2.3 setup complete!${RESET}"
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
echo "  Run inference:"
echo "    cd ${INSTALL_DIR}"
echo "    uv run python inference_new.py \\"
echo "      --prompt \"A cinematic mountain valley at dawn\" \\"
echo "      --resolution 16:9 --duration 6 --output out.mp4"
echo ""
