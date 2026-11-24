# ComfyUI-CivitAI-Missing-Models

A ComfyUI custom node & frontend extension that **automatically detects missing models** in loaded workflows, searches **CivitAI (`civitai.com` and `civitai.red`) and HuggingFace (`huggingface.co`)**, downloads them at maximum speed using **`aria2c`**, and lets you **upload local LoRAs/models directly from your browser** into `models/loras/` (or any model folder).

## Features

- **Automatic Workflow Scanning**:
  - Scans ComfyUI LiteGraph workflows (including nested subgraphs and `properties.models` metadata) and API prompt JSONs.
  - Supports standard loaders (`CheckpointLoaderSimple`, `LoraLoader`, `VAELoader`, `ControlNetLoader`, `CLIPLoader`, `DualCLIPLoader`, `UNETLoader`, `UnetLoaderGGUF`, `UpscaleModelLoader`, etc.), `rgthree` Power Lora Loader, and generic custom nodes referencing `.safetensors`, `.ckpt`, `.gguf`, `.pt`, `.pth`, `.bin`, `.onnx`, or `.sft` files.
  - Automatically scans when you load or drop a workflow and shows a toast + toolbar badge with the missing count.
- **CivitAI (`.com` + `.red`) & HuggingFace Search + Download**:
  - Clean segmented **Search Provider** toggle (`Both` | `CivitAI` | `HuggingFace`) right in the modal header.
  - Queries `https://civitai.com`, `https://civitai.red`, and/or `https://huggingface.co`, merging and ranking candidates with clear domain badges.
  - Direct URLs (`civitai.com/...`, `civitai.red/...`, `huggingface.co/...`, or `owner/repo` shorthand) automatically route to the matching provider regardless of the toggle.
  - Supports both a **CivitAI API Key** (`CIVITAI_API_KEY` / `CAI_API_KEY`) and a **HuggingFace Access Token** (`HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`) for gated/NSFW/private repos, each with a built-in **Test** button in Settings.
- **Fast Multi-Connection Downloads via `aria2c`**:
  - Pre-resolves CivitAI and HuggingFace redirects to signed CDN URLs so `aria2c` can saturate your connection with `-x 16 -s 16` without S3/R2/XetHub auth-header redirect errors.
  - Supports resumable downloads (`--continue=true`), concurrency limits, cancellation, and live progress bars (percentage, speed, active connections, ETA) streamed over ComfyUI's WebSocket.
- **Browser LoRA / Model Upload (Chunked)**:
  - Drag-and-drop or browse local `.safetensors` / `.ckpt` / `.gguf` files from your machine straight into `models/loras/` (with optional subfolder and folder selector, or click **Upload** right on any missing model card).
  - Uploads in 4 MB chunks to bypass reverse-proxy body limits (RunPod/Vast.ai/Cloudflare) with live MB/s progress and cancel support.
  - Automatically refreshes ComfyUI's model dropdowns when downloads or uploads complete.

## Installation

### 1. Install `aria2`

- **macOS (Homebrew)**:
  ```bash
  brew install aria2
  ```
- **Ubuntu / Debian**:
  ```bash
  sudo apt update && sudo apt install -y aria2
  ```
- **Windows (winget / Scoop)**:
  ```powershell
  winget install aria2.aria2
  # or: scoop install aria2
  ```

### 2. Install the Custom Node

Copy or clone this folder into your ComfyUI `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone <repo-url> ComfyUI-CivitAI-Missing-Models
```

*(ComfyUI already bundles `aiohttp`, so no extra Python packages are usually required, though `pip install -r requirements.txt` is provided.)*

Restart ComfyUI.

### 3. Configure Your CivitAI API Key

Many CivitAI downloads (including mature/NSFW models on `civitai.red` and gated models) require an API token:

1. Generate an API key at **Account Settings → API Keys** on CivitAI.
2. Either:
   - Click **Missing Models → Settings** in ComfyUI and paste your key (saved locally in `config.json` with `0600` permissions), **or**
   - Export `CIVITAI_API_KEY` (or `CAI_API_KEY`) in the environment before launching ComfyUI.

## Usage

1. Load any workflow in ComfyUI. If models are missing, a toast notification appears and the **Missing Models** toolbar button shows the count.
2. Click **Missing Models** (or **Find & Download** on the toast).
3. Click **Search All** to query `civitai.com` and `civitai.red` for every missing file.
4. Review the ranked matches (or pick an alternate version from the dropdown / paste a CivitAI URL) and click **Download (aria2)** or **Download All**.
5. Once finished, ComfyUI's node dropdowns refresh automatically so you can queue your workflow right away.

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```
