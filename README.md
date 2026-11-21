# ComfyUI-CivitAI-Missing-Models

A ComfyUI custom node & frontend extension that **automatically detects missing models** in loaded workflows, searches **both CivitAI domains (`civitai.com` and `civitai.red`)**, and downloads them at maximum speed using **`aria2c`** directly into your ComfyUI model directories.

## Features

- **Automatic Workflow Scanning**:
  - Scans ComfyUI LiteGraph workflows (including nested subgraphs and `properties.models` metadata) and API prompt JSONs.
  - Supports standard loaders (`CheckpointLoaderSimple`, `LoraLoader`, `VAELoader`, `ControlNetLoader`, `CLIPLoader`, `DualCLIPLoader`, `UNETLoader`, `UnetLoaderGGUF`, `UpscaleModelLoader`, etc.), `rgthree` Power Lora Loader, and generic custom nodes referencing `.safetensors`, `.ckpt`, `.gguf`, `.pt`, `.pth`, `.bin`, `.onnx`, or `.sft` files.
  - Automatically scans when you load or drop a workflow and shows a toast + toolbar badge with the missing count.
- **Dual-Domain CivitAI Search & Download (`civitai.com` + `civitai.red`)**:
  - Queries both `https://civitai.com` and `https://civitai.red` concurrently, merges/deduplicates candidates, and tags which domain(s) host each model.
  - Supports pasting a direct `civitai.com` or `civitai.red` URL (or custom search query) per missing model.
  - Automatically falls back to the counterpart domain if a download is gated/unavailable on one domain.
- **Fast Multi-Connection Downloads via `aria2c`**:
  - Pre-resolves CivitAI redirects to the signed CDN URL so `aria2c` can saturate your connection with `-x 16 -s 16` without S3/R2 auth-header redirect errors.
  - Supports resumable downloads (`--continue=true`), concurrency limits, cancellation, and live progress bars (percentage, speed, active connections, ETA) streamed over ComfyUI's WebSocket.
  - Automatically refreshes ComfyUI's model dropdowns when downloads complete.

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
