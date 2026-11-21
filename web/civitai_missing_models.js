import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "CivitAI.MissingModels";
const API_BASE = "/civitai-downloader";

function loadStylesheet() {
  if (document.getElementById("cmm-styles")) return;
  const link = document.createElement("link");
  link.id = "cmm-styles";
  link.rel = "stylesheet";
  link.type = "text/css";
  link.href = new URL("./civitai_missing_models.css", import.meta.url).href;
  document.head.appendChild(link);
}

function formatBytes(bytes) {
  if (!bytes || bytes <= 0) return "Unknown size";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let idx = 0;
  let val = bytes;
  while (val >= 1024 && idx < units.length - 1) {
    val /= 1024;
    idx++;
  }
  return `${val.toFixed(idx >= 2 ? 2 : 0)} ${units[idx]}`;
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

class CivitAIMissingModelsUI {
  constructor() {
    this.status = { aria2: { available: false }, config: { domains: ["https://civitai.com", "https://civitai.red"] } };
    this.missing = [];
    this.totalReferenced = 0;
    this.searchResults = new Map(); // modelId -> { loading, candidates, selectedIdx, customQuery, error }
    this.jobsByModelKey = new Map(); // `${folder}::${filename.toLowerCase()}` -> job
    this.jobsById = new Map(); // job_id -> job
    this.settingsOpen = false;
    this.keyTestMessage = "";
    this.keyTestValid = null;
    this.modalOverlay = null;
    this.toolbarBtn = null;
    this.toastEl = null;
  }

  async init() {
    loadStylesheet();
    await this.fetchStatus();
    await this.fetchDownloads();
    this.registerSidebarTab();
    this.injectToolbarButton();
    this.bindProgressEvents();
    this.hookWorkflowLoad();
    console.log("[CivitAI Missing Models] Extension initialized.");
  }

  registerSidebarTab() {
    if (!app.extensionManager || typeof app.extensionManager.registerSidebarTab !== "function") {
      return;
    }
    try {
      app.extensionManager.registerSidebarTab({
        id: "civitai-missing-models",
        icon: "pi pi-download",
        title: "Missing Models",
        tooltip: "CivitAI Missing Models Downloader (aria2)",
        type: "custom",
        render: (el) => {
          this.sidebarEl = el;
          this.renderSidebarContent();
        },
      });
    } catch (err) {
      console.warn("[CivitAI Missing Models] Failed to register sidebar tab:", err);
    }
  }

  renderSidebarContent() {
    if (!this.sidebarEl) return;
    const aria2 = this.status?.aria2 || {};
    const cfg = this.status?.config || {};
    const count = this.missing.length;

    this.sidebarEl.innerHTML = `
      <div style="padding:14px;display:flex;flex-direction:column;gap:12px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#e6edf3;">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
          <strong style="font-size:14px;">CivitAI Missing Models</strong>
          <span class="cmm-badge ${count > 0 ? "" : "cmm-badge-ok"}">${count}</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;">
          <span class="cmm-pill ${aria2.available ? "cmm-pill-ok" : "cmm-pill-red"}">
            ${aria2.available ? "aria2c Ready" : "aria2c Missing"}
          </span>
          <span class="cmm-pill ${cfg.has_api_key ? "cmm-pill-ok" : "cmm-pill-warn"}">
            ${cfg.has_api_key ? "API Key Set" : "No API Key"}
          </span>
        </div>
        <div style="font-size:12px;color:#8b949e;line-height:1.4;">
          Scans your active workflow for missing models, searches both <code>civitai.com</code> and <code>civitai.red</code>, and downloads with <code>aria2c</code>.
        </div>
        <div style="display:flex;flex-direction:column;gap:8px;">
          <button type="button" class="cmm-btn cmm-btn-blue" data-sb="open">Open Missing Models Panel</button>
          <button type="button" class="cmm-btn cmm-btn-primary" data-sb="scan-search">Scan &amp; Search CivitAI</button>
          <button type="button" class="cmm-btn" data-sb="settings">API Key &amp; Settings</button>
        </div>
        ${
          count > 0
            ? `<div style="border-top:1px solid #30363d;padding-top:10px;display:flex;flex-direction:column;gap:6px;">
                <div style="font-size:11px;font-weight:600;color:#8b949e;text-transform:uppercase;">Missing (${count})</div>
                ${this.missing
                  .slice(0, 8)
                  .map(
                    (m) =>
                      `<div style="font-size:12px;font-family:ui-monospace,monospace;word-break:break-all;padding:4px 6px;border-radius:4px;background:#0d1117;border:1px solid #30363d;">
                        ${escapeHtml(m.basename || m.filename)}
                        <span style="color:#8b949e;font-family:sans-serif;"> (${escapeHtml(m.folder)})</span>
                      </div>`
                  )
                  .join("")}
                ${count > 8 ? `<div style="font-size:11px;color:#8b949e;">+${count - 8} more...</div>` : ""}
              </div>`
            : ""
        }
      </div>
    `;

    this.sidebarEl.querySelector('[data-sb="open"]')?.addEventListener("click", () => this.openModal());
    this.sidebarEl.querySelector('[data-sb="scan-search"]')?.addEventListener("click", () => this.openModal({ autoSearch: true }));
    this.sidebarEl.querySelector('[data-sb="settings"]')?.addEventListener("click", () => {
      this.settingsOpen = true;
      this.openModal();
    });
  }

  async apiJson(path, options = {}) {
    const resp = await api.fetchApi(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    return data;
  }

  async fetchStatus() {
    try {
      this.status = await this.apiJson("/status");
    } catch (err) {
      console.warn("[CivitAI Missing Models] Failed to fetch status:", err);
    }
  }

  async fetchDownloads() {
    try {
      const data = await this.apiJson("/downloads");
      for (const job of data.jobs || []) {
        this.recordJob(job);
      }
    } catch (err) {
      console.warn("[CivitAI Missing Models] Failed to fetch downloads:", err);
    }
  }

  recordJob(job) {
    if (!job || !job.job_id) return;
    this.jobsById.set(job.job_id, job);
    const key = `${job.folder}::${String(job.filename || "").toLowerCase()}`;
    const existing = this.jobsByModelKey.get(key);
    if (!existing || (job.updated_at || 0) >= (existing.updated_at || 0)) {
      this.jobsByModelKey.set(key, job);
    }
  }

  bindProgressEvents() {
    api.addEventListener("civitai_missing_models.progress", (event) => {
      const job = event.detail;
      if (!job) return;
      this.recordJob(job);
      if (this.modalOverlay) {
        this.updateJobProgressInDom(job);
      }
      if (job.status === "completed") {
        this.refreshComfyNodeLists();
      }
    });
  }

  renderProgressHtml(job) {
    if (!job) return "";
    const fillClass =
      job.status === "completed" ? "done" : job.status === "failed" || job.status === "cancelled" ? "error" : "";
    const active = ["queued", "resolving", "downloading"].includes(job.status);
    return `
      <div class="cmm-progress-wrap">
        <div class="cmm-progress-bar-bg">
          <div class="cmm-progress-bar-fill ${fillClass}" style="width: ${job.percent || 0}%"></div>
        </div>
        <div class="cmm-progress-meta">
          <span>
            <strong>${escapeHtml(job.status.toUpperCase())}</strong>
            ${job.domain ? `via ${escapeHtml(job.domain)}` : ""}
            ${job.downloaded ? `&bull; ${escapeHtml(job.downloaded)}${job.total ? ` / ${escapeHtml(job.total)}` : ""} (${job.percent}%)` : ""}
            ${job.speed ? `&bull; ${escapeHtml(job.speed)}` : ""}
            ${job.connections ? `&bull; ${job.connections} conn` : ""}
            ${job.eta ? `&bull; ETA ${escapeHtml(job.eta)}` : ""}
            ${job.error ? `&bull; <span style="color:#ff7b72">${escapeHtml(job.error)}</span>` : ""}
          </span>
          ${
            active
              ? `<button type="button" class="cmm-btn cmm-btn-danger" data-cancel-job="${escapeHtml(job.job_id)}">Cancel</button>`
              : ""
          }
        </div>
      </div>
    `;
  }

  updateJobProgressInDom(job) {
    if (!this.modalOverlay || !job) return;
    const modelKey = `${job.folder}::${String(job.filename || "").toLowerCase()}`;
    const escapedKey = CSS.escape(modelKey);
    const card = this.modalOverlay.querySelector(`.cmm-card[data-model-id="${escapedKey}"]`);
    if (!card) return;

    const slot = card.querySelector(`[data-progress-slot="${escapedKey}"]`);
    if (slot) {
      slot.innerHTML = this.renderProgressHtml(job);
      const cancelBtn = slot.querySelector("[data-cancel-job]");
      if (cancelBtn) {
        cancelBtn.addEventListener("click", () => {
          const jobId = cancelBtn.getAttribute("data-cancel-job");
          if (jobId) this.cancelDownload(jobId);
        });
      }
    }

    const dlBtn = card.querySelector(`[data-download="${escapedKey}"]`);
    if (dlBtn) {
      const aria2Available = Boolean(this.status?.aria2?.available);
      const isDownloading = ["queued", "resolving", "downloading"].includes(job.status);
      const isDone = job.status === "completed";
      dlBtn.disabled = !aria2Available || isDownloading || isDone;
      dlBtn.textContent = isDone ? "Downloaded" : isDownloading ? "Downloading..." : "Download (aria2)";
    }
  }

  hookWorkflowLoad() {
    const origLoadGraphData = app.loadGraphData?.bind(app);
    if (!origLoadGraphData) return;

    app.loadGraphData = async (...args) => {
      const res = await origLoadGraphData(...args);
      if (this.status?.config?.auto_scan_on_load !== false) {
        setTimeout(() => this.scanCurrentWorkflow({ showToast: true }), 400);
      }
      return res;
    };
  }

  injectToolbarButton() {
    if (this.toolbarBtn) return;

    const btn = document.createElement("button");
    btn.className = "cmm-toolbar-btn";
    btn.title = "Detect missing workflow models, search CivitAI (.com & .red), and download with aria2";
    btn.innerHTML = `<span>Missing Models</span><span class="cmm-badge cmm-badge-ok" style="display:none">0</span>`;
    btn.addEventListener("click", () => this.openModal());
    this.toolbarBtn = btn;

    // Try ComfyUI top menu bar variants (legacy & transition) if present
    const attach = () => {
      const topBar =
        app.menu?.settingsGroup?.element?.parentElement ||
        document.querySelector(".comfyui-menu-right") ||
        document.querySelector(".comfyui-menu .flex") ||
        document.querySelector(".comfy-menu");
      if (topBar && !btn.isConnected) {
        topBar.prepend(btn);
        return true;
      }
      return false;
    };

    if (!attach()) {
      const observer = new MutationObserver(() => {
        if (attach()) observer.disconnect();
      });
      observer.observe(document.body, { childList: true, subtree: true });
    }
  }

  updateBadge() {
    if (!this.toolbarBtn) return;
    const badge = this.toolbarBtn.querySelector(".cmm-badge");
    if (!badge) return;
    const count = this.missing.length;
    if (count > 0) {
      badge.style.display = "inline-flex";
      badge.textContent = String(count);
      badge.classList.remove("cmm-badge-ok");
    } else {
      badge.style.display = "none";
    }
  }

  showToast(count) {
    this.dismissToast();
    const toast = document.createElement("div");
    toast.className = "cmm-toast";
    toast.innerHTML = `
      <div><strong>${count} missing model${count === 1 ? "" : "s"}</strong> detected in this workflow.</div>
      <button type="button" data-action="open">Find & Download</button>
      <button type="button" class="cmm-toast-close" data-action="close">&times;</button>
    `;
    toast.querySelector('[data-action="open"]').addEventListener("click", () => {
      this.dismissToast();
      this.openModal({ autoSearch: true });
    });
    toast.querySelector('[data-action="close"]').addEventListener("click", () => this.dismissToast());
    document.body.appendChild(toast);
    this.toastEl = toast;

    setTimeout(() => {
      if (this.toastEl === toast) this.dismissToast();
    }, 10000);
  }

  dismissToast() {
    if (this.toastEl) {
      this.toastEl.remove();
      this.toastEl = null;
    }
  }

  async scanCurrentWorkflow({ showToast = false } = {}) {
    if (!app.graph) return;
    const workflow = app.graph.serialize();
    try {
      const res = await this.apiJson("/scan", {
        method: "POST",
        body: JSON.stringify({ workflow }),
      });
      this.missing = res.missing || [];
      this.totalReferenced = res.total_referenced || 0;
      this.updateBadge();
      this.renderSidebarContent();
      if (showToast && this.missing.length > 0 && !this.modalOverlay) {
        this.showToast(this.missing.length);
      }
      if (this.modalOverlay) {
        this.renderModalContent();
      }
    } catch (err) {
      console.error("[CivitAI Missing Models] Scan failed:", err);
    }
  }

  async searchOne(model, customQuery = "") {
    const state = this.searchResults.get(model.id) || { selectedIdx: 0 };
    state.loading = true;
    state.error = "";
    if (customQuery !== undefined) state.customQuery = customQuery;
    this.searchResults.set(model.id, state);
    this.renderModalContent();

    try {
      const res = await this.apiJson("/search", {
        method: "POST",
        body: JSON.stringify({
          filename: model.filename,
          folder: model.folder,
          known_url: model.known_url || "",
          custom_query: state.customQuery || "",
        }),
      });
      state.loading = false;
      state.candidates = res.candidates || [];
      state.selectedIdx = 0;
      state.domainsSearched = res.domains_searched || [];
    } catch (err) {
      state.loading = false;
      state.error = err.message;
    }
    this.searchResults.set(model.id, state);
    this.renderModalContent();
  }

  async searchAll() {
    if (!this.missing.length) return;
    for (const m of this.missing) {
      const st = this.searchResults.get(m.id) || { selectedIdx: 0 };
      st.loading = true;
      st.error = "";
      this.searchResults.set(m.id, st);
    }
    this.renderModalContent();

    try {
      const res = await this.apiJson("/search", {
        method: "POST",
        body: JSON.stringify({ items: this.missing }),
      });
      for (const item of res.results || []) {
        this.searchResults.set(item.id, {
          loading: false,
          candidates: item.candidates || [],
          selectedIdx: 0,
          domainsSearched: item.domains_searched || [],
          error: item.error || "",
          customQuery: "",
        });
      }
    } catch (err) {
      for (const m of this.missing) {
        const st = this.searchResults.get(m.id) || {};
        st.loading = false;
        st.error = err.message;
        this.searchResults.set(m.id, st);
      }
    }
    this.renderModalContent();
  }

  async startDownloadForModel(model) {
    const st = this.searchResults.get(model.id);
    const cand = st?.candidates?.[st.selectedIdx || 0];
    if (!cand) return;

    try {
      const res = await this.apiJson("/download", {
        method: "POST",
        body: JSON.stringify({
          filename: model.filename,
          folder: model.folder,
          download_url: cand.download_url,
          version_id: cand.version_id,
          model_name: cand.model_name,
          primary_domain: cand.primary_domain,
        }),
      });
      if (res.job) {
        this.recordJob(res.job);
      }
    } catch (err) {
      alert(`Download failed to start: ${err.message}`);
    }
    this.renderModalContent();
  }

  async downloadAllMatched() {
    for (const model of this.missing) {
      const st = this.searchResults.get(model.id);
      const cand = st?.candidates?.[st.selectedIdx || 0];
      const job = this.jobsByModelKey.get(model.id);
      if (cand && (!job || job.status === "failed" || job.status === "cancelled")) {
        await this.startDownloadForModel(model);
      }
    }
  }

  async cancelDownload(jobId) {
    try {
      const res = await this.apiJson(`/downloads/${encodeURIComponent(jobId)}/cancel`, {
        method: "POST",
      });
      if (res.job) this.recordJob(res.job);
    } catch (err) {
      console.error("[CivitAI Missing Models] Cancel failed:", err);
    }
    this.renderModalContent();
  }

  async refreshComfyNodeLists() {
    try {
      if (typeof app.refreshComboInNodes === "function") {
        await app.refreshComboInNodes();
      }
    } catch (err) {
      console.warn("[CivitAI Missing Models] Could not refresh node combos:", err);
    }
  }

  async saveSettingsFromForm(formEl) {
    const apiKeyInput = formEl.querySelector('[name="civitai_api_key"]');
    const domCom = formEl.querySelector('[name="domain_com"]').checked;
    const domRed = formEl.querySelector('[name="domain_red"]').checked;
    const domains = [];
    if (domCom) domains.push("https://civitai.com");
    if (domRed) domains.push("https://civitai.red");

    const payload = {
      domains,
      aria2c_path: formEl.querySelector('[name="aria2c_path"]').value.trim(),
      aria2c_connections: Number(formEl.querySelector('[name="aria2c_connections"]').value),
      aria2c_splits: Number(formEl.querySelector('[name="aria2c_splits"]').value),
      aria2c_max_concurrent: Number(formEl.querySelector('[name="aria2c_max_concurrent"]').value),
      auto_scan_on_load: formEl.querySelector('[name="auto_scan_on_load"]').checked,
    };

    // Only overwrite stored key if the user typed something new
    const newKey = apiKeyInput.value.trim();
    if (newKey) {
      payload.civitai_api_key = newKey;
    }

    try {
      const res = await this.apiJson("/config", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      this.status.config = res.config;
      this.status.aria2 = res.aria2;
      this.settingsOpen = false;
      this.renderModalContent();
    } catch (err) {
      alert(`Failed to save settings: ${err.message}`);
    }
  }

  async clearApiKey() {
    try {
      const res = await this.apiJson("/config", {
        method: "POST",
        body: JSON.stringify({ civitai_api_key: "" }),
      });
      this.status.config = res.config;
      this.keyTestMessage = "";
      this.keyTestValid = null;
      this.renderModalContent();
    } catch (err) {
      alert(`Failed to clear API key: ${err.message}`);
    }
  }

  async testApiKey(candidateKey = "") {
    this.keyTestMessage = "Testing API key...";
    this.keyTestValid = null;
    this.renderModalContent();

    try {
      const res = await this.apiJson("/test-key", {
        method: "POST",
        body: JSON.stringify(candidateKey ? { civitai_api_key: candidateKey } : {}),
      });
      if (res.valid) {
        this.keyTestValid = true;
        this.keyTestMessage = `Valid! Authenticated as ${res.username} (${res.domain}).`;
      } else {
        this.keyTestValid = false;
        this.keyTestMessage = res.error || "Invalid API key.";
      }
    } catch (err) {
      this.keyTestValid = false;
      this.keyTestMessage = err.message;
    }
    this.renderModalContent();
  }

  async syncNativeSetting(partialConfig) {
    try {
      const res = await this.apiJson("/config", {
        method: "POST",
        body: JSON.stringify(partialConfig),
      });
      this.status.config = res.config;
      this.status.aria2 = res.aria2;
      if (this.modalOverlay) {
        this.renderModalContent();
      }
    } catch (err) {
      console.warn("[CivitAI Missing Models] Failed to sync setting:", err);
    }
  }

  async openModal({ autoSearch = false } = {}) {
    this.dismissToast();
    await this.fetchStatus();
    await this.scanCurrentWorkflow({ showToast: false });

    if (!this.modalOverlay) {
      const overlay = document.createElement("div");
      overlay.className = "cmm-overlay";
      overlay.addEventListener("click", (e) => {
        if (e.target === overlay) this.closeModal();
      });
      const modal = document.createElement("div");
      modal.className = "cmm-modal";
      overlay.appendChild(modal);
      document.body.appendChild(overlay);
      this.modalOverlay = overlay;
    }

    this.renderModalContent();
    if (autoSearch && this.missing.length > 0) {
      this.searchAll();
    }
  }

  closeModal() {
    if (this.modalOverlay) {
      this.modalOverlay.remove();
      this.modalOverlay = null;
    }
  }

  renderModalContent() {
    if (!this.modalOverlay) return;
    const modal = this.modalOverlay.querySelector(".cmm-modal");
    const prevBody = modal.querySelector(".cmm-body");
    const savedScrollTop = prevBody ? prevBody.scrollTop : 0;
    const cfg = this.status.config || {};
    const aria2 = this.status.aria2 || {};
    const domains = cfg.domains || ["https://civitai.com", "https://civitai.red"];
    const hasCom = domains.includes("https://civitai.com");
    const hasRed = domains.includes("https://civitai.red");

    const matchedCount = this.missing.filter((m) => (this.searchResults.get(m.id)?.candidates?.length || 0) > 0).length;

    modal.innerHTML = `
      <div class="cmm-header">
        <div class="cmm-title-group">
          <h2 class="cmm-title">CivitAI Missing Models</h2>
          <span class="cmm-pill ${aria2.available ? "cmm-pill-ok" : "cmm-pill-red"}">
            ${aria2.available ? "aria2c Ready" : "aria2c Missing"}
          </span>
          <span class="cmm-pill ${cfg.has_api_key ? "cmm-pill-ok" : "cmm-pill-warn"}">
            ${cfg.has_api_key ? `API Key (${escapeHtml(cfg.api_key_preview)})` : "No API Key"}
          </span>
          ${hasCom ? `<span class="cmm-pill cmm-pill-blue">civitai.com</span>` : ""}
          ${hasRed ? `<span class="cmm-pill cmm-pill-red">civitai.red</span>` : ""}
        </div>
        <div class="cmm-actions">
          <button type="button" class="cmm-btn" data-act="rescan">Rescan</button>
          <button type="button" class="cmm-btn cmm-btn-blue" data-act="search-all" ${this.missing.length ? "" : "disabled"}>
            Search All (${this.missing.length})
          </button>
          <button type="button" class="cmm-btn cmm-btn-primary" data-act="download-all" ${matchedCount && aria2.available ? "" : "disabled"}>
            Download All (${matchedCount})
          </button>
          <button type="button" class="cmm-btn" data-act="toggle-settings">Settings</button>
          <button type="button" class="cmm-btn" data-act="close">&times;</button>
        </div>
      </div>

      ${
        !aria2.available
          ? `<div class="cmm-banner">
              <strong>aria2c not found.</strong> Install it for fast multi-connection downloads:
              macOS: <code>brew install aria2</code> &bull;
              Ubuntu/Debian: <code>sudo apt install aria2</code> &bull;
              Windows: <code>winget install aria2.aria2</code> (or set its path in Settings).
            </div>`
          : ""
      }

      ${
        this.settingsOpen
          ? `<form class="cmm-settings" data-settings-form>
              <div class="cmm-field">
                <label>CivitAI API Key ${cfg.has_api_key ? `(saved: ${escapeHtml(cfg.api_key_preview)})` : "(required for gated/NSFW downloads)"}</label>
                <div style="display:flex;gap:6px;">
                  <input class="cmm-input" style="flex:1" type="password" name="civitai_api_key" placeholder="Paste new CivitAI API key..." />
                  <button type="button" class="cmm-btn" data-act="test-key">Test</button>
                  ${cfg.has_api_key ? `<button type="button" class="cmm-btn" data-act="clear-key">Clear</button>` : ""}
                </div>
                ${
                  this.keyTestMessage
                    ? `<div style="font-size:11px;color:${this.keyTestValid === true ? "#56d364" : this.keyTestValid === false ? "#ff7b72" : "#8b949e"}">${escapeHtml(this.keyTestMessage)}</div>`
                    : ""
                }
              </div>
              <div class="cmm-field">
                <label>CivitAI Domains to Query</label>
                <div class="cmm-checkbox-row">
                  <label><input type="checkbox" name="domain_com" ${hasCom ? "checked" : ""} /> civitai.com</label>
                  <label><input type="checkbox" name="domain_red" ${hasRed ? "checked" : ""} /> civitai.red</label>
                </div>
              </div>
              <div class="cmm-field">
                <label>aria2c Executable Path</label>
                <input class="cmm-input" type="text" name="aria2c_path" value="${escapeHtml(cfg.aria2c_path || "aria2c")}" />
              </div>
              <div class="cmm-field">
                <label>Connections (-x) / Splits (-s) / Concurrent</label>
                <div style="display:flex;gap:6px;">
                  <input class="cmm-input" style="width:64px" type="number" min="1" max="16" name="aria2c_connections" value="${cfg.aria2c_connections ?? 16}" title="Max connections per server" />
                  <input class="cmm-input" style="width:64px" type="number" min="1" max="64" name="aria2c_splits" value="${cfg.aria2c_splits ?? 16}" title="Splits per file" />
                  <input class="cmm-input" style="width:64px" type="number" min="1" max="8" name="aria2c_max_concurrent" value="${cfg.aria2c_max_concurrent ?? 2}" title="Concurrent downloads" />
                </div>
              </div>
              <div class="cmm-field">
                <label>Workflow Scan</label>
                <div class="cmm-checkbox-row">
                  <label><input type="checkbox" name="auto_scan_on_load" ${cfg.auto_scan_on_load !== false ? "checked" : ""} /> Auto-scan when loading workflows</label>
                </div>
              </div>
              <div class="cmm-field" style="justify-content:flex-end;">
                <button type="submit" class="cmm-btn cmm-btn-primary">Save Settings</button>
              </div>
            </form>`
          : ""
      }

      <div class="cmm-body">
        ${
          this.missing.length === 0
            ? `<div class="cmm-empty">
                All <strong>${this.totalReferenced}</strong> model${this.totalReferenced === 1 ? "" : "s"} referenced by this workflow are installed locally.
              </div>`
            : this.missing.map((m) => this.renderModelCardHtml(m, aria2.available)).join("")
        }
      </div>

      <div class="cmm-footer">
        <span>Referenced: ${this.totalReferenced} &bull; Missing: ${this.missing.length}</span>
        <div style="display:flex;gap:8px;">
          <button type="button" class="cmm-btn" data-act="refresh-nodes">Refresh ComfyUI Model Lists</button>
        </div>
      </div>
    `;

    this.bindModalHandlers(modal);
    const nextBody = modal.querySelector(".cmm-body");
    if (nextBody && savedScrollTop > 0) {
      nextBody.scrollTop = savedScrollTop;
    }
  }

  renderModelCardHtml(model, aria2Available) {
    const st = this.searchResults.get(model.id) || {};
    const candidates = st.candidates || [];
    const selectedIdx = st.selectedIdx || 0;
    const cand = candidates[selectedIdx];
    const job = this.jobsByModelKey.get(model.id);

    let candidateSection = "";
    if (st.loading) {
      candidateSection = `<div class="cmm-candidate-sub">Searching civitai.com &amp; civitai.red...</div>`;
    } else if (st.error) {
      candidateSection = `<div class="cmm-pill cmm-pill-red">Search error: ${escapeHtml(st.error)}</div>`;
    } else if (cand) {
      const domainBadges = (cand.domains || [cand.primary_domain])
        .map((d) => `<span class="cmm-pill ${d.includes(".red") ? "cmm-pill-red" : "cmm-pill-blue"}">${escapeHtml(d)}</span>`)
        .join("");

      const selectorHtml =
        candidates.length > 1
          ? `<select class="cmm-select" data-select-cand="${escapeHtml(model.id)}">
              ${candidates
                .map(
                  (c, idx) =>
                    `<option value="${idx}" ${idx === selectedIdx ? "selected" : ""}>
                      ${escapeHtml(c.model_name)} — ${escapeHtml(c.file_name)} (${Math.round(c.score)}%, ${formatBytes(c.size_bytes)})
                    </option>`
                )
                .join("")}
            </select>`
          : "";

      const isDownloading = job && ["queued", "resolving", "downloading"].includes(job.status);
      const isDone = job && job.status === "completed";

      candidateSection = `
        ${selectorHtml}
        <div class="cmm-candidate">
          ${cand.preview_url ? `<img class="cmm-thumb" src="${escapeHtml(cand.preview_url)}" alt="" loading="lazy" />` : ""}
          <div class="cmm-candidate-info">
            <a class="cmm-candidate-title" href="${escapeHtml(cand.page_url)}" target="_blank" rel="noopener noreferrer">
              ${escapeHtml(cand.model_name)} &mdash; ${escapeHtml(cand.version_name)}
            </a>
            <div class="cmm-candidate-sub">
              <span>${escapeHtml(cand.file_name)}</span>
              <span>&bull; ${formatBytes(cand.size_bytes)}</span>
              ${cand.base_model ? `<span class="cmm-pill">${escapeHtml(cand.base_model)}</span>` : ""}
              <span class="cmm-pill ${cand.score >= 85 ? "cmm-pill-ok" : "cmm-pill-warn"}">${Math.round(cand.score)}% ${escapeHtml(cand.match_reason)}</span>
              ${domainBadges}
              ${cand.nsfw ? `<span class="cmm-pill cmm-pill-red">NSFW</span>` : ""}
            </div>
          </div>
          <div>
            <button
              type="button"
              class="cmm-btn cmm-btn-primary"
              data-download="${escapeHtml(model.id)}"
              ${!aria2Available || isDownloading || isDone ? "disabled" : ""}
            >
              ${isDone ? "Downloaded" : isDownloading ? "Downloading..." : "Download (aria2)"}
            </button>
          </div>
        </div>
      `;
    } else if (st.candidates) {
      candidateSection = `<div class="cmm-candidate-sub">No matches found on CivitAI. Try editing the search term or pasting a CivitAI URL above.</div>`;
    }

    const progressSection = `<div data-progress-slot="${escapeHtml(model.id)}">${this.renderProgressHtml(job)}</div>`;

    return `
      <div class="cmm-card" data-model-id="${escapeHtml(model.id)}">
        <div class="cmm-card-top">
          <div>
            <div class="cmm-model-filename">${escapeHtml(model.filename)}</div>
            <div class="cmm-meta-row">
              <span class="cmm-pill">${escapeHtml(model.folder)}</span>
              <span>Nodes: ${escapeHtml((model.node_types || []).join(", "))}</span>
              ${
                model.local_matches?.length
                  ? `<span class="cmm-pill cmm-pill-warn">Found locally at: ${escapeHtml(model.local_matches[0])}</span>`
                  : ""
              }
            </div>
          </div>
          <div class="cmm-search-bar">
            <input
              class="cmm-input"
              type="text"
              data-query-input="${escapeHtml(model.id)}"
              value="${escapeHtml(st.customQuery || "")}"
              placeholder="Custom search or CivitAI URL..."
            />
            <button type="button" class="cmm-btn cmm-btn-blue" data-search-one="${escapeHtml(model.id)}">
              Search
            </button>
          </div>
        </div>
        ${candidateSection}
        ${progressSection}
      </div>
    `;
  }

  bindModalHandlers(modal) {
    modal.querySelector('[data-act="close"]')?.addEventListener("click", () => this.closeModal());
    modal.querySelector('[data-act="rescan"]')?.addEventListener("click", () => this.scanCurrentWorkflow());
    modal.querySelector('[data-act="search-all"]')?.addEventListener("click", () => this.searchAll());
    modal.querySelector('[data-act="download-all"]')?.addEventListener("click", () => this.downloadAllMatched());
    modal.querySelector('[data-act="toggle-settings"]')?.addEventListener("click", () => {
      this.settingsOpen = !this.settingsOpen;
      this.renderModalContent();
    });
    modal.querySelector('[data-act="clear-key"]')?.addEventListener("click", () => this.clearApiKey());
    modal.querySelector('[data-act="test-key"]')?.addEventListener("click", () => {
      const input = modal.querySelector('[name="civitai_api_key"]');
      this.testApiKey(input ? input.value.trim() : "");
    });
    modal.querySelector('[data-act="refresh-nodes"]')?.addEventListener("click", async () => {
      await this.refreshComfyNodeLists();
      await this.scanCurrentWorkflow();
    });

    const settingsForm = modal.querySelector("[data-settings-form]");
    if (settingsForm) {
      settingsForm.addEventListener("submit", (e) => {
        e.preventDefault();
        this.saveSettingsFromForm(settingsForm);
      });
    }

    const modelById = new Map(this.missing.map((m) => [m.id, m]));

    modal.querySelectorAll("[data-search-one]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.getAttribute("data-search-one");
        const model = modelById.get(id);
        const input = modal.querySelector(`[data-query-input="${CSS.escape(id)}"]`);
        if (model) this.searchOne(model, input ? input.value.trim() : "");
      });
    });

    modal.querySelectorAll("[data-query-input]").forEach((input) => {
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          const id = input.getAttribute("data-query-input");
          const model = modelById.get(id);
          if (model) this.searchOne(model, input.value.trim());
        }
      });
    });

    modal.querySelectorAll("[data-select-cand]").forEach((sel) => {
      sel.addEventListener("change", () => {
        const id = sel.getAttribute("data-select-cand");
        const st = this.searchResults.get(id);
        if (st) {
          st.selectedIdx = Number(sel.value) || 0;
          this.renderModalContent();
        }
      });
    });

    modal.querySelectorAll("[data-download]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.getAttribute("data-download");
        const model = modelById.get(id);
        if (model) this.startDownloadForModel(model);
      });
    });

    modal.querySelectorAll("[data-cancel-job]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const jobId = btn.getAttribute("data-cancel-job");
        if (jobId) this.cancelDownload(jobId);
      });
    });
  }
}

const ui = new CivitAIMissingModelsUI();

app.registerExtension({
  name: EXTENSION_NAME,
  commands: [
    {
      id: "CivitAI.MissingModels.Open",
      label: "Open Missing Models Downloader",
      icon: "pi pi-download",
      function: () => ui.openModal(),
    },
    {
      id: "CivitAI.MissingModels.ScanAndSearch",
      label: "Scan Workflow & Search CivitAI",
      icon: "pi pi-search",
      function: () => ui.openModal({ autoSearch: true }),
    },
  ],
  menuCommands: [
    {
      path: ["Extensions", "CivitAI Missing Models"],
      commands: ["CivitAI.MissingModels.Open", "CivitAI.MissingModels.ScanAndSearch"],
    },
  ],
  keybindings: [
    {
      combo: { key: "m", ctrl: true, shift: true },
      commandId: "CivitAI.MissingModels.Open",
    },
  ],
  settings: [
    {
      id: "CivitAI.MissingModels.ApiKey",
      name: "CivitAI API Key",
      category: ["CivitAI Missing Models", "Authentication", "API Key"],
      type: "text",
      defaultValue: "",
      tooltip: "Personal CivitAI API key for downloading gated/NSFW models from civitai.com and civitai.red.",
      onChange: (newVal) => {
        if (typeof newVal === "string" && newVal.trim()) {
          ui.syncNativeSetting({ civitai_api_key: newVal.trim() });
        }
      },
    },
    {
      id: "CivitAI.MissingModels.AutoScan",
      name: "Auto-scan workflows on load",
      category: ["CivitAI Missing Models", "Behavior", "Auto Scan"],
      type: "boolean",
      defaultValue: true,
      tooltip: "Automatically check for missing models when loading or dropping a workflow.",
      onChange: (newVal) => {
        if (typeof newVal === "boolean") {
          ui.syncNativeSetting({ auto_scan_on_load: newVal });
        }
      },
    },
    {
      id: "CivitAI.MissingModels.Aria2Connections",
      name: "aria2c max connections (-x)",
      category: ["CivitAI Missing Models", "Downloads", "Connections"],
      type: "number",
      defaultValue: 16,
      attrs: { min: 1, max: 16, step: 1 },
      tooltip: "Maximum parallel connections per server used by aria2c.",
      onChange: (newVal) => {
        if (newVal !== undefined && newVal !== null) {
          ui.syncNativeSetting({ aria2c_connections: Number(newVal) });
        }
      },
    },
  ],
  async setup() {
    await ui.init();
  },
});
