/* 歌词管理页面：通过 AstrBotPluginView bridge 调用插件 API。
 * 不直接访问 Dashboard 的 cookie / DOM，所有数据都走 bridge。
 */
const MAX_BYTES = 5 * 1024 * 1024;
const ALLOWED_EXT = /\.(txt|json|lrc)$/i;

const state = {
  files: [],
  stats: { songs: 0, lines: 0, grams: 0, files: 0 },
  busy: false,
};

const $ = (id) => document.getElementById(id);

function bridge() {
  return window.AstrBotPluginView || window.AstrBotPluginPage || null;
}

function fmtBytes(bytes) {
  const n = Number(bytes) || 0;
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

function fmtTime(seconds) {
  const n = Number(seconds) || 0;
  if (!n) return "";
  const d = new Date(n * 1000);
  const pad = (v) => String(v).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

let toastTimer = 0;
function toast(message, kind = "info", duration = 3200) {
  const el = $("toast");
  if (!el) return;
  el.textContent = message;
  el.className = `toast ${kind}`;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    el.className = "toast hidden";
  }, duration);
}

function setBanner(message, kind = "warn") {
  const el = $("banner");
  if (!el) return;
  if (!message) {
    el.className = "banner hidden";
    el.textContent = "";
    return;
  }
  el.textContent = message;
  el.className = `banner ${kind}`;
}

function setBusy(busy) {
  state.busy = busy;
  for (const id of ["upload-btn", "reload-btn", "refresh-btn"]) {
    const el = $(id);
    if (el) el.disabled = busy;
  }
}

// 页面在 sandbox iframe 里，window.confirm 会被拦截，这里用自绘确认框。
function confirmDialog(message, okLabel = "确定") {
  return new Promise((resolve) => {
    const modal = $("modal");
    const text = $("modal-text");
    const ok = $("modal-ok");
    const cancel = $("modal-cancel");
    if (!modal || !text || !ok || !cancel) {
      resolve(false);
      return;
    }
    text.textContent = message;
    ok.textContent = okLabel;
    modal.classList.remove("hidden");

    const cleanup = () => {
      modal.classList.add("hidden");
      ok.removeEventListener("click", onOk);
      cancel.removeEventListener("click", onCancel);
      modal.removeEventListener("click", onBackdrop);
      document.removeEventListener("keydown", onKey);
    };
    const finish = (value) => {
      cleanup();
      resolve(value);
    };
    const onOk = () => finish(true);
    const onCancel = () => finish(false);
    const onBackdrop = (event) => {
      if (event.target === modal) finish(false);
    };
    const onKey = (event) => {
      if (event.key === "Escape") finish(false);
    };

    ok.addEventListener("click", onOk);
    cancel.addEventListener("click", onCancel);
    modal.addEventListener("click", onBackdrop);
    document.addEventListener("keydown", onKey);
    ok.focus();
  });
}

async function callGet(endpoint, params) {
  const api = bridge();
  if (!api) throw new Error("页面桥接未加载，请从 AstrBot 插件详情页打开此页面。");
  return api.apiGet(endpoint, params);
}

async function callPost(endpoint, body) {
  const api = bridge();
  if (!api) throw new Error("页面桥接未加载，请从 AstrBot 插件详情页打开此页面。");
  return api.apiPost(endpoint, body);
}

function renderStats(payload) {
  const list = Array.isArray(payload.files) ? payload.files : [];
  const stats = {
    songs: payload.songs || 0,
    lines: payload.lines || 0,
    grams: payload.grams || 0,
    files: list.length || payload.file_count || 0,
  };
  state.stats = stats;
  $("stat-songs").textContent = String(stats.songs);
  $("stat-lines").textContent = String(stats.lines);
  $("stat-grams").textContent = String(stats.grams);
  $("stat-files").textContent = String(stats.files);
  $("file-count").textContent = stats.files ? `（${stats.files} 个）` : "";
  $("loaded-at").textContent = payload.loaded_at
    ? `最近加载：${fmtTime(payload.loaded_at)}`
    : "";
  if (payload.lyrics_dir) {
    $("dir").textContent = `歌词目录：${payload.lyrics_dir}`;
    $("dir").title = payload.lyrics_dir;
  }
  if (Array.isArray(payload.errors) && payload.errors.length) {
    setBanner(`有 ${payload.errors.length} 个文件解析失败：${payload.errors.slice(0, 3).join("；")}`, "warn");
  } else {
    setBanner("");
  }
}

function makeCell(text, className) {
  const td = document.createElement("td");
  if (className) td.className = className;
  td.textContent = text;
  return td;
}

function renderFiles(files) {
  state.files = files;
  const tbody = $("files-body");
  tbody.replaceChildren();

  const hasFiles = files.length > 0;
  $("files-table").classList.toggle("hidden", !hasFiles);
  $("empty").classList.toggle("hidden", hasFiles);

  for (const file of files) {
    const tr = document.createElement("tr");

    const nameCell = document.createElement("td");
    const nameWrap = document.createElement("div");
    nameWrap.className = "file-name";
    const icon = document.createElement("span");
    icon.className = "file-icon";
    const lower = String(file.name).toLowerCase();
    icon.textContent = lower.endsWith(".json") ? "🧾" : lower.endsWith(".lrc") ? "⏱" : "📄";
    const nameText = document.createElement("span");
    nameText.textContent = file.name;
    nameWrap.append(icon, nameText);
    nameCell.append(nameWrap);
    if (file.error) {
      const err = document.createElement("span");
      err.className = "tag err";
      err.textContent = "解析失败";
      err.title = file.error;
      nameCell.append(err);
    }
    tr.append(nameCell);

    tr.append(makeCell(fmtBytes(file.size), "right"));
    tr.append(makeCell(String(file.songs || 0), "right"));
    tr.append(makeCell(String(file.lines || 0), "right"));
    tr.append(makeCell(file.mtime || "", "muted"));

    const actionCell = document.createElement("td");
    actionCell.className = "right";
    const del = document.createElement("button");
    del.type = "button";
    del.className = "btn danger small";
    del.textContent = "删除";
    del.addEventListener("click", () => deleteFile(file.name));
    actionCell.append(del);
    tr.append(actionCell);

    tbody.append(tr);
  }
}

async function refresh() {
  try {
    const payload = await callGet("files");
    const list = Array.isArray(payload.files) ? payload.files : [];
    renderStats(payload);
    renderFiles(list);
    if ((payload.songs || 0) === 0 && list.length > 0 && !(payload.errors || []).length) {
      setBanner(
        "文件已经在歌词库里，但没有解析到歌词。请检查格式：txt 每首要以《歌名》开头，json 要有 songs/lines。",
        "warn"
      );
    }
    return payload;
  } catch (err) {
    setBanner(`加载失败：${err.message || err}`, "err");
    toast(`加载失败：${err.message || err}`, "err", 5000);
    return null;
  }
}

async function uploadFiles(fileList) {
  const api = bridge();
  if (!api) {
    setBanner("页面桥接未加载，请从 AstrBot 插件详情页打开此页面。", "err");
    return;
  }
  const files = Array.from(fileList || []);
  if (!files.length) return;

  const accepted = [];
  const rejected = [];
  for (const file of files) {
    if (!ALLOWED_EXT.test(file.name)) {
      rejected.push(`${file.name}：只支持 .txt / .json / .lrc`);
    } else if (file.size > MAX_BYTES) {
      rejected.push(`${file.name}：超过 5 MB`);
    } else if (file.size === 0) {
      rejected.push(`${file.name}：文件是空的`);
    } else {
      accepted.push(file);
    }
  }
  if (!accepted.length) {
    setBanner(rejected.join("；"), "err");
    toast("没有可上传的文件", "err", 4000);
    return;
  }

  setBusy(true);
  setBanner("");
  let ok = 0;
  const failed = [...rejected];

  for (let i = 0; i < accepted.length; i += 1) {
    const file = accepted[i];
    toast(`正在上传 ${i + 1}/${accepted.length}：${file.name}`, "info", 1500);
    try {
      await api.upload("upload", file);
      ok += 1;
    } catch (err) {
      failed.push(`${file.name}：${err.message || err}`);
    }
  }

  await refresh();
  const totalSongs = state.stats.songs || 0;
  toast(
    `已上传 ${ok} 个文件，当前 ${totalSongs} 首`,
    failed.length ? "warn" : "ok",
    4500
  );
  if (failed.length) {
    setBanner(`部分文件没上传成功：${failed.join("；")}`, "warn");
  }
}

async function deleteFile(name) {
  if (state.busy) return;
  const okay = await confirmDialog(
    `确定要删除「${name}」吗？删除后需要重新上传才能恢复。`,
    "删除"
  );
  if (!okay) return;

  setBusy(true);
  try {
    await callPost("delete", { name });
    const payload = await refresh();
    const songs = payload ? payload.songs : 0;
    toast(`已删除「${name}」，当前 ${songs} 首`, "ok", 4000);
  } catch (err) {
    setBanner(`删除失败：${err.message || err}`, "err");
    toast(`删除失败：${err.message || err}`, "err", 5000);
  } finally {
    setBusy(false);
  }
}

async function reloadLibrary() {
  if (state.busy) return;
  setBusy(true);
  try {
    await callPost("reload", {});
    const payload = await refresh();
    const songs = payload ? payload.songs : 0;
    toast(`歌词库已重载，当前 ${songs} 首`, "ok", 4000);
  } catch (err) {
    setBanner(`重载失败：${err.message || err}`, "err");
    toast(`重载失败：${err.message || err}`, "err", 5000);
  } finally {
    setBusy(false);
  }
}

function bindEvents() {
  const input = $("file-input");
  $("upload-btn").addEventListener("click", () => {
    if (state.busy) return;
    input.value = "";
    input.click();
  });
  input.addEventListener("change", () => {
    uploadFiles(input.files).finally(() => {
      input.value = "";
      setBusy(false);
    });
  });
  $("reload-btn").addEventListener("click", reloadLibrary);
  $("refresh-btn").addEventListener("click", async () => {
    if (state.busy) return;
    setBusy(true);
    try {
      await refresh();
      toast("已刷新", "ok", 1800);
    } finally {
      setBusy(false);
    }
  });
}

async function waitForBridge(timeoutMs = 2000) {
  const start = Date.now();
  while (!bridge() && Date.now() - start < timeoutMs) {
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  return bridge();
}

async function init() {
  bindEvents();
  const api = await waitForBridge();
  if (!api) {
    setBanner(
      "没有检测到 AstrBot 页面桥接。请从「控制台 → 插件 → 歌词识别 → 歌词管理」打开这个页面。",
      "err"
    );
    $("upload-btn").disabled = true;
    $("reload-btn").disabled = true;
    $("refresh-btn").disabled = true;
    return;
  }
  try {
    const info = await api.ready();
    if (info && info.isDark) {
      document.documentElement.dataset.theme = "dark";
    }
  } catch (err) {
    // ready 失败不阻塞页面，下面照常拉数据
    console.warn("bridge.ready 失败：", err);
  }
  await refresh();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
