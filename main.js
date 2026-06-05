const { app, BrowserWindow, ipcMain, dialog, Menu } = require("electron");
const fs = require("fs");
const path = require("path");

let win;

function createWindow() {
  win = new BrowserWindow({
    width: 1180,
    height: 880,
    minWidth: 900,
    minHeight: 640,
    backgroundColor: "#0b0d10",
    title: "Sapere-DNA Studio",
    icon: path.join(__dirname, "build", "icon.png"),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  win.setMenuBarVisibility(false);
  win.loadFile(path.join(__dirname, "src", "index.html"));
}

/* ---- salvataggio file nativo ("Salva con nome" di Windows) ---- */
ipcMain.handle("save-file", async (_evt, { suggestedName, content, filters }) => {
  const result = await dialog.showSaveDialog(win, {
    title: "Salva il file",
    defaultPath: suggestedName,
    filters: filters || [{ name: "Tutti i file", extensions: ["*"] }]
  });
  if (result.canceled || !result.filePath) return { ok: false, canceled: true };
  try {
    fs.writeFileSync(result.filePath, content, "utf8");
    return { ok: true, path: result.filePath };
  } catch (err) {
    return { ok: false, error: String(err) };
  }
});

/* ---- apertura di un file di testo/JSON ---- */
ipcMain.handle("open-text", async () => {
  const result = await dialog.showOpenDialog(win, {
    title: "Apri un file",
    properties: ["openFile"],
    filters: [
      { name: "Testo / JSON", extensions: ["txt", "json", "md", "text", "dna"] },
      { name: "Tutti i file", extensions: ["*"] }
    ]
  });
  if (result.canceled || !result.filePaths.length) return { ok: false, canceled: true };
  try {
    const content = fs.readFileSync(result.filePaths[0], "utf8");
    return { ok: true, content, path: result.filePaths[0] };
  } catch (err) {
    return { ok: false, error: String(err) };
  }
});

/* ---- store persistente (genoma di lavoro + archivio) in userData ---- */
function storePath() { return path.join(app.getPath("userData"), "sapere-dna-store.json"); }
function readStore() {
  try { return JSON.parse(fs.readFileSync(storePath(), "utf8")); }
  catch (e) { return {}; }
}
ipcMain.handle("store-get", async (_evt, key) => {
  const s = readStore();
  return { ok: true, value: (key in s) ? s[key] : null };
});
ipcMain.handle("store-set", async (_evt, { key, value }) => {
  try {
    const s = readStore();
    s[key] = value;
    fs.writeFileSync(storePath(), JSON.stringify(s), "utf8");
    return { ok: true };
  } catch (err) {
    return { ok: false, error: String(err) };
  }
});

app.whenReady().then(() => { Menu.setApplicationMenu(null); createWindow(); });
app.on("window-all-closed", () => { if (process.platform !== "darwin") app.quit(); });
app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
