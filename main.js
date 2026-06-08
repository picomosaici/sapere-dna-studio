const { app, BrowserWindow, ipcMain, dialog, Menu } = require("electron");
const fs = require("fs");
const path = require("path");
const zlib = require("zlib");

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

/* ---- estrazione testo da .docx, senza dipendenze (solo zlib di Node) ----
   Un .docx è uno ZIP; il testo sta in word/document.xml dentro i tag <w:t>,
   con <w:tab/> per i tab e <w:br/> per gli a capo. Leggiamo la directory
   centrale dello ZIP (offset/dimensioni affidabili), troviamo document.xml,
   lo decomprimiamo (deflate) e scandiamo ogni paragrafo in ordine. */
function extractDocxText(buf) {
  // 1) trova la fine della directory centrale (EOCD: firma 0x06054b50)
  let eocd = -1;
  for (let i = buf.length - 22; i >= 0 && i >= buf.length - 22 - 65536; i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error("Il file non è un .docx valido (ZIP senza EOCD)");
  const cdOffset = buf.readUInt32LE(eocd + 16);
  const cdCount = buf.readUInt16LE(eocd + 10);

  // 2) scorri la directory centrale e trova word/document.xml
  let p = cdOffset, target = null;
  for (let n = 0; n < cdCount; n++) {
    if (buf.readUInt32LE(p) !== 0x02014b50) break;
    const method = buf.readUInt16LE(p + 10);
    const compSize = buf.readUInt32LE(p + 20);
    const fnLen = buf.readUInt16LE(p + 28);
    const extraLen = buf.readUInt16LE(p + 30);
    const commentLen = buf.readUInt16LE(p + 32);
    const localOff = buf.readUInt32LE(p + 42);
    const name = buf.toString("utf8", p + 46, p + 46 + fnLen);
    if (name === "word/document.xml") { target = { method, compSize, localOff }; break; }
    p += 46 + fnLen + extraLen + commentLen;
  }
  if (!target) throw new Error("Nel .docx manca word/document.xml");

  // 3) salta l'header locale (nome + extra) per arrivare ai dati compressi
  const lo = target.localOff;
  if (buf.readUInt32LE(lo) !== 0x04034b50) throw new Error("Header locale dello ZIP non valido");
  const dataStart = lo + 30 + buf.readUInt16LE(lo + 26) + buf.readUInt16LE(lo + 28);
  const comp = buf.slice(dataStart, dataStart + target.compSize);
  const xml = (target.method === 8 ? zlib.inflateRawSync(comp) : comp).toString("utf8");

  // 4) testo: ogni <w:p> è un paragrafo; dentro, in ordine, testo + tab + a capo
  const unesc = (s) => s
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&apos;/g, "'")
    .replace(/&#x([0-9a-fA-F]+);/g, (_, h) => String.fromCharCode(parseInt(h, 16)))
    .replace(/&#(\d+);/g, (_, dd) => String.fromCharCode(+dd))
    .replace(/&amp;/g, "&");
  const paras = xml.split(/<w:p[ >]/);
  const lines = [];
  for (let i = 1; i < paras.length; i++) {
    const seg = paras[i];
    let out = ""; const re = /<w:t\b[^>]*>([\s\S]*?)<\/w:t>|<w:tab\b[^>]*\/>|<w:br\b[^>]*\/>|<w:cr\b[^>]*\/>/g; let m;
    while ((m = re.exec(seg))) {
      if (m[1] !== undefined) out += unesc(m[1]);
      else if (m[0].indexOf("<w:tab") === 0) out += "\t";
      else out += "\n";
    }
    lines.push(out);
  }
  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

/* ---- apertura di un file di testo/JSON/Word ---- */
ipcMain.handle("open-text", async () => {
  const result = await dialog.showOpenDialog(win, {
    title: "Apri un file",
    properties: ["openFile"],
    filters: [
      { name: "Testo / Word / JSON", extensions: ["txt", "json", "md", "text", "dna", "docx"] },
      { name: "Tutti i file", extensions: ["*"] }
    ]
  });
  if (result.canceled || !result.filePaths.length) return { ok: false, canceled: true };
  const fp = result.filePaths[0];
  try {
    const content = /\.docx$/i.test(fp) ? extractDocxText(fs.readFileSync(fp)) : fs.readFileSync(fp, "utf8");
    return { ok: true, content, path: fp };
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