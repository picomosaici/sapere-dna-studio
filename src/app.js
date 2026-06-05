/* ============================================================
   Sapere-DNA Studio · APP (orchestratore)
   - guscio/navigazione, laboratorio del genoma, archivio
   - persistenza (Electron store, con fallback localStorage)
   - helper condivisi usati da codifica.js / decodifica.js
   ============================================================ */
(function () {
  "use strict";

  /* ---- ponte nativo (con fallback per uso fuori da Electron) ---- */
  const native = window.nativeAPI || {
    async saveFile(name, content) {
      const blob = new Blob([content], { type: "text/plain" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob); a.download = name; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1500);
      return { ok: true, path: name };
    },
    openText() {
      return new Promise((res) => {
        const inp = document.createElement("input");
        inp.type = "file";
        inp.onchange = () => {
          const f = inp.files[0]; if (!f) return res({ ok: false });
          const r = new FileReader();
          r.onload = () => res({ ok: true, content: String(r.result), path: f.name });
          r.readAsText(f);
        };
        inp.click();
      });
    },
    async storeGet(key) { try { return { ok: true, value: JSON.parse(localStorage.getItem("sdna:" + key)) }; } catch (e) { return { ok: true, value: null }; } },
    async storeSet(key, value) { try { localStorage.setItem("sdna:" + key, JSON.stringify(value)); return { ok: true }; } catch (e) { return { ok: false }; } }
  };

  /* ---- helper ---- */
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const fmt = (n) => Number(n).toLocaleString("it-IT");
  const humanSize = (b) => b < 1024 ? b + " B" : b < 1048576 ? (b / 1024).toFixed(1) + " KB" : (b / 1048576).toFixed(2) + " MB";
  const cleanName = (s) => (s || "").trim().replace(/\s+/g, "-").replace(/[^\p{L}\p{N}._-]+/gu, "-").replace(/-+/g, "-").replace(/^-+|-+$/g, "");
  const slug = (s) => (s.trim().slice(0, 40).replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-+|-+$/g, "") || "voce").toLowerCase();
  const stamp = () => { const d = new Date(), p = (x) => String(x).padStart(2, "0"); return d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate()) + "-" + p(d.getHours()) + p(d.getMinutes()) + p(d.getSeconds()); };

  function showLoader(t, s) { $("loader-title").textContent = t; $("loader-sub").textContent = s || ""; $("overlay").classList.add("on"); }
  function hideLoader() { $("overlay").classList.remove("on"); }
  let toastTimer;
  function toast(msg, kind) {
    const el = $("toast"); el.textContent = msg; el.className = "toast on" + (kind ? " " + kind : "");
    clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove("on"), 2200);
  }

  async function saveNative(content, suggested, ext, what) {
    showLoader("Preparazione del file", (what || "") + "  ·  " + humanSize(new Blob([content]).size));
    await new Promise(r => setTimeout(r, 160)); hideLoader();
    const filters = ext.endsWith(".json") ? [{ name: "JSON", extensions: ["json"] }] : [{ name: "Testo", extensions: ["txt"] }];
    const res = await native.saveFile(suggested + ext, content, filters);
    if (res && res.ok) { showLoader("Salvato ✓", res.path || ""); setTimeout(hideLoader, 850); }
  }

  function colorizeSegs(segs) {
    let html = "";
    for (const s of segs) {
      if (s.role === "space") html += s.text === "\n" ? "<br>" : " ";
      else if (s.role === "gene") html += '<span class="seg-gene">' + esc(s.text) + "</span>";
      else if (s.role === "lit") html += '<span class="seg-lit">' + esc(s.text) + "</span>";
      else if (s.role === "raw") html += '<span class="seg-raw">' + esc(s.text) + "</span>";
    }
    return html;
  }

  /* ---- oggetto condiviso ---- */
  const G = window.SapereDNAGenoma, C = window.SapereDNA, R = window.SapereDNARete, T = window.SapereDNATrasformatore, Gen = window.SapereDNAGeneratore;  const DEMO = window.GENOMA_IT_DEMO;
  const APP = {
    G, C, R, T, Gen, esc, fmt, humanSize, cleanName, slug, stamp,
    showLoader, hideLoader, toast, saveNative, openText: () => native.openText(),
    colorizeSegs, addToArchive,
    genome: null, dict: null, archive: []
  };
  window.APP = APP;

  /* ---- genoma attivo ---- */
  function setGenome(meta, persist) {
    APP.genome = { id: meta.id || "genoma", version: meta.version || "0", words: (meta.words || []).slice() };
    APP.dict = G.loadDictionary(APP.genome.words, APP.genome);
    refreshChip();
    if (document.getElementById("view-gen").classList.contains("active")) renderGenome();
    if (persist !== false) native.storeSet("genome", APP.genome);
  }
  function refreshChip() {
    $("gc-name").textContent = APP.genome.id + " v" + APP.genome.version;
    $("gc-meta").textContent = fmt(APP.dict.size) + " geni · " + APP.dict.hash.slice(0, 8);
  }

  /* ============================================================
     LABORATORIO DEL GENOMA
     ============================================================ */
  function renderGenome() { renderOverview(); renderBrowser(); renderBuild(); }

  function renderOverview() {
    const d = APP.dict;
    const ranges = G.tiers.ranges;
    const tierRows = ranges.map((r, i) => {
      const present = Math.max(0, Math.min(d.size, r.to + 1) - r.from);
      const capForTier = r.to - r.from + 1;
      const w = Math.round(present / Math.max(1, capForTier) * 100);
      return `<div class="trow"><span class="tlab">${r.bases} basi · liv. ${i + 1}</span>
        <div class="ttrack"><div class="tfill" style="width:${present ? Math.max(3, w) : 0}%"></div></div>
        <span class="tval">${fmt(present)}</span></div>`;
    }).join("");
    const fillPct = Math.min(100, d.size / d.capacity * 100);

    $("gen-overview").innerHTML = `
      <div class="out-label"><span>Genoma di riferimento attivo</span>
        <span class="badge ${d.over ? "bad" : "ok"}">${d.over ? "oltre capacità!" : "impronta " + d.hash.slice(0, 10)}</span></div>
      <div class="gen-grid">
        <div class="stat-card"><div class="v">${fmt(d.size)}</div><div class="k">geni (parole)</div></div>
        <div class="stat-card"><div class="v">${fmt(d.capacity)}</div><div class="k">capacità max</div></div>
        <div class="stat-card"><div class="v">${fillPct < 0.1 ? fillPct.toFixed(2) : fillPct.toFixed(1)}%</div><div class="k">riempimento</div></div>
        <div class="stat-card"><div class="v">${G.RADIX}</div><div class="k">cifre (codoni)</div></div>
      </div>
      <div class="out-label" style="margin-top:16px">Distribuzione per costo dell'indirizzo</div>
      <div class="tier-bars">${tierRows}</div>
      <div class="row mt">
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <input class="fin" id="gen-id" value="${esc(APP.genome.id)}" style="width:150px" title="identità del genoma">
          <input class="fin" id="gen-ver" value="${esc(APP.genome.version)}" style="width:110px" title="versione">
          <button class="btn ghost" id="gen-rename">Aggiorna identità</button>
        </div>
      </div>
      <div class="row mt">
        <button class="btn" id="gen-export">Esporta genoma .json</button>
        <button class="btn ghost" id="gen-import">Importa genoma .json</button>
        <button class="btn ghost" id="gen-reset">Ripristina dimostrativo</button>
      </div>`;

    $("gen-rename").addEventListener("click", () => {
      setGenome({ id: cleanName($("gen-id").value) || "genoma", version: $("gen-ver").value.trim() || "0", words: APP.genome.words });
      toast("Identità del genoma aggiornata", "ok");
    });
    $("gen-export").addEventListener("click", () =>
      saveNative(JSON.stringify({ format: "sapere-dna-genome", id: APP.genome.id, version: APP.genome.version, words: APP.genome.words }, null, 0),
        cleanName(APP.genome.id) + "-v" + APP.genome.version, ".json", "Sto esportando il genoma «" + APP.genome.id + "» (" + fmt(d.size) + " geni)"));
    $("gen-import").addEventListener("click", importGenome);
    $("gen-reset").addEventListener("click", () => { setGenome({ id: DEMO.id, version: DEMO.version, words: DEMO.words }); toast("Genoma dimostrativo ripristinato", "ok"); });
  }

  async function importGenome() {
    const r = await native.openText(); if (!r || !r.ok) return;
    let data; try { data = JSON.parse(r.content); } catch (e) { toast("File non valido (atteso .json).", "bad"); return; }
    const words = Array.isArray(data) ? data : data.words;
    if (!Array.isArray(words) || !words.length) { toast("Genoma non valido: manca la lista di parole.", "bad"); return; }
    setGenome({ id: data.id || "importato", version: data.version || "0", words });
    toast("Genoma importato: " + fmt(APP.dict.size) + " geni", "ok");
  }

  function renderBrowser() {
    $("gen-browser").innerHTML = `
      <div class="out-label">Esplora i geni e i loro indirizzi (loci)</div>
      <input class="fin gene-search" id="gen-search" placeholder="cerca una parola…" autocomplete="off">
      <div class="gene-list">
        <div class="ghead"><span>locus</span><span>parola (gene)</span><span>indirizzo (codoni)</span><span>basi</span></div>
        <div class="gbody" id="gen-rows"></div>
      </div>`;
    const draw = (q) => {
      const words = APP.genome.words;
      const rows = [];
      const cap = 600;
      let shown = 0, matched = 0;
      for (let i = 0; i < words.length; i++) {
        if (q && words[i].toLowerCase().indexOf(q) < 0) continue;
        matched++;
        if (shown >= cap) continue;
        const addr = G.addressOf(i);
        rows.push(`<div class="grow"><span class="grank">#${i}</span><span class="gword">${esc(words[i])}</span><span class="gaddr">${addr}</span><span class="gtier">${addr.length}</span></div>`);
        shown++;
      }
      const el = $("gen-rows");
      if (!matched) { el.innerHTML = '<div class="gene-empty">Nessun gene trovato.</div>'; return; }
      el.innerHTML = rows.join("") + (matched > cap ? `<div class="gene-empty">…e altri ${fmt(matched - cap)} (affina la ricerca)</div>` : "");
    };
    draw("");
    $("gen-search").addEventListener("input", (e) => draw(e.target.value.trim().toLowerCase()));
  }

  function renderBuild() {
    $("gen-build").innerHTML = `
      <div class="out-label">Costruisci il genoma</div>
      <div class="note" style="margin-bottom:10px">Aggiungi parole a mano, oppure <b>ricava un genoma da un corpus</b>: incolla molto testo e l'app conta le frequenze e ordina le parole (le più comuni prendono gli indirizzi più corti) — proprio come faresti per il dizionario grande del progetto.</div>

      <div class="flabel" style="margin-top:6px"><span>Aggiungi parole <span style="text-transform:none;color:var(--parch-faint)">(separate da spazi o a-capo)</span></span></div>
      <textarea id="gen-addwords" placeholder="es. fotone entropia quark biologia…" style="min-height:70px"></textarea>
      <div class="row mt"><button class="btn ghost" id="gen-add">Aggiungi al genoma</button></div>

      <div class="flabel" style="margin-top:18px"><span>Costruisci da corpus <span class="flink" id="gen-load-corpus">apri file di testo…</span></span></div>
      <textarea id="gen-corpus" placeholder="incolla qui un testo grande (un libro, un insieme di documenti…)" style="min-height:110px"></textarea>
      <div class="row mt" style="align-items:center">
        <label style="font-family:var(--font-mono);font-size:.7rem;color:var(--parch-dim)">frequenza minima
          <input class="fin" id="gen-mincount" value="1" style="width:64px;display:inline-block;margin-left:6px"></label>
        <label style="font-family:var(--font-mono);font-size:.7rem;color:var(--parch-dim);display:flex;align-items:center;gap:6px">
          <input type="checkbox" id="gen-merge"> unisci al genoma attuale</label>
      </div>
      <div class="row mt">
        <button class="btn" id="gen-build-go">Costruisci genoma</button>
        <span id="gen-build-info" style="font-family:var(--font-mono);font-size:.68rem;color:var(--parch-dim)"></span>
      </div>`;

    $("gen-add").addEventListener("click", () => {
      const toks = $("gen-addwords").value.split(/\s+/).filter(x => x.length).map(t => /\p{L}/u.test(t[0]) ? t.toLowerCase() : t);
      if (!toks.length) return;
      const seen = new Set(APP.genome.words);
      const merged = APP.genome.words.slice();
      let added = 0;
      for (const t of toks) if (!seen.has(t)) { seen.add(t); merged.push(t); added++; }
      setGenome({ id: APP.genome.id, version: APP.genome.version, words: merged });
      $("gen-addwords").value = "";
      toast(added + " parole aggiunte", "ok");
    });

    $("gen-load-corpus").addEventListener("click", async () => { const r = await native.openText(); if (r && r.ok) $("gen-corpus").value = r.content; });

    $("gen-build-go").addEventListener("click", () => {
      const corpus = $("gen-corpus").value;
      if (corpus.trim().length < 20) { toast("Incolla un po' più di testo per stimare le frequenze.", "bad"); return; }
      const minCount = Math.max(1, parseInt($("gen-mincount").value, 10) || 1);
      const merge = $("gen-merge").checked;
      const words = G.buildGenome(corpus, { minCount, seed: merge ? APP.genome.words : null });
      const newVer = stamp();
      setGenome({ id: merge ? APP.genome.id : "corpus", version: newVer, words });
      $("gen-build-info").textContent = fmt(words.length) + " geni · v" + newVer;
      toast("Genoma costruito: " + fmt(words.length) + " geni", "ok");
    });
  }

  /* ============================================================
     ARCHIVIO
     ============================================================ */
  function addToArchive(entry) {
    entry.id = "e" + Date.now() + Math.floor(Math.random() * 1000);
    APP.archive.unshift(entry);
    native.storeSet("archive", APP.archive);
    if ($("view-arc").classList.contains("active")) renderArchive();
  }

  function renderArchive() {
    const list = APP.archive;
    let body;
    if (!list.length) {
      body = '<div class="gene-empty">L\'archivio è vuoto. Incidi una frase e premi «Aggiungi all\'archivio», oppure importa un file.</div>';
    } else {
      body = list.map((e, i) => {
        const m = e.mode === "genoma" ? "genoma" : e.mode === "mente" ? "mente" : "classica";
        return `<div class="entry"><div class="meta">
          <div class="et"><span class="mtag ${m}">${m}</span>${esc(e.title || "(senza titolo)")}</div>
          <div class="es">${fmt(e.chars || 0)} caratteri · ${e.codons ? fmt(e.codons) + " codoni · " : ""}${new Date(e.created).toLocaleString("it-IT")}</div>
        </div><div class="acts">
          <button class="mini" data-act="json" data-i="${i}">.json</button>
          <button class="mini" data-act="del" data-i="${i}">✕</button>
        </div></div>`;
      }).join("");
    }
    $("arc-panel").innerHTML = `
      <div class="row" style="margin-bottom:14px">
        <button class="btn ghost" id="arc-export">Esporta intero archivio</button>
        <button class="btn ghost" id="arc-import">Importa archivio</button>
      </div>
      <div id="arc-list">${body}</div>`;

    $("arc-export").addEventListener("click", () => {
      if (!APP.archive.length) { toast("Archivio vuoto.", "bad"); return; }
      saveNative(JSON.stringify({ version: 3, entries: APP.archive }, null, 2), "archivio-sapere-dna-" + stamp(), ".json", "Sto esportando l'intero archivio (" + APP.archive.length + " voci)");
    });
    $("arc-import").addEventListener("click", async () => {
      const r = await native.openText(); if (!r || !r.ok) return;
      try { const data = JSON.parse(r.content); const inc = Array.isArray(data) ? data : (data.entries || []); if (!inc.length) throw 0; APP.archive = inc.concat(APP.archive); native.storeSet("archive", APP.archive); renderArchive(); toast(inc.length + " voci importate", "ok"); }
      catch (e) { toast("File non valido: serve un archivio .json di questo strumento.", "bad"); }
    });
    $("arc-list").addEventListener("click", (ev) => {
      const b = ev.target.closest("button"); if (!b) return;
      const i = +b.dataset.i, e = APP.archive[i]; if (!e) return;
      if (b.dataset.act === "del") { APP.archive.splice(i, 1); native.storeSet("archive", APP.archive); renderArchive(); }
      else if (b.dataset.act === "json") {
        const name = cleanName(e.title || "voce") || slug(e.title || "voce");
        saveNative(JSON.stringify(e, null, 2), name + "-" + (e.mode || "voce"), ".json", "Sto esportando «" + (e.title || "").slice(0, 30) + "»");
      }
    });
  }

  /* ============================================================
     NAVIGAZIONE
     ============================================================ */
  const VIEWS = ["gen", "cod", "dec", "arc", "men", "tra", "gnr"];
  function showView(v) {
    VIEWS.forEach(k => $("view-" + k).classList.toggle("active", k === v));
    document.querySelectorAll(".nav-item").forEach(n => n.classList.toggle("active", n.dataset.view === v));
    if (v === "gen") renderGenome();
    if (v === "arc") renderArchive();
  }
  document.querySelector(".side").addEventListener("click", (e) => {
    const item = e.target.closest(".nav-item"); if (!item) return;
    showView(item.dataset.view);
  });

  /* ============================================================
     AVVIO
     ============================================================ */
  (async function init() {
    // genoma: dal disco se presente, altrimenti dimostrativo
    let stored = null;
    try { const g = await native.storeGet("genome"); stored = g && g.value; } catch (e) {}
    if (stored && Array.isArray(stored.words) && stored.words.length) setGenome(stored, false);
    else setGenome({ id: DEMO.id, version: DEMO.version, words: DEMO.words }, false);

    // archivio
    try { const a = await native.storeGet("archive"); if (a && Array.isArray(a.value)) APP.archive = a.value; } catch (e) {}

    // controller
    window.initCodifica(APP);
    window.initDecodifica(APP);
    window.initMente(APP);
    window.initTrasformatore(APP);
    window.initGeneratore(APP);

    showView("gen");
  })();
})();
