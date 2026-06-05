/* ============================================================
   Sapere-DNA Studio · CODIFICA (controller)
   Espone window.initCodifica(APP)
   ============================================================ */
(function () {
  "use strict";

  window.initCodifica = function (APP) {
    const G = APP.G, C = APP.C;
    const panel = document.getElementById("cod-panel");
    let mode = "genoma";
    let last = null; // ultimo risultato per i salvataggi

    panel.innerHTML = `
      <div class="seg" id="cod-seg">
        <button data-m="genoma" class="on">Genoma</button>
        <button data-m="classica">Classica</button>
      </div>
      <div class="flabel"><span class="accent" id="cod-modehint">Filamento unico, parole come geni</span></div>
      <textarea id="cod-src" placeholder="Scrivi o incolla qui la conoscenza da incidere. Anche testi lunghi.">In montagna l'acqua bolle a 90 gradi perché la pressione è più bassa.
Il sapere non vive nelle basi: vive nell'interprete che le legge.</textarea>
      <div class="flabel" style="margin-top:14px"><span>Nome base dei file <span style="text-transform:none;font-style:italic;color:var(--parch-faint)">(facoltativo)</span></span></div>
      <input type="text" id="cod-base" class="fin" placeholder="es. acqua-montagna" autocomplete="off">
      <div class="row mt">
        <button class="btn teal" id="cod-go">Incidi nel filo &#10038;</button>
        <button class="btn teal ghost" id="cod-clear">Pulisci</button>
      </div>
      <div class="progress" id="cod-prog"><div class="bar"><div class="fill" id="cod-fill"></div></div><div class="lab" id="cod-plab"></div></div>
      <div id="cod-out"></div>`;

    const $ = (id) => document.getElementById(id);
    const out = $("cod-out");

    $("cod-seg").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      mode = b.dataset.m;
      [...$("cod-seg").children].forEach(c => c.classList.toggle("on", c === b));
      $("cod-modehint").textContent = mode === "genoma"
        ? "Filamento unico, parole come geni"
        : "Doppio binario: DNA + indice gematrico (lettere ebraiche)";
      out.innerHTML = ""; last = null;
    });
    $("cod-clear").addEventListener("click", () => { $("cod-src").value = ""; out.innerHTML = ""; last = null; });
    $("cod-go").addEventListener("click", run);

    function suggestName(kind) {
      const base = $("cod-base").value.trim();
      if (base) return APP.cleanName(base) + (kind ? "-" + kind : "");
      return APP.slug($("cod-src").value || "voce") + (kind ? "-" + kind : "") + "-" + APP.stamp();
    }

    async function run() {
      const text = $("cod-src").value;
      if (!text.length) { out.innerHTML = '<p class="err">Scrivi o apri del testo da incidere.</p>'; return; }
      const go = $("cod-go"); go.disabled = true; out.innerHTML = "";
      const prog = $("cod-prog"), fill = $("cod-fill"), plab = $("cod-plab");
      prog.classList.add("on");
      const pcb = (f) => { fill.style.width = (f * 100).toFixed(1) + "%"; plab.textContent = "codifica… " + ((f * 100) | 0) + "%"; };
      const t0 = performance.now();

      if (mode === "genoma") {
        const e = await G.encode(text, APP.dict, pcb);
        const d = await G.decode(e, APP.dict);
        const cls = await G.classicSize(text);
        const t1 = performance.now();
        prog.classList.remove("on"); fill.style.width = "0";
        last = { mode: "genoma", entry: e, text };
        renderGenoma(e, d, cls, t1 - t0);
      } else {
        const e = await C.encode(text, pcb);
        const t1 = performance.now();
        prog.classList.remove("on"); fill.style.width = "0";
        last = { mode: "classica", res: e, text };
        renderClassica(e, t1 - t0);
      }
      go.disabled = false;
    }

    function renderGenoma(e, d, cls, ms) {
      const segs = G.trace(e.dna, APP.dict);
      const gBytes = e.dna.length, cBytes = cls.total, maxB = Math.max(gBytes, cBytes, 1);
      const pct = cBytes ? Math.round(gBytes / cBytes * 100) : 100;
      const strandShown = e.dna.length > 4500 ? e.dna.slice(0, 4500) : e.dna;
      const trunc = e.dna.length > 4500 ? ' <span style="color:var(--parch-dim)">…(vista troncata; calcolo completo)</span>' : "";

      out.innerHTML = `
        <div class="out-label"><span>La conoscenza, colorata per come è stata incisa</span>
          <span class="${d.verified === false ? "badge bad" : "badge ok"}">${d.verified === false ? "&#9888; non combacia" : "&#10003; riletta identica"}</span></div>
        <div class="box text">${APP.colorizeSegs(segs)}</div>
        <div class="legend">
          <span><span class="dot" style="background:var(--gold-bright)"></span><b>gene</b> (parola nel genoma)</span>
          <span><span class="dot" style="background:var(--violet-bright)"></span><b>sillabato</b> (fuori genoma)</span>
          <span><span class="dot" style="background:var(--red)"></span><b>via di fuga</b> (fuori tavola)</span>
        </div>
        <div class="out-label"><span>Filamento di DNA — uno solo, niente indice (${APP.fmt(e.dna.length)} basi)</span>
          <button class="mini" id="cod-save-strand">salva filamento</button></div>
        <div class="box mono strand">${APP.esc(strandShown)}${trunc}</div>
        <div class="statbar">
          <span>geni: <b>${APP.fmt(e.gene_hits)}</b></span>
          <span>sillabati: <b>${APP.fmt(e.lit_chars)}</b></span>
          ${e.raw_chars ? `<span>via di fuga: <b>${APP.fmt(e.raw_chars)}</b></span>` : ""}
          <span>codoni: <b>${APP.fmt(e.codons)}</b></span>
          <span>tempo: <b>${ms.toFixed(0)} ms</b></span>
        </div>
        <div class="out-label" style="margin-top:18px">Dimensione a confronto (meno è meglio)</div>
        <div class="cmp">
          <div class="crow"><span class="cname">Genoma</span><div class="track"><div class="fill g" style="width:${Math.max(8, Math.round(gBytes / maxB * 100))}%">${APP.fmt(gBytes)} byte</div></div></div>
          <div class="crow"><span class="cname">Classica</span><div class="track"><div class="fill c" style="width:${Math.max(8, Math.round(cBytes / maxB * 100))}%">${APP.fmt(cBytes)} byte</div></div></div>
        </div>
        <div class="verdict">Il Genoma occupa il <b>${pct}%</b> della Classica${pct < 100 ? " — risparmio del " + (100 - pct) + "%." : "."}</div>
        <div class="row mt">
          <button class="btn teal" id="cod-arc">&#9733; Aggiungi all'archivio</button>
          <button class="btn teal ghost" id="cod-save-json">Salva voce .json</button>
        </div>`;

      $("cod-save-strand").addEventListener("click", () =>
        APP.saveNative(e.dna, suggestName("filamento"), ".dna.txt", "Sto preparando il filamento di DNA (" + APP.fmt(e.dna.length) + " basi)"));
      $("cod-save-json").addEventListener("click", () =>
        APP.saveNative(JSON.stringify(buildGenomaEntry(e), null, 2), suggestName(""), ".json", "Sto componendo la voce d'archivio Genoma (filamento + impronte di integrità)"));
      $("cod-arc").addEventListener("click", () => { APP.addToArchive(buildGenomaEntry(e)); APP.toast("Aggiunta all'archivio", "ok"); });
    }

    function buildGenomaEntry(e) {
      return Object.assign({
        title: (last.text.trim().slice(0, 60) || "(senza titolo)"),
        phrase: last.text, created: new Date().toISOString()
      }, e);
    }

    function renderClassica(e, ms) {
      const hp = C.hebPreview(e.dna);
      const hebNote = hp.truncated ? `<span style="color:var(--parch-dim);font-style:italic;text-transform:none">(disegnate ${APP.fmt(hp.shown)} di ${APP.fmt(hp.total)} lettere; DNA e indice sotto sono completi)</span>` : "";
      const warn = e.unmapped.length ? `<p class="err">${e.unmapped.length} caratteri fuori tavola, ignorati: ${APP.esc(e.unmapped.slice(0, 30).map(c => c === "\n" ? "⏎" : c === " " ? "␣" : c === "\t" ? "⇥" : c).join(" "))}. (In modalità Genoma sarebbero salvati con la via di fuga.)</p>` : "";

      out.innerHTML = `
        <div class="out-label">Lettere ebraiche ${hebNote}</div>
        <div class="box heb">${hp.txt}</div>
        <div class="out-label"><span>Sequenza di DNA — completa (${APP.fmt(e.bases)} basi)</span>
          <button class="mini" id="cl-save-dna">salva DNA</button></div>
        <div class="box mono dna">${APP.esc(e.dna)}</div>
        <div class="out-label"><span>Indice di recupero — completo (${APP.fmt(e.count)} numeri)</span>
          <button class="mini" id="cl-save-idx">salva indice</button></div>
        <div class="box mono idx">${APP.esc(e.idx)}</div>
        <div class="statbar">
          <span>caratteri: <b>${APP.fmt(e.count)}</b></span>
          <span>basi: <b>${APP.fmt(e.bases)}</b></span>
          <span>codoni: <b>${APP.fmt(e.bases / 3)}</b></span>
          <span>tempo: <b>${ms.toFixed(0)} ms</b></span>
        </div>
        <div class="row mt">
          <button class="btn teal" id="cl-arc">&#9733; Aggiungi all'archivio</button>
          <button class="btn teal ghost" id="cl-save-json">Salva voce .json</button>
        </div>${warn}`;

      $("cl-save-dna").addEventListener("click", () =>
        APP.saveNative(e.dna, suggestName("dna"), ".txt", "Sto preparando la sequenza di DNA (" + APP.fmt(e.bases) + " basi)"));
      $("cl-save-idx").addEventListener("click", () =>
        APP.saveNative(e.idx, suggestName("indice"), ".txt", "Sto scrivendo l'indice di recupero (" + APP.fmt(e.count) + " numeri)"));
      $("cl-save-json").addEventListener("click", () =>
        APP.saveNative(JSON.stringify(buildClassicaEntry(e), null, 2), suggestName(""), ".json", "Sto componendo la voce d'archivio Classica"));
      $("cl-arc").addEventListener("click", () => { APP.addToArchive(buildClassicaEntry(e)); APP.toast("Aggiunta all'archivio", "ok"); });
    }

    function buildClassicaEntry(e) {
      return {
        mode: "classica", title: (last.text.trim().slice(0, 60) || "(senza titolo)"),
        phrase: last.text, dna: e.dna, idx: e.idx, chars: e.count, bases: e.bases,
        codoni: (e.dna.length / 3) | 0,
        numeri_indice: e.idx.trim().split(/\s+/).filter(x => x).length,
        created: new Date().toISOString()
      };
    }
  };
})();
