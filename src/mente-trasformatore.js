/* ============================================================
   Sapere-DNA Studio · MENTE — TRANSFORMER (controller)
   Pilota dal pannello il modello di punta per l'interpretabilità:
   addestramento · SAE · confronto con la verità di base ·
   tracciato (filamento) · intervento causale · sigillo.
   Espone window.initTrasformatore(APP)
   ============================================================ */
(function () {
  "use strict";

  const TASKS = {
    lettura:  { label: "Lettura",            cls: 4,  desc: "Distingue <b>quali</b> concetti ci sono (c'è acqua? c'è fuoco?). Costringe il modello a tenere i concetti separati: è qui che il recupero contro la verità di base è più pulito." },
    xor:      { label: "XOR · parità",       cls: 2,  desc: "Vero se è presente <b>esattamente uno</b> tra acqua e fuoco. Compito non-lineare e «duro»: il modello tende a <i>intrecciare</i> i concetti — e il sistema lo dichiara invece di fingere chiarezza." },
    anelli:   { label: "Anelli · conteggio", cls: 3,  desc: "In quale fascia cade il <b>numero</b> di concetti presenti. Il modello comprime l'identità in una quantità: ottimo contrasto." },
    completo: { label: "Completo · 16",       cls: 16, desc: "Legge <b>tutte e quattro</b> le feature come bit: sovrapposizione spinta, concetti più difficili da districare." }
  };
  const FEAT = ["acqua", "fuoco", "terra", "aria"];
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  window.initTrasformatore = function (APP) {
    const G = APP.G, T = APP.T, C = APP.C;
    const panel = document.getElementById("tra-panel");
    if (!T) { panel.innerHTML = '<p class="err">Modulo trasformatore non caricato (manca trasformatore.js).</p>'; return; }

    let model = null, corpus = null, sae = null, genome = null;
    let curSeq = null, lastTrace = null, lastInterv = null, lastSeal = null;
    let curTask = "lettura";

    panel.innerHTML = `
      <div class="note" style="margin-bottom:12px">Un <b>Transformer</b> vero ma minuscolo — embedding, una testa di auto-attenzione, MLP, testa di classificazione — reso trasparente con i tre strumenti di Sapere-DNA. Legge un corpus di cui <b>conosciamo le regole</b>: così possiamo <b>verificare</b> se l'interpretazione è giusta, non solo raccontarla. È la prova che il metodo funziona, non un aneddoto.</div>

      <div class="out-label">Compito da imparare</div>
      <div class="row" id="tra-tasks" style="gap:8px;flex-wrap:wrap"></div>
      <div class="callout-mini" id="tra-task-desc" style="margin-top:8px"></div>

      <div class="row mt" style="align-items:flex-end;gap:16px">
        <label class="men-f">passi<input class="fin" id="tra-steps" value="5000" style="width:90px"></label>
        <label class="men-f">restart<input class="fin" id="tra-restart" value="2" style="width:60px"></label>
        <label class="men-f">dati<input class="fin" id="tra-data" value="1500" style="width:80px"></label>
        <label class="men-f">larghezza MLP<input class="fin" id="tra-h" value="32" style="width:70px"></label>
      </div>
      <div class="row mt">
        <button class="btn amber" id="tra-train">Addestra il Transformer</button>
        <span id="tra-status" style="font-family:var(--font-mono);font-size:.7rem;color:var(--parch-dim)"></span>
      </div>

      <div id="tra-body" style="display:none">

        <div class="out-label" style="margin-top:20px"><span>Confronto con la verità di base</span> <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">i concetti estratti combaciano con quelli che <i>noi</i> abbiamo piantato?</span></div>
        <div class="men-clean-grid">
          <div class="seal-box">
            <div style="color:var(--amber-bright);margin-bottom:6px">Concetti (categorie) recuperati</div>
            <div id="tra-gt-classes"></div>
          </div>
          <div class="seal-box">
            <div style="color:var(--violet-bright);margin-bottom:6px">Singole feature (in sovrapposizione)</div>
            <div id="tra-gt-feats"></div>
          </div>
        </div>
        <div class="callout-mini" id="tra-gt-note" style="margin-top:10px"></div>

        <div class="out-label" style="margin-top:20px">Genoma dei concetti</div>
        <div class="men-hash" id="tra-genhash" style="margin-bottom:8px"></div>
        <div class="men-list">
          <div class="men-row men-head"><span>loc.</span><span>concetto</span><span>combacia con (F1)</span><span>freq</span><span>ind.</span></div>
          <div class="men-rows" id="tra-genrows"></div>
        </div>

        <div class="out-label" style="margin-top:22px"><span>Fai una domanda al Transformer</span> <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">scegli quali concetti mettere nella sequenza, poi leggi <i>come</i> ragiona</span></div>
        <div class="row" id="tra-toggles" style="gap:14px;flex-wrap:wrap;align-items:center"></div>
        <div class="row mt">
          <button class="btn amber" id="tra-ask">Componi e interroga</button>
          <button class="btn amber ghost" id="tra-rand">Pesca a caso</button>
        </div>

        <div id="tra-trace-wrap" style="display:none">
          <div class="out-label" style="margin-top:18px"><span id="tra-seqline"></span><span id="tra-verdict"></span></div>
          <div class="box mono strand" id="tra-filament" style="color:var(--amber-bright)"></div>

          <div class="out-label">Concetti accesi <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">forte → debole</span></div>
          <div id="tra-fired" style="margin-bottom:6px"></div>

          <div class="out-label" style="margin-top:14px">Intervento — uno alla volta <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">spengo un concetto e guardo se la risposta cambia (prova di causa)</span></div>
          <div id="tra-interv"></div>
          <div class="out-label" style="margin-top:10px">Intervento — a gruppi <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">stesso concetto in più posti, spenti insieme</span></div>
          <div id="tra-groups"></div>
          <div class="men-fid-set">
            <div class="men-fid"><div class="men-fid-lab">fedeltà singola</div><div class="men-fid-track"><div class="men-fid-fill" id="tra-fid1"></div></div><div class="men-fid-val" id="tra-fidv1"></div></div>
            <div class="men-fid"><div class="men-fid-lab">fedeltà di gruppo</div><div class="men-fid-track"><div class="men-fid-fill grp" id="tra-fid2"></div></div><div class="men-fid-val" id="tra-fidv2"></div></div>
            <div class="men-fid"><div class="men-fid-lab">fedeltà graduata</div><div class="men-fid-track"><div class="men-fid-fill grad" id="tra-fid3"></div></div><div class="men-fid-val" id="tra-fidv3"></div></div>
          </div>

          <div class="out-label" style="margin-top:14px">Sigillo di fedeltà</div>
          <div class="seal-box" id="tra-seal"></div>

          <div class="out-label" style="margin-top:14px"><span>Ponte con Sapere-DNA</span> <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">lo stesso testo, inciso col metodo classico</span></div>
          <div class="seal-box" id="tra-bridge"></div>

          <div class="row mt">
            <button class="btn amber" id="tra-arc">&#9733; Aggiungi all'archivio</button>
            <button class="btn amber ghost" id="tra-json">Salva tracciato .json</button>
          </div>
        </div>
      </div>`;

    const $ = (id) => document.getElementById(id);

    /* ---- selettore del compito ---- */
    function renderTasks() {
      $("tra-tasks").innerHTML = Object.keys(TASKS).map(k =>
        `<button class="btn amber ${k === curTask ? "" : "ghost"}" data-task="${k}">${TASKS[k].label}</button>`
      ).join("");
      $("tra-task-desc").innerHTML = TASKS[curTask].desc;
    }
    $("tra-tasks").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-task]"); if (!b) return;
      curTask = b.dataset.task; renderTasks();
    });
    renderTasks();

    /* ---- interruttori delle feature ---- */
    $("tra-toggles").innerHTML = FEAT.map(f =>
      `<label class="men-f" style="flex-direction:row;gap:6px;align-items:center;text-transform:none">
        <input type="checkbox" id="tra-f-${f}"> ${f}</label>`
    ).join("");

    /* ---- verità di base ---- */
    function bar(f1, color) {
      const w = Math.max(2, f1 * 100).toFixed(0);
      return `<div class="track" style="display:inline-block;width:120px"><div class="fill" style="width:${w}%;background:${color}"></div></div>`;
    }
    function renderGround() {
      const gt = genome.gt;
      $("tra-gt-classes").innerHTML = gt.recoveredClasses.map(r =>
        `<div style="display:flex;align-items:center;gap:8px;margin:3px 0">
          <span style="min-width:120px">${APP.esc(r.klass)}</span>
          ${bar(r.f1, "var(--amber-bright)")}
          <span style="font-family:var(--font-mono);font-size:.66rem;color:var(--parch-dim)">${r.f1.toFixed(2)} ${r.recovered ? "&#10003;" : ""}</span>
        </div>`).join("");
      $("tra-gt-feats").innerHTML = gt.recovered.map(r =>
        `<div style="display:flex;align-items:center;gap:8px;margin:3px 0">
          <span style="min-width:120px">${APP.esc(r.feature)}</span>
          ${bar(r.f1, "var(--violet-bright)")}
          <span style="font-family:var(--font-mono);font-size:.66rem;color:var(--parch-dim)">${r.f1.toFixed(2)} ${r.recovered ? "&#10003;" : ""}</span>
        </div>`).join("");
      const cls = (gt.classRecovery * 100).toFixed(0), mono = (gt.monosemanticity * 100).toFixed(0);
      $("tra-gt-note").innerHTML = `&#10003; Verifica <b>misurata</b>, non raccontata: al punto di decisione il SAE recupera <b style="color:var(--amber-bright)">${cls}%</b> delle categorie note, e i concetti puliti sono <b style="color:var(--amber-bright)">${mono}%</b>. Le singole feature appaiono più deboli perché vivono <i>in sovrapposizione</i> (composte dentro le categorie) — ed è esattamente il fenomeno che l'interpretabilità studia. La verità di base esiste perché il corpus è sintetico: è il laboratorio dove l'idea si può <i>provare</i>.`;
    }

    function renderGenome() {
      $("tra-genhash").textContent = genome.dict.id + " v" + genome.dict.version + " · impronta " + genome.dict.hash.slice(0, 10);
      $("tra-genrows").innerHTML = genome.genes.map((g, rank) => {
        if (g.freq === 0) return "";
        const col = g.monosemantic ? "var(--amber-bright)" : "var(--parch-faint)";
        return `<div class="men-row"><span>#${rank}</span><span class="nm" style="color:${col}">${APP.esc(g.name)}</span><span style="font-family:var(--font-mono);font-size:.66rem">${g.topF1.toFixed(2)}</span><span class="fr">${APP.fmt(g.freq)}</span><span class="a">${G.addressOf(rank)}</span></div>`;
      }).join("");
    }

    /* ---- tracciato + intervento + sigillo ---- */
    function renderTrace() {
      const trueName = corpus.labelNames[curSeq.label], predName = corpus.labelNames[lastTrace.pred];
      $("tra-seqline").textContent = `«${T.seqText(curSeq)}» · vera: ${trueName} · predetta: ${predName}`;
      $("tra-verdict").innerHTML = lastTrace.pred === curSeq.label
        ? ' <span class="badge ok">&#10003; risposta corretta</span>'
        : ' <span class="badge warn">&#9888; diversa dal vero</span>';

      const SP = G.CTRL.SP;
      if (!lastTrace.fired.length) $("tra-filament").innerHTML = '<span style="color:var(--parch-faint)">(nessun concetto acceso per questa sequenza)</span>';
      else $("tra-filament").innerHTML = lastTrace.fired.map((f, k) => {
        const addr = G.addressOf(genome.dict.index["c" + f.m + "_" + f.name]);
        return '<span style="color:var(--amber-bright);font-weight:700">' + addr + '</span>' +
          (k < lastTrace.fired.length - 1 ? '<span style="color:var(--parch-faint)">' + SP + '</span>' : "");
      }).join("");

      const maxAct = Math.max(1e-6, ...lastTrace.fired.map(f => f.act));
      $("tra-fired").innerHTML = lastTrace.fired.map(f =>
        `<div class="men-fired-row"><span class="nm">${APP.esc(f.name)} <span style="color:var(--parch-faint)">c${f.m}</span></span><div class="track"><div class="fill" style="width:${(f.act / maxAct * 100).toFixed(0)}%"></div></div><span class="v">${f.act.toFixed(2)}</span></div>`
      ).join("") || '<span style="color:var(--parch-faint);font-family:var(--font-mono);font-size:.7rem">nessuno</span>';

      $("tra-interv").innerHTML = lastInterv.rows.map(r =>
        `<div class="men-iv-row ${r.causal ? "causal" : ""}">spengo <b style="color:var(--parch)">${APP.esc(r.name)}</b> &rarr; <b style="color:${r.causal ? "var(--amber-bright)" : "var(--parch-faint)"}">${corpus.labelNames[r.newPred]}</b> ${r.causal ? '<span class="flag">&#9888; cambia</span>' : '<span style="color:var(--parch-faint)">= invariata</span>'}<span class="men-drop">&minus;${(r.dropPct * 100).toFixed(0)}% sicurezza</span></div>`
      ).join("") || '<span style="color:var(--parch-faint);font-family:var(--font-mono);font-size:.7rem">—</span>';

      $("tra-groups").innerHTML = lastInterv.groups.map(g =>
        `<div class="men-iv-row ${g.causal ? "causal" : ""}">spengo il gruppo <b style="color:var(--parch)">${APP.esc(g.name)}</b> <span style="color:var(--parch-faint)">(${g.neurons.length})</span> &rarr; <b style="color:${g.causal ? "var(--amber-bright)" : "var(--parch-faint)"}">${corpus.labelNames[g.newPred]}</b> ${g.causal ? '<span class="flag">&#9888; causale</span>' : '<span style="color:var(--parch-faint)">= invariata</span>'}<span class="men-drop">&minus;${(g.dropPct * 100).toFixed(0)}%</span></div>`
      ).join("") || '<span style="color:var(--parch-faint);font-family:var(--font-mono);font-size:.7rem">—</span>';

      const setFid = (f, v, val) => { $(f).style.width = (val * 100).toFixed(0) + "%"; $(v).textContent = (val * 100).toFixed(0) + "%"; };
      setFid("tra-fid1", "tra-fidv1", lastInterv.fidelitySingle);
      setFid("tra-fid2", "tra-fidv2", lastInterv.fidelityGroup);
      setFid("tra-fid3", "tra-fidv3", lastInterv.fidelityGraded);

      const segs = G.trace(lastTrace.filament, genome.dict).filter(s => s.role === "gene").map(s => s.text);
      const orig = lastTrace.fired.map(f => "c" + f.m + "_" + f.name);
      const verified = JSON.stringify(segs) === JSON.stringify(orig);
      $("tra-seal").innerHTML = `impronta genoma &nbsp;&nbsp;&nbsp;<b>${lastSeal.genomeHash}</b><br>impronta tracciato <b>${lastSeal.traceHash}</b><br>rilettura col genoma &rarr; <b>${verified ? "identica &#10003;" : "NON combacia &#9888;"}</b>`;

      $("tra-trace-wrap").style.display = "block";
      updateBridge();
    }

    async function updateBridge() {
      $("tra-bridge").innerHTML = '<span style="color:var(--parch-faint)">incido nel filo…</span>';
      try {
        const enc = await C.encode(T.seqText(curSeq));
        const shown = enc.dna.length > 90 ? enc.dna.slice(0, 90) + "…" : enc.dna;
        $("tra-bridge").innerHTML = `testo &nbsp;<b style="color:var(--parch)">«${APP.esc(T.seqText(curSeq))}»</b><br>` +
          `DNA classico &nbsp;<span style="color:var(--teal-bright);word-break:break-all">${shown}</span><br>` +
          `<span style="color:var(--parch-faint)">${APP.fmt(enc.count)} codoni · lo stesso corpus che il Transformer legge, scritto come filamento.</span>`;
      } catch (e) { $("tra-bridge").innerHTML = '<span style="color:var(--parch-faint)">—</span>'; }
    }

    function deriveAndRender() {
      lastTrace = T.derive(model, curSeq, genome, G);
      lastInterv = T.intervene(model, curSeq, genome, lastTrace);
      lastSeal = T.seal(genome, lastTrace.filament, G);
      renderTrace();
    }

    function buildEntry() {
      return {
        mode: "mente",
        title: "transformer " + curTask + " · " + corpus.labelNames[lastTrace.pred],
        task: curTask, sequence: T.seqText(curSeq),
        present: curSeq.present, true_label: corpus.labelNames[curSeq.label], pred: corpus.labelNames[lastTrace.pred],
        dict_id: genome.dict.id, dict_version: genome.dict.version, dict_hash: genome.dict.hash,
        filament: lastTrace.filament, trace_hash: lastSeal.traceHash,
        fired: lastTrace.fired.map(f => ({ concept: f.m, name: f.name, act: +f.act.toFixed(4) })),
        ground_truth: { classRecovery: +genome.gt.classRecovery.toFixed(3), monosemanticity: +genome.gt.monosemanticity.toFixed(3) },
        fidelity: { single: +lastInterv.fidelitySingle.toFixed(3), group: +lastInterv.fidelityGroup.toFixed(3), graded: +lastInterv.fidelityGraded.toFixed(3) },
        chars: lastTrace.fired.length, codons: (lastTrace.filament.length / 3) | 0,
        created: new Date().toISOString()
      };
    }

    /* ---- eventi ---- */
    $("tra-train").addEventListener("click", async () => {
      const steps = Math.max(800, parseInt($("tra-steps").value, 10) || 5000);
      const restarts = Math.max(1, parseInt($("tra-restart").value, 10) || 2);
      const N = Math.max(400, parseInt($("tra-data").value, 10) || 1500);
      const h = Math.max(8, parseInt($("tra-h").value, 10) || 32);
      APP.showLoader("Addestramento del Transformer", TASKS[curTask].label + " · " + APP.fmt(steps) + " passi × " + restarts + " restart");
      await sleep(50);
      corpus = T.makeCorpus({ N, L: 8, task: curTask });
      const fitRes = T.fit(corpus, { d: 24, h }, { steps, lr: 0.01, restarts });
      model = fitRes.model;
      APP.showLoader("Estrazione dei concetti", "dizionario sparso (SAE) + confronto con la verità");
      await sleep(50);
      const A = T.representationMatrix(model, corpus);
      sae = T.trainSAE(A, { M: 16, steps: Math.max(4000, steps + 2000) });
      genome = T.extractGenome(model, sae, corpus, G);
      curSeq = null; lastTrace = null;
      renderGround(); renderGenome();
      $("tra-status").textContent = "accuratezza " + (fitRes.accuracy * 100).toFixed(1) + "% · " +
        genome.gt.activeConcepts + " concetti attivi · monosem. " + (genome.gt.monosemanticity * 100).toFixed(0) + "%";
      $("tra-body").style.display = "block";
      $("tra-trace-wrap").style.display = "none";
      FEAT.forEach(f => { const el = $("tra-f-" + f); if (el) el.checked = false; });
      APP.hideLoader();
    });

    $("tra-ask").addEventListener("click", () => {
      if (!model || !genome) return;
      const present = FEAT.map(f => $("tra-f-" + f).checked ? 1 : 0);
      const matches = corpus.seqs.filter(s => s.present.every((v, i) => v === present[i]));
      curSeq = matches.length ? matches[(Math.random() * matches.length) | 0] : corpus.seqs[(Math.random() * corpus.seqs.length) | 0];
      if (!matches.length) APP.toast("Nessuna sequenza con esattamente quei concetti: ne pesco una a caso.", "warn");
      deriveAndRender();
    });

    $("tra-rand").addEventListener("click", () => {
      if (!model || !genome) return;
      curSeq = corpus.seqs[(Math.random() * corpus.seqs.length) | 0];
      curSeq.present.forEach((v, i) => { const el = $("tra-f-" + FEAT[i]); if (el) el.checked = !!v; });
      deriveAndRender();
    });

    $("tra-arc").addEventListener("click", () => { if (!lastTrace) return; APP.addToArchive(buildEntry()); APP.toast("Tracciato aggiunto all'archivio", "ok"); });
    $("tra-json").addEventListener("click", () => { if (!lastTrace) return; APP.saveNative(JSON.stringify(buildEntry(), null, 2), APP.slug("transformer-" + curTask + "-" + corpus.labelNames[lastTrace.pred]) + "-" + APP.stamp(), ".json", "Sto salvando il tracciato del Transformer (" + lastTrace.fired.length + " concetti)"); });
  };
})();
