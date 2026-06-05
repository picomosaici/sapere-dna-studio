/* ============================================================
   Sapere-DNA Studio · MENTE — GENERATORE (controller)
   Pilota il modello generativo reale: corpus + vocabolario-genoma,
   addestramento, salva/carica modello su disco, estrazione concetti
   (SAE), generazione, e — su ogni parola generata — filamento,
   intervento causale e sigillo. Fuori dal laboratorio: niente
   verità di base, ma prova di causa + sigillo.
   Espone window.initGeneratore(APP)
   ============================================================ */
(function () {
  "use strict";
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  window.initGeneratore = function (APP) {
    const G = APP.G, Gen = APP.Gen, C = APP.C;
    const panel = document.getElementById("gG-panel");
    if (!Gen) { panel.innerHTML = '<p class="err">Modulo generatore non caricato (manca generatore.js).</p>'; return; }

    let model = null, vocab = null, ids = null, sae = null, genome = null;
    let lastGen = null, lastTrace = null, lastInterv = null, lastSeal = null, curK = -1;

    panel.innerHTML = `
      <div class="note" style="margin-bottom:12px">Un piccolo <b>Transformer causale a parole</b> il cui vocabolario <b>è il genoma</b>: impara da un corpus vero a prevedere la parola seguente e genera testo. Qui la <b>verità di base non c'è</b> — non sappiamo quali siano i concetti «giusti». Restano le due prove di Sapere-DNA che valgono nel mondo reale: l'<b>intervento causale</b> (spengo un concetto e vedo se la parola cambia) e il <b>sigillo</b> (il tracciato è legato al calcolo, non è una storia inventata). Il modello non sarà eloquente: <i>non è la fluenza la dimostrazione, è che possiamo aprirlo e provarlo.</i></div>

      <div class="out-label">1 · Vocabolario (genoma attivo) e corpus</div>
      <div class="men-hash" id="gG-vocab-info"></div>
      <textarea id="gG-corpus" class="fin" style="width:100%;min-height:90px;margin-top:8px" placeholder="Incolla qui il testo del corpus, oppure caricalo da file…"></textarea>
      <div class="row mt">
        <button class="btn green ghost" id="gG-load-corpus">Carica testo…</button>
        <button class="btn green" id="gG-prep">Prepara vocabolario e corpus</button>
        <span id="gG-prep-status" style="font-family:var(--font-mono);font-size:.7rem;color:var(--parch-dim)"></span>
      </div>

      <div id="gG-after-prep" style="display:none">
        <div class="out-label" style="margin-top:18px">2 · Modello</div>
        <div class="row" style="align-items:flex-end;gap:16px">
          <label class="men-f">tetto vocab.<input class="fin" id="gG-cap" value="1500" style="width:80px"></label>
          <label class="men-f">contesto<input class="fin" id="gG-lc" value="16" style="width:60px"></label>
          <label class="men-f">dim. modello<input class="fin" id="gG-d" value="32" style="width:60px"></label>
          <label class="men-f">larghezza MLP<input class="fin" id="gG-h" value="64" style="width:70px"></label>
          <label class="men-f">passi<input class="fin" id="gG-steps" value="3000" style="width:80px"></label>
        </div>
        <div class="row mt">
          <button class="btn green" id="gG-train">Addestra</button>
          <button class="btn green ghost" id="gG-save">Salva modello…</button>
          <button class="btn green ghost" id="gG-load">Carica modello…</button>
          <span id="gG-status" style="font-family:var(--font-mono);font-size:.7rem;color:var(--parch-dim)"></span>
        </div>

        <div id="gG-after-train" style="display:none">
          <div class="out-label" style="margin-top:18px">3 · Concetti <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">dal vettore che decide la parola seguente</span></div>
          <div class="row"><button class="btn green ghost" id="gG-sae">Estrai concetti (SAE)</button><span id="gG-sae-status" style="font-family:var(--font-mono);font-size:.66rem;color:var(--parch-dim)"></span></div>
          <div id="gG-sae-out" style="display:none">
            <div class="callout-mini" id="gG-honesty" style="margin:8px 0"></div>
            <div class="men-hash" id="gG-genhash" style="margin-bottom:8px"></div>
            <div class="men-list">
              <div class="men-row men-head"><span>loc.</span><span>concetto (favorisce →)</span><span>freq</span><span>ind.</span></div>
              <div class="men-rows" id="gG-genrows"></div>
            </div>
          </div>

          <div class="out-label" style="margin-top:20px">4 · Genera <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">poi clicca una parola generata per leggerne la decisione</span></div>
          <textarea id="gG-prime" class="fin" style="width:100%;min-height:48px" placeholder="Avvio, es: «La simmetria dell'universo»"></textarea>
          <div class="row mt" style="align-items:flex-end;gap:16px">
            <label class="men-f">temperatura<input class="fin" id="gG-temp" value="0.8" style="width:60px"></label>
            <label class="men-f">parole<input class="fin" id="gG-len" value="30" style="width:60px"></label>
            <button class="btn green" id="gG-gen">Genera</button>
          </div>
          <div class="box" id="gG-out" style="margin-top:10px;line-height:1.9;display:none"></div>

          <div id="gG-trace-wrap" style="display:none">
            <div class="out-label" style="margin-top:16px"><span id="gG-seqline"></span></div>
            <div class="box mono strand" id="gG-filament" style="color:var(--green-bright)"></div>
            <div class="out-label">Concetti accesi <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">forte → debole</span></div>
            <div id="gG-fired" style="margin-bottom:6px"></div>
            <div class="out-label" style="margin-top:14px">Intervento causale <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">spengo un concetto e guardo se la parola cambia</span></div>
            <div id="gG-interv"></div>
            <div class="men-fid-set">
              <div class="men-fid"><div class="men-fid-lab">fedeltà singola</div><div class="men-fid-track"><div class="men-fid-fill" id="gG-fid1"></div></div><div class="men-fid-val" id="gG-fidv1"></div></div>
              <div class="men-fid"><div class="men-fid-lab">fedeltà graduata</div><div class="men-fid-track"><div class="men-fid-fill grad" id="gG-fid3"></div></div><div class="men-fid-val" id="gG-fidv3"></div></div>
            </div>
            <div class="out-label" style="margin-top:14px">Sigillo</div>
            <div class="seal-box" id="gG-seal"></div>
            <div class="out-label" style="margin-top:14px">Ponte con Sapere-DNA <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">il contesto, inciso col metodo classico</span></div>
            <div class="seal-box" id="gG-bridge"></div>
            <div class="row mt">
              <button class="btn green" id="gG-arc">&#9733; Aggiungi all'archivio</button>
              <button class="btn green ghost" id="gG-json">Salva tracciato .json</button>
            </div>
          </div>
        </div>
      </div>`;

    const $ = (id) => document.getElementById(id);

    function vocabInfo() {
      const d = APP.dict;
      $("gG-vocab-info").innerHTML = d
        ? `genoma attivo: <b>${APP.esc(d.id)}</b> v${APP.esc(d.version)} · ${APP.fmt(d.size)} geni · impronta ${d.hash.slice(0, 8)} <span style="color:var(--parch-faint)">— il vocabolario userà i primi «tetto» geni</span>`
        : '<span style="color:var(--gold-bright)">&#9888; Nessun genoma attivo: caricane o costruiscine uno nella sezione Genoma, poi torna qui.</span>';
    }
    vocabInfo();

    /* ============================================================
       Web Worker — sposta i lavori pesanti (addestramento, SAE)
       fuori dal thread dell'interfaccia, così la finestra resta viva.
       Se il Worker non parte (es. restrizioni file://), si RICADE
       in automatico sul percorso sincrono: l'app non si rompe mai.
       ============================================================ */
    let worker = null, workerBroken = false;
    function getWorker() {
      if (workerBroken) return null;
      if (worker) return worker;
      try { worker = new Worker("generatore-worker.js"); }
      catch (e) { workerBroken = true; worker = null; }
      return worker;
    }
    function runInWorker(message, onProgress) {
      return new Promise((resolve, reject) => {
        const w = getWorker();
        if (!w) { reject(new Error("worker non disponibile")); return; }
        const cleanup = () => { w.removeEventListener("message", onMsg); w.removeEventListener("error", onErr); };
        const onMsg = (e) => {
          const d = e.data || {};
          if (d.type === "progress") { if (onProgress) onProgress(d); }
          else if (d.type === "done") { cleanup(); resolve(d); }
          else if (d.type === "error") { cleanup(); reject(new Error(d.message || "errore nel worker")); }
        };
        const onErr = (ev) => {
          cleanup(); workerBroken = true;
          try { w.terminate(); } catch (_) {}
          worker = null;
          reject(new Error("worker non avviato" + (ev && ev.message ? ": " + ev.message : "")));
        };
        w.addEventListener("message", onMsg);
        w.addEventListener("error", onErr);
        w.postMessage(message);
      });
    }
    const loaderSub = (text) => { const s = document.getElementById("loader-sub"); if (s) s.textContent = text; };

    /* ---- 1 · corpus ---- */
    $("gG-load-corpus").addEventListener("click", async () => {
      const r = await APP.openText(); if (r && r.ok) { $("gG-corpus").value = r.content; APP.toast("Testo caricato", "ok"); }
    });

    $("gG-prep").addEventListener("click", () => {
      const d = APP.dict;
      if (!d) { vocabInfo(); APP.toast("Manca il genoma attivo", "warn"); return; }
      const text = $("gG-corpus").value.trim();
      if (!text) { APP.toast("Incolla o carica il testo del corpus", "warn"); return; }
      const cap = Math.max(50, parseInt($("gG-cap").value, 10) || 1500);
      vocab = Gen.makeVocab(d, { cap });
      ids = Gen.encode(text, d, vocab, G);
      const unk = ids.filter(i => i === vocab.unk).length;
      $("gG-prep-status").textContent = "vocabolario " + vocab.V + " · corpus " + APP.fmt(ids.length) + " token · ignoti " + (100 * unk / Math.max(1, ids.length)).toFixed(1) + "%";
      model = null; sae = null; genome = null; lastGen = null;
      $("gG-after-prep").style.display = "block";
      $("gG-after-train").style.display = "none";
      $("gG-sae-out").style.display = "none";
      $("gG-out").style.display = "none";
      $("gG-trace-wrap").style.display = "none";
    });
    
    /* ---- 2 · modello ---- */
    $("gG-train").addEventListener("click", async () => {
      if (!vocab || !ids) { APP.toast("Prepara prima vocabolario e corpus", "warn"); return; }
      const cap = Math.max(50, parseInt($("gG-cap").value, 10) || 1500);
      if (cap !== vocab.cap) { vocab = Gen.makeVocab(APP.dict, { cap }); ids = Gen.encode($("gG-corpus").value, APP.dict, vocab, G); }
      const Lc = Math.max(4, parseInt($("gG-lc").value, 10) || 16);
      const d = Math.max(8, parseInt($("gG-d").value, 10) || 32);
      const h = Math.max(8, parseInt($("gG-h").value, 10) || 64);
      const steps = Math.max(500, parseInt($("gG-steps").value, 10) || 3000);
      const cfg = { Lc, d, h };
      APP.showLoader("Addestramento del generatore", APP.fmt(steps) + " passi · vocab " + vocab.V + " · contesto " + Lc);
      await sleep(50);

      let ppl0, ppl1;
      try {
        const res = await runInWorker(
          { job: "train", cfg, vocab, ids, steps, lr: 0.01, pplSlice: 4000 },
          (p) => loaderSub("addestramento " + Math.min(99, Math.round(p.frac * 100)) + "% · passo "
            + APP.fmt(Math.round(p.frac * steps)) + "/" + APP.fmt(steps)
            + (typeof p.loss === "number" ? " · perdita " + p.loss.toFixed(2) : "")
            + " — la finestra resta viva")
        );
        model = Gen.deserialize(res.weights, vocab);
        ppl0 = res.ppl0; ppl1 = res.ppl1;
      } catch (e) {
        // ricaduta: tutto sul thread principale, identico al comportamento storico
        loaderSub("addestramento sul thread principale… un momento");
        await sleep(20);
        model = new Gen.LM(vocab, cfg);
        ppl0 = Gen.perplexity(model, ids.slice(0, 4000));
        const r = Gen.train(model, ids, { steps, lr: 0.01 });
        ppl1 = Gen.perplexity(model, ids.slice(0, 4000));
      }
      sae = null; genome = null;
      $("gG-status").textContent = "perplessità " + ppl0.toFixed(0) + " → " + ppl1.toFixed(0) + " (più bassa = ha imparato)";
      $("gG-after-train").style.display = "block";
      $("gG-sae-out").style.display = "none";
      $("gG-out").style.display = "none";
      $("gG-trace-wrap").style.display = "none";
      APP.hideLoader();
    });

    $("gG-save").addEventListener("click", () => {
      if (!model) { APP.toast("Niente modello da salvare", "warn"); return; }
      const blob = JSON.stringify(Gen.serialize(model, vocab));
      APP.saveNative(blob, APP.slug("modello-generatore-" + (vocab.dict_id || "corpus")) + "-" + APP.stamp(), ".json", "Sto salvando i pesi del modello (" + (blob.length / 1024 | 0) + " KB)");
    });

    $("gG-load").addEventListener("click", async () => {
      if (!APP.dict) { APP.toast("Serve il genoma attivo per ricostruire il vocabolario", "warn"); return; }
      const r = await APP.openText(); if (!r || !r.ok) return;
      let obj; try { obj = JSON.parse(r.content); } catch (e) { APP.toast("File non valido", "warn"); return; }
      if (obj.format !== "sapere-dna-generator") { APP.toast("Non è un modello del generatore", "warn"); return; }
      vocab = Gen.makeVocab(APP.dict, { cap: obj.vocab.cap });
      if (obj.vocab.dict_hash && obj.vocab.dict_hash !== APP.dict.hash) APP.toast("Attenzione: il modello è stato addestrato su un genoma diverso", "warn");
      model = Gen.deserialize(obj, vocab);
      $("gG-cap").value = obj.vocab.cap; $("gG-lc").value = obj.cfg.Lc; $("gG-d").value = obj.cfg.d; $("gG-h").value = obj.cfg.h;
      sae = null; genome = null;
      // serve comunque il corpus per estrarre i concetti
      if ($("gG-corpus").value.trim()) ids = Gen.encode($("gG-corpus").value, APP.dict, vocab, G);
      $("gG-status").textContent = "modello caricato · vocab " + vocab.V + " · contesto " + obj.cfg.Lc;
      $("gG-after-prep").style.display = "block";
      $("gG-after-train").style.display = "block";
      $("gG-sae-out").style.display = "none";
      $("gG-out").style.display = "none";
      $("gG-trace-wrap").style.display = "none";
      APP.toast("Modello caricato", "ok");
    });

    /* ---- 3 · concetti ---- */
    $("gG-sae").addEventListener("click", async () => {
      if (!model) { APP.toast("Addestra o carica prima un modello", "warn"); return; }
      if (!ids || ids.length < 50) { APP.toast("Serve il testo del corpus (preparalo sopra)", "warn"); return; }
      APP.showLoader("Estrazione dei concetti", "dizionario sparso (SAE) sull'ultima posizione");
      await sleep(50);

      let A;
      try {
        const weights = Gen.serialize(model, vocab);   // pesi correnti → worker
        const res = await runInWorker(
          { job: "sae", vocab, weights, ids, N: 1500, M: 24, steps: 8000 },
          (p) => loaderSub("estrazione concetti " + Math.round(p.frac * 100) + "% — la finestra resta viva")
        );
        A = res.A;
        sae = Gen.buildSAEInterface(res.saeRaw);
      } catch (e) {
        // ricaduta: tutto sul thread principale, identico al comportamento storico
        loaderSub("estrazione concetti sul thread principale… un momento");
        await sleep(20);
        A = Gen.lastPosMatrix(model, ids, { N: 1500 });
        sae = Gen.trainSAE(A, { M: 24, steps: 8000 });
      }
      genome = Gen.extractGenome(model, sae, A, vocab, G);
      const mono = genome.genes.filter(g => g.freq > 0).length;
      $("gG-honesty").innerHTML = `Qui <b>non c'è verità di base</b>: i nomi dei concetti sono indicativi (le parole che ciascuno favorisce). La qualità si misura come <b>monosemanticità</b> e, soprattutto, con l'<b>intervento causale</b> qui sotto. Ricostruzione SAE grossolana (rmse <b>${sae.rmse.toFixed(2)}</b>) — l'ultima posizione su testo vero è ad alta entropia: è il prezzo, onesto, di essere fuori dal laboratorio.`;
      $("gG-genhash").textContent = genome.dict.id + " v" + genome.dict.version + " · impronta " + genome.dict.hash.slice(0, 10);
      $("gG-genrows").innerHTML = genome.genes.map((g, rank) => g.freq === 0 ? "" :
        `<div class="men-row"><span>#${rank}</span><span class="nm">${APP.esc(g.favors.slice(0, 3).join(" / "))}</span><span class="fr">${APP.fmt(g.freq)}</span><span class="a">${G.addressOf(rank)}</span></div>`
      ).join("");
      $("gG-sae-status").textContent = "fatto · " + sae.M + " concetti · attivi medi " + sae.avgActive.toFixed(2);
      $("gG-sae-out").style.display = "block";
      APP.hideLoader();
    });

    /* ---- 4 · genera ---- */
    $("gG-gen").addEventListener("click", async () => {
      if (!model) { APP.toast("Addestra o carica prima un modello", "warn"); return; }
      const prime = $("gG-prime").value.trim();
      const temp = Math.max(0, parseFloat($("gG-temp").value) || 0.8);
      const len = Math.max(1, parseInt($("gG-len").value, 10) || 30);
      const primeIds = Gen.encode(prime, APP.dict, vocab, G);
      APP.showLoader("Generazione", len + " parole · temperatura " + temp);
      await sleep(40);
      lastGen = Gen.generate(model, primeIds, { maxLen: len, temperature: temp });
      const primeHtml = '<span style="color:var(--parch-dim)">' + APP.esc(prime) + '</span> ';
      const genHtml = lastGen.steps.map((s, k) => {
        const w = vocab.words[s.chosen] || "□";
        return `<span class="genword" data-k="${k}" style="cursor:pointer;border-bottom:1px dotted var(--green-bright);color:var(--parch)">${APP.esc(w)}</span>`;
      }).join(" ");
      $("gG-out").innerHTML = primeHtml + genHtml +
        (genome ? '' : '<div style="color:var(--parch-faint);font-size:.7rem;margin-top:8px">Estrai i concetti (passo 3) per poter cliccare una parola e leggerne la decisione.</div>');
      $("gG-out").style.display = "block";
      $("gG-trace-wrap").style.display = "none";
      APP.hideLoader();
    });

    $("gG-out").addEventListener("click", (e) => {
      const el = e.target.closest(".genword"); if (!el) return;
      if (!genome) { APP.toast("Estrai prima i concetti (passo 3)", "warn"); return; }
      curK = parseInt(el.dataset.k, 10);
      document.querySelectorAll(".genword").forEach(x => x.style.background = "");
      el.style.background = "rgba(127,206,154,.18)";
      const step = lastGen.steps[curK];
      lastTrace = Gen.derive(model, step.ctx, genome, vocab, G);
      lastInterv = Gen.intervene(model, genome, lastTrace, vocab);
      lastSeal = Gen.seal(genome, lastTrace.filament, G);
      renderTrace(step);
    });

    function renderTrace(step) {
      const chosen = vocab.words[step.chosen] || "□";
      const ctxText = step.ctx.map(i => vocab.words[i] || "□").join(" ");
      $("gG-seqline").innerHTML = `contesto: «${APP.esc(ctxText)}» &rarr; scelta: <b style="color:var(--green-bright)">${APP.esc(chosen)}</b> · più probabile: <b>${APP.esc(lastTrace.predWord)}</b>`;

      const SP = G.CTRL.SP;
      $("gG-filament").innerHTML = lastTrace.fired.length
        ? lastTrace.fired.map((f, k) => '<span style="color:var(--green-bright);font-weight:700">' + G.addressOf(genome.dict.index[genome.conceptKey(genome.ch.concept[f.m])]) + '</span>' + (k < lastTrace.fired.length - 1 ? '<span style="color:var(--parch-faint)">' + SP + '</span>' : "")).join("")
        : '<span style="color:var(--parch-faint)">(nessun concetto acceso)</span>';

      const maxAct = Math.max(1e-6, ...lastTrace.fired.map(f => f.act));
      $("gG-fired").innerHTML = lastTrace.fired.map(f =>
        `<div class="men-fired-row"><span class="nm">${APP.esc(f.favors.slice(0, 3).join("/"))} <span style="color:var(--parch-faint)">c${f.m}</span></span><div class="track"><div class="fill" style="width:${(f.act / maxAct * 100).toFixed(0)}%"></div></div><span class="v">${f.act.toFixed(2)}</span></div>`
      ).join("") || '<span style="color:var(--parch-faint);font-family:var(--font-mono);font-size:.7rem">nessuno</span>';

      $("gG-interv").innerHTML = lastInterv.rows.map(r =>
        `<div class="men-iv-row ${r.causal ? "causal" : ""}">spengo <b style="color:var(--parch)">${APP.esc(r.favors.slice(0, 2).join("/"))}</b> &rarr; «<b style="color:${r.causal ? "var(--green-bright)" : "var(--parch-faint)"}">${APP.esc(r.newWord)}</b>» ${r.causal ? '<span class="flag">&#9888; cambia</span>' : '<span style="color:var(--parch-faint)">= invariata</span>'}<span class="men-drop">&minus;${(r.dropPct * 100).toFixed(0)}%</span></div>`
      ).join("") || '<span style="color:var(--parch-faint);font-family:var(--font-mono);font-size:.7rem">—</span>';

      const setFid = (f, v, val) => { $(f).style.width = (val * 100).toFixed(0) + "%"; $(v).textContent = (val * 100).toFixed(0) + "%"; };
      setFid("gG-fid1", "gG-fidv1", lastInterv.fidelitySingle);
      setFid("gG-fid3", "gG-fidv3", lastInterv.fidelityGraded);

      const segs = G.trace(lastTrace.filament, genome.dict).filter(s => s.role === "gene").map(s => s.text);
      const orig = lastTrace.fired.map(f => genome.conceptKey(genome.ch.concept[f.m]));
      const verified = JSON.stringify(segs) === JSON.stringify(orig);
      $("gG-seal").innerHTML = `impronta genoma &nbsp;&nbsp;&nbsp;<b>${lastSeal.genomeHash}</b><br>impronta tracciato <b>${lastSeal.traceHash}</b><br>rilettura col genoma &rarr; <b>${verified ? "identica &#10003;" : "NON combacia &#9888;"}</b>`;

      $("gG-trace-wrap").style.display = "block";
      updateBridge(ctxText);
    }

    async function updateBridge(ctxText) {
      $("gG-bridge").innerHTML = '<span style="color:var(--parch-faint)">incido nel filo…</span>';
      try {
        const enc = await C.encode(ctxText);
        const shown = enc.dna.length > 90 ? enc.dna.slice(0, 90) + "…" : enc.dna;
        $("gG-bridge").innerHTML = `DNA classico <span style="color:var(--teal-bright);word-break:break-all">${shown}</span><br><span style="color:var(--parch-faint)">${APP.fmt(enc.count)} codoni · lo stesso contesto, scritto come filamento.</span>`;
      } catch (e) { $("gG-bridge").innerHTML = '<span style="color:var(--parch-faint)">—</span>'; }
    }

    function buildEntry() {
      const step = lastGen.steps[curK];
      return {
        mode: "mente",
        title: "generazione · «" + (vocab.words[step.chosen] || "□") + "»",
        context: step.ctx.map(i => vocab.words[i] || "□").join(" "),
        chosen: vocab.words[step.chosen] || "□", most_likely: lastTrace.predWord,
        dict_id: genome.dict.id, dict_hash: genome.dict.hash,
        filament: lastTrace.filament, trace_hash: lastSeal.traceHash,
        fired: lastTrace.fired.map(f => ({ concept: f.m, favors: f.favors, act: +f.act.toFixed(4) })),
        fidelity: { single: +lastInterv.fidelitySingle.toFixed(3), graded: +lastInterv.fidelityGraded.toFixed(3) },
        created: new Date().toISOString()
      };
    }
    $("gG-arc").addEventListener("click", () => { if (!lastTrace) return; APP.addToArchive(buildEntry()); APP.toast("Tracciato aggiunto all'archivio", "ok"); });
    $("gG-json").addEventListener("click", () => { if (!lastTrace) return; APP.saveNative(JSON.stringify(buildEntry(), null, 2), APP.slug("generazione-" + lastTrace.predWord) + "-" + APP.stamp(), ".json", "Sto salvando il tracciato di generazione"); });
  };
})();
