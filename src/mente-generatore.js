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
    const Mot = window.SapereDNAMotoreEmisferi;       // motore a emisferi = Cassandra
    const ENGINE = "emisferi";                        // il banco usa Cassandra
    const panel = document.getElementById("gG-panel");
    if (!Gen) { panel.innerHTML = '<p class="err">Modulo generatore non caricato (manca generatore.js).</p>'; return; }
    if (!Mot) { panel.innerHTML = '<p class="err">Motore di Cassandra non caricato (mancano emisferi.js / motore-emisferi.js).</p>'; return; }

    let model = null, vocab = null, ids = null, sae = null, genome = null;
    let lastGen = null, lastTrace = null, lastInterv = null, lastSeal = null, lastDeep = null, curK = -1, modelSig = null;
    let peekSAE = null, peekWords = null;   // SAE dei primitivi del 1° blocco (sbirciata) + mappa concetto→parole, per il tracciamento profondo

    panel.innerHTML = `
      <div class="note" style="margin-bottom:12px"><b>Cassandra</b> — un piccolo <b>Transformer causale a parole</b> a <b>due emisferi</b> (neuroni cablati dalla regola del π) uniti da un <b>corpo calloso</b>, il cui vocabolario <b>è il genoma</b>: impara da un corpus vero a prevedere la parola seguente e genera testo. La <b>lente legge il calloso</b>, il punto dove tutto si ricompone. Qui la <b>verità di base non c'è</b> — non sappiamo quali siano i concetti «giusti». Restano le due prove di Sapere-DNA che valgono nel mondo reale: l'<b>intervento causale</b> (spengo un concetto e vedo se la parola cambia) e il <b>sigillo</b> (il tracciato è legato al calcolo, non è una storia inventata). Il modello non sarà eloquente: <i>non è la fluenza la dimostrazione, è che possiamo aprirlo e provarlo.</i></div>

      <div class="out-label">1 · Vocabolario (genoma attivo) e corpus</div>
      <div class="men-hash" id="gG-vocab-info"></div>
      <textarea id="gG-corpus" class="fin" style="width:100%;min-height:90px;margin-top:8px" placeholder="Incolla qui il testo del corpus, oppure caricalo da file…"></textarea>
      <div class="row mt">
        <button class="btn green ghost" id="gG-load-corpus">Carica testo…</button>
        <button class="btn green" id="gG-prep">Prepara vocabolario e corpus</button>
        <button class="btn green ghost" id="gG-load-top">Carica modello salvato…</button>
        <span id="gG-prep-status" style="font-family:var(--font-mono);font-size:.7rem;color:var(--parch-dim)"></span>
      </div>

      <div id="gG-after-prep" style="display:none">
        <div class="out-label" style="margin-top:18px">2 · Modello</div>
        <div class="row" style="align-items:flex-end;gap:16px">
          <label class="men-f">tetto vocab.<input class="fin" id="gG-cap" value="1500" style="width:80px"></label>
          <label class="men-f">contesto<input class="fin" id="gG-lc" value="16" style="width:60px"></label>
          <label class="men-f">dim. residuo<input class="fin" id="gG-d" value="32" style="width:60px"></label>
          <label class="men-f">lato emisfero<input class="fin" id="gG-hemi" value="12" style="width:60px"></label>
          <label class="men-f">lato calloso<input class="fin" id="gG-cal" value="8" style="width:60px"></label>
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
          <div class="row" style="align-items:flex-end;gap:16px"><label class="men-f">concetti del dizionario (M)<input class="fin" id="gG-M" value="48" style="width:70px"></label><label class="men-f">concetti attivi (K · TopK)<input class="fin" id="gG-K" value="8" style="width:70px"></label><button class="btn green ghost" id="gG-sae">Estrai concetti (SAE)</button><span id="gG-sae-status" style="font-family:var(--font-mono);font-size:.66rem;color:var(--parch-dim)"></span></div>
          <div class="note" style="margin:6px 0;font-size:.72rem">Per studiare la <b>sovrapposizione</b> servono <b>più concetti che dimensioni</b>: in Cassandra la lente legge il <b>corpo calloso</b>, quindi imposta <b>M</b> sopra il numero di neuroni del calloso (<b>lato calloso × lato calloso</b>). Con M &gt; nC il dizionario è <i>sovra-completo</i> — è lo strumento che rende osservabile l'impacchettamento. <b>K</b> impone quanti concetti possono accendersi per decisione (TopK): più piccolo = concetti più puliti. Metti K=0 per la vecchia sparsità morbida (L1).</div>
          <div id="gG-sae-out" style="display:none">
            <div class="callout-mini" id="gG-honesty" style="margin:8px 0"></div>
            <div class="men-hash" id="gG-genhash" style="margin-bottom:8px"></div>
            <div class="men-list">
              <div class="men-row men-head"><span>loc.</span><span>concetto (favorisce →)</span><span>freq</span><span>ind.</span></div>
              <div class="men-rows" id="gG-genrows"></div>
            </div>

            <div class="out-label" style="margin-top:20px">Concetti primitivi del 1° blocco <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">la sbirciata sul calloso intermedio</span></div>
            <div class="row"><button class="btn green ghost" id="gG-peek">Sbircia il 1° blocco</button><span id="gG-peek-status" style="font-family:var(--font-mono);font-size:.66rem;color:var(--parch-dim)"></span></div>
            <div class="note" style="margin:6px 0;font-size:.72rem">Cassandra ha più lastre: il <b>1° blocco</b> forma concetti «primitivi» che il blocco finale ricompone. Qui li leggiamo per <b>cosa li accende</b> (la parola più recente del contesto in cui scattano) — <i>non</i> per quale parola fanno dire, perché il 1° blocco non sceglie la parola direttamente. Il «cosa fanno a valle» è il <b>tracciamento causale profondo</b>, un passo futuro.</div>
            <div id="gG-peek-out" style="display:none">
              <div class="callout-mini" id="gG-peek-info" style="margin:8px 0"></div>
              <div id="gG-peek-rows" style="margin-top:6px"></div>
            </div>

            <div class="out-label" style="margin-top:20px">5 · Sovrapposizione <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">più concetti causali che dimensioni</span></div>
            <div class="row"><button class="btn green ghost" id="gG-super">Analizza sovrapposizione</button><span id="gG-super-status" style="font-family:var(--font-mono);font-size:.66rem;color:var(--parch-dim)"></span></div>
            <div id="gG-super-out" style="display:none">
              <div class="callout-mini" id="gG-super-pre" style="margin:8px 0"></div>
              <div class="out-label" style="margin-top:8px">Geometria delle direzioni-concetto <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">il dizionario come vettori in d dimensioni</span></div>
              <div class="row" style="gap:18px;align-items:flex-start;flex-wrap:wrap">
                <canvas id="gG-gram" width="224" height="224" style="background:rgba(8,10,13,0.7);border:1px solid var(--line);border-radius:6px;image-rendering:pixelated"></canvas>
                <div id="gG-super-geo" style="font-family:var(--font-mono);font-size:.72rem;line-height:1.8;color:var(--parch);min-width:230px"></div>
              </div>
              <div class="out-label" style="margin-top:16px">Censimento causale <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">spengo ogni concetto su molti contesti veri</span></div>
              <div id="gG-super-cen"></div>
              <div class="callout-mini" id="gG-super-verdict" style="margin-top:10px;border-left-color:var(--green);background:rgba(127,206,154,0.07)"></div>
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
            <div class="out-label" style="margin-top:14px">Tracciamento causale profondo <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">spengo un primitivo del 1° blocco e guardo la parola passando per il blocco finale</span></div>
            <div id="gG-deep"></div>
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
      // Il modello già addestrato si CONSERVA se è compatibile con questo
      // genoma + vocabolario: in tal caso si va dritti ai concetti, senza
      // riaddestrare. Si riparte da zero solo se non c'è un modello compatibile
      // (genoma nuovo, oppure cap del vocabolario cambiato).
      const reuse = !!model && modelSig === (d.hash + "|" + vocab.cap) && model.V === vocab.V;
      $("gG-after-prep").style.display = "block";
      $("gG-sae-out").style.display = "none";
      $("gG-super-out").style.display = "none"; $("gG-super-status").textContent = "";
      $("gG-out").style.display = "none";
      $("gG-trace-wrap").style.display = "none";
      peekSAE = null; lastDeep = null;   // il dizionario dei primitivi va rifatto sul corpus/modello correnti
      if (reuse) {
        sae = null; genome = null;   // i concetti si ri-estraggono sul corpus corrente
        $("gG-status").textContent = "modello già addestrato riusato · vocab " + vocab.V + " · contesto " + model.Lc + " — nessun riaddestramento necessario";
        $("gG-after-train").style.display = "block";
        APP.toast("Modello riusato: vai pure a estrarre i concetti", "ok");
      } else {
        model = null; sae = null; genome = null; lastGen = null; modelSig = null;
        $("gG-after-train").style.display = "none";
      }
    });
    
    /* ---- 2 · modello ---- */
    $("gG-train").addEventListener("click", async () => {
      if (!vocab || !ids) { APP.toast("Prepara prima vocabolario e corpus", "warn"); return; }
      const cap = Math.max(50, parseInt($("gG-cap").value, 10) || 1500);
      if (cap !== vocab.cap) { vocab = Gen.makeVocab(APP.dict, { cap }); ids = Gen.encode($("gG-corpus").value, APP.dict, vocab, G); }
      const Lc = Math.max(4, parseInt($("gG-lc").value, 10) || 16);
      const d = Math.max(8, parseInt($("gG-d").value, 10) || 32);
      const hemi = Math.max(3, parseInt($("gG-hemi").value, 10) || 12);
      const cal = Math.max(2, parseInt($("gG-cal").value, 10) || 8);
      const steps = Math.max(500, parseInt($("gG-steps").value, 10) || 3000);
      const cfg = { Lc, d, hemiW: hemi, hemiH: hemi, calW: cal, calH: cal };
      APP.showLoader("Addestramento di Cassandra", APP.fmt(steps) + " passi · vocab " + vocab.V + " · contesto " + Lc);
      await sleep(50);

      let ppl0, ppl1;
      try {
        const res = await runInWorker(
          { job: "train", engine: ENGINE, cfg, vocab, ids, steps, lr: 0.01, pplSlice: 4000 },
          (p) => loaderSub("addestramento " + Math.min(99, Math.round(p.frac * 100)) + "% · passo "
            + APP.fmt(Math.round(p.frac * steps)) + "/" + APP.fmt(steps)
            + (typeof p.loss === "number" ? " · perdita " + p.loss.toFixed(2) : "")
            + " — la finestra resta viva")
        );
        model = Mot.deserialize(res.weights, vocab);
        ppl0 = res.ppl0; ppl1 = res.ppl1;
      } catch (e) {
        // ricaduta: tutto sul thread principale, l'app non si rompe mai
        loaderSub("addestramento sul thread principale… un momento");
        await sleep(20);
        model = new Mot.LMEmisferi(vocab, cfg);
        ppl0 = Gen.perplexity(model, ids.slice(0, 4000));
        Gen.train(model, ids, { steps, lr: 0.01 });
        ppl1 = Gen.perplexity(model, ids.slice(0, 4000));
      }
      sae = null; genome = null;
      peekSAE = null;   // i primitivi del 1° blocco appartenevano al modello precedente
      modelSig = APP.dict.hash + "|" + vocab.cap;
      $("gG-status").textContent = "perplessità " + ppl0.toFixed(0) + " → " + ppl1.toFixed(0) + " (più bassa = ha imparato)";
      $("gG-after-train").style.display = "block";
      $("gG-sae-out").style.display = "none";
      $("gG-out").style.display = "none";
      $("gG-trace-wrap").style.display = "none";
      APP.hideLoader();
    });

    $("gG-save").addEventListener("click", () => {
      if (!model) { APP.toast("Niente modello da salvare", "warn"); return; }
      const blob = JSON.stringify(Mot.serialize(model, vocab));
      APP.saveNative(blob, APP.slug("cassandra-" + (vocab.dict_id || "corpus")) + "-" + APP.stamp(), ".json", "Sto salvando i pesi del modello (" + (blob.length / 1024 | 0) + " KB)");
    });

    async function loadModelFromFile() {
      if (!APP.dict) { APP.toast("Serve il genoma attivo per ricostruire il vocabolario", "warn"); return; }
      const r = await APP.openText(); if (!r || !r.ok) return;
      let obj; try { obj = JSON.parse(r.content); } catch (e) { APP.toast("File non valido", "warn"); return; }
      if (obj.format !== "sapere-dna-cassandra") { APP.toast("Non è un modello di Cassandra", "warn"); return; }
      vocab = Gen.makeVocab(APP.dict, { cap: obj.vocab.cap });
      if (obj.vocab.dict_hash && obj.vocab.dict_hash !== APP.dict.hash) APP.toast("Attenzione: il modello è stato addestrato su un genoma diverso", "warn");
      model = Mot.deserialize(obj, vocab);
      modelSig = (obj.vocab.dict_hash || APP.dict.hash) + "|" + vocab.cap;
      peekSAE = null;   // modello nuovo: i primitivi del 1° blocco vanno ri-sbirciati
      $("gG-cap").value = obj.vocab.cap; $("gG-lc").value = obj.cfg.Lc; $("gG-d").value = obj.cfg.d;
      $("gG-hemi").value = obj.cfg.hemiW; $("gG-cal").value = obj.cfg.calW;
      sae = null; genome = null;
      // serve comunque il corpus per estrarre i concetti
      if ($("gG-corpus").value.trim()) ids = Gen.encode($("gG-corpus").value, APP.dict, vocab, G);
      $("gG-status").textContent = "modello caricato · vocab " + vocab.V + " · contesto " + obj.cfg.Lc;
      $("gG-after-prep").style.display = "block";
      $("gG-after-train").style.display = "block";
      $("gG-sae-out").style.display = "none";
      $("gG-super-out").style.display = "none"; $("gG-super-status").textContent = "";
      $("gG-out").style.display = "none";
      $("gG-trace-wrap").style.display = "none";
      APP.toast("Modello caricato", "ok");
    }
    $("gG-load").addEventListener("click", loadModelFromFile);
    $("gG-load-top").addEventListener("click", loadModelFromFile);

    /* ---- 3 · concetti ---- */
    $("gG-sae").addEventListener("click", async () => {
      if (!model) { APP.toast("Addestra o carica prima un modello", "warn"); return; }
      if (!ids || ids.length < 50) { APP.toast("Serve il testo del corpus (preparalo sopra)", "warn"); return; }
      const M = Math.max(2, parseInt($("gG-M").value, 10) || 48);
      const K = Math.max(0, parseInt($("gG-K").value, 10) || 0);
      $("gG-peek-out").style.display = "none"; $("gG-peek-status").textContent = ""; peekSAE = null;   // la sbirciata precedente non vale più
      APP.showLoader("Estrazione dei concetti", M + " concetti" + (K ? " · TopK=" + K : " · L1") + " · dizionario sparso (SAE)");
      await sleep(50);

      let A;
      try {
        const weights = Mot.serialize(model, vocab);   // pesi correnti → worker
        const res = await runInWorker(
          { job: "sae", engine: ENGINE, vocab, weights, ids, N: 1500, M: M, k: K, steps: 8000 },
          (p) => loaderSub("estrazione concetti " + Math.round(p.frac * 100) + "% — la finestra resta viva")
        );
        A = res.A;
        sae = Gen.buildSAEInterface(res.saeRaw);
      } catch (e) {
        // ricaduta: tutto sul thread principale, identico al comportamento storico
        loaderSub("estrazione concetti sul thread principale… un momento");
        await sleep(20);
        A = Gen.lastPosMatrix(model, ids, { N: 1500 });
        sae = Gen.trainSAE(A, { M: M, k: K, steps: 8000 });
      }
      genome = Gen.extractGenome(model, sae, A, vocab, G);
      let q = null;
      try { loaderSub("misuro la qualità della SAE… frazione di perdita recuperata"); await sleep(20); q = Gen.saeQuality(model, sae, ids, { samples: 160 }); } catch (e) { }
      const sparsLabel = sae.k ? ("TopK · " + sae.k + " concetti attivi per decisione") : "L1 · sparsità morbida";
      const recPct = q ? (q.lossRecovered * 100).toFixed(0) + "%" : "—";
      $("gG-honesty").innerHTML = `Qui <b>non c'è verità di base</b>: i nomi dei concetti sono indicativi (le parole che ciascuno favorisce). La qualità si misura come <b>monosemanticità</b> e, soprattutto, con l'<b>intervento causale</b> qui sotto.<br><span style="color:var(--parch-faint)">Dizionario: <b>${sparsLabel}</b> · ricostruzione rmse <b>${sae.rmse.toFixed(2)}</b> · <b>perdita recuperata ${recPct}</b> · feature morte <b>${sae.dead || 0}</b>/${sae.M}.</span>`;
      $("gG-genhash").textContent = genome.dict.id + " v" + genome.dict.version + " · impronta " + genome.dict.hash.slice(0, 10);
      $("gG-genrows").innerHTML = genome.genes.map((g, rank) => g.freq === 0 ? "" :
        `<div class="men-row"><span>#${rank}</span><span class="nm">${APP.esc(g.favors.slice(0, 3).join(" / "))}</span><span class="fr">${APP.fmt(g.freq)}</span><span class="a">${G.addressOf(rank)}</span></div>`
      ).join("");
      $("gG-sae-status").textContent = "fatto · " + sae.M + " concetti · " + (sae.k ? "TopK=" + sae.k : "attivi medi " + sae.avgActive.toFixed(2)) + (sae.M > model.d ? " · sovra-completo (M>d)" : "");
      $("gG-sae-out").style.display = "block";
      $("gG-super-out").style.display = "none"; $("gG-super-status").textContent = "";
      APP.hideLoader();
    });

    $("gG-peek").addEventListener("click", async () => {
      if (!model) { APP.toast("Addestra o carica prima Cassandra", "warn"); return; }
      if ((model.L || 1) < 2) { peekSAE = null; $("gG-peek-info").innerHTML = "Questo modello ha un <b>solo blocco</b>: non c'è un calloso intermedio da sbirciare."; $("gG-peek-rows").innerHTML = ""; $("gG-peek-out").style.display = "block"; return; }
      if (!ids || ids.length < 50) { APP.toast("Serve il testo del corpus (preparalo sopra)", "warn"); return; }
      const M = Math.max(2, parseInt($("gG-M").value, 10) || 48);
      const K = Math.max(0, parseInt($("gG-K").value, 10) || 0);
      APP.showLoader("Sbirciata sul 1° blocco", "estraggo i concetti primitivi del calloso intermedio");
      await sleep(50);
      let pk;
      try {
        const weights = Mot.serialize(model, vocab);
        const res = await runInWorker(
          { job: "peek", engine: ENGINE, vocab, weights, ids, block: 0, N: 800, M: M, k: K, steps: 6000 },
          (p) => loaderSub("sbirciata " + Math.round((p.frac || 0) * 100) + "% — la finestra resta viva")
        );
        pk = res.peek;
      } catch (e) {
        loaderSub("sbirciata sul thread principale… un momento");
        await sleep(20);
        pk = Gen.peekConcepts(model, ids, vocab, { N: 400, block: 0, M: M, k: K, steps: 3000 });
      }
      const live = pk.concepts.length;
      $("gG-peek-info").innerHTML = `Calloso del <b>1° blocco</b> · dizionario <b>${pk.M}</b> concetti${pk.k ? (" · TopK " + pk.k) : ""} · <b>${live}</b> con un significato leggibile · feature morte <b>${pk.dead}</b>/${pk.M}.<br><span style="color:var(--parch-faint)">Ogni concetto è descritto da ciò che lo <b>accende</b> di più — sono i mattoni che il blocco finale ricompone.</span>`;
      $("gG-peek-rows").innerHTML = pk.concepts.map(c =>
        `<div style="display:flex;gap:12px;padding:4px 0;border-bottom:1px solid var(--line);font-size:.8rem"><span style="font-family:var(--font-mono);color:var(--teal-bright);min-width:44px">#${c.concept}</span><span style="color:var(--parch)">si accende su ${c.words.map(w => `«<b>${APP.esc(w.word)}</b>»`).join(", ")}</span></div>`
      ).join("") || '<span style="color:var(--parch-faint)">nessun concetto con un significato leggibile</span>';
      $("gG-peek-status").textContent = "fatto · " + live + " concetti vivi";
      $("gG-peek-out").style.display = "block";
      // tengo la SAE dei primitivi (e la mappa concetto→parole) per il tracciamento profondo
      peekSAE = pk.saeRaw ? Gen.buildSAEInterface(pk.saeRaw) : null;
      peekWords = new Map(pk.concepts.map(c => [c.concept, c.words]));
      if (lastTrace && curK >= 0) renderDeep(lastGen.steps[curK]);   // decisione già aperta → aggiorno subito la traccia profonda
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

      renderDeep(step);
      $("gG-trace-wrap").style.display = "block";
      updateBridge(ctxText);
    }

    /* ---- tracciamento causale profondo (sonda d'onestà sul 1° blocco) ----
       Spegne ogni primitivo del calloso intermedio e propaga l'effetto fino
       alla testa. Se la parola non cambia, è una VERITÀ, non un fallimento:
       qui la decisione non passa dal 1° blocco. Niente falsa attribuzione. */
    function renderDeep(step) {
      const box = $("gG-deep"); if (!box) return;
      const faint = (t) => '<div style="color:var(--parch-faint);font-family:var(--font-mono);font-size:.72rem">' + t + '</div>';
      if ((model.L || 1) < 2) { box.innerHTML = faint("Cassandra ha un solo blocco: niente calloso intermedio da attraversare."); lastDeep = null; return; }
      if (!peekSAE) { box.innerHTML = faint("Premi prima «Sbircia il 1° blocco» (passo 3): serve il dizionario dei primitivi per poterli spegnere."); lastDeep = null; return; }
      const dt = Gen.deepTrace(model, step.ctx, peekSAE, vocab, { block: 0 });
      lastDeep = dt;
      if (!dt) { box.innerHTML = faint("Niente blocco a valle da attraversare per questo modello."); return; }
      const label = (m) => { const w = peekWords && peekWords.get(m); return w && w.length ? w.slice(0, 3).map(x => APP.esc(x.word)).join("/") : ("primitivo #" + m); };
      const rowsHtml = dt.rows.length ? dt.rows.map(r =>
        `<div class="men-iv-row ${r.causal ? "causal" : ""}">spengo <b style="color:var(--parch)">${label(r.m)}</b> <span style="color:var(--parch-faint)">#${r.m}</span> &rarr; «<b style="color:${r.causal ? "var(--green-bright)" : "var(--parch-faint)"}">${APP.esc(r.newWord)}</b>» ${r.causal ? '<span class="flag">&#9888; cambia</span>' : '<span style="color:var(--parch-faint)">= invariata</span>'}<span class="men-drop">&minus;${(r.dropPct * 100).toFixed(0)}%</span></div>`
      ).join("") : faint("nessun primitivo acceso su questo contesto");
      const anyCausal = dt.rows.some(r => r.causal);
      const verdict = (!anyCausal && dt.fidelityGraded < 0.01)
        ? `<div class="callout-mini" style="margin-top:8px"><b>Il 1° blocco qui è aggirato.</b> Spegnere i suoi primitivi non muove «<b>${APP.esc(dt.baseWord)}</b>»: la decisione non passa di lì — il flusso residuo la ricostruisce dal contesto, e il blocco finale ci arriva da solo. <span style="color:var(--parch-faint)">Lo strumento dice il vero anche quando il vero è «non conta»: niente falsa attribuzione a concetti che non causano nulla.</span></div>`
        : `<div class="callout-mini" style="margin-top:8px;border-left-color:var(--green);background:rgba(127,206,154,0.07)"><b>I primitivi del 1° blocco contano a valle.</b> Spegnerli sposta la parola finale (spegnendoli tutti: «<b>${APP.esc(dt.baseWord)}</b>» &rarr; «<b>${APP.esc(dt.allWord)}</b>», calo <b>${(dt.fidelityGraded * 100).toFixed(0)}%</b>). Questo mattone intermedio porta qualcosa che il resto non ricostruisce.</div>`;
      box.innerHTML =
        `<div style="font-family:var(--font-mono);font-size:.72rem;color:var(--parch-faint);margin-bottom:4px">parola predetta: «<b style="color:var(--parch)">${APP.esc(dt.baseWord)}</b>» (conf ${(dt.baseConf * 100).toFixed(0)}%) · attraverso il blocco finale</div>`
        + rowsHtml + verdict;
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

    /* ---- 5 · sovrapposizione (Asse 1) ---- */
    function gramColor(c) {
      const a = Math.min(1, Math.abs(c));
      if (c >= 0) return "rgb(" + Math.round(14 + a * 36) + "," + Math.round(34 + a * 158) + "," + Math.round(32 + a * 122) + ")"; // teal: allineate
      return "rgb(" + Math.round(40 + a * 192) + "," + Math.round(32 + a * 130) + "," + Math.round(18 + a * 22) + ")";           // oro: opposte
    }

    function renderSuper(geo, cen) {
      const d = geo.d, M = geo.M;
      $("gG-super-pre").innerHTML = geo.overcomplete
        ? "Dizionario <b>sovra-completo</b>: <b>" + M + "</b> concetti in uno spazio di sole <b>" + d + "</b> dimensioni (M &gt; d). È la <i>condizione</i> per osservare la sovrapposizione — lo strumento ha più caselle delle dimensioni. La <b>prova</b> è il censimento causale qui sotto."
        : '<span style="color:var(--gold-bright)">&#9888; Qui M = ' + M + ' &le; d = ' + d + ': il dizionario <b>non</b> è sovra-completo, quindi la sovrapposizione non può comparire per costruzione. Aumenta i <b>concetti (M)</b> oltre la <b>dim. modello (d)</b> e ri-estrai.</span>';

      const cv = $("gG-gram");
      if (cv && cv.getContext) {
        const cx = cv.getContext("2d"), W = cv.width, H = cv.height, cw = W / M, ch = H / M;
        cx.clearRect(0, 0, W, H);
        for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) { cx.fillStyle = gramColor(geo.gram[i][j]); cx.fillRect(Math.floor(j * cw), Math.floor(i * ch), Math.ceil(cw), Math.ceil(ch)); }
      }
      $("gG-super-geo").innerHTML =
        "M = <b>" + M + "</b> concetti · d = <b>" + d + "</b> dim.<br>" +
        "interferenza media |cos| <b>" + geo.offdiagMean.toFixed(2) + "</b><br>" +
        "interferenza massima |cos| <b>" + geo.offdiagMax.toFixed(2) + "</b><br>" +
        "coppie quasi-antipodali <b>" + geo.antipodalCount + "</b><br>" +
        "coppie interferenti <b>" + geo.interferingCount + "</b><br>" +
        '<span style="color:var(--parch-faint)">teal = allineate · oro = opposte · scuro = ortogonali</span>';

      const top = cen.perConcept.slice(0, 12);
      $("gG-super-cen").innerHTML = top.length ? top.map(function (c) {
        return '<div class="men-fired-row"><span class="nm">' + APP.esc(c.favors.slice(0, 3).join("/")) + ' <span style="color:var(--parch-faint)">c' + c.m + '</span></span><div class="track"><div class="fill" style="width:' + (c.rate * 100).toFixed(0) + '%' + (c.robust ? '' : ';opacity:.45') + '"></div></div><span class="v">' + (c.rate * 100).toFixed(0) + '%</span></div>';
      }).join("") : '<span style="color:var(--parch-faint);font-family:var(--font-mono);font-size:.7rem">nessun concetto causale nei contesti sondati</span>';

      $("gG-super-verdict").innerHTML = cen.superposition
        ? '<b style="color:var(--green-bright)">Sovrapposizione dimostrata.</b> <b>' + cen.causalRobust + '</b> concetti sono singolarmente causali in modo robusto, in uno spazio di sole <b>' + d + '</b> dimensioni: il modello ha impacchettato <b>più feature che dimensioni</b>. <span style="color:var(--parch-faint)">(causali almeno una volta: ' + cen.causalEver + ')</span>'
        : '<b>' + cen.causalRobust + '</b> concetti causali robusti, contro <b>' + d + '</b> dimensioni: non oltre d, quindi qui la sovrapposizione <b>non è dimostrata</b>. Prova ad aumentare M, oppure ad addestrare di più il modello. <span style="color:var(--parch-faint)">(causali almeno una volta: ' + cen.causalEver + ')</span>';
    }

    $("gG-super").addEventListener("click", async () => {
      if (!genome) { APP.toast("Estrai prima i concetti (passo 3)", "warn"); return; }
      if (!ids || ids.length < 50) { APP.toast("Serve il testo del corpus per sondare il modello", "warn"); return; }
      APP.showLoader("Analisi della sovrapposizione", "geometria dei concetti + censimento causale");
      await sleep(50);
      try {
        const geo = Gen.decoderGeometry(genome.sae, {});
        loaderSub("censimento causale su contesti veri… la finestra può fermarsi un istante");
        await sleep(20);
        const cen = Gen.causalCensus(model, genome, vocab, ids, G, { samples: 100 });
        renderSuper(geo, cen);
        $("gG-super-out").style.display = "block";
        $("gG-super-status").textContent = "fatto · " + cen.samples + " contesti sondati";
      } catch (e) {
        APP.toast("Analisi non riuscita: " + ((e && e.message) || e), "warn");
      }
      APP.hideLoader();
    });
  };
})();