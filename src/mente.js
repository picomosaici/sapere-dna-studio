/* ============================================================
   Sapere-DNA Studio · MENTE (controller della scatola trasparente)
   Espone window.initMente(APP)
   ============================================================ */
(function () {
  "use strict";

  window.initMente = function (APP) {
    const G = APP.G, R = APP.R;
    const panel = document.getElementById("men-panel");
    if (!R) { panel.innerHTML = '<p class="err">Modulo rete non caricato (manca rete.js).</p>'; return; }

    let net = null, genome = null, data = null, point = null, lastTrace = null, lastInterv = null, lastSeal = null;

    panel.innerHTML = `
      <div class="note" style="margin-bottom:12px">Una piccola rete neurale addestrabile (classificatore) agganciata a Sapere-DNA. La rete fa il suo lavoro (separa i punti del piano in quattro quadranti); poi <b>estraiamo dai suoi neuroni un genoma di concetti</b>, e per ogni punto incidiamo il <b>tracciato del ragionamento</b> come un vero filamento, lo mettiamo alla prova con l'<b>intervento</b> e lo <b>sigilliamo</b>. È il banco dove l'idea smette di essere teoria.</div>

      <div class="row" style="align-items:flex-end;gap:16px">
        <label class="men-f">nodi nascosti<input class="fin" id="men-hidden" value="8" style="width:70px"></label>
        <label class="men-f">iterazioni<input class="fin" id="men-iters" value="80000" style="width:95px"></label>
        <label class="men-f">dati<input class="fin" id="men-data" value="4000" style="width:80px"></label>
        <label class="men-f">tasso appr.<input class="fin" id="men-lr" value="0.3" style="width:70px"></label>
      </div>
      <div class="row mt">
        <button class="btn blue" id="men-train">Inizializza e addestra</button>
        <span id="men-status" style="font-family:var(--font-mono);font-size:.7rem;color:var(--parch-dim)"></span>
      </div>

      <div id="men-body" style="display:none">
        <div class="men-grid">
          <div>
            <div class="out-label"><span>La rete al lavoro</span> <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">clicca un punto per leggerne il ragionamento</span></div>
            <canvas id="men-canvas" width="360" height="360" class="men-canvas"></canvas>
          </div>
          <div>
            <div class="out-label">Genoma dei concetti</div>
            <div class="men-hash" id="men-genhash" style="margin-bottom:8px"></div>
            <div class="men-list">
              <div class="men-row men-head"><span>loc.</span><span>concetto (neurone)</span><span>pesi</span><span>freq</span><span>ind.</span></div>
              <div class="men-rows" id="men-genrows"></div>
            </div>
          </div>
        </div>

        <div class="out-label" style="margin-top:20px">Pulizia della ridondanza <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">la "sovrapposizione": stessi concetti in più neuroni</span></div>
        <div class="row"><button class="btn blue ghost" id="men-clean">Analizza (PCA + dizionario sparso)</button><span id="men-clean-status" style="font-family:var(--font-mono);font-size:.66rem;color:var(--parch-dim)"></span></div>
        <div id="men-clean-out" style="display:none">
          <div class="men-clean-grid">
            <div class="seal-box">
              <div style="color:var(--blue-bright);margin-bottom:4px">PCA · base ortogonale</div>
              <span id="men-pca-line"></span><br>
              <span id="men-pca-bars"></span>
            </div>
            <div class="seal-box">
              <div style="color:var(--violet-bright);margin-bottom:4px">Dizionario sparso · concetti puliti</div>
              <span id="men-sparse-line"></span>
            </div>
          </div>
          <div class="callout-mini" id="men-clean-note"></div>
        </div>

        <div id="men-trace-wrap" style="display:none">
          <div class="out-label" style="margin-top:20px"><span id="men-point"></span><span id="men-verdict-badge"></span></div>
          <div class="box mono strand" id="men-filament" style="color:var(--blue-bright)"></div>

          <div class="out-label">Concetti accesi <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">forte → debole</span></div>
          <div id="men-fired" style="margin-bottom:6px"></div>

          <div class="out-label" style="margin-top:14px">Intervento — uno alla volta <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">spengo un concetto e guardo se la risposta cambia</span></div>
          <div id="men-interv"></div>
          <div class="out-label" style="margin-top:10px">Intervento — a gruppi <span style="text-transform:none;color:var(--parch-faint);font-size:.62rem">stesso concetto in più neuroni, spenti insieme — scioglie la ridondanza</span></div>
          <div id="men-groups"></div>
          <div class="men-fid-set">
            <div class="men-fid"><div class="men-fid-lab">fedeltà singola</div><div class="men-fid-track"><div class="men-fid-fill" id="men-fid1"></div></div><div class="men-fid-val" id="men-fidv1"></div></div>
            <div class="men-fid"><div class="men-fid-lab">fedeltà di gruppo</div><div class="men-fid-track"><div class="men-fid-fill grp" id="men-fid2"></div></div><div class="men-fid-val" id="men-fidv2"></div></div>
            <div class="men-fid"><div class="men-fid-lab">fedeltà graduata</div><div class="men-fid-track"><div class="men-fid-fill grad" id="men-fid3"></div></div><div class="men-fid-val" id="men-fidv3"></div></div>
          </div>

          <div class="out-label" style="margin-top:14px">Sigillo di fedeltà</div>
          <div class="seal-box" id="men-seal"></div>

          <div class="row mt">
            <button class="btn blue" id="men-arc">&#9733; Aggiungi all'archivio</button>
            <button class="btn blue ghost" id="men-json">Salva tracciato .json</button>
          </div>
        </div>
      </div>`;

    const $ = (id) => document.getElementById(id);
    const canvas = $("men-canvas"), cx = canvas.getContext("2d");

    function hexA(hex, a) { const n = parseInt(hex.slice(1), 16); return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")"; }

    function draw() {
      const W = canvas.width, H = canvas.height, N = 48, cw = W / N, ch = H / N;
      cx.clearRect(0, 0, W, H);
      if (net) {
        for (let gx = 0; gx < N; gx++) for (let gy = 0; gy < N; gy++) {
          const x = (gx + 0.5) / N * 2 - 1, y = (gy + 0.5) / N * 2 - 1;
          const pred = R.oneHotDecode(net.feedForward([x, y]));
          cx.fillStyle = hexA(R.COLORS[pred], 0.5);
          const pxv = (x + 1) / 2 * W, pyv = H - (y + 1) / 2 * H;
          cx.fillRect(pxv - cw / 2, pyv - ch / 2, cw + 1, ch + 1);
        }
      }
      cx.strokeStyle = "rgba(159,176,168,0.4)"; cx.lineWidth = 1;
      cx.beginPath(); cx.moveTo(0, H / 2); cx.lineTo(W, H / 2); cx.moveTo(W / 2, 0); cx.lineTo(W / 2, H); cx.stroke();
      if (point) {
        const pxv = (point[0] + 1) / 2 * W, pyv = H - (point[1] + 1) / 2 * H;
        cx.beginPath(); cx.arc(pxv, pyv, 7, 0, 2 * Math.PI); cx.strokeStyle = "#fff"; cx.lineWidth = 2.5; cx.stroke();
        cx.beginPath(); cx.arc(pxv, pyv, 3, 0, 2 * Math.PI); cx.fillStyle = "#fff"; cx.fill();
      }
    }

    function renderGenome() {
      $("men-genhash").textContent = genome.dict.id + " v" + genome.dict.version + " · impronta " + genome.dict.hash.slice(0, 10);
      $("men-genrows").innerHTML = genome.genes.map((g, rank) =>
        `<div class="men-row"><span>#${rank}</span><span class="nm">${APP.esc(g.name)} <span style="color:var(--parch-faint)">n${g.neuron}</span></span><span>[${g.weights[0].toFixed(1)}, ${g.weights[1].toFixed(1)}]</span><span class="fr">${APP.fmt(g.freq)}</span><span class="a">${G.addressOf(rank)}</span></div>`
      ).join("");
    }

    function renderTrace() {
      const [x, y] = point, tl = R.labelOf(x, y);
      $("men-point").textContent = `Tracciato del punto (${x.toFixed(2)}, ${y.toFixed(2)}) · vero: ${tl} · predetto: ${lastTrace.pred}`;
      $("men-verdict-badge").innerHTML = lastTrace.pred === tl ? '<span class="badge ok">&#10003; classificato giusto</span>' : '<span class="badge warn">&#9888; diverso dal vero</span>';

      const SP = G.CTRL.SP;
      if (!lastTrace.fired.length) { $("men-filament").innerHTML = '<span style="color:var(--parch-faint)">(nessun concetto acceso per questo punto)</span>'; }
      else {
        let html = "";
        lastTrace.fired.forEach((f, k) => {
          html += '<span style="color:var(--blue-bright);font-weight:700">' + G.addressOf(genome.dict.index[R.geneKey(f.neuron, f.name)]) + '</span>';
          if (k < lastTrace.fired.length - 1) html += '<span style="color:var(--parch-faint)">' + SP + '</span>';
        });
        $("men-filament").innerHTML = html;
      }

      $("men-fired").innerHTML = lastTrace.fired.map(f =>
        `<div class="men-fired-row"><span class="nm">${APP.esc(f.name)} <span style="color:var(--parch-faint)">n${f.neuron}</span></span><div class="track"><div class="fill" style="width:${(f.act * 100).toFixed(0)}%"></div></div><span class="v">${f.act.toFixed(3)}</span></div>`
      ).join("") || '<span style="color:var(--parch-faint);font-family:var(--font-mono);font-size:.7rem">nessuno</span>';

      $("men-interv").innerHTML = lastInterv.rows.map(r => {
        const col = R.COLORS[r.newPred] || "#888";
        return `<div class="men-iv-row ${r.causal ? "causal" : ""}"><span class="sw" style="background:${col}"></span>spengo <b style="color:var(--parch)">n${r.neuron}</b> (${APP.esc(r.name)}) → <b style="color:${col}">${r.newPred}</b> ${r.causal ? '<span class="flag">&#9888; cambia</span>' : '<span style="color:var(--parch-faint)">= invariata</span>'}<span class="men-drop">&minus;${(r.dropPct * 100).toFixed(0)}% sicurezza</span></div>`;
      }).join("") || '<span style="color:var(--parch-faint);font-family:var(--font-mono);font-size:.7rem">—</span>';

      $("men-groups").innerHTML = lastInterv.groups.map(g => {
        const col = R.COLORS[g.newPred] || "#888";
        return `<div class="men-iv-row ${g.causal ? "causal" : ""}"><span class="sw" style="background:${col}"></span>spengo il gruppo <b style="color:var(--parch)">${APP.esc(g.name)}</b> <span style="color:var(--parch-faint)">(${g.neurons.length})</span> → <b style="color:${col}">${g.newPred}</b> ${g.causal ? '<span class="flag">&#9888; gruppo causale</span>' : '<span style="color:var(--parch-faint)">= invariata</span>'}<span class="men-drop">&minus;${(g.dropPct * 100).toFixed(0)}%</span></div>`;
      }).join("") || '<span style="color:var(--parch-faint);font-family:var(--font-mono);font-size:.7rem">—</span>';

      const setFid = (f, v, val) => { $(f).style.width = (val * 100).toFixed(0) + "%"; $(v).textContent = (val * 100).toFixed(0) + "%"; };
      setFid("men-fid1", "men-fidv1", lastInterv.fidelitySingle);
      setFid("men-fid2", "men-fidv2", lastInterv.fidelityGroup);
      setFid("men-fid3", "men-fidv3", lastInterv.fidelityGraded);

      const segs = G.trace(lastTrace.filament, genome.dict).filter(s => s.role === "gene").map(s => s.text);
      const orig = lastTrace.fired.map(f => R.geneKey(f.neuron, f.name));
      const verified = JSON.stringify(segs) === JSON.stringify(orig);
      $("men-seal").innerHTML = `impronta genoma &nbsp;&nbsp;&nbsp;<b>${lastSeal.genomeHash}</b><br>impronta tracciato <b>${lastSeal.traceHash}</b><br>rilettura col genoma → <b>${verified ? "identica &#10003;" : "NON combacia &#9888;"}</b>`;

      $("men-trace-wrap").style.display = "block";
    }

    function deriveAndRender() {
      lastTrace = R.derive(net, point, genome, G);
      lastInterv = R.intervene(net, point, lastTrace);
      lastSeal = R.seal(genome, lastTrace.filament, G);
      renderTrace();
      draw();
    }

    function buildEntry() {
      const [x, y] = point;
      return {
        mode: "mente",
        title: "ragionamento " + lastTrace.pred + " @(" + x.toFixed(2) + "," + y.toFixed(2) + ")",
        point: [x, y], true_label: R.labelOf(x, y), pred: lastTrace.pred,
        dict_id: genome.dict.id, dict_version: genome.dict.version, dict_hash: genome.dict.hash,
        filament: lastTrace.filament, trace_hash: lastSeal.traceHash,
        fired: lastTrace.fired.map(f => ({ neuron: f.neuron, name: f.name, act: +f.act.toFixed(4) })),
        intervention: { single: lastInterv.rows, groups: lastInterv.groups },
        fidelity: { single: +lastInterv.fidelitySingle.toFixed(3), group: +lastInterv.fidelityGroup.toFixed(3), graded: +lastInterv.fidelityGraded.toFixed(3) },
        chars: lastTrace.fired.length, codons: (lastTrace.filament.length / 3) | 0,
        created: new Date().toISOString()
      };
    }

    $("men-train").addEventListener("click", async () => {
      const hidden = Math.max(3, parseInt($("men-hidden").value, 10) || 8);
      const iters = Math.max(1000, parseInt($("men-iters").value, 10) || 80000);
      const dataN = Math.max(500, parseInt($("men-data").value, 10) || 4000);
      const lr = parseFloat($("men-lr").value) || 0.3;
      APP.showLoader("Addestramento della rete", APP.fmt(iters) + " iterazioni · " + hidden + " nodi nascosti");
      await new Promise(r => setTimeout(r, 40));
      net = new R.NeuralNetwork(2, hidden, 4, lr);
      data = R.makeData(dataN);
      for (let i = 0; i < iters; i++) { const d = data[(Math.random() * data.length) | 0]; net.train([d.x, d.y], R.oneHotEncode(d.label)); }
      let ok = 0; for (const d of data) if (R.oneHotDecode(net.feedForward([d.x, d.y])) === d.label) ok++;
      genome = R.extractGenome(net, data, G);
      point = null; lastTrace = null;
      renderGenome();
      $("men-status").textContent = "accuratezza " + (100 * ok / data.length).toFixed(1) + "% · " + genome.dict.size + " concetti estratti";
      $("men-body").style.display = "block";
      $("men-trace-wrap").style.display = "none";
      draw();
      APP.hideLoader();
    });

    canvas.addEventListener("click", (ev) => {
      if (!net || !genome) return;
      const rect = canvas.getBoundingClientRect();
      const x = (ev.clientX - rect.left) / rect.width * 2 - 1;
      const y = (1 - (ev.clientY - rect.top) / rect.height) * 2 - 1;
      point = [x, y];
      deriveAndRender();
    });

    $("men-arc").addEventListener("click", () => { if (!lastTrace) return; APP.addToArchive(buildEntry()); APP.toast("Tracciato aggiunto all'archivio", "ok"); });
    $("men-json").addEventListener("click", () => { if (!lastTrace) return; APP.saveNative(JSON.stringify(buildEntry(), null, 2), APP.slug("ragionamento-" + lastTrace.pred) + "-" + APP.stamp(), ".json", "Sto salvando il tracciato di ragionamento (" + lastTrace.fired.length + " concetti)"); });

    $("men-clean").addEventListener("click", async () => {
      if (!net || !genome) return;
      APP.showLoader("Analisi della ridondanza", "PCA + dizionario sparso");
      await new Promise(r => setTimeout(r, 40));
      const a = R.analyzeRedundancy(net, data, { M: 2 * net.hiddenSize, steps: 25000 });
      const p = a.pca, sp = a.sparse;
      $("men-pca-line").innerHTML = `<b>${p.H}</b> neuroni → <b style="color:var(--blue-bright)">${p.effDim}</b> concetti indipendenti (95% varianza) · coppie quasi-identiche (|corr|&gt;0.9): <b>${p.pairs}</b> · corr. max <b>${p.maxOff.toFixed(2)}</b>`;
      $("men-pca-bars").innerHTML = '<span style="color:var(--parch-faint)">varianza per componente: </span>' + p.ratio.map((r, i) =>
        `<span style="display:inline-block;width:${Math.max(2, r * 90)}px;height:9px;background:${i < p.effDim ? "var(--blue-bright)" : "rgba(159,176,168,.3)"};border-radius:2px;margin:0 2px;vertical-align:middle" title="${(r * 100).toFixed(0)}%"></span>`
      ).join("");
      $("men-sparse-line").innerHTML = `ogni punto usa in media <b style="color:var(--violet-bright)">${sp.avgActive.toFixed(1)}</b> concetti su <b>${sp.M}</b> · ricostruzione RMSE <b>${sp.rmse.toFixed(3)}</b><br><span style="color:var(--parch-faint)">codici sparsi e puliti: ogni concetto si accende per una cosa sola (monosemanticità)</span>`;
      $("men-clean-note").innerHTML = `&#10003; La ridondanza è <b>misurata</b>: i ${p.H} neuroni codificano solo ${p.effDim} concetti reali. È la sovrapposizione, resa visibile — e PCA/dizionario sparso la sciolgono in una base senza doppioni, dove la fedeltà "singola" torna significativa.`;
      $("men-clean-out").style.display = "block";
      $("men-clean-status").textContent = "fatto · " + p.effDim + " concetti reali su " + p.H + " neuroni";
      APP.hideLoader();
    });
  };
})();