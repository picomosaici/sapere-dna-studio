/* ============================================================
   Sapere-DNA Studio · EMISFERI — motore del cablaggio "π"
   ------------------------------------------------------------
   Il cuore NUOVO del Generatore: i neuroni stanno su GRIGLIE 2D
   e si cablano con la regola di Mirko —
     un neurone si connette a TUTTI quelli la cui DISTANZA sul
     piano cade su un anello di π  (π, 2π, 3π … kπ) entro una
     tolleranza. Su un piano la regola disegna anelli concentrici:
     un mosaico quasi-periodico (π è irrazionale → non si ripete).

   Struttura:
     · EMISFERO SINISTRO  → griglia 2D, cablato al suo interno;
     · EMISFERO DESTRO    → griglia 2D, cablato al suo interno;
     · i due emisferi NON si parlano: nessun filo diretto sx↔dx;
     · CORPO CALLOSO      → griglia 2D; riceve da ENTRAMBI con la
       stessa regola π; è l'unico che ricompone in un vettore.
       → su questo vettore poggia la LENTE (SAE): per derive /
         intervieni / sigilla / sovrapposizione non cambia nulla.

   Niente matrice densa salvata: gli archi si RIGENERANO dalla
   regola; si conservano solo i pochi pesi degli archi attivi.
   Tutto a mano, leggibile, Float32, nessuna dipendenza.

   Espone  window.SapereDNAEmisferi  (module.exports in Node).
   Auto-test:  node emisferi.js
   ============================================================ */
(function (root) {
  "use strict";

  const PI = Math.PI;
  const relu = (x) => x > 0 ? x : 0;

  /* PRNG deterministico (mulberry32): stessa seed → stessi pesi.
     Serve perché il modello dev'essere riproducibile e sigillabile. */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* coordinate (x,y) dei neuroni di una griglia W×H, ordine riga-per-riga.
     L'INDICE del neurone è  y*W + x. */
  function gridCoords(W, H) {
    const c = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) c.push([x, y]);
    return c;
  }

  /* archi "π": coppie (i in A) → (j in B) la cui distanza euclidea
     cade entro `tol` da un multiplo di π (kπ, con 1 ≤ k ≤ maxK).
     selfDirected=true → stesso gruppo: solo i<j ("in avanti"),
     mai l'auto-anello, mai un doppione. Funzione DETERMINISTICA. */
  function piEdges(A, B, opt) {
    opt = opt || {};
    const tol = opt.tol == null ? 0.35 : opt.tol;
    const maxK = opt.maxK == null ? 6 : opt.maxK;
    const self = !!opt.selfDirected;
    const edges = [];
    for (let i = 0; i < A.length; i++) {
      const ax = A[i][0], ay = A[i][1];
      const jStart = self ? i + 1 : 0;
      for (let j = jStart; j < B.length; j++) {
        const dx = B[j][0] - ax, dy = B[j][1] - ay;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const k = Math.round(dist / PI);
        if (k < 1 || k > maxK) continue;
        if (Math.abs(dist - k * PI) <= tol) edges.push({ i: i, j: j, k: k });
      }
    }
    return edges;
  }

  /* strato sparso: out[j] = bias[j] + Σ_{archi i→j} peso * in[i].
     Si conserva UN solo Float32 per arco (più i bias): niente densa. */
  function makeSparse(edges, nIn, nOut, seed) {
    const w = new Float32Array(edges.length);
    const bias = new Float32Array(nOut);
    const avgIn = Math.max(1, edges.length / Math.max(1, nOut));
    const scale = 0.5 / Math.sqrt(avgIn);
    const r = mulberry32(seed >>> 0);
    for (let e = 0; e < edges.length; e++) w[e] = (r() * 2 - 1) * scale;
    return {
      nIn: nIn, nOut: nOut, edges: edges, w: w, bias: bias,
      forward: function (input) {
        const out = new Float32Array(nOut);
        out.set(bias);
        for (let e = 0; e < edges.length; e++) out[edges[e].j] += w[e] * input[edges[e].i];
        return out;
      }
    };
  }

  /* costruisce l'intera struttura dal solo CABLAGGIO (regola + griglie).
     I due emisferi sono spazi-indice separati: qui non si chiama MAI
     piEdges(sinistro, destro) → l'assenza di fili sx↔dx è per costruzione. */
  function build(cfg) {
    cfg = cfg || {};
    const HW = cfg.hemiW || 12, HH = cfg.hemiH || 12;   // griglia di un emisfero
    const CW = cfg.calW  || 8,  CH = cfg.calH  || 8;    // griglia del corpo calloso
    const tol = cfg.tol == null ? 0.35 : cfg.tol;       // larghezza dell'anello π
    const maxK = cfg.maxK == null ? 6 : cfg.maxK;       // quanti anelli (kπ) al massimo

    const hemi = gridCoords(HW, HH);
    const cal  = gridCoords(CW, CH);
    const nH = hemi.length, nC = cal.length;

    const wire = { tol: tol, maxK: maxK };
    const edgesL    = piEdges(hemi, hemi, { tol: tol, maxK: maxK, selfDirected: true });
    const edgesR    = piEdges(hemi, hemi, { tol: tol, maxK: maxK, selfDirected: true });
    const edgesLtoC = piEdges(hemi, cal,  { tol: tol, maxK: maxK });
    const edgesRtoC = piEdges(hemi, cal,  { tol: tol, maxK: maxK });

    return {
      cfg: { HW: HW, HH: HH, CW: CW, CH: CH, tol: tol, maxK: maxK },
      nH: nH, nC: nC, hemi: hemi, cal: cal,
      left:  makeSparse(edgesL,    nH, nH, 0x11),
      right: makeSparse(edgesR,    nH, nH, 0x22),
      lToC:  makeSparse(edgesLtoC, nH, nC, 0x33),
      rToC:  makeSparse(edgesRtoC, nH, nC, 0x44)
    };
  }

  /* passaggio in avanti della struttura:
     ingresso sinistro e destro (uno per emisfero) → ciascun emisfero
     elabora per conto suo → ENTRAMBI scrivono nel calloso, che è
     l'unico a ricomporre. Ritorna anche `cal`: il vettore della lente. */
  function forward(net, leftIn, rightIn) {
    const lAct = net.left.forward(leftIn);
    for (let i = 0; i < lAct.length; i++) lAct[i] = relu(lAct[i]);
    const rAct = net.right.forward(rightIn);
    for (let i = 0; i < rAct.length; i++) rAct[i] = relu(rAct[i]);
    const fromL = net.lToC.forward(lAct);
    const fromR = net.rToC.forward(rAct);
    const cal = new Float32Array(net.nC);
    for (let j = 0; j < net.nC; j++) cal[j] = relu(fromL[j] + fromR[j]);
    return { lAct: lAct, rAct: rAct, cal: cal };
  }

  /* radiografia strutturale: serve a noi (e al test) per VERIFICARE
     che il cablaggio rispetti le regole di Mirko, non solo a fidarsi. */
  function describe(net) {
    const inL = new Int32Array(net.nC), inR = new Int32Array(net.nC);
    net.lToC.edges.forEach((e) => inL[e.j]++);
    net.rToC.edges.forEach((e) => inR[e.j]++);
    let calFromBoth = 0, calFromNone = 0;
    for (let j = 0; j < net.nC; j++) {
      if (inL[j] > 0 && inR[j] > 0) calFromBoth++;
      if (inL[j] === 0 && inR[j] === 0) calFromNone++;
    }
    const hist = {};
    const add = (es) => es.forEach((e) => { hist[e.k] = (hist[e.k] || 0) + 1; });
    add(net.left.edges); add(net.right.edges); add(net.lToC.edges); add(net.rToC.edges);
    const edges = net.left.edges.length + net.right.edges.length +
                  net.lToC.edges.length + net.rToC.edges.length;
    const weights = edges;                       // un peso per arco
    const biases  = net.nH * 2 + net.nC * 2;      // bias dei quattro strati
    return {
      neuroni: { sinistro: net.nH, destro: net.nH, calloso: net.nC, totale: net.nH * 2 + net.nC },
      archi: {
        dentro_sinistro: net.left.edges.length,
        dentro_destro:   net.right.edges.length,
        sinistro_to_calloso: net.lToC.edges.length,
        destro_to_calloso:   net.rToC.edges.length,
        totale: edges
      },
      parametri_allenabili: weights + biases,
      memoria_pesi_KB: +(weights * 4 / 1024).toFixed(1),
      anelli_pi_usati: hist,
      calloso_che_riceve_da_entrambi: calFromBoth + "/" + net.nC,
      calloso_isolato: calFromNone + "/" + net.nC
    };
  }

  const API = {
    PI: PI, mulberry32: mulberry32, gridCoords: gridCoords,
    piEdges: piEdges, makeSparse: makeSparse,
    build: build, forward: forward, describe: describe
  };

  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else root.SapereDNAEmisferi = API;

  /* ============================================================
     AUTO-TEST  (gira con:  node emisferi.js)
     Verifica le regole di Mirko, non le dà per scontate.
     ============================================================ */
  if (typeof require !== "undefined" && require.main === module) {
    const ok = (cond, msg) => console.log((cond ? "  PASS  " : "  FALLITO  ") + msg);
    console.log("\n=== EMISFERI · prova del cablaggio π ===\n");

    const net = build({ hemiW: 12, hemiH: 12, calW: 8, calH: 8, tol: 0.35, maxK: 6 });
    const d = describe(net);
    console.log("Neuroni:", JSON.stringify(d.neuroni));
    console.log("Archi:  ", JSON.stringify(d.archi));
    console.log("Parametri allenabili:", d.parametri_allenabili, " · pesi:", d.memoria_pesi_KB, "KB");
    console.log("Anelli π (k → quanti archi):", JSON.stringify(d.anelli_pi_usati));
    console.log("Calloso che riceve da ENTRAMBI:", d.calloso_che_riceve_da_entrambi);
    console.log("Calloso isolato (da nessuno):  ", d.calloso_isolato);
    console.log("");

    // 1) ogni arco cade davvero su un anello π entro la tolleranza
    let fuoriAnello = 0;
    const checkRing = (coordsA, coordsB, es) => es.forEach((e) => {
      const dx = coordsB[e.j][0] - coordsA[e.i][0], dy = coordsB[e.j][1] - coordsA[e.i][1];
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (Math.abs(dist - e.k * Math.PI) > 0.35 + 1e-9) fuoriAnello++;
    });
    checkRing(net.hemi, net.hemi, net.left.edges);
    checkRing(net.hemi, net.hemi, net.right.edges);
    checkRing(net.hemi, net.cal, net.lToC.edges);
    checkRing(net.hemi, net.cal, net.rToC.edges);
    ok(fuoriAnello === 0, "ogni arco cade su un multiplo di π (kπ) entro la tolleranza");

    // 2) nessun arco interno è un auto-anello e ogni arco va "in avanti" (i<j)
    let autoAnello = 0, indietro = 0;
    [net.left.edges, net.right.edges].forEach((es) => es.forEach((e) => {
      if (e.i === e.j) autoAnello++;
      if (e.i >= e.j) indietro++;
    }));
    ok(autoAnello === 0, "nessun neurone è connesso a se stesso");
    ok(indietro === 0, "dentro un emisfero gli archi vanno solo in avanti (i<j)");

    // 3) i due emisferi sono spazi separati: nessun filo diretto sx↔dx
    //    (per costruzione non esiste alcuno strato fra left e right)
    const haStratoCrociato = ("leftToRight" in net) || ("rightToLeft" in net);
    ok(!haStratoCrociato, "i due emisferi NON si parlano (nessun filo diretto sx↔dx)");

    // 4) il calloso riceve davvero da entrambi
    ok(d.calloso_isolato.split("/")[0] === "0", "nessun neurone del calloso resta isolato");
    const both = parseInt(d.calloso_che_riceve_da_entrambi.split("/")[0], 10);
    ok(both > 0, "esistono neuroni del calloso alimentati da SINISTRA e DESTRA insieme");

    // 5) la regola è deterministica: ricostruendo → stessi archi e stessi pesi
    const net2 = build({ hemiW: 12, hemiH: 12, calW: 8, calH: 8, tol: 0.35, maxK: 6 });
    const sameEdges = net2.left.edges.length === net.left.edges.length &&
                      net2.lToC.edges.length === net.lToC.edges.length;
    let sameW = true;
    for (let e = 0; e < net.left.w.length; e++) if (net.left.w[e] !== net2.left.w[e]) { sameW = false; break; }
    ok(sameEdges && sameW, "il cablaggio si rigenera identico (riproducibile/sigillabile)");

    // 6) il passaggio in avanti produce un vettore-calloso valido (quello della lente)
    const r = mulberry32(7);
    const li = new Float32Array(net.nH), ri = new Float32Array(net.nH);
    for (let i = 0; i < net.nH; i++) { li[i] = r() * 2 - 1; ri[i] = r() * 2 - 1; }
    const f = forward(net, li, ri);
    let finito = true, attivi = 0;
    for (let j = 0; j < f.cal.length; j++) { if (!isFinite(f.cal[j])) finito = false; if (f.cal[j] > 0) attivi++; }
    ok(f.cal.length === net.nC && finito, "il calloso emette un vettore finito di lunghezza " + net.nC + " (input della lente)");
    console.log("        calloso: " + attivi + "/" + net.nC + " neuroni attivi su un ingresso casuale\n");

    console.log("=== fine prova ===\n");
  }

})(typeof self !== "undefined" ? self : this);
