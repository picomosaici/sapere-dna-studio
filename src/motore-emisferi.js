/* ============================================================
   Sapere-DNA Studio · MOTORE A EMISFERI (Cassandra)
   ------------------------------------------------------------
   Transformer causale a parole montato sul cablaggio π, a PIÙ
   LASTRE (blocchi) sovrapposte e con TESTE D'ATTENZIONE DIVISE
   fra i due emisferi. Ogni blocco:
     residuo → nHeads TESTE di attenzione causale (metà «sinistre»,
     metà «destre») → due EMISFERI π slegati, ciascuno alimentato
     dal contesto del PROPRIO gruppo di teste → CORPO CALLOSO
     che ricompone.
   I blocchi intermedi RIPROIETTANO il proprio calloso sul flusso
   residuo (nC→d) e lo sommano, passando il testimone al blocco
   dopo. Solo l'ULTIMO calloso è TERMINALE: lo leggono la testa
   (per scegliere la parola) e la LENTE (SAE).

   TESTE DIVISE SX/DX (la novità di questo formato): le prime
   nHeads/2 teste occupano la prima metà dei canali del contesto,
   le altre la seconda. Ogni emisfero legge S = il proprio
   spicchio dell'INGRESSO del blocco + il contesto del PROPRIO
   gruppo di teste. Dentro un blocco le teste dell'altro gruppo
   NON POSSONO fisicamente toccarlo: spegnere le teste sinistre
   lascia l'emisfero destro identico al bit (provato nei test).
   La corrispondenza testa→emisfero è fisica, non di comodo.
   (Prima la divisione era lo spacchettamento dei canali del
   residuo DOPO la proiezione Wo, che però mescola tutto: quella
   metà «sinistra» non aveva un significato sinistro.)

   Perché S include anche l'ingresso del blocco e non SOLO il
   contesto: l'abbiamo misurato. Con gli emisferi a solo-contesto
   il passaggio-identità del residuo sparisce dal percorso del
   gradiente e il calloso INTERMEDIO muore in addestramento
   (ReLU a 0%: la sbirciata resta cieca). Con S = ingresso +
   contesto l'autostrada del gradiente resta aperta e il calloso
   intermedio resta vivo quanto prima. Onestà: l'ingresso del
   blocco è condiviso dai due emisferi; l'ESCLUSIVA per emisfero
   è il contesto del suo gruppo di teste.

   Il flusso residuo fra i blocchi resta MESCOLATO normalmente
   dalla proiezione Wo (serve all'addestramento). Conseguenza
   onesta: nel blocco TERMINALE il residuo in uscita non serve a
   nessuno (gli emisferi leggono S, non il residuo uscente),
   quindi lì Wo non esiste — niente pesi morti nel modello.

   SUPERVISIONE PROFONDA (misurata su corpus reale, Pinocchio):
   sul testo vero il calloso INTERMEDIO moriva di fame durante
   l'addestramento (ReLU a 0%): al modello conveniva scavalcarlo
   col residuo, e sbirciata e tracciamento profondo restavano
   ciechi. Nessuna cura passiva (bias, scala, leaky) lo salvava.
   La cura che funziona: durante l'addestramento la testa legge
   ANCHE i callosi intermedi (stessa Wout, possibile perché ogni
   calloso è largo nC) con peso LAMBDA_PROF=0.3 — così il 1°
   blocco ha una ragione diretta di restare vivo. Misurato:
   calloso intermedio dal 0% al ~17% vivo, 14 primitivi leggibili
   (prima zero), bypass nel tracciamento profondo da quasi-sempre
   a 1/20, al costo di ~5 punti di perplessità. L'INFERENZA è
   IDENTICA: la parola la sceglie solo il calloso terminale, la
   supervisione profonda esiste solo dentro la perdita.
   Sempre da quelle misure, due inizializzazioni di fabbrica:
   bias del calloso a +0.1 (i neuroni nascono vivi) e riproiezione
   calloso→residuo a scala 2 (più incentivo e gradiente al blocco
   intermedio fin dall'inizio).

   Perché più lastre: con due blocchi il modello può fare
   INDUZIONE (riconoscere uno schema già visto nel contesto e
   continuarlo) e COMPOSIZIONE (concetti astratti costruiti su
   quelli primitivi) — fenomeni che con un blocco solo non
   esistono. La profondità qui serve a sbloccare comportamento
   interpretabile nuovo, non a fare numero.

   Idea-chiave che NON rompe nulla: il calloso TERMINALE è il
   vettore di rappresentazione del modello. Quindi forward()→{n,X2},
   logitsAt(rep) e model.d (= nC) funzionano come prima, e
   derive/intervieni/sigilla/sovrapposizione/qualità girano IDENTICI.
   X2[ultima] È il calloso terminale.

   I due emisferi di ogni blocco partono dalla STESSA distribuzione
   (stessa scala, sorteggio diverso): nessun ruolo deciso a mano →
   la divisione emerge dall'addestramento. Anche i blocchi partono
   con sorteggi diversi (seed per-blocco) e si differenziano da soli.

   SBIRCIATA: forward() espone anche `cals` (il calloso di OGNI
   blocco) e il modello offre callosiAt(ids) → i concetti del 1°
   blocco si possono guardare accanto a quelli del 2°.

   blocks = numero di lastre (default 2).
   nHeads = teste d'attenzione (default 4 → 2 sinistre + 2 destre);
   deve essere PARI e dividere d: se non lo è, viene corretto da
   solo al valore valido più vicino verso il basso.
   Pesi in Float64 (compatibilità salva/carica).
   FORMATO FILE v3: i modelli salvati prima delle teste divise
   (v1/v2) hanno un'architettura diversa e NON si possono
   rileggere — il caricamento lo dichiara onestamente.
   Espone window.SapereDNAMotoreEmisferi (module.exports in Node).
   Auto-test: node motore-emisferi.js
   ============================================================ */
(function (root) {
  "use strict";

  const E = (typeof require !== "undefined") ? require("./emisferi.js") : root.SapereDNAEmisferi;

  const rnd = (a, b) => a + Math.random() * (b - a);
  const matZeros = (r, c) => Array.from({ length: r }, () => new Float64Array(c));
  const randMat = (r, c, s) => { const M = matZeros(r, c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) M[i][j] = rnd(-s, s); return M; };
  const relu = (x) => x > 0 ? x : 0;
  function softmaxInto(arr) { let mx = -Infinity; for (let i = 0; i < arr.length; i++) if (arr[i] > mx) mx = arr[i]; let s = 0; for (let i = 0; i < arr.length; i++) { arr[i] = Math.exp(arr[i] - mx); s += arr[i]; } const inv = 1 / (s || 1); for (let i = 0; i < arr.length; i++) arr[i] *= inv; return arr; }
  function argmax(a) { let mi = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[mi]) mi = i; return mi; }

  // init dei pesi sparsi identico a emisferi.makeSparse (scala 0.5/√gradoMedio),
  // ma con seed scelto da noi → stessa distribuzione, sorteggio diverso per blocco.
  function initSparse(edges, nOut, seed) {
    const w = new Float64Array(edges.length);
    const avgIn = Math.max(1, edges.length / Math.max(1, nOut));
    const scale = 0.5 / Math.sqrt(avgIn);
    const r = E.mulberry32(seed >>> 0);
    for (let e = 0; e < edges.length; e++) w[e] = (r() * 2 - 1) * scale;
    return w;
  }

  class Adam {
    constructor(lr) { this.lr = lr == null ? 0.01 : lr; this.b1 = 0.9; this.b2 = 0.999; this.eps = 1e-8; this.t = 0; this.m = new Map(); this.v = new Map(); }
    step(tensors) {
      this.t++; const bc1 = 1 - Math.pow(this.b1, this.t), bc2 = 1 - Math.pow(this.b2, this.t);
      for (const [key, P, Gd] of tensors) {
        if (!this.m.has(key)) { this.m.set(key, P.map(r => new Float64Array(r.length))); this.v.set(key, P.map(r => new Float64Array(r.length))); }
        const m = this.m.get(key), v = this.v.get(key);
        for (let i = 0; i < P.length; i++) for (let j = 0; j < P[i].length; j++) {
          const g = Gd[i][j];
          m[i][j] = this.b1 * m[i][j] + (1 - this.b1) * g;
          v[i][j] = this.b2 * v[i][j] + (1 - this.b2) * g * g;
          P[i][j] -= this.lr * (m[i][j] / bc1) / (Math.sqrt(v[i][j] / bc2) + this.eps);
        }
      }
    }
  }

  /* ============================================================
     MODELLO · transformer causale a più lastre a emisferi,
     con teste d'attenzione divise sx/dx
     ============================================================ */
  class LMEmisferi {
    constructor(vocab, cfg) {
      cfg = cfg || {};
      this.V = vocab.V;
      this.vocabUnk = vocab.unk == null ? null : vocab.unk;
      this.Lc = cfg.Lc || 16;
      this.dmod = cfg.d || 32; if (this.dmod % 2 !== 0) this.dmod += 1;
      this.dh = this.dmod / 2;
      this.L = Math.max(1, cfg.blocks || 2);              // numero di lastre (default 2)

      // teste d'attenzione: PARI (metà sx, metà dx) e divisore esatto di d.
      // Se il valore chiesto non va bene, scendo al PARI valido più vicino
      // (2 divide sempre d, che è pari per costruzione).
      let nHd = Math.max(2, cfg.nHeads || 4);
      if (nHd % 2 !== 0) nHd -= 1;
      while (nHd > 2 && this.dmod % nHd !== 0) nHd -= 2;
      this.nHeads = nHd;
      this.hd = this.dmod / this.nHeads;                  // larghezza di una testa

      this.wire = {
        hemiW: cfg.hemiW || 12, hemiH: cfg.hemiH || 12,
        calW: cfg.calW || 8, calH: cfg.calH || 8,
        tol: cfg.tol == null ? 0.35 : cfg.tol, maxK: cfg.maxK == null ? 6 : cfg.maxK
      };
      const net = E.build(this.wire);
      this.nH = net.nH; this.nC = net.nC;
      this.edgesL = net.left.edges; this.edgesR = net.right.edges;
      this.edgesLC = net.lToC.edges; this.edgesRC = net.rToC.edges;

      this.d = this.nC;   // la lente e la testa vedono il calloso TERMINALE

      const d = this.dmod, dh = this.dh, nH = this.nH, nC = this.nC, V = this.V;
      const se = 0.3, sw = 1 / Math.sqrt(d), so = 1 / Math.sqrt(nC), sin = 1 / Math.sqrt(dh);
      const sproj = (cfg.projScale != null ? cfg.projScale : 2) / Math.sqrt(nC);   // scala del ramo riproiezione (2 di fabbrica: misurato, dà incentivo e gradiente al calloso intermedio)

      // peso della supervisione profonda (0 = spenta): vedi il commento di testa
      this.lambdaProf = cfg.lambdaProf == null ? 0.3 : cfg.lambdaProf;

      // pesi CONDIVISI
      this.Wemb = randMat(V, d, se);
      this.Wpos = randMat(this.Lc, d, se);
      this.Wout = randMat(nC, V, so); this.bout = matZeros(1, V);

      // una lastra per blocco
      this.blk = [];
      for (let i = 0; i < this.L; i++) {
        const terminal = (i === this.L - 1);
        const b = {
          terminal: terminal,
          Wq: randMat(d, d, sw), Wk: randMat(d, d, sw), Wv: randMat(d, d, sw),
          WinL: randMat(dh, nH, sin), bInL: matZeros(1, nH), WinR: randMat(dh, nH, sin), bInR: matZeros(1, nH),
          wL: [initSparse(this.edgesL, nH, 0x11 + i * 0x1000)], biasL: matZeros(1, nH),
          wR: [initSparse(this.edgesR, nH, 0x22 + i * 0x1000)], biasR: matZeros(1, nH),
          wLC: [initSparse(this.edgesLC, nC, 0x33 + i * 0x1000)], biasLC: matZeros(1, nC),
          wRC: [initSparse(this.edgesRC, nC, 0x44 + i * 0x1000)], biasRC: matZeros(1, nC)
        };
        // i neuroni del calloso NASCONO VIVI (bias +0.1): con la ReLU pura un
        // neurone che parte spento può restare congelato per sempre (misurato)
        for (let m = 0; m < nC; m++) { b.biasLC[0][m] = 0.1; b.biasRC[0][m] = 0.1; }
        // Wo (mescola il contesto sul residuo) e la riproiezione calloso→residuo
        // esistono SOLO nei blocchi non terminali: il blocco terminale non scrive
        // su un residuo che nessuno legge — gli emisferi leggono il contesto.
        if (!terminal) {
          b.Wo = randMat(d, d, sw);
          b.WoutCal = randMat(nC, d, sproj); b.bOutCal = matZeros(1, d);
        }
        this.blk.push(b);
      }
    }

    /* ---- pezzi di un blocco (la matematica già collaudata, per blocco) ---- */
    // Attenzione causale a nHeads teste: ogni testa lavora sulla propria
    // fettina di canali [h·hd, (h+1)·hd). Le prime nHeads/2 teste sono il
    // gruppo SINISTRO (canali 0..dh-1 del contesto), le altre il DESTRO.
    // AttO (contesto mescolato da Wo, per il residuo) si calcola solo dove
    // Wo esiste, cioè nei blocchi non terminali.
    _attnFwd(b, X, n) {
      const d = this.dmod, nHd = this.nHeads, hd = this.hd;
      const Q = matZeros(n, d), K = matZeros(n, d), Vv = matZeros(n, d);
      for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) { let q = 0, kk = 0, vv = 0; for (let j = 0; j < d; j++) { q += X[p][j] * b.Wq[j][k]; kk += X[p][j] * b.Wk[j][k]; vv += X[p][j] * b.Wv[j][k]; } Q[p][k] = q; K[p][k] = kk; Vv[p][k] = vv; }
      const scale = 1 / Math.sqrt(hd);
      const Att = [];                                   // una mappa di attenzione PER TESTA
      for (let h = 0; h < nHd; h++) {
        const off = h * hd, A = matZeros(n, n);
        for (let i = 0; i < n; i++) { const row = A[i]; for (let j = 0; j <= i; j++) { let s = 0; for (let kk = 0; kk < hd; kk++) s += Q[i][off + kk] * K[j][off + kk]; row[j] = s * scale; } softmaxInto(row.subarray(0, i + 1)); }
        Att.push(A);
      }
      const Ctx = matZeros(n, d);                       // contesto: canali a fette per testa
      for (let h = 0; h < nHd; h++) {
        const off = h * hd, A = Att[h];
        for (let i = 0; i < n; i++) for (let kk = 0; kk < hd; kk++) { const k = off + kk; let s = 0; for (let j = 0; j <= i; j++) s += A[i][j] * Vv[j][k]; Ctx[i][k] = s; }
      }
      let AttO = null;
      if (b.Wo) {
        AttO = matZeros(n, d);
        for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) { let s = 0; for (let j = 0; j < d; j++) s += Ctx[i][j] * b.Wo[j][k]; AttO[i][k] = s; }
      }
      return { AttO, Q, K, V: Vv, Att, Ctx };
    }

    // All'indietro: il gradiente degli emisferi (dS) si sdoppia sulle due
    // strade di S = rIn + Ctx — il passaggio-identità verso il residuo
    // entrante (l'autostrada che tiene vivo il calloso intermedio) e il
    // contesto delle teste. A questo si somma, nei blocchi non terminali,
    // il ramo del residuo via Wo (dX1). Poi si torna per testa attraverso
    // la softmax di ciascuna, fino a Wq/Wk/Wv e al residuo entrante.
    _attnBwd(b, X, cache, dX1, dS, gb) {
      const d = this.dmod, n = X.length, nHd = this.nHeads, hd = this.hd;
      const dX = matZeros(n, d);
      const dCtx = matZeros(n, d);
      if (dX1) {
        for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) dX[p][k] += dX1[p][k];   // residuo passante (X1 = X + AttO)
        for (let p = 0; p < n; p++) for (let j = 0; j < d; j++) { let s = 0; for (let k = 0; k < d; k++) { gb.Wo[j][k] += cache.Ctx[p][j] * dX1[p][k]; s += b.Wo[j][k] * dX1[p][k]; } dCtx[p][j] += s; }
      }
      if (dS) for (let p = 0; p < n; p++) for (let j = 0; j < d; j++) { dCtx[p][j] += dS[p][j]; dX[p][j] += dS[p][j]; }   // S = rIn + Ctx: il gradiente si sdoppia
      const dQ = matZeros(n, d), dK = matZeros(n, d), dV = matZeros(n, d);
      const scale = 1 / Math.sqrt(hd);
      for (let h = 0; h < nHd; h++) {
        const off = h * hd, A = cache.Att[h];
        const dAtt = matZeros(n, n);
        for (let i = 0; i < n; i++) for (let kk = 0; kk < hd; kk++) { const k = off + kk, gg = dCtx[i][k]; if (gg === 0) continue; for (let j = 0; j <= i; j++) { dAtt[i][j] += gg * cache.V[j][k]; dV[j][k] += gg * A[i][j]; } }
        for (let i = 0; i < n; i++) { let dot = 0; for (let j = 0; j <= i; j++) dot += dAtt[i][j] * A[i][j]; for (let j = 0; j <= i; j++) { const ds = A[i][j] * (dAtt[i][j] - dot) * scale; if (ds === 0) continue; for (let kk = 0; kk < hd; kk++) { const k = off + kk; dQ[i][k] += ds * cache.K[j][k]; dK[j][k] += ds * cache.Q[i][k]; } } }
      }
      for (let p = 0; p < n; p++) for (let j = 0; j < d; j++) { let sx = 0; for (let k = 0; k < d; k++) { gb.Wq[j][k] += X[p][j] * dQ[p][k]; gb.Wk[j][k] += X[p][j] * dK[p][k]; gb.Wv[j][k] += X[p][j] * dV[p][k]; sx += b.Wq[j][k] * dQ[p][k] + b.Wk[j][k] * dK[p][k] + b.Wv[j][k] * dV[p][k]; } dX[p][j] += sx; }
      return dX;
    }

    // Gli emisferi leggono S = INGRESSO del blocco + CONTESTO delle teste:
    // la prima metà dei canali (teste sinistre) va all'emisfero sinistro,
    // la seconda (teste destre) al destro. L'ingresso è condiviso; il
    // contesto del gruppo è l'esclusiva di ciascun emisfero.
    _mlpFwd(b, S, n) {
      const dh = this.dh, nH = this.nH, nC = this.nC;
      const preLin = matZeros(n, nH), hLin = matZeros(n, nH), preRin = matZeros(n, nH), hRin = matZeros(n, nH);
      const preLout = matZeros(n, nH), hLout = matZeros(n, nH), preRout = matZeros(n, nH), hRout = matZeros(n, nH);
      const preC = matZeros(n, nC), cal = matZeros(n, nC);
      const wL = b.wL[0], wR = b.wR[0], wLC = b.wLC[0], wRC = b.wRC[0];
      const eL = this.edgesL, eR = this.edgesR, eLC = this.edgesLC, eRC = this.edgesRC;
      for (let p = 0; p < n; p++) {
        for (let a = 0; a < nH; a++) { let sL = b.bInL[0][a], sR = b.bInR[0][a]; for (let k = 0; k < dh; k++) { sL += S[p][k] * b.WinL[k][a]; sR += S[p][dh + k] * b.WinR[k][a]; } preLin[p][a] = sL; hLin[p][a] = relu(sL); preRin[p][a] = sR; hRin[p][a] = relu(sR); }
        for (let bb = 0; bb < nH; bb++) { preLout[p][bb] = b.biasL[0][bb]; preRout[p][bb] = b.biasR[0][bb]; }
        for (let e = 0; e < eL.length; e++) preLout[p][eL[e].j] += wL[e] * hLin[p][eL[e].i];
        for (let e = 0; e < eR.length; e++) preRout[p][eR[e].j] += wR[e] * hRin[p][eR[e].i];
        for (let bb = 0; bb < nH; bb++) { hLout[p][bb] = relu(preLout[p][bb]); hRout[p][bb] = relu(preRout[p][bb]); }
        for (let m = 0; m < nC; m++) preC[p][m] = b.biasLC[0][m] + b.biasRC[0][m];
        for (let e = 0; e < eLC.length; e++) preC[p][eLC[e].j] += wLC[e] * hLout[p][eLC[e].i];
        for (let e = 0; e < eRC.length; e++) preC[p][eRC[e].j] += wRC[e] * hRout[p][eRC[e].i];
        for (let m = 0; m < nC; m++) cal[p][m] = relu(preC[p][m]);
      }
      return { cal, preLin, hLin, preRin, hRin, preLout, hLout, preRout, hRout, preC };
    }

    _mlpBwd(b, S, cache, dCal, gb) {
      const d = this.dmod, dh = this.dh, nH = this.nH, nC = this.nC, n = S.length;
      const dS = matZeros(n, d);   // gradiente sul CONTESTO delle teste
      const wL = b.wL[0], wR = b.wR[0], wLC = b.wLC[0], wRC = b.wRC[0];
      const eL = this.edgesL, eR = this.edgesR, eLC = this.edgesLC, eRC = this.edgesRC;
      for (let p = 0; p < n; p++) {
        const dPreC = new Float64Array(nC);
        for (let m = 0; m < nC; m++) dPreC[m] = cache.preC[p][m] > 0 ? dCal[p][m] : 0;
        const dHLout = new Float64Array(nH), dHRout = new Float64Array(nH);
        for (let m = 0; m < nC; m++) { gb.biasLC[0][m] += dPreC[m]; gb.biasRC[0][m] += dPreC[m]; }
        for (let e = 0; e < eLC.length; e++) { const i = eLC[e].i, m = eLC[e].j; gb.wLC[0][e] += cache.hLout[p][i] * dPreC[m]; dHLout[i] += wLC[e] * dPreC[m]; }
        for (let e = 0; e < eRC.length; e++) { const i = eRC[e].i, m = eRC[e].j; gb.wRC[0][e] += cache.hRout[p][i] * dPreC[m]; dHRout[i] += wRC[e] * dPreC[m]; }
        const dPreLout = new Float64Array(nH), dPreRout = new Float64Array(nH);
        for (let bb = 0; bb < nH; bb++) { dPreLout[bb] = cache.preLout[p][bb] > 0 ? dHLout[bb] : 0; dPreRout[bb] = cache.preRout[p][bb] > 0 ? dHRout[bb] : 0; }
        const dHLin = new Float64Array(nH), dHRin = new Float64Array(nH);
        for (let bb = 0; bb < nH; bb++) { gb.biasL[0][bb] += dPreLout[bb]; gb.biasR[0][bb] += dPreRout[bb]; }
        for (let e = 0; e < eL.length; e++) { const i = eL[e].i, bb = eL[e].j; gb.wL[0][e] += cache.hLin[p][i] * dPreLout[bb]; dHLin[i] += wL[e] * dPreLout[bb]; }
        for (let e = 0; e < eR.length; e++) { const i = eR[e].i, bb = eR[e].j; gb.wR[0][e] += cache.hRin[p][i] * dPreRout[bb]; dHRin[i] += wR[e] * dPreRout[bb]; }
        for (let a = 0; a < nH; a++) {
          const dpl = cache.preLin[p][a] > 0 ? dHLin[a] : 0;
          const dpr = cache.preRin[p][a] > 0 ? dHRin[a] : 0;
          gb.bInL[0][a] += dpl; gb.bInR[0][a] += dpr;
          for (let k = 0; k < dh; k++) { gb.WinL[k][a] += S[p][k] * dpl; dS[p][k] += b.WinL[k][a] * dpl; gb.WinR[k][a] += S[p][dh + k] * dpr; dS[p][dh + k] += b.WinR[k][a] * dpr; }
        }
      }
      return dS;
    }

    _zeroBlockGrad(b) {
      const d = this.dmod, dh = this.dh, nH = this.nH, nC = this.nC;
      const g = {
        Wq: matZeros(d, d), Wk: matZeros(d, d), Wv: matZeros(d, d),
        WinL: matZeros(dh, nH), bInL: matZeros(1, nH), WinR: matZeros(dh, nH), bInR: matZeros(1, nH),
        wL: [new Float64Array(b.wL[0].length)], biasL: matZeros(1, nH),
        wR: [new Float64Array(b.wR[0].length)], biasR: matZeros(1, nH),
        wLC: [new Float64Array(b.wLC[0].length)], biasLC: matZeros(1, nC),
        wRC: [new Float64Array(b.wRC[0].length)], biasRC: matZeros(1, nC)
      };
      if (!b.terminal) { g.Wo = matZeros(d, d); g.WoutCal = matZeros(nC, d); g.bOutCal = matZeros(1, d); }
      return g;
    }

    /* ---- forward su una finestra: attraversa tutte le lastre ---- */
    forward(ids, keep) {
      const n = ids.length, d = this.dmod, nC = this.nC, L = this.L;
      let r = matZeros(n, d);
      for (let p = 0; p < n; p++) { const e = this.Wemb[ids[p]], po = this.Wpos[p]; for (let k = 0; k < d; k++) r[p][k] = e[k] + po[k]; }
      const cals = [];
      const caches = keep ? [] : null;
      let X2 = null;
      for (let i = 0; i < L; i++) {
        const b = this.blk[i];
        const aF = this._attnFwd(b, r, n);
        const S = matZeros(n, d);                       // sorgente degli emisferi: ingresso del blocco + contesto del proprio gruppo
        for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) S[p][k] = r[p][k] + aF.Ctx[p][k];
        const mF = this._mlpFwd(b, S, n);
        cals.push(mF.cal);
        if (keep) caches.push({ rIn: r, attn: aF, S: S, mlp: mF });
        if (b.terminal) { X2 = mF.cal; break; }
        const rOut = matZeros(n, d);                    // residuo: X1 = r + AttO, poi riproiezione del calloso
        for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) { let s = b.bOutCal[0][k]; for (let m = 0; m < nC; m++) s += mF.cal[p][m] * b.WoutCal[m][k]; rOut[p][k] = r[p][k] + aF.AttO[p][k] + s; }
        r = rOut;
      }
      const out = { n: n, X2: X2, cals: cals };
      if (keep) Object.assign(out, { caches: caches, ids: ids });
      return out;
    }

    // calloso dell'ULTIMA posizione per OGNI blocco (sbirciata sui livelli di concetti)
    callosiAt(ids) { const f = this.forward(ids); return f.cals.map(c => Float64Array.from(c[f.n - 1])); }

    // forward CONTROFATTUALE: identico a forward(), ma al blocco `atBlock` sostituisce
    // il calloso dell'ULTIMA posizione con `calLastNew` (vettore di nC valori) PRIMA
    // della riproiezione, poi prosegue attraverso i blocchi a valle e restituisce il
    // calloso TERMINALE dell'ultima posizione. È il motore del tracciamento causale
    // profondo: spegnere un concetto «primitivo» di un blocco intermedio e vedere la
    // parola finale cambiare passando per il blocco dopo.
    // NB: deve restare ALLINEATO a forward() — se cambia uno, va cambiato anche l'altro.
    forwardFromCal(ids, atBlock, calLastNew) {
      const n = ids.length, d = this.dmod, nC = this.nC, L = this.L;
      let r = matZeros(n, d);
      for (let p = 0; p < n; p++) { const e = this.Wemb[ids[p]], po = this.Wpos[p]; for (let k = 0; k < d; k++) r[p][k] = e[k] + po[k]; }
      let X2 = null;
      for (let i = 0; i < L; i++) {
        const b = this.blk[i];
        const aF = this._attnFwd(b, r, n);
        const S = matZeros(n, d);
        for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) S[p][k] = r[p][k] + aF.Ctx[p][k];
        const mF = this._mlpFwd(b, S, n);
        let cal = mF.cal;
        if (i === atBlock && calLastNew) {          // inietto il calloso controfattuale SOLO all'ultima posizione
          cal = mF.cal.slice();                     // copia di righe-riferimento: cambio solo l'ultima
          cal[n - 1] = Float64Array.from(calLastNew);
        }
        if (b.terminal) { X2 = cal; break; }
        const rOut = matZeros(n, d);
        for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) { let s = b.bOutCal[0][k]; for (let m = 0; m < nC; m++) s += cal[p][m] * b.WoutCal[m][k]; rOut[p][k] = r[p][k] + aF.AttO[p][k] + s; }
        r = rOut;
      }
      return Float64Array.from(X2[n - 1]);
    }

    logitsAt(rep) { const V = this.V, lg = new Float64Array(V); for (let c = 0; c < V; c++) { let s = this.bout[0][c]; for (let m = 0; m < this.nC; m++) s += rep[m] * this.Wout[m][c]; lg[c] = s; } return lg; }
    nextProbs(ids) { const f = this.forward(ids); return softmaxInto(this.logitsAt(f.X2[f.n - 1])); }

    lossOnly(ids) {
      const f = this.forward(ids); const T = ids.length - 1; if (T < 1) return 0;
      let loss = 0;
      for (let p = 0; p < T; p++) { const pr = softmaxInto(this.logitsAt(f.X2[p])); loss += -Math.log(Math.max(1e-12, pr[ids[p + 1]])); }
      // supervisione profonda: anche i callosi INTERMEDI provano a predire la
      // parola dopo (stessa testa Wout, peso lambdaProf) — tiene vivo il 1° blocco
      if (this.lambdaProf > 0) for (let i = 0; i < this.L - 1; i++) {
        for (let p = 0; p < T; p++) { const pr = softmaxInto(this.logitsAt(f.cals[i][p])); loss += this.lambdaProf * -Math.log(Math.max(1e-12, pr[ids[p + 1]])); }
      }
      return loss / T;
    }

    lossAndGrads(ids) {
      const n = ids.length, d = this.dmod, nC = this.nC, V = this.V, L = this.L;
      const f = this.forward(ids, true);
      const T = n - 1, invT = 1 / Math.max(1, T);
      let loss = 0;

      const g = { Wemb: matZeros(V, d), Wpos: matZeros(this.Lc, d), Wout: matZeros(nC, V), bout: matZeros(1, V), blk: [] };
      for (let i = 0; i < L; i++) g.blk.push(this._zeroBlockGrad(this.blk[i]));

      // testa → gradiente sul calloso TERMINALE
      const dCalT = matZeros(n, nC);
      for (let p = 0; p < T; p++) {
        const probs = softmaxInto(this.logitsAt(f.X2[p])); const tgt = ids[p + 1];
        loss += -Math.log(Math.max(1e-12, probs[tgt]));
        const dl = probs; dl[tgt] -= 1; for (let c = 0; c < V; c++) dl[c] *= invT;
        for (let c = 0; c < V; c++) { const gc = dl[c]; if (gc === 0) continue; g.bout[0][c] += gc; for (let m = 0; m < nC; m++) g.Wout[m][c] += f.X2[p][m] * gc; }
        for (let m = 0; m < nC; m++) { let s = 0; for (let c = 0; c < V; c++) s += this.Wout[m][c] * dl[c]; dCalT[p][m] += s; }
      }
      loss *= invT;

      // backprop a ritroso attraverso le lastre
      let drOut = null;   // gradiente sul residuo USCENTE dal blocco (per i non-terminali)
      for (let i = L - 1; i >= 0; i--) {
        const b = this.blk[i], gb = g.blk[i], cache = f.caches[i];
        let dCal, dX1 = null;
        if (b.terminal) {
          dCal = dCalT;
        } else {
          // rOut = (rIn + AttO) + m(calloso) → il residuo torna a dX1, e m torna al calloso
          dX1 = matZeros(n, d);
          for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) dX1[p][k] = drOut[p][k];
          dCal = matZeros(n, nC);
          for (let p = 0; p < n; p++) {
            for (let k = 0; k < d; k++) gb.bOutCal[0][k] += drOut[p][k];
            for (let m = 0; m < nC; m++) { let s = 0; for (let k = 0; k < d; k++) { gb.WoutCal[m][k] += cache.mlp.cal[p][m] * drOut[p][k]; s += b.WoutCal[m][k] * drOut[p][k]; } dCal[p][m] = s; }
          }
          // supervisione profonda: la testa legge anche QUESTO calloso (peso lambdaProf).
          // Spinge gradiente su Wout/bout e direttamente sul calloso intermedio.
          if (this.lambdaProf > 0) for (let p = 0; p < T; p++) {
            const cal = cache.mlp.cal[p];
            const probs = softmaxInto(this.logitsAt(cal)); const tgt = ids[p + 1];
            loss += this.lambdaProf * invT * -Math.log(Math.max(1e-12, probs[tgt]));
            const dl = probs; dl[tgt] -= 1; for (let c = 0; c < V; c++) dl[c] *= this.lambdaProf * invT;
            for (let c = 0; c < V; c++) { const gc = dl[c]; if (gc === 0) continue; g.bout[0][c] += gc; for (let m = 0; m < nC; m++) g.Wout[m][c] += cal[m] * gc; }
            for (let m = 0; m < nC; m++) { let s = 0; for (let c = 0; c < V; c++) s += this.Wout[m][c] * dl[c]; dCal[p][m] += s; }
          }
        }
        const dS = this._mlpBwd(b, cache.S, cache.mlp, dCal, gb);          // gradiente degli emisferi sulla sorgente S = rIn + Ctx
        drOut = this._attnBwd(b, cache.rIn, cache.attn, dX1, dS, gb);      // → gradiente sul residuo entrante = uscente del blocco precedente
      }
      // drOut = gradiente su (embedding + posizione)
      for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) { g.Wemb[ids[p]][k] += drOut[p][k]; g.Wpos[p][k] += drOut[p][k]; }

      return { loss: loss, grads: g };
    }

    tensors(g) {
      const out = [
        ["Wemb", this.Wemb, g.Wemb], ["Wpos", this.Wpos, g.Wpos],
        ["Wout", this.Wout, g.Wout], ["bout", this.bout, g.bout]
      ];
      for (let i = 0; i < this.L; i++) {
        const b = this.blk[i], gb = g.blk[i], P = "b" + i + ".";
        out.push(
          [P + "Wq", b.Wq, gb.Wq], [P + "Wk", b.Wk, gb.Wk], [P + "Wv", b.Wv, gb.Wv],
          [P + "WinL", b.WinL, gb.WinL], [P + "bInL", b.bInL, gb.bInL], [P + "WinR", b.WinR, gb.WinR], [P + "bInR", b.bInR, gb.bInR],
          [P + "wL", b.wL, gb.wL], [P + "biasL", b.biasL, gb.biasL], [P + "wR", b.wR, gb.wR], [P + "biasR", b.biasR, gb.biasR],
          [P + "wLC", b.wLC, gb.wLC], [P + "biasLC", b.biasLC, gb.biasLC], [P + "wRC", b.wRC, gb.wRC], [P + "biasRC", b.biasRC, gb.biasRC]
        );
        if (!b.terminal) out.push([P + "Wo", b.Wo, gb.Wo], [P + "WoutCal", b.WoutCal, gb.WoutCal], [P + "bOutCal", b.bOutCal, gb.bOutCal]);
      }
      return out;
    }

    trainStep(ids, opt) { const r = this.lossAndGrads(ids); opt.step(this.tensors(r.grads)); return r.loss; }
  }

  function train(model, ids, opts) {
    opts = opts || {};
    const steps = opts.steps || 4000, Lc = model.Lc;
    const L = model.L || 1;
    const lr = (opts.lr || 0.01) / L;   // passo scalato con la profondità: a passo pieno i modelli profondi
                                        // vanno in stallo (die-off delle ReLU) — più lastre → passo più piccolo
    const opt = new Adam(lr); let loss = 0, seen = 0;
    if (ids.length < 2) return { loss: 0, perplexity: Infinity };
    for (let t = 0; t < steps; t++) {
      const maxStart = Math.max(1, ids.length - Lc);
      const start = (Math.random() * maxStart) | 0;
      const win = ids.slice(start, start + Lc);
      if (win.length < 2) continue;
      loss = (loss * seen + model.trainStep(win, opt)) / (seen + 1); seen++;
      if (opts.onProgress && (t % 200 === 0)) opts.onProgress(t / steps, loss);
    }
    return { loss: loss, perplexity: Math.exp(loss) };
  }

  /* ---- persistenza: salva/carica i pesi (il cablaggio π si rigenera dal cfg) ----
     FORMATO v3 (teste divise sx/dx). I file v1/v2 appartengono all'architettura
     precedente (emisferi che leggevano il residuo, Wo anche nel blocco terminale):
     rileggerli qui produrrebbe un modello DIVERSO da quello salvato — quindi il
     caricamento li rifiuta con un messaggio onesto, invece di fingere che vada bene. */
  const BKEYS = ["Wq", "Wk", "Wv", "WinL", "bInL", "WinR", "bInR", "wL", "biasL", "wR", "biasR", "wLC", "biasLC", "wRC", "biasRC"];
  function serialize(model, vocab) {
    const m2 = (M) => M.map(r => Array.from(r));
    const blocks = model.blk.map((b) => {
      const o = {}; for (const key of BKEYS) o[key] = m2(b[key]);
      if (!b.terminal) { o.Wo = m2(b.Wo); o.WoutCal = m2(b.WoutCal); o.bOutCal = m2(b.bOutCal); }
      return o;
    });
    return {
      format: "sapere-dna-cassandra", version: 3,
      cfg: { V: model.V, Lc: model.Lc, d: model.dmod, blocks: model.L, nHeads: model.nHeads, lambdaProf: model.lambdaProf, hemiW: model.wire.hemiW, hemiH: model.wire.hemiH, calW: model.wire.calW, calH: model.wire.calH, tol: model.wire.tol, maxK: model.wire.maxK },
      vocab: { V: vocab.V, cap: vocab.cap, unk: vocab.unk, dict_id: vocab.dict_id, dict_version: vocab.dict_version, dict_hash: vocab.dict_hash },
      shared: { Wemb: m2(model.Wemb), Wpos: m2(model.Wpos), Wout: m2(model.Wout), bout: m2(model.bout) },
      blockWeights: blocks
    };
  }
  function deserialize(obj, vocab) {
    if ((obj.version || 1) < 3) {
      throw new Error(
        "Questo file è di una Cassandra precedente (formato v" + (obj.version || 1) + "): " +
        "da allora l'architettura è cambiata — le teste d'attenzione sono divise sx/dx e gli emisferi " +
        "leggono il contesto delle proprie teste, non più il residuo. Caricare i vecchi pesi qui " +
        "produrrebbe un modello diverso da quello che avevi salvato. Riaddestra il modello (pochi minuti) e risalvalo."
      );
    }
    const F = (a) => a.map(r => Float64Array.from(r));
    const m = new LMEmisferi(vocab, obj.cfg);
    m.Wemb = F(obj.shared.Wemb); m.Wpos = F(obj.shared.Wpos); m.Wout = F(obj.shared.Wout); m.bout = F(obj.shared.bout);
    for (let i = 0; i < m.L; i++) {
      const b = m.blk[i], o = obj.blockWeights[i];
      for (const key of BKEYS) b[key] = F(o[key]);
      if (!b.terminal) { b.Wo = F(o.Wo); b.WoutCal = F(o.WoutCal); b.bOutCal = F(o.bOutCal); }
    }
    return m;
  }

  const API = { LMEmisferi, Adam, train, serialize, deserialize, argmax };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else root.SapereDNAMotoreEmisferi = API;

  /* ============================================================
     AUTO-TEST  (node motore-emisferi.js)
     ============================================================ */
  if (typeof require !== "undefined" && require.main === module) {
    const Gen = require("./generatore.js");
    const ok = (c, m) => console.log((c ? "  PASS  " : "  FALLITO  ") + m);
    console.log("\n=== MOTORE A EMISFERI · teste divise sx/dx · prove ===\n");

    // 1) CONTROLLO NUMERICO DEL GRADIENTE su modellini a 2 e 3 blocchi (4 teste)
    function gradCheck(L) {
      const vsmall = { V: 7, unk: null, words: ["a", "b", "c", "d", "e", "f", "g"] };
      const mg = new LMEmisferi(vsmall, { d: 8, Lc: 6, blocks: L, nHeads: 4, hemiW: 4, hemiH: 4, calW: 3, calH: 3, tol: 0.5, maxK: 2 });
      const win = [1, 3, 0, 5, 2];
      const r0 = mg.lossAndGrads(win); const l0 = mg.lossOnly(win);
      const eps = 1e-5; let maxRel = 0, checked = 0, gomiti = 0;
      for (const [key, P, G] of mg.tensors(r0.grads)) {
        for (let i = 0; i < P.length; i++) for (let j = 0; j < P[i].length; j++) {
          if (((i * 131 + j * 17) % 5) !== 0) continue;
          const save = P[i][j];
          P[i][j] = save + eps; const lp = mg.lossOnly(win);
          P[i][j] = save - eps; const lm = mg.lossOnly(win);
          P[i][j] = save;
          const sp = (lp - l0) / eps, sm = (l0 - lm) / eps;
          if (Math.abs(sp - sm) > 0.0025 * (Math.abs(sp) + Math.abs(sm)) + 1e-6) { gomiti++; continue; }
          const num = (lp - lm) / (2 * eps), ana = G[i][j];
          if (Math.abs(num) < 1e-6 && Math.abs(ana) < 1e-6) continue;   // gradiente trascurabile: l'errore relativo non ha senso
          const rel = Math.abs(num - ana) / (Math.abs(num) + Math.abs(ana) + 1e-9);
          if (rel > maxRel) maxRel = rel; checked++;
        }
      }
      return { maxRel, checked, gomiti, params: mg.tensors(r0.grads).length };
    }
    // sentinella della supervisione profonda: la perdita deve essere IDENTICA
    // fra lossOnly e lossAndGrads (se divergono, il gradiente mente)
    {
      const vs = { V: 7, unk: null, words: ["a", "b", "c", "d", "e", "f", "g"] };
      const ms = new LMEmisferi(vs, { d: 8, Lc: 6, blocks: 3, nHeads: 4, hemiW: 4, hemiH: 4, calW: 3, calH: 3, tol: 0.5, maxK: 2 });
      const w = [1, 3, 0, 5, 2];
      const dlt = Math.abs(ms.lossOnly(w) - ms.lossAndGrads(w).loss);
      ok(dlt < 1e-12, "supervisione profonda: perdita identica fra le due strade (scarto " + dlt.toExponential(1) + ")");
    }
    const g2 = gradCheck(2), g3 = gradCheck(3);
    ok(g2.maxRel < 1.2e-2, "2 blocchi · 4 teste · gradiente corretto (errore max " + g2.maxRel.toExponential(2) + " su " + g2.checked + " parametri lisci · " + g2.params + " tensori)");
    ok(g3.maxRel < 1.2e-2, "3 blocchi · 4 teste · gradiente corretto (errore max " + g3.maxRel.toExponential(2) + " su " + g3.checked + " parametri lisci)");

    // 2) CORRISPONDENZA ESATTA testa→emisfero: azzero il contesto delle teste
    //    SINISTRE all'ultima posizione → l'emisfero DESTRO non si accorge di nulla.
    {
      const W0 = ["a", "b", "c", "d", "e"];
      const v0 = { V: 5, unk: null, words: W0 };
      const m0 = new LMEmisferi(v0, { d: 8, Lc: 6, blocks: 1, nHeads: 4, hemiW: 4, hemiH: 4, calW: 3, calH: 3 });
      const ctx0 = [0, 2, 1, 3];
      const f0 = m0.forward(ctx0, true);
      const S0 = f0.caches[0].S, Ctx = f0.caches[0].attn.Ctx;
      const SAbl = S0.map(r => Float64Array.from(r));
      for (let k = 0; k < m0.dh; k++) SAbl[ctx0.length - 1][k] -= Ctx[ctx0.length - 1][k];   // spengo il contesto delle teste SINISTRE all'ultima posizione
      const mA = m0._mlpFwd(m0.blk[0], S0, ctx0.length);
      const mB = m0._mlpFwd(m0.blk[0], SAbl, ctx0.length);
      let dR = 0, dL = 0;
      const last = ctx0.length - 1;
      for (let a = 0; a < m0.nH; a++) { dR = Math.max(dR, Math.abs(mA.hRin[last][a] - mB.hRin[last][a])); dL = Math.max(dL, Math.abs(mA.hLin[last][a] - mB.hLin[last][a])); }
      ok(dR === 0 && dL > 0, "teste sinistre spente → emisfero destro IDENTICO (scarto " + dR + ") · sinistro cambia (scarto max " + dL.toExponential(1) + ")");
    }

    // 3) ADDESTRAMENTO a 2 blocchi · 4 teste su corpus-giocattolo
    const W = ["il", "gatto", "cane", "dorme", "corre", "sul", "tappeto", "tronco", "."];
    const vocab = { V: W.length, unk: null, words: W, cap: 1500, dict_id: "x", dict_version: "0", dict_hash: "h" };
    const idx = (w) => W.indexOf(w); const ids = [];
    for (let s = 0; s < 700; s++) { const so = Math.random() < .5 ? "gatto" : "cane"; const ve = Math.random() < .5 ? "dorme" : "corre"; const dv = Math.random() < .5 ? "tappeto" : "tronco";["il", so, ve, "sul", "il", dv, "."].forEach(w => ids.push(idx(w))); }
    const model = new LMEmisferi(vocab, { d: 24, Lc: 8, blocks: 2, nHeads: 4, hemiW: 8, hemiH: 8, calW: 6, calH: 6 });
    console.log("\nmodello: " + model.L + " blocchi · d(residuo) " + model.dmod + " · " + model.nHeads + " teste (larghe " + model.hd + ") · nH/emisfero " + model.nH + " · calloso(=d lente) " + model.nC);
    const ppl0 = Gen.perplexity(model, ids);
    train(model, ids, { steps: 3000, lr: 0.01 });
    const ppl1 = Gen.perplexity(model, ids);
    console.log("perplessità: prima " + ppl0.toFixed(2) + " → dopo " + ppl1.toFixed(2) + "  (vocabolario " + W.length + ")");
    ok(ppl1 < ppl0 * 0.6, "il modello a 2 blocchi · 4 teste impara (perplessità scesa nettamente)");

    // 4) generazione + lente sul calloso terminale
    const gen = Gen.generate(model, [idx("il"), idx("gatto")], { maxLen: 8, temperature: 0.5 });
    console.log("genera: \"" + gen.ids.map(i => W[i]).join(" ") + "\"");
    const A = Gen.lastPosMatrix(model, ids, { N: 400 });
    ok(A[0].length === model.nC, "la lente legge il calloso TERMINALE (dim " + model.nC + ")");
    const sae = Gen.trainSAE(A, { M: 24, k: 6, steps: 2500 });
    const q = Gen.saeQuality(model, sae, ids, { samples: 150 });
    console.log("SAE sul calloso terminale → perdita recuperata " + (q.lossRecovered * 100).toFixed(0) + "% · feature morte " + sae.dead + "/" + sae.M);
    ok(q.lossRecovered >= 0 && q.lossRecovered <= 1, "qualità (perdita recuperata) ben definita anche a 2 blocchi");

    // 5) SBIRCIATA: il calloso di ogni blocco è accessibile, i due livelli
    //    DIFFERISCONO e — la falla che ci è già costata cara — il calloso
    //    INTERMEDIO deve essere VIVO dopo l'addestramento, non un muro di zeri.
    const ctx = [idx("il"), idx("gatto"), idx("corre"), idx("sul"), idx("il")];
    const cs = model.callosiAt(ctx);
    let diff = 0; for (let m = 0; m < model.nC; m++) diff += Math.abs(cs[0][m] - cs[1][m]);
    let vivi = 0, tot = 0;
    for (let s = 0; s < 50; s++) {
      const start = (Math.random() * Math.max(1, ids.length - model.Lc)) | 0;
      const fv = model.forward(ids.slice(start, start + model.Lc));
      const c0 = fv.cals[0][fv.n - 1];
      for (let m = 0; m < model.nC; m++) { tot++; if (c0[m] > 1e-9) vivi++; }
    }
    console.log("sbirciata: differenza L1 fra i callosi " + diff.toFixed(2) + " · calloso intermedio vivo al " + (100 * vivi / tot).toFixed(0) + "%");
    ok(cs.length === 2 && diff > 1e-3, "sbirciata sul calloso₁ disponibile e distinta dal calloso terminale");
    ok(vivi > 0, "il calloso INTERMEDIO è vivo dopo l'addestramento (" + (100 * vivi / tot).toFixed(0) + "% di neuroni attivi)");

    // 6) TRACCIAMENTO PROFONDO allineato: forwardFromCal senza iniezione = forward
    const X2ref = model.forward(ctx).X2[ctx.length - 1];
    const X2cf = model.forwardFromCal(ctx, -1, null);
    let mdc = 0; for (let m = 0; m < model.nC; m++) mdc = Math.max(mdc, Math.abs(X2ref[m] - X2cf[m]));
    ok(mdc < 1e-12, "forwardFromCal allineato a forward (scarto " + mdc.toExponential(1) + ")");

    // 7) SALVA/CARICA (formato v3 con teste)
    const blob = serialize(model, vocab);
    const model2 = deserialize(JSON.parse(JSON.stringify(blob)), vocab);
    const fa = model.forward(ctx).X2[ctx.length - 1], fb = model2.forward(ctx).X2[ctx.length - 1];
    let md = 0; for (let m = 0; m < fa.length; m++) md = Math.max(md, Math.abs(fa[m] - fb[m]));
    ok(blob.version === 3 && blob.cfg.blocks === 2 && blob.cfg.nHeads === 4 && md < 1e-9, "salva/carica fedele (v3 · nHeads nel cfg · scarto " + md.toExponential(1) + ")");

    // 8) ONESTÀ SUI VECCHI FILE: un v2 viene rifiutato con un messaggio chiaro
    let rifiutato = false, msg = "";
    try { deserialize({ format: "sapere-dna-cassandra", version: 2, cfg: {}, blockWeights: [] }, vocab); }
    catch (e) { rifiutato = true; msg = e.message; }
    ok(rifiutato && msg.indexOf("Riaddestra") >= 0, "i file v1/v2 vengono rifiutati onestamente («…Riaddestra il modello…»)");

    // 9) AUTOCORREZIONE di nHeads: dispari o non divisore di d → scende al valido
    const mh = new LMEmisferi({ V: 5, unk: null }, { d: 20, Lc: 4, blocks: 1, nHeads: 6, hemiW: 3, hemiH: 3, calW: 2, calH: 2 });
    ok(mh.nHeads === 4 && mh.hd === 5, "nHeads non valido (6 con d=20) corretto da solo a " + mh.nHeads);

    console.log("\n=== fine prove ===\n");
  }

})(typeof self !== "undefined" ? self : this);