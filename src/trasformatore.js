/* ============================================================
   Sapere-DNA Studio · TRASFORMATORE
   (modello di punta per l'interpretabilità — accanto al classificatore)
   ------------------------------------------------------------
   Un piccolo Transformer VERO e completamente ispezionabile:
     embedding token + posizione · 1 blocco di auto-attenzione
     (una testa) · MLP · pooling medio · testa di classificazione.

   Legge un CORPUS SINTETICO le cui regole le conosciamo noi
   (parole-concetto presenti/assenti) → quindi abbiamo una
   VERITÀ DI BASE contro cui validare l'interpretabilità.
   Compito: "xor" (parità di due concetti) oppure "anelli"
   (in quale fascia cade il numero di concetti presenti).

   Trasparenza, con gli stessi tre strumenti di Sapere-DNA:
     · GENOMA   → un SAE (dizionario sparso) districa l'attività
                  interna in concetti distinti e nominabili;
     · ESPRESSIONE → per ogni sequenza, quali concetti si accendono
                  e in che ordine = il tracciato, inciso come filamento;
     · IMPRONTA → il motore di SDNA sigilla genoma e tracciato.
   In più, la prova che manca a quasi tutti: il CONFRONTO con la
   verità di base, che dice quanto i concetti estratti sono reali.

   Espone window.SapereDNATrasformatore  (module.exports in Node)
   ============================================================ */
(function (root) {
  "use strict";

  /* ---------- piccola algebra ---------- */
  const rnd = (a, b) => a + Math.random() * (b - a);
  function matZeros(r, c) { return Array.from({ length: r }, () => new Float64Array(c)); }
  function vecZeros(n) { return new Float64Array(n); }
  function randMat(r, c, s) { const M = matZeros(r, c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) M[i][j] = rnd(-s, s); return M; }
  function relu(x) { return x > 0 ? x : 0; }
  function softmaxInto(arr) {
    let mx = -Infinity; for (let i = 0; i < arr.length; i++) if (arr[i] > mx) mx = arr[i];
    let s = 0; for (let i = 0; i < arr.length; i++) { arr[i] = Math.exp(arr[i] - mx); s += arr[i]; }
    const inv = 1 / (s || 1); for (let i = 0; i < arr.length; i++) arr[i] *= inv; return arr;
  }

  /* ---------- Adam, un parametro-tensore alla volta ---------- */
  class Adam {
    constructor(lr) { this.lr = lr == null ? 0.01 : lr; this.b1 = 0.9; this.b2 = 0.999; this.eps = 1e-8; this.t = 0; this.m = new Map(); this.v = new Map(); }
    step(tensors) {
      this.t++;
      const bc1 = 1 - Math.pow(this.b1, this.t), bc2 = 1 - Math.pow(this.b2, this.t);
      for (const [key, P, G] of tensors) {
        if (!this.m.has(key)) { this.m.set(key, P.map(r => new Float64Array(r.length))); this.v.set(key, P.map(r => new Float64Array(r.length))); }
        const m = this.m.get(key), v = this.v.get(key);
        for (let i = 0; i < P.length; i++) for (let j = 0; j < P[i].length; j++) {
          const g = G[i][j];
          m[i][j] = this.b1 * m[i][j] + (1 - this.b1) * g;
          v[i][j] = this.b2 * v[i][j] + (1 - this.b2) * g * g;
          const mh = m[i][j] / bc1, vh = v[i][j] / bc2;
          P[i][j] -= this.lr * mh / (Math.sqrt(vh) + this.eps);
        }
      }
    }
  }

  /* ============================================================
     CORPUS SINTETICO — con verità di base
     ============================================================ */
  // parole-concetto (le 4 "feature": presenti/assenti per ogni sequenza)
  const FEATURES = ["acqua", "fuoco", "terra", "aria"];
  // riempitivi/distrattori: occupano posto, non contano per l'etichetta
  const FILLERS = ["il", "e", "di", "che", "una", "luce", "tempo", "pietra", "vento", "sale"];
  const VOCAB = FEATURES.concat(FILLERS);
  const WORD2ID = {}; VOCAB.forEach((w, i) => { WORD2ID[w] = i; });

  const RING_LABELS = ["nessuno-uno", "due", "tre-quattro"]; // fasce del conteggio
  function ringOf(count) { if (count <= 1) return 0; if (count === 2) return 1; return 2; }

  // costruisce N sequenze di lunghezza L con feature presenti a caso (verità di base)
  function makeCorpus(opts) {
    opts = opts || {};
    const N = opts.N || 2000, L = opts.L || 8, task = opts.task || "lettura";
    const seqs = [];
    for (let s = 0; s < N; s++) {
      const present = FEATURES.map(() => Math.random() < 0.5 ? 1 : 0); // verità di base
      const tokens = [];
      // colloca le feature presenti
      FEATURES.forEach((w, fi) => { if (present[fi]) tokens.push(WORD2ID[w]); });
      // riempi fino a L con riempitivi
      while (tokens.length < L) tokens.push(WORD2ID[FILLERS[(Math.random() * FILLERS.length) | 0]]);
      // mescola le posizioni: la rete DEVE attendere al contenuto, non alla posizione
      for (let i = tokens.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; const t = tokens[i]; tokens[i] = tokens[j]; tokens[j] = t; }
      tokens.length = L;
      let label;
      if (task === "anelli") label = ringOf(present.reduce((a, b) => a + b, 0));
      else if (task === "lettura") label = present[0] * 2 + present[1]; // 4 classi: distingue QUALI concetti, non quanti
      else if (task === "completo") label = present[0] * 8 + present[1] * 4 + present[2] * 2 + present[3]; // 16 classi: legge tutte e 4 le feature
      else label = present[0] ^ present[1]; // XOR: parità dei primi due (terra/aria sono distrattori)
      seqs.push({ tokens, present, label });
    }
    const numClasses = task === "anelli" ? RING_LABELS.length : task === "lettura" ? 4 : task === "completo" ? 16 : 2;
    const labelNames = task === "anelli" ? RING_LABELS.slice()
      : task === "lettura" ? ["nessuno", "fuoco", "acqua", "acqua+fuoco"]
      : task === "completo" ? Array.from({ length: 16 }, (_, i) => FEATURES.filter((_, fi) => (i >> (3 - fi)) & 1).join("+") || "nessuno")
      : ["pari", "dispari"];
    return { seqs, N, L, task, numClasses, labelNames, features: FEATURES.slice() };
  }

  // testo leggibile della sequenza (per genoma del corpus e per le stringhe di DNA classiche)
  function seqText(seq) { return seq.tokens.map(t => VOCAB[t]).join(" "); }
  function corpusText(corpus, max) { const n = Math.min(corpus.seqs.length, max || corpus.seqs.length); const out = []; for (let i = 0; i < n; i++) out.push(seqText(corpus.seqs[i])); return out.join("\n"); }

  /* ============================================================
     TRANSFORMER (1 blocco encoder, una testa) — tutto ispezionabile
     ============================================================ */
  class Transformer {
    constructor(cfg) {
      cfg = cfg || {};
      this.V = VOCAB.length;
      this.L = cfg.L || 8;
      this.d = cfg.d || 24;
      this.h = cfg.h || 32;       // larghezza MLP
      this.C = cfg.C || 2;        // classi
      const se = 0.3, sw = 1 / Math.sqrt(this.d), sm = 1 / Math.sqrt(this.h);
      this.Wemb = randMat(this.V, this.d, se);   // embedding dei token
      this.Wpos = randMat(this.L, this.d, se);   // embedding di posizione
      this.Wq = randMat(this.d, this.d, sw);
      this.Wk = randMat(this.d, this.d, sw);
      this.Wv = randMat(this.d, this.d, sw);
      this.Wo = randMat(this.d, this.d, sw);     // proiezione d'uscita dell'attenzione
      this.W1 = randMat(this.d, this.h, sw);     // MLP su
      this.b1 = matZeros(1, this.h);
      this.W2 = randMat(this.h, this.d, sm);     // MLP giù
      this.b2 = matZeros(1, this.d);
      this.Wc = randMat(this.d, this.C, sw);     // testa di classificazione
      this.bc = matZeros(1, this.C);
    }

    // forward completo; con keep=true conserva tutte le attivazioni per il backprop
    forward(tokens, keep) {
      const L = this.L, d = this.d, h = this.h, C = this.C;
      // 1) input = embedding token + posizione
      const X = matZeros(L, d);
      for (let p = 0; p < L; p++) { const emb = this.Wemb[tokens[p]], pos = this.Wpos[p]; for (let k = 0; k < d; k++) X[p][k] = emb[k] + pos[k]; }
      // 2) attenzione (una testa)
      const Q = matZeros(L, d), K = matZeros(L, d), Vv = matZeros(L, d);
      for (let p = 0; p < L; p++) for (let k = 0; k < d; k++) {
        let q = 0, kk = 0, vv = 0;
        for (let j = 0; j < d; j++) { q += X[p][j] * this.Wq[j][k]; kk += X[p][j] * this.Wk[j][k]; vv += X[p][j] * this.Wv[j][k]; }
        Q[p][k] = q; K[p][k] = kk; Vv[p][k] = vv;
      }
      const scale = 1 / Math.sqrt(d);
      const Att = matZeros(L, L);            // pesi di attenzione (riga = query)
      for (let i = 0; i < L; i++) {
        const row = Att[i];
        for (let j = 0; j < L; j++) { let s = 0; for (let k = 0; k < d; k++) s += Q[i][k] * K[j][k]; row[j] = s * scale; }
        softmaxInto(row);
      }
      const Ctx = matZeros(L, d);            // contesto = Att · V
      for (let i = 0; i < L; i++) for (let k = 0; k < d; k++) { let s = 0; for (let j = 0; j < L; j++) s += Att[i][j] * Vv[j][k]; Ctx[i][k] = s; }
      const AttO = matZeros(L, d);           // proiezione d'uscita
      for (let i = 0; i < L; i++) for (let k = 0; k < d; k++) { let s = 0; for (let j = 0; j < d; j++) s += Ctx[i][j] * this.Wo[j][k]; AttO[i][k] = s; }
      const X1 = matZeros(L, d);             // residuo 1
      for (let p = 0; p < L; p++) for (let k = 0; k < d; k++) X1[p][k] = X[p][k] + AttO[p][k];
      // 3) MLP con residuo
      const Hpre = matZeros(L, h), Hact = matZeros(L, h);
      for (let p = 0; p < L; p++) for (let m = 0; m < h; m++) { let s = this.b1[0][m]; for (let k = 0; k < d; k++) s += X1[p][k] * this.W1[k][m]; Hpre[p][m] = s; Hact[p][m] = relu(s); }
      const Mout = matZeros(L, d);
      for (let p = 0; p < L; p++) for (let k = 0; k < d; k++) { let s = this.b2[0][k]; for (let m = 0; m < h; m++) s += Hact[p][m] * this.W2[m][k]; Mout[p][k] = s; }
      const X2 = matZeros(L, d);             // residuo 2 (flusso residuo finale)
      for (let p = 0; p < L; p++) for (let k = 0; k < d; k++) X2[p][k] = X1[p][k] + Mout[p][k];
      // 4) pooling medio sulle posizioni → vettore di rappresentazione
      const pooled = vecZeros(d);
      for (let p = 0; p < L; p++) for (let k = 0; k < d; k++) pooled[k] += X2[p][k] / L;
      // 5) testa di classificazione
      const logits = new Float64Array(C);
      for (let c = 0; c < C; c++) { let s = this.bc[0][c]; for (let k = 0; k < d; k++) s += pooled[k] * this.Wc[k][c]; logits[c] = s; }
      const probs = softmaxInto(logits.slice());
      const out = { probs, pooled, pred: argmax(probs) };
      if (keep) Object.assign(out, { X, Q, K, V: Vv, Att, Ctx, AttO, X1, Hpre, Hact, Mout, X2, tokens });
      return out;
    }

    // un passo di addestramento (cross-entropy) su una sequenza; ritorna la perdita
    trainStep(tokens, label, opt) {
      const L = this.L, d = this.d, h = this.h, C = this.C;
      const f = this.forward(tokens, true);
      const loss = -Math.log(Math.max(1e-12, f.probs[label]));
      // gradiente sui logit
      const dLogit = new Float64Array(C); for (let c = 0; c < C; c++) dLogit[c] = f.probs[c] - (c === label ? 1 : 0);
      // testa
      const gWc = matZeros(d, C), gbc = matZeros(1, C), dPooled = vecZeros(d);
      for (let c = 0; c < C; c++) { gbc[0][c] = dLogit[c]; for (let k = 0; k < d; k++) { gWc[k][c] = dPooled[k]; } }
      for (let k = 0; k < d; k++) { let s = 0; for (let c = 0; c < C; c++) { gWc[k][c] = f.pooled[k] * dLogit[c]; s += this.Wc[k][c] * dLogit[c]; } dPooled[k] = s; }
      // pooling → dX2 (ogni posizione riceve dPooled/L)
      const dX2 = matZeros(L, d);
      for (let p = 0; p < L; p++) for (let k = 0; k < d; k++) dX2[p][k] = dPooled[k] / L;
      // residuo 2: dX1 += dX2 ; dMout = dX2
      const dX1 = matZeros(L, d), dMout = matZeros(L, d);
      for (let p = 0; p < L; p++) for (let k = 0; k < d; k++) { dX1[p][k] += dX2[p][k]; dMout[p][k] = dX2[p][k]; }
      // MLP giù
      const gW2 = matZeros(h, d), gb2 = matZeros(1, d), dHact = matZeros(L, h);
      for (let p = 0; p < L; p++) for (let k = 0; k < d; k++) gb2[0][k] += dMout[p][k];
      for (let p = 0; p < L; p++) { for (let m = 0; m < h; m++) { let s = 0; for (let k = 0; k < d; k++) { gW2[m][k] += f.Hact[p][m] * dMout[p][k]; s += this.W2[m][k] * dMout[p][k]; } dHact[p][m] = s; } }
      // ReLU
      const dHpre = matZeros(L, h);
      for (let p = 0; p < L; p++) for (let m = 0; m < h; m++) dHpre[p][m] = f.Hpre[p][m] > 0 ? dHact[p][m] : 0;
      // MLP su → dX1
      const gW1 = matZeros(d, h), gb1 = matZeros(1, h);
      for (let p = 0; p < L; p++) for (let m = 0; m < h; m++) gb1[0][m] += dHpre[p][m];
      for (let p = 0; p < L; p++) { for (let k = 0; k < d; k++) { let s = 0; for (let m = 0; m < h; m++) { gW1[k][m] += f.X1[p][k] * dHpre[p][m]; s += this.W1[k][m] * dHpre[p][m]; } dX1[p][k] += s; } }
      // residuo 1: dX += dX1 ; dAttO = dX1
      const dX = matZeros(L, d), dAttO = matZeros(L, d);
      for (let p = 0; p < L; p++) for (let k = 0; k < d; k++) { dX[p][k] += dX1[p][k]; dAttO[p][k] = dX1[p][k]; }
      // proiezione Wo: AttO[p][k] = Σ_j Ctx[p][j]·Wo[j][k]
      const gWo = matZeros(d, d), dCtx = matZeros(L, d);
      for (let p = 0; p < L; p++) { for (let j = 0; j < d; j++) { let s = 0; for (let k = 0; k < d; k++) { gWo[j][k] += f.Ctx[p][j] * dAttO[p][k]; s += this.Wo[j][k] * dAttO[p][k]; } dCtx[p][j] = s; } }
      // Ctx = Att·V  →  dAtt e dV
      const dAtt = matZeros(L, L), dV = matZeros(L, d);
      for (let i = 0; i < L; i++) for (let k = 0; k < d; k++) { const g = dCtx[i][k]; for (let j = 0; j < L; j++) { dAtt[i][j] += g * f.V[j][k]; dV[j][k] += g * f.Att[i][j]; } }
      // softmax sulle righe di Att  →  dScores
      const dScores = matZeros(L, L);
      const scale = 1 / Math.sqrt(d);
      for (let i = 0; i < L; i++) {
        let dot = 0; for (let j = 0; j < L; j++) dot += dAtt[i][j] * f.Att[i][j];
        for (let j = 0; j < L; j++) dScores[i][j] = f.Att[i][j] * (dAtt[i][j] - dot) * scale;
      }
      // scores = Q·Kᵀ  →  dQ, dK
      const dQ = matZeros(L, d), dK = matZeros(L, d);
      for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) { const g = dScores[i][j]; for (let k = 0; k < d; k++) { dQ[i][k] += g * f.K[j][k]; dK[j][k] += g * f.Q[i][k]; } }
      // Q,K,V = X·Wq,Wk,Wv  →  gradienti e dX
      const gWq = matZeros(d, d), gWk = matZeros(d, d), gWv = matZeros(d, d);
      for (let p = 0; p < L; p++) for (let j = 0; j < d; j++) {
        let sx = 0;
        for (let k = 0; k < d; k++) {
          gWq[j][k] += f.X[p][j] * dQ[p][k];
          gWk[j][k] += f.X[p][j] * dK[p][k];
          gWv[j][k] += f.X[p][j] * dV[p][k];
          sx += this.Wq[j][k] * dQ[p][k] + this.Wk[j][k] * dK[p][k] + this.Wv[j][k] * dV[p][k];
        }
        dX[p][j] += sx;
      }
      // X = Wemb[token] + Wpos[p]  →  gradienti embedding
      const gWemb = matZeros(this.V, d), gWpos = matZeros(L, d);
      for (let p = 0; p < L; p++) for (let k = 0; k < d; k++) { gWemb[tokens[p]][k] += dX[p][k]; gWpos[p][k] += dX[p][k]; }

      opt.step([
        ["Wemb", this.Wemb, gWemb], ["Wpos", this.Wpos, gWpos],
        ["Wq", this.Wq, gWq], ["Wk", this.Wk, gWk], ["Wv", this.Wv, gWv], ["Wo", this.Wo, gWo],
        ["W1", this.W1, gW1], ["b1", this.b1, gb1], ["W2", this.W2, gW2], ["b2", this.b2, gb2],
        ["Wc", this.Wc, gWc], ["bc", this.bc, gbc]
      ]);
      return loss;
    }
  }
  function argmax(a) { let mi = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[mi]) mi = i; return mi; }

  // addestra il modello sul corpus; ritorna {accuracy, loss}
  function train(model, corpus, opts) {
    opts = opts || {};
    const steps = opts.steps || 6000, lr = opts.lr || 0.01;
    const opt = new Adam(lr);
    const S = corpus.seqs, N = S.length;
    let loss = 0;
    for (let t = 0; t < steps; t++) {
      const s = S[(Math.random() * N) | 0];
      loss = 0.99 * loss + 0.01 * model.trainStep(s.tokens, s.label, opt);
      if (opts.onProgress && (t % 500 === 0)) opts.onProgress(t / steps, loss);
    }
    return { accuracy: accuracy(model, corpus), loss };
  }
  function accuracy(model, corpus) {
    let ok = 0; for (const s of corpus.seqs) if (model.forward(s.tokens).pred === s.label) ok++;
    return ok / corpus.seqs.length;
  }

  // addestra più modelli da inizializzazioni diverse e tiene il migliore
  // (XOR si incastra spesso in un minimo a caso: i restart lo risolvono)
  function fit(corpus, cfg, opts) {
    opts = opts || {};
    const restarts = opts.restarts || 4;
    let best = null, bestAcc = -1, bestLoss = 0;
    for (let r = 0; r < restarts; r++) {
      const m = new Transformer(Object.assign({ C: corpus.numClasses, L: corpus.L }, cfg || {}));
      const res = train(m, corpus, opts);
      if (res.accuracy > bestAcc) { bestAcc = res.accuracy; best = m; bestLoss = res.loss; }
      if (opts.onRestart) opts.onRestart(r + 1, restarts, res.accuracy);
      if (bestAcc >= 0.99) break;
    }
    return { model: best, accuracy: bestAcc, loss: bestLoss };
  }

  /* ============================================================
     SAE — dizionario sparso che districa l'attività interna
     (ritorna pesi, codici e capacità di codifica: serve a derivare,
      intervenire, confrontare con la verità di base)
     ============================================================ */
  function representationMatrix(model, corpus) {
    const A = []; for (const s of corpus.seqs) { const r = model.forward(s.tokens).pooled; A.push(Float64Array.from(r)); } return A;
  }
  function trainSAE(A, opts) {
    opts = opts || {};
    const d = A[0].length, N = A.length;
    const M = opts.M || 16, steps = opts.steps || 9000, lr = opts.lr || 0.02, l1 = opts.l1 == null ? 0.012 : opts.l1;
    // normalizza (media/scarto) per stabilità
    const mean = new Float64Array(d), std = new Float64Array(d);
    for (const r of A) for (let k = 0; k < d; k++) mean[k] += r[k] / N;
    for (const r of A) for (let k = 0; k < d; k++) std[k] += (r[k] - mean[k]) ** 2 / N;
    for (let k = 0; k < d; k++) std[k] = Math.sqrt(std[k]) || 1;
    const norm = (r) => { const z = new Float64Array(d); for (let k = 0; k < d; k++) z[k] = (r[k] - mean[k]) / std[k]; return z; };
    const We = randMat(M, d, 0.1), Wd = randMat(d, M, 0.1), be = matZeros(1, M), bd = matZeros(1, d);
    for (let m = 0; m < M; m++) be[0][m] = 0.05; // parti con codici un po' attivi
    // colonne del decoder a norma unitaria (stabilizza l'L1: niente codici che muoiono)
    const normalizeDecoder = () => { for (let m = 0; m < M; m++) { let nrm = 0; for (let k = 0; k < d; k++) nrm += Wd[k][m] * Wd[k][m]; nrm = Math.sqrt(nrm) || 1; for (let k = 0; k < d; k++) Wd[k][m] /= nrm; } };
    normalizeDecoder();
    const opt = new Adam(lr);
    const encode = (z) => { const code = new Float64Array(M); for (let m = 0; m < M; m++) { let s = be[0][m]; for (let k = 0; k < d; k++) s += We[m][k] * z[k]; code[m] = relu(s); } return code; };
    for (let t = 0; t < steps; t++) {
      const z = norm(A[(Math.random() * N) | 0]);
      const pre = new Float64Array(M), code = new Float64Array(M);
      for (let m = 0; m < M; m++) { let s = be[0][m]; for (let k = 0; k < d; k++) s += We[m][k] * z[k]; pre[m] = s; code[m] = relu(s); }
      const rec = new Float64Array(d);
      for (let k = 0; k < d; k++) { let s = bd[0][k]; for (let m = 0; m < M; m++) s += Wd[k][m] * code[m]; rec[k] = s; }
      const dRec = new Float64Array(d); for (let k = 0; k < d; k++) dRec[k] = 2 * (rec[k] - z[k]) / d;
      const gWd = matZeros(d, M), gbd = matZeros(1, d), dCode = new Float64Array(M);
      for (let k = 0; k < d; k++) { gbd[0][k] = dRec[k]; for (let m = 0; m < M; m++) { gWd[k][m] = dRec[k] * code[m]; dCode[m] += Wd[k][m] * dRec[k]; } }
      for (let m = 0; m < M; m++) dCode[m] += l1 * (code[m] > 0 ? 1 : 0);
      const gWe = matZeros(M, d), gbe = matZeros(1, M);
      for (let m = 0; m < M; m++) { const g = pre[m] > 0 ? dCode[m] : 0; gbe[0][m] = g; for (let k = 0; k < d; k++) gWe[m][k] = g * z[k]; }
      opt.step([["We", We, gWe], ["be", be, gbe], ["Wd", Wd, gWd], ["bd", bd, gbd]]);
      if ((t & 31) === 0) normalizeDecoder();
    }
    normalizeDecoder();
    // statistiche
    let active = 0, rmse = 0;
    for (const r of A) { const z = norm(r), code = encode(z); active += code.reduce((a, v) => a + (v > 1e-2 ? 1 : 0), 0);
      const rec = new Float64Array(d); for (let k = 0; k < d; k++) { let s = bd[0][k]; for (let m = 0; m < M; m++) s += Wd[k][m] * code[m]; rec[k] = s; }
      let e = 0; for (let k = 0; k < d; k++) e += (rec[k] - z[k]) ** 2; rmse += Math.sqrt(e / d); }
    const decode = (code) => { const rec = new Float64Array(d); for (let k = 0; k < d; k++) { let s = bd[0][k]; for (let m = 0; m < M; m++) s += Wd[k][m] * code[m]; rec[k] = s; } return rec; };
    const denorm = (z) => { const r = new Float64Array(d); for (let k = 0; k < d; k++) r[k] = z[k] * std[k] + mean[k]; return r; };
    return { M, d, We, Wd, be, bd, mean, std, norm, denorm, encode, decode, avgActive: active / N, rmse: rmse / N };
  }

  /* ============================================================
     PONTE DI TRASPARENZA — genoma, espressione, impronta + verità
     ============================================================ */
  // CONFRONTO CON LA VERITÀ DI BASE: a quale feature reale corrisponde ogni concetto?
  function groundTruthMatch(model, sae, corpus, thresh) {
    thresh = thresh == null ? 0.5 : thresh;
    const ALIGN = 0.8; // soglia: concetto "pulito" se F1 ≥ 0.8 con un generatore noto
    const F = corpus.features.length, M = sae.M, N = corpus.seqs.length, C = corpus.numClasses;
    const codes = corpus.seqs.map(s => sae.encode(sae.norm(model.forward(s.tokens).pooled)));
    // soglia adattiva per concetto = metà del max
    const maxc = new Float64Array(M); codes.forEach(c => { for (let m = 0; m < M; m++) if (c[m] > maxc[m]) maxc[m] = c[m]; });
    const fires = (c, m) => c[m] > Math.max(1e-3, thresh * maxc[m]);
    // F1 di "concetto m acceso" contro un predicato booleano sulle sequenze
    const f1Against = (m, pred) => {
      let tp = 0, fp = 0, fn = 0;
      for (let n = 0; n < N; n++) { const on = fires(codes[n], m), gt = pred(n); if (on && gt) tp++; else if (on && !gt) fp++; else if (!on && gt) fn++; }
      const prec = tp / (tp + fp || 1), rec = tp / (tp + fn || 1); return { f1: 2 * prec * rec / (prec + rec || 1), prec, rec };
    };
    const featPred = (fi) => (n) => corpus.seqs[n].present[fi] === 1;
    const classPred = (cl) => (n) => corpus.seqs[n].label === cl;

    const concept = []; const freq = new Float64Array(M);
    // matrici per il report
    const featBest = new Float64Array(F);   // miglior F1 per feature (qualsiasi concetto)
    const classBest = new Float64Array(C);
    const featBestM = new Int32Array(F).fill(-1), classBestM = new Int32Array(C).fill(-1);
    for (let m = 0; m < M; m++) {
      let fr = 0; for (let n = 0; n < N; n++) if (fires(codes[n], m)) fr++; freq[m] = fr;
      // miglior feature e miglior classe per questo concetto
      let bF = { i: -1, f1: 0, prec: 0, rec: 0 };
      for (let fi = 0; fi < F; fi++) { const r = f1Against(m, featPred(fi)); if (r.f1 > bF.f1) bF = { i: fi, f1: r.f1, prec: r.prec, rec: r.rec }; if (r.f1 > featBest[fi]) { featBest[fi] = r.f1; featBestM[fi] = m; } }
      let bC = { i: -1, f1: 0 };
      for (let cl = 0; cl < C; cl++) { const r = f1Against(m, classPred(cl)); if (r.f1 > bC.f1) bC = { i: cl, f1: r.f1 }; if (r.f1 > classBest[cl]) { classBest[cl] = r.f1; classBestM[cl] = m; } }
      // il concetto si allinea al generatore (feature o classe) col F1 più alto
      const toClass = bC.f1 > bF.f1;
      const topF1 = Math.max(bF.f1, bC.f1);
      const aligned = topF1 >= ALIGN && fr > 0;
      const label = !aligned ? (fr > 0 ? "misto" : "spento")
        : toClass ? ("classe:" + corpus.labelNames[bC.i]) : ("rileva:" + corpus.features[bF.i]);
      concept.push({ m, freq: fr, name: label,
        feature: aligned && !toClass ? corpus.features[bF.i] : null,
        klass: aligned && toClass ? bC.i : null,
        featureF1: bF.f1, classF1: bC.f1, topF1, prec: bF.prec, rec: bF.rec, monosemantic: aligned });
    }
    const recovered = corpus.features.map((fname, fi) => ({ feature: fname, f1: featBest[fi], concept: featBestM[fi], recovered: featBest[fi] >= ALIGN }));
    const recoveredClasses = corpus.labelNames.map((cname, cl) => ({ klass: cname, f1: classBest[cl], concept: classBestM[cl], recovered: classBest[cl] >= ALIGN }));
    const activeConcepts = concept.filter(c => c.freq > 0).length;
    const monoCount = concept.filter(c => c.monosemantic).length;
    return {
      concept, recovered, recoveredClasses, F, M, C, ALIGN, fires, codes, maxc,
      recoveredCount: recovered.filter(r => r.recovered).length,
      recoveredClassCount: recoveredClasses.filter(r => r.recovered).length,
      featureRecovery: recovered.filter(r => r.recovered).length / F,
      classRecovery: recoveredClasses.filter(r => r.recovered).length / C,
      monosemanticity: activeConcepts ? monoCount / activeConcepts : 0,
      activeConcepts
    };
  }

  // GENOMA dei concetti (ordinati per frequenza, indirizzi reali del motore SDNA)
  function extractGenome(model, sae, corpus, G) {
    const gt = groundTruthMatch(model, sae, corpus);
    const order = [...Array(sae.M).keys()].sort((a, b) => gt.concept[b].freq - gt.concept[a].freq);
    const genes = order.map(m => gt.concept[m]);
    const conceptKey = (c) => "c" + c.m + "_" + c.name;
    const words = genes.map(conceptKey);
    const dict = G.loadDictionary(words, { id: "concetti-transformer", version: corpus.task });
    return { dict, order, genes, words, gt, sae, conceptKey };
  }

  // ESPRESSIONE: tracciato di una sequenza — quali concetti si accendono, in ordine
  function derive(model, seq, genome, G) {
    const f = model.forward(seq.tokens);
    const code = genome.sae.encode(genome.sae.norm(f.pooled));
    const fired = [];
    for (let m = 0; m < genome.sae.M; m++) if (genome.gt.fires(code, m)) {
      const c = genome.gt.concept[m];
      fired.push({ m, name: c.name, feature: c.feature, act: code[m], causalTruth: false });
    }
    fired.sort((a, b) => b.act - a.act);
    const SP = G.CTRL.SP;
    const filament = fired.map(x => {
      const c = genome.gt.concept[x.m];
      return G.addressOf(genome.dict.index[genome.conceptKey(c)]);
    }).join(SP);
    return { out: f.probs, pred: f.pred, predName: corpusName(seq, model, f.pred), code, fired, filament };
  }
  function corpusName(seq, model, idx) { return idx; } // l'etichetta numerica; il nome lo dà il controller via labelNames

  // INTERVENTO: spegne i concetti nello spazio del SAE e rilegge la testa (3 letture)
  //  baseline = testa sulla ricostruzione SAE (isola l'effetto del concetto dall'errore di ricostruzione)
  function intervene(model, seq, genome, trace) {
    const sae = genome.sae, d = sae.d, M = sae.M;
    const headOn = (vecD) => { // testa di classificazione su un vettore d-dim
      const C = model.C, logits = new Float64Array(C);
      for (let c = 0; c < C; c++) { let s = model.bc[0][c]; for (let k = 0; k < d; k++) s += vecD[k] * model.Wc[k][c]; logits[c] = s; }
      return softmaxInto(logits);
    };
    const code0 = trace.code.slice();
    const baseRec = sae.denorm(sae.decode(code0));
    const baseProbs = headOn(baseRec), basePred = argmax(baseProbs), baseConf = baseProbs[basePred];
    const norm = (after) => baseConf > 0 ? Math.max(0, Math.min(1, (baseConf - after) / baseConf)) : 0;

    // (1) singola
    const rows = trace.fired.map(x => {
      const c = code0.slice(); c[x.m] = 0;
      const p = headOn(sae.denorm(sae.decode(c))); const np = argmax(p);
      return { m: x.m, name: x.name, feature: x.feature, newPred: np, causal: np !== basePred, confAfter: p[basePred], dropPct: norm(p[basePred]) };
    });
    const fidelitySingle = trace.fired.length ? rows.filter(r => r.causal).length / trace.fired.length : 0;
    // (2) di gruppo (per nome di concetto)
    const byName = {}; trace.fired.forEach(x => { (byName[x.name] = byName[x.name] || []).push(x.m); });
    const groups = Object.keys(byName).map(name => {
      const ms = byName[name], c = code0.slice(); ms.forEach(m => c[m] = 0);
      const p = headOn(sae.denorm(sae.decode(c))); const np = argmax(p);
      return { name, neurons: ms, newPred: np, causal: np !== basePred, confAfter: p[basePred], dropPct: norm(p[basePred]) };
    });
    const fidelityGroup = groups.length ? groups.filter(g => g.causal).length / groups.length : 0;
    // (3) graduata: spegne tutti i concetti accesi
    let fidelityGraded = 0, allPred = basePred, allConf = baseConf;
    if (trace.fired.length) { const c = code0.slice(); trace.fired.forEach(x => c[x.m] = 0); const p = headOn(sae.denorm(sae.decode(c))); allConf = p[basePred]; allPred = argmax(p); fidelityGraded = norm(allConf); }
    return { basePred, baseConf, rows, groups, allPred, allConf, fidelitySingle, fidelityGroup, fidelityGraded };
  }

  function seal(genome, filament, G) { return { genomeHash: genome.dict.hash, traceHash: G.fingerprint(filament) }; }

  const API = {
    VOCAB, FEATURES, FILLERS, RING_LABELS,
    Transformer, makeCorpus, seqText, corpusText,
    train, fit, accuracy, representationMatrix, trainSAE,
    groundTruthMatch, extractGenome, derive, intervene, seal, argmax
  };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else root.SapereDNATrasformatore = API;

})(typeof window !== "undefined" ? window : this);
