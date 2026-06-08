/* ============================================================
   Sapere-DNA Studio · GENERATORE (modello generativo reale)
   ------------------------------------------------------------
   Un piccolo Transformer CAUSALE a livello di PAROLA, il cui
   vocabolario È il genoma di Sapere-DNA: ogni parola un gene,
   con il suo indirizzo DNA. Impara da un corpus VERO a predire
   la parola successiva, e genera testo continuando un avvio.

   Fuori dal laboratorio la "verità di base" non esiste più: non
   sappiamo quali siano i concetti "veri". Restano però le due
   prove di Sapere-DNA che valgono nel mondo reale:
     · INTERVENTO CAUSALE — spengo un concetto e guardo se la
       parola scelta cambia: dimostro che quel concetto conta;
     · SIGILLO — il tracciato è legato al calcolo reale e
       rileggibile col genoma, quindi non è una storia inventata.
   Più la monosemanticità del SAE come misura di QUALITÀ.

   Espone window.SapereDNAGeneratore  (module.exports in Node)
   ============================================================ */
(function (root) {
  "use strict";

  /* ---------- algebra (come gli altri motori) ---------- */
  const rnd = (a, b) => a + Math.random() * (b - a);
  const matZeros = (r, c) => Array.from({ length: r }, () => new Float64Array(c));
  const randMat = (r, c, s) => { const M = matZeros(r, c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) M[i][j] = rnd(-s, s); return M; };
  const relu = (x) => x > 0 ? x : 0;
  function softmaxInto(arr) { let mx = -Infinity; for (let i = 0; i < arr.length; i++) if (arr[i] > mx) mx = arr[i]; let s = 0; for (let i = 0; i < arr.length; i++) { arr[i] = Math.exp(arr[i] - mx); s += arr[i]; } const inv = 1 / (s || 1); for (let i = 0; i < arr.length; i++) arr[i] *= inv; return arr; }
  function argmax(a) { let mi = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[mi]) mi = i; return mi; }

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
     VOCABOLARIO = il genoma (parole-gene), con tetto + <ignoto>
     ============================================================ */
  // dict = genoma caricato (dict.words rank→parola, dict.index parola→rank, freq-ordinato)
  function makeVocab(dict, opts) {
    opts = opts || {};
    const cap = Math.min(dict.size, opts.cap || 1500);
    const unk = cap;                 // id riservato per parole fuori dal tetto/ignote
    const V = cap + 1;
    const words = dict.words.slice(0, cap);
    return { V, cap, unk, words, dict_id: dict.id, dict_version: dict.version, dict_hash: dict.hash };
  }
  // testo → id locali (usa il tokenizzatore del genoma)
  function encode(text, dict, vocab, G) {
    const toks = G.genomeTokens(text), ids = [];
    for (const t of toks) { const r = dict.index[t]; ids.push((r !== undefined && r < vocab.cap) ? r : vocab.unk); }
    return ids;
  }
  function decode(ids, vocab) { return ids.map(i => i === vocab.unk ? "□" : vocab.words[i]).join(" "); }

  /* ============================================================
     MODELLO · Transformer causale (1 blocco, una testa)
     ============================================================ */
  class LM {
    constructor(vocab, cfg) {
      cfg = cfg || {};
      this.V = vocab.V;
      this.Lc = cfg.Lc || 16;     // finestra di contesto
      this.d = cfg.d || 32;
      this.h = cfg.h || 64;
      this.vocabUnk = vocab.unk;
      const se = 0.3, sw = 1 / Math.sqrt(this.d), so = 1 / Math.sqrt(this.d);
      this.Wemb = randMat(this.V, this.d, se);
      this.Wpos = randMat(this.Lc, this.d, se);
      this.Wq = randMat(this.d, this.d, sw); this.Wk = randMat(this.d, this.d, sw);
      this.Wv = randMat(this.d, this.d, sw); this.Wo = randMat(this.d, this.d, sw);
      this.W1 = randMat(this.d, this.h, sw); this.b1 = matZeros(1, this.h);
      this.W2 = randMat(this.h, this.d, 1 / Math.sqrt(this.h)); this.b2 = matZeros(1, this.d);
      this.Wout = randMat(this.d, this.V, so); this.bout = matZeros(1, this.V);
    }

    // forward su una finestra di token (lunghezza n ≤ Lc), attenzione CAUSALE.
    // keep=true conserva le attivazioni per il backprop.
    forward(ids, keep) {
      const n = ids.length, d = this.d, h = this.h;
      const X = matZeros(n, d);
      for (let p = 0; p < n; p++) { const e = this.Wemb[ids[p]], po = this.Wpos[p]; for (let k = 0; k < d; k++) X[p][k] = e[k] + po[k]; }
      const Q = matZeros(n, d), K = matZeros(n, d), Vv = matZeros(n, d);
      for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) { let q = 0, kk = 0, vv = 0; for (let j = 0; j < d; j++) { q += X[p][j] * this.Wq[j][k]; kk += X[p][j] * this.Wk[j][k]; vv += X[p][j] * this.Wv[j][k]; } Q[p][k] = q; K[p][k] = kk; Vv[p][k] = vv; }
      const scale = 1 / Math.sqrt(d);
      const Att = matZeros(n, n);
      for (let i = 0; i < n; i++) { const row = Att[i]; for (let j = 0; j <= i; j++) { let s = 0; for (let k = 0; k < d; k++) s += Q[i][k] * K[j][k]; row[j] = s * scale; } const valid = row.subarray(0, i + 1); softmaxInto(valid); }
      const Ctx = matZeros(n, d);
      for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) { let s = 0; for (let j = 0; j <= i; j++) s += Att[i][j] * Vv[j][k]; Ctx[i][k] = s; }
      const AttO = matZeros(n, d);
      for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) { let s = 0; for (let j = 0; j < d; j++) s += Ctx[i][j] * this.Wo[j][k]; AttO[i][k] = s; }
      const X1 = matZeros(n, d);
      for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) X1[p][k] = X[p][k] + AttO[p][k];
      const Hpre = matZeros(n, h), Hact = matZeros(n, h);
      for (let p = 0; p < n; p++) for (let m = 0; m < h; m++) { let s = this.b1[0][m]; for (let k = 0; k < d; k++) s += X1[p][k] * this.W1[k][m]; Hpre[p][m] = s; Hact[p][m] = relu(s); }
      const Mout = matZeros(n, d);
      for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) { let s = this.b2[0][k]; for (let m = 0; m < h; m++) s += Hact[p][m] * this.W2[m][k]; Mout[p][k] = s; }
      const X2 = matZeros(n, d);
      for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) X2[p][k] = X1[p][k] + Mout[p][k];
      const out = { n, X2 };
      if (keep) Object.assign(out, { X, Q, K, V: Vv, Att, Ctx, AttO, X1, Hpre, Hact, ids });
      return out;
    }

    // distribuzione della parola successiva dalla posizione p (default: ultima)
    logitsAt(X2row) { const V = this.V, lg = new Float64Array(V); for (let c = 0; c < V; c++) { let s = this.bout[0][c]; for (let k = 0; k < this.d; k++) s += X2row[k] * this.Wout[k][c]; lg[c] = s; } return lg; }
    nextProbs(ids) { const f = this.forward(ids); return softmaxInto(this.logitsAt(f.X2[f.n - 1])); }

    // un passo di addestramento su una finestra: predici tok[p+1] da posizione p
    trainStep(ids, opt) {
      const n = ids.length, d = this.d, h = this.h, V = this.V;
      const f = this.forward(ids, true);
      let loss = 0; const T = n - 1, invT = 1 / Math.max(1, T);
      const dX2 = matZeros(n, d);
      const gWout = matZeros(d, V), gbout = matZeros(1, V);
      for (let p = 0; p < T; p++) {
        const probs = softmaxInto(this.logitsAt(f.X2[p])); const tgt = ids[p + 1];
        loss += -Math.log(Math.max(1e-12, probs[tgt]));
        const dl = probs; dl[tgt] -= 1; for (let c = 0; c < V; c++) dl[c] *= invT;   // dLogit della perdita MEDIA
        for (let c = 0; c < V; c++) { const g = dl[c]; if (g === 0) continue; gbout[0][c] += g; for (let k = 0; k < d; k++) gWout[k][c] += f.X2[p][k] * g; }
        for (let k = 0; k < d; k++) { let s = 0; for (let c = 0; c < V; c++) s += this.Wout[k][c] * dl[c]; dX2[p][k] += s; }
      }
      loss *= invT;
      // ---- backprop nel blocco (come trasformatore, ma per-posizione e causale) ----
      const dX1 = matZeros(n, d), dMout = matZeros(n, d);
      for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) { dX1[p][k] += dX2[p][k]; dMout[p][k] = dX2[p][k]; }
      const gW2 = matZeros(h, d), gb2 = matZeros(1, d), dHact = matZeros(n, h);
      for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) gb2[0][k] += dMout[p][k];
      for (let p = 0; p < n; p++) for (let m = 0; m < h; m++) { let s = 0; for (let k = 0; k < d; k++) { gW2[m][k] += f.Hact[p][m] * dMout[p][k]; s += this.W2[m][k] * dMout[p][k]; } dHact[p][m] = s; }
      const dHpre = matZeros(n, h);
      for (let p = 0; p < n; p++) for (let m = 0; m < h; m++) dHpre[p][m] = f.Hpre[p][m] > 0 ? dHact[p][m] : 0;
      const gW1 = matZeros(d, h), gb1 = matZeros(1, h);
      for (let p = 0; p < n; p++) for (let m = 0; m < h; m++) gb1[0][m] += dHpre[p][m];
      for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) { let s = 0; for (let m = 0; m < h; m++) { gW1[k][m] += f.X1[p][k] * dHpre[p][m]; s += this.W1[k][m] * dHpre[p][m]; } dX1[p][k] += s; }
      const dX = matZeros(n, d), dAttO = matZeros(n, d);
      for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) { dX[p][k] += dX1[p][k]; dAttO[p][k] = dX1[p][k]; }
      const gWo = matZeros(d, d), dCtx = matZeros(n, d);
      for (let p = 0; p < n; p++) for (let j = 0; j < d; j++) { let s = 0; for (let k = 0; k < d; k++) { gWo[j][k] += f.Ctx[p][j] * dAttO[p][k]; s += this.Wo[j][k] * dAttO[p][k]; } dCtx[p][j] = s; }
      const dAtt = matZeros(n, n), dV = matZeros(n, d);
      for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) { const g = dCtx[i][k]; for (let j = 0; j <= i; j++) { dAtt[i][j] += g * f.V[j][k]; dV[j][k] += g * f.Att[i][j]; } }
      const dScores = matZeros(n, n), scale = 1 / Math.sqrt(d);
      for (let i = 0; i < n; i++) { let dot = 0; for (let j = 0; j <= i; j++) dot += dAtt[i][j] * f.Att[i][j]; for (let j = 0; j <= i; j++) dScores[i][j] = f.Att[i][j] * (dAtt[i][j] - dot) * scale; }
      const dQ = matZeros(n, d), dK = matZeros(n, d);
      for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { const g = dScores[i][j]; for (let k = 0; k < d; k++) { dQ[i][k] += g * f.K[j][k]; dK[j][k] += g * f.Q[i][k]; } }
      const gWq = matZeros(d, d), gWk = matZeros(d, d), gWv = matZeros(d, d);
      for (let p = 0; p < n; p++) for (let j = 0; j < d; j++) { let sx = 0; for (let k = 0; k < d; k++) { gWq[j][k] += f.X[p][j] * dQ[p][k]; gWk[j][k] += f.X[p][j] * dK[p][k]; gWv[j][k] += f.X[p][j] * dV[p][k]; sx += this.Wq[j][k] * dQ[p][k] + this.Wk[j][k] * dK[p][k] + this.Wv[j][k] * dV[p][k]; } dX[p][j] += sx; }
      const gWemb = matZeros(this.V, d), gWpos = matZeros(this.Lc, d);
      for (let p = 0; p < n; p++) for (let k = 0; k < d; k++) { gWemb[ids[p]][k] += dX[p][k]; gWpos[p][k] += dX[p][k]; }
      opt.step([
        ["Wemb", this.Wemb, gWemb], ["Wpos", this.Wpos, gWpos],
        ["Wq", this.Wq, gWq], ["Wk", this.Wk, gWk], ["Wv", this.Wv, gWv], ["Wo", this.Wo, gWo],
        ["W1", this.W1, gW1], ["b1", this.b1, gb1], ["W2", this.W2, gW2], ["b2", this.b2, gb2],
        ["Wout", this.Wout, gWout], ["bout", this.bout, gbout]
      ]);
      return loss;
    }
  }

  // addestra su un flusso di id: finestre casuali di lunghezza Lc
  function train(model, ids, opts) {
    opts = opts || {};
    const steps = opts.steps || 6000, lr = opts.lr || 0.01, Lc = model.Lc;
    const opt = new Adam(lr); let loss = 0, seen = 0;
    if (ids.length < 2) return { loss: 0, perplexity: Infinity };
    const now = () => (typeof Date !== "undefined" ? Date.now() : 0);
    const report = (t) => { if (opts.onProgress) opts.onProgress(t / steps, loss, Math.exp(loss)); };
    let lastReport = now();
    report(0);
    for (let t = 0; t < steps; t++) {
      const maxStart = Math.max(1, ids.length - Lc);
      const start = (Math.random() * maxStart) | 0;
      const win = ids.slice(start, start + Lc);
      if (win.length < 2) continue;
      loss = (loss * seen + model.trainStep(win, opt)) / (seen + 1); seen++;
      const t1 = now();
      if (t1 - lastReport >= 150) { report(t + 1); lastReport = t1; }
    }
    report(steps);
    return { loss, perplexity: Math.exp(loss) };
  }

  // perplexity su tutto il testo (qualità del modello linguistico)
  function perplexity(model, ids) {
    const Lc = model.Lc; let nll = 0, cnt = 0;
    for (let start = 0; start + 1 < ids.length; start += Lc) {
      const win = ids.slice(start, start + Lc);
      if (win.length < 2) break;
      const f = model.forward(win);
      for (let p = 0; p < win.length - 1; p++) { const pr = softmaxInto(model.logitsAt(f.X2[p])); nll += -Math.log(Math.max(1e-12, pr[win[p + 1]])); cnt++; }
    }
    return cnt ? Math.exp(nll / cnt) : Infinity;
  }

  // GENERAZIONE: continua un avvio, campionando con temperatura.
  // registra, per ogni passo, contesto + rappresentazione dell'ultima posizione + scelta
  function generate(model, primeIds, opts) {
    opts = opts || {};
    const maxLen = opts.maxLen || 40, temp = opts.temperature == null ? 0.8 : opts.temperature, Lc = model.Lc;
    let ctx = primeIds.slice(); if (!ctx.length) ctx = [0];
    const steps = [];
    for (let s = 0; s < maxLen; s++) {
      const win = ctx.slice(Math.max(0, ctx.length - Lc));
      const f = model.forward(win);
      const rep = Float64Array.from(f.X2[f.n - 1]);
      const lg = model.logitsAt(rep);
      if (model.vocabUnk != null) lg[model.vocabUnk] = -1e9;   // non generiamo mai <ignoto>
      if (temp > 0) for (let c = 0; c < lg.length; c++) lg[c] /= temp;
      const pr = softmaxInto(lg.slice());
      let chosen; if (temp <= 0) chosen = argmax(pr);
      else { let r = Math.random(), acc = 0; chosen = pr.length - 1; for (let c = 0; c < pr.length; c++) { acc += pr[c]; if (r <= acc) { chosen = c; break; } } }
      const top = [...pr.keys()].sort((a, b) => pr[b] - pr[a]).slice(0, 5).map(i => ({ id: i, p: pr[i] }));
      steps.push({ ctx: win.slice(), rep, chosen, top });
      ctx.push(chosen);
      if (model.vocabUnk != null && chosen === model.vocabUnk) {/* continua comunque */ }
    }
    return { ids: ctx, generated: ctx.slice(primeIds.length), steps };
  }

  /* ============================================================
     SAE sull'ultima posizione (ciò che decide la parola seguente)
     — fuori dal laboratorio: niente verità di base, ma intervento
     causale + sigillo + monosemanticità.
     ============================================================ */
  function lastPosMatrix(model, ids, opts) {
    opts = opts || {}; const Lc = model.Lc, N = opts.N || 1500, A = [];
    const maxStart = Math.max(1, ids.length - 2);
    for (let s = 0; s < N; s++) {
      const start = (Math.random() * maxStart) | 0;
      const win = ids.slice(start, Math.min(ids.length, start + Lc));
      if (win.length < 2) continue;
      const f = model.forward(win); A.push(Float64Array.from(f.X2[f.n - 1]));
    }
    return A;
  }
  function trainSAE(A, opts) {
    opts = opts || {}; const d = A[0].length, N = A.length;
    const M = opts.M || 32, steps = opts.steps || 9000, lr = opts.lr || 0.02, l1 = opts.l1 == null ? 0.004 : opts.l1;
    const k = Math.max(0, opts.k || 0);   // TopK: 0 = sparsità morbida L1 (storico) · >0 = TopK netto
    const mean = new Float64Array(d), std = new Float64Array(d);
    for (const r of A) for (let i = 0; i < d; i++) mean[i] += r[i] / N;
    for (const r of A) for (let i = 0; i < d; i++) std[i] += (r[i] - mean[i]) ** 2 / N;
    for (let i = 0; i < d; i++) std[i] = Math.sqrt(std[i]) || 1;
    const norm = (r) => { const z = new Float64Array(d); for (let i = 0; i < d; i++) z[i] = (r[i] - mean[i]) / std[i]; return z; };
    const We = randMat(M, d, 0.1), Wd = randMat(d, M, 0.1), be = matZeros(1, M), bd = matZeros(1, d);
    for (let m = 0; m < M; m++) be[0][m] = 0.05;
    const normDec = () => { for (let m = 0; m < M; m++) { let nr = 0; for (let i = 0; i < d; i++) nr += Wd[i][m] * Wd[i][m]; nr = Math.sqrt(nr) || 1; for (let i = 0; i < d; i++) Wd[i][m] /= nr; } };
    normDec(); const opt = new Adam(lr);
    // TopK: tiene solo i k concetti più accesi (gli altri a zero) — sparsità netta, niente penalità morbida
    const topk = (c) => { if (!k || k >= M) return c; const idx = new Array(M); for (let m = 0; m < M; m++) idx[m] = m; idx.sort((a, b) => c[b] - c[a]); for (let r = k; r < M; r++) c[idx[r]] = 0; return c; };
    const encode = (z) => { const c = new Float64Array(M); for (let m = 0; m < M; m++) { let s = be[0][m]; for (let i = 0; i < d; i++) s += We[m][i] * z[i]; c[m] = relu(s); } return k ? topk(c) : c; };
    const fires = new Float64Array(M);                       // accensioni nell'intervallo corrente (per le feature morte)
    const resampleEvery = Math.max(800, (steps / 6) | 0);
    for (let t = 0; t < steps; t++) {
      const z = norm(A[(Math.random() * N) | 0]); const pre = new Float64Array(M), code = new Float64Array(M);
      for (let m = 0; m < M; m++) { let s = be[0][m]; for (let i = 0; i < d; i++) s += We[m][i] * z[i]; pre[m] = s; code[m] = relu(s); }
      if (k) topk(code);                                     // maschera sparsa TopK
      for (let m = 0; m < M; m++) if (code[m] > 0) fires[m]++;
      const rec = new Float64Array(d); for (let i = 0; i < d; i++) { let s = bd[0][i]; for (let m = 0; m < M; m++) s += Wd[i][m] * code[m]; rec[i] = s; }
      const dRec = new Float64Array(d); for (let i = 0; i < d; i++) dRec[i] = 2 * (rec[i] - z[i]) / d;
      const gWd = matZeros(d, M), gbd = matZeros(1, d), dCode = new Float64Array(M);
      for (let i = 0; i < d; i++) { gbd[0][i] = dRec[i]; for (let m = 0; m < M; m++) { gWd[i][m] = dRec[i] * code[m]; dCode[m] += Wd[i][m] * dRec[i]; } }
      if (!k) for (let m = 0; m < M; m++) dCode[m] += l1 * (code[m] > 0 ? 1 : 0);   // L1 SOLO in modalità storica
      const gWe = matZeros(M, d), gbe = matZeros(1, M);
      // il gradiente del codificatore passa solo dai concetti VIVI (relu acceso e, in TopK, scelti)
      for (let m = 0; m < M; m++) { const g = (k ? code[m] > 0 : pre[m] > 0) ? dCode[m] : 0; gbe[0][m] = g; for (let i = 0; i < d; i++) gWe[m][i] = g * z[i]; }
      opt.step([["We", We, gWe], ["be", be, gbe], ["Wd", Wd, gWd], ["bd", bd, gbd]]);
      if ((t & 31) === 0) normDec();
      // RIANIMA le feature morte: ogni tanto reinizializza quelle mai accese nell'intervallo
      if (k && t > 0 && t % resampleEvery === 0) {
        for (let m = 0; m < M; m++) if (fires[m] === 0) { for (let i = 0; i < d; i++) { We[m][i] = rnd(-0.1, 0.1); Wd[i][m] = rnd(-0.1, 0.1); } be[0][m] = 0.1; }
        normDec();
        for (let m = 0; m < M; m++) fires[m] = 0;
      }
    }
    normDec();
    // statistiche finali + conteggio feature morte (sull'intero A)
    let active = 0, rmse = 0; const fireA = new Float64Array(M);
    for (const r of A) { const z = norm(r), c = encode(z); for (let m = 0; m < M; m++) if (c[m] > 1e-2) { active++; fireA[m]++; } const rec = new Float64Array(d); for (let i = 0; i < d; i++) { let s = bd[0][i]; for (let m = 0; m < M; m++) s += Wd[i][m] * c[m]; rec[i] = s; } let e = 0; for (let i = 0; i < d; i++) e += (rec[i] - z[i]) ** 2; rmse += Math.sqrt(e / d); }
    let dead = 0; for (let m = 0; m < M; m++) if (fireA[m] === 0) dead++;
    const decode = (c) => { const rec = new Float64Array(d); for (let i = 0; i < d; i++) { let s = bd[0][i]; for (let m = 0; m < M; m++) s += Wd[i][m] * c[m]; rec[i] = s; } return rec; };
    const denorm = (z) => { const r = new Float64Array(d); for (let i = 0; i < d; i++) r[i] = z[i] * std[i] + mean[i]; return r; };
    return { M, d, k, dead, We, Wd, be, bd, mean, std, norm, denorm, encode, decode, avgActive: active / N, rmse: rmse / N };
  }

  /* ============================================================
     CONCETTI: dai un nome ai concetti tramite le parole-seguenti
     che ciascuno favorisce (interpretazione, non verità di base)
     ============================================================ */
  function characterizeConcepts(model, sae, A, vocab) {
    const M = sae.M, codes = A.map(r => sae.encode(sae.norm(r)));
    const maxc = new Float64Array(M); codes.forEach(c => { for (let m = 0; m < M; m++) if (c[m] > maxc[m]) maxc[m] = c[m]; });
    const fires = (c, m) => c[m] > 0.12;
    const concept = [];
    for (let m = 0; m < M; m++) {
      // effetto del concetto sulla distribuzione d'uscita: direzione decoder → logit
      const dir = new Float64Array(sae.d); for (let k = 0; k < sae.d; k++) dir[k] = sae.Wd[k][m] * sae.std[k];
      const lg = model.logitsAt(dir);
      if (vocab.unk != null) lg[vocab.unk] = -1e9;
      const top = [...lg.keys()].sort((a, b) => lg[b] - lg[a]).slice(0, 4).map(i => vocab.words[i] || "□");
      let fr = 0; for (const c of codes) if (fires(c, m)) fr++;
      concept.push({ m, freq: fr, favors: top, name: "concetto→" + top.slice(0, 2).join("/") });
    }
    return { concept, fires, codes, maxc };
  }

  function extractGenome(model, sae, A, vocab, G) {
    const ch = characterizeConcepts(model, sae, A, vocab);
    const order = [...Array(sae.M).keys()].sort((a, b) => ch.concept[b].freq - ch.concept[a].freq);
    const genes = order.map(m => ch.concept[m]);
    const conceptKey = (c) => "k" + c.m + "_" + c.favors.slice(0, 2).join("-");
    const words = genes.map(conceptKey);
    const dict = G.loadDictionary(words, { id: "concetti-generatore", version: vocab.dict_version || "0" });
    return { dict, order, genes, words, ch, sae, conceptKey };
  }

  // ESPRESSIONE: per un contesto, la parola scelta + concetti accesi + filamento
  function derive(model, ctxIds, genome, vocab, G) {
    const f = model.forward(ctxIds);
    const rep = Float64Array.from(f.X2[f.n - 1]);
    const lg = model.logitsAt(rep); if (vocab.unk != null) lg[vocab.unk] = -1e9;
    const probs = softmaxInto(lg);
    const pred = argmax(probs);
    const code = genome.sae.encode(genome.sae.norm(rep));
    const fired = [];
    for (let m = 0; m < genome.sae.M; m++) if (genome.ch.fires(code, m)) fired.push({ m, name: genome.ch.concept[m].name, favors: genome.ch.concept[m].favors, act: code[m] });
    fired.sort((a, b) => b.act - a.act);
    const SP = G.CTRL.SP;
    const filament = fired.map(x => G.addressOf(genome.dict.index[genome.conceptKey(genome.ch.concept[x.m])])).join(SP);
    return { rep, code, probs, pred, predWord: vocab.words[pred] || "□", fired, filament };
  }

  // INTERVENTO CAUSALE: parto dalla rappresentazione VERA (niente errore di
  // ricostruzione nel riferimento) e tolgo SOLO il contributo del concetto:
  //   z' = norm(rep) − code[m]·decoder[:,m] ; poi rieseguo la testa.
  // Guardo se la PAROLA scelta cambia e quanto cala la sua probabilità.
  function intervene(model, genome, trace, vocab) {
    const sae = genome.sae, d = sae.d;
    const head = (vecD) => { const lg = model.logitsAt(vecD); if (vocab.unk != null) lg[vocab.unk] = -1e9; const p = softmaxInto(lg); return { w: argmax(p), p }; };
    const z0 = sae.norm(trace.rep);
    const base = head(trace.rep);                 // riferimento = modello reale
    const baseConf = base.p[base.w];
    const drop = (after) => baseConf > 0 ? Math.max(0, Math.min(1, (baseConf - after) / baseConf)) : 0;
    const ablate = (ms) => { const z = z0.slice(); for (const m of ms) for (let k = 0; k < d; k++) z[k] -= trace.code[m] * sae.Wd[k][m]; return head(sae.denorm(z)); };
    const rows = trace.fired.map(x => {
      const r = ablate([x.m]);
      return { m: x.m, name: x.name, favors: x.favors, newWord: vocab.words[r.w] || "□", causal: r.w !== base.w, dropPct: drop(r.p[base.w]) };
    });
    const fidelitySingle = trace.fired.length ? rows.filter(r => r.causal).length / trace.fired.length : 0;
    let allConf = baseConf, allWord = base.w;
    if (trace.fired.length) { const r = ablate(trace.fired.map(x => x.m)); allConf = r.p[base.w]; allWord = r.w; }
    const fidelityGraded = drop(allConf);
    return { baseWord: vocab.words[base.w] || "□", baseConf, rows, fidelitySingle, fidelityGraded, allWord: vocab.words[allWord] || "□" };
  }

  function seal(genome, filament, G) { return { genomeHash: genome.dict.hash, traceHash: G.fingerprint(filament) }; }

  /* ============================================================
     PERSISTENZA — salva/carica il modello (pesi) su disco come JSON
     ============================================================ */
  const KEYS = ["Wemb", "Wpos", "Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2", "Wout", "bout"];
  function serialize(model, vocab) {
    const w = {}; for (const k of KEYS) w[k] = model[k].map(r => Array.from(r));
    return { format: "sapere-dna-generator", version: 1, cfg: { V: model.V, Lc: model.Lc, d: model.d, h: model.h },
      vocab: { V: vocab.V, cap: vocab.cap, unk: vocab.unk, dict_id: vocab.dict_id, dict_version: vocab.dict_version, dict_hash: vocab.dict_hash }, weights: w };
  }
function deserialize(obj, vocab) {
    const m = new LM(vocab, obj.cfg); for (const k of KEYS) m[k] = obj.weights[k].map(r => Float64Array.from(r)); return m;
  }

  /* ============================================================
     SAE attraverso il confine del Worker
     ------------------------------------------------------------
     L'oggetto SAE di trainSAE contiene CLOSURE (norm/encode/…)
     che non si possono trasferire tra thread. Lo smontiamo nei
     soli DATI (saeToRaw) per inviarlo, e ricostruiamo le stesse
     closure sul thread principale (buildSAEInterface) — identiche
     a quelle di trainSAE, così extractGenome/derive si comportano
     esattamente come nel percorso sincrono.
     ============================================================ */
  function saeToRaw(sae) {
    const m2 = (M) => M.map(r => Array.from(r));
    return { M: sae.M, d: sae.d, k: sae.k || 0, dead: sae.dead || 0, We: m2(sae.We), Wd: m2(sae.Wd), be: m2(sae.be), bd: m2(sae.bd),
      mean: Array.from(sae.mean), std: Array.from(sae.std), avgActive: sae.avgActive, rmse: sae.rmse };
  }
  function buildSAEInterface(raw) {
    const M = raw.M, d = raw.d, k = raw.k || 0, dead = raw.dead || 0;
    const We = raw.We.map(r => Float64Array.from(r)), Wd = raw.Wd.map(r => Float64Array.from(r));
    const be = raw.be.map(r => Float64Array.from(r)), bd = raw.bd.map(r => Float64Array.from(r));
    const mean = Float64Array.from(raw.mean), std = Float64Array.from(raw.std);
    const norm = (r) => { const z = new Float64Array(d); for (let i = 0; i < d; i++) z[i] = (r[i] - mean[i]) / std[i]; return z; };
    const denorm = (z) => { const r = new Float64Array(d); for (let i = 0; i < d; i++) r[i] = z[i] * std[i] + mean[i]; return r; };
    const topk = (c) => { if (!k || k >= M) return c; const idx = new Array(M); for (let m = 0; m < M; m++) idx[m] = m; idx.sort((a, b) => c[b] - c[a]); for (let r = k; r < M; r++) c[idx[r]] = 0; return c; };
    const encode = (z) => { const c = new Float64Array(M); for (let m = 0; m < M; m++) { let s = be[0][m]; for (let i = 0; i < d; i++) s += We[m][i] * z[i]; c[m] = relu(s); } return k ? topk(c) : c; };
    const decode = (c) => { const rec = new Float64Array(d); for (let i = 0; i < d; i++) { let s = bd[0][i]; for (let m = 0; m < M; m++) s += Wd[i][m] * c[m]; rec[i] = s; } return rec; };
    return { M, d, k, dead, We, Wd, be, bd, mean, std, norm, denorm, encode, decode, avgActive: raw.avgActive, rmse: raw.rmse };
  }

  /* ============================================================
     QUALITÀ DELLA SAE — "perdita recuperata" (loss recovered)
     ------------------------------------------------------------
     Quanta capacità predittiva del modello sopravvive se sostituisco
     la rappresentazione VERA con la sua RICOSTRUZIONE fatta dai soli
     concetti. Misuro su contesti veri la perdita (next-token) con:
       · la rappresentazione vera        → L_vera   (pavimento basso)
       · la ricostruzione della SAE      → L_ricostr.
       · la rappresentazione media       → L_media  (nessuna informazione)
     recuperata = (L_media − L_ricostr.) / (L_media − L_vera), in [0,1].
     1 = i concetti catturano tutto ciò che serve a decidere; 0 = inutili.
     È la prova quantitativa che i concetti sono il contenuto, non un disegno.
     ============================================================ */
  function saeQuality(model, sae, ids, opts) {
    opts = opts || {}; const S = opts.samples || 160, Lc = model.Lc;
    const maxStart = Math.max(1, ids.length - 2);
    const meanRep = sae.denorm(new Float64Array(sae.d));   // z=0 → rappresentazione media
    const nll = (rep, tgt) => { const p = softmaxInto(model.logitsAt(rep)); return -Math.log(Math.max(1e-12, p[tgt])); };
    let Lvera = 0, Lric = 0, Lmedia = 0, cnt = 0;
    for (let s = 0; s < S; s++) {
      const start = (Math.random() * maxStart) | 0;
      const end = Math.min(ids.length, start + Lc);
      if (end >= ids.length) continue;                     // serve la parola VERA successiva
      const win = ids.slice(start, end);
      if (win.length < 1) continue;
      const tgt = ids[end];
      const f = model.forward(win);
      const rep = Float64Array.from(f.X2[f.n - 1]);
      const recon = sae.denorm(sae.decode(sae.encode(sae.norm(rep))));
      Lvera += nll(rep, tgt); Lric += nll(recon, tgt); Lmedia += nll(meanRep, tgt); cnt++;
    }
    if (!cnt) return { lossRecovered: 0, Lvera: 0, Lric: 0, Lmedia: 0, samples: 0 };
    Lvera /= cnt; Lric /= cnt; Lmedia /= cnt;
    const denom = Lmedia - Lvera;
    const lossRecovered = denom > 1e-9 ? Math.max(0, Math.min(1, (Lmedia - Lric) / denom)) : 0;
    return { lossRecovered, Lvera, Lric, Lmedia, samples: cnt };
  }
  
  /* ============================================================
     SOVRAPPOSIZIONE (Asse 1) — la prova che il modello impacchetta
     più concetti che dimensioni.
       · decoderGeometry: le direzioni-decoder dei concetti vivono in
         uno spazio di sole d dimensioni. Se i concetti sono M>d non
         possono essere tutte ortogonali → interferenza (coseni
         fuori-diagonale ≠ 0), con coppie quasi-antipodali. È lo
         STRUMENTO che rende osservabile la sovrapposizione.
       · causalCensus: su molti contesti veri, quanti concetti sono
         SINGOLARMENTE causali (spegnerli cambia la parola)? Se sono
         più di d, il modello ha davvero impacchettato più feature
         che dimensioni: questa è la PROVA. Riusa derive+intervene,
         così la matematica è identica al percorso a singolo passo.
     ============================================================ */
  function decoderGeometry(sae, opts) {
    opts = opts || {};
    const tauA = opts.antipodal == null ? 0.4 : opts.antipodal;   // coppie quasi-antipodali: coseno < -tauA
    const tauI = opts.interfere == null ? 0.4 : opts.interfere;   // coppie interferenti: |coseno| > tauI
    const topN = opts.topN || 8;
    const M = sae.M, d = sae.d, Wd = sae.Wd;
    const nrm = new Float64Array(M);
    for (let m = 0; m < M; m++) { let s = 0; for (let k = 0; k < d; k++) s += Wd[k][m] * Wd[k][m]; nrm[m] = Math.sqrt(s) || 1; }
    const gram = Array.from({ length: M }, () => new Float64Array(M));
    let sumAbs = 0, cnt = 0, maxAbs = 0, antipodal = 0, interfering = 0;
    const pairs = [];
    for (let i = 0; i < M; i++) {
      gram[i][i] = 1;
      for (let j = i + 1; j < M; j++) {
        let dot = 0; for (let k = 0; k < d; k++) dot += Wd[k][i] * Wd[k][j];
        const c = dot / (nrm[i] * nrm[j]);
        gram[i][j] = c; gram[j][i] = c;
        const a = Math.abs(c);
        sumAbs += a; cnt++; if (a > maxAbs) maxAbs = a;
        if (c < -tauA) antipodal++;
        if (a > tauI) interfering++;
        pairs.push({ i, j, cos: c });
      }
    }
    pairs.sort((p, q) => Math.abs(q.cos) - Math.abs(p.cos));
    return {
      M, d, overcomplete: M > d, gram,
      offdiagMean: cnt ? sumAbs / cnt : 0, offdiagMax: maxAbs,
      antipodalCount: antipodal, interferingCount: interfering, topPairs: pairs.slice(0, topN)
    };
  }

  function causalCensus(model, genome, vocab, ids, G, opts) {
    opts = opts || {};
    const S = opts.samples || 100, minSeen = opts.minSeen || 3;
    const Lc = model.Lc, M = genome.sae.M, d = model.d;
    const seen = new Float64Array(M), causal = new Float64Array(M), dropSum = new Float64Array(M);
    const maxStart = Math.max(1, ids.length - 2);
    let used = 0;
    for (let s = 0; s < S; s++) {
      const start = (Math.random() * maxStart) | 0;
      const win = ids.slice(start, Math.min(ids.length, start + Lc));
      if (win.length < 2) continue;
      const trace = derive(model, win, genome, vocab, G);
      used++;
      if (!trace.fired.length) continue;
      const iv = intervene(model, genome, trace, vocab);
      for (const r of iv.rows) { seen[r.m]++; if (r.causal) causal[r.m]++; dropSum[r.m] += r.dropPct; }
    }
    const perConcept = [];
    let causalEver = 0, causalRobust = 0;
    for (let m = 0; m < M; m++) {
      if (seen[m] === 0) continue;
      const rate = causal[m] / seen[m];
      const robust = seen[m] >= minSeen && rate >= 0.5;
      if (causal[m] >= 1) causalEver++;
      if (robust) causalRobust++;
      perConcept.push({ m, seen: seen[m], causal: causal[m], rate, avgDrop: dropSum[m] / seen[m], robust, favors: genome.ch.concept[m].favors });
    }
    perConcept.sort((a, b) => b.causal - a.causal || b.rate - a.rate);
    return { samples: used, d, M, overcomplete: M > d, causalEver, causalRobust, superposition: causalRobust > d, perConcept };
  }

  /* ============================================================
     SBIRCIATA · concetti "primitivi" di un blocco intermedio
     ------------------------------------------------------------
     I blocchi intermedi (es. il 1° in Cassandra a 2 lastre) formano
     concetti che il blocco finale ricompone. Non producono parola
     direttamente, quindi NON li caratterizziamo per "quale parola
     favoriscono" (sarebbe falso), ma per COSA LI ACCENDE: la parola
     più recente del contesto in cui si attivano di più. È un ritratto
     onesto di una feature intermedia. Il "cosa fanno a valle" è il
     tracciamento causale profondo, un passo a sé.
     opts.block = indice del blocco (0 = primo). Se il modello non ha
     `cals`, ricade sul vettore terminale (innocuo).
     ============================================================ */
  function characterizeByInput(sae, A, tok, vocab, opts) {
    opts = opts || {};
    const topC = opts.topConcepts || 14, topW = opts.topWords || 4, M = sae.M;
    const use = new Float64Array(M);
    const wordAct = []; for (let m = 0; m < M; m++) wordAct.push(new Map());
    for (let i = 0; i < A.length; i++) {
      const c = sae.encode(sae.norm(A[i])), w = tok[i];
      for (let m = 0; m < M; m++) if (c[m] > 1e-3) { use[m] += c[m]; wordAct[m].set(w, (wordAct[m].get(w) || 0) + c[m]); }
    }
    const words = (vocab && vocab.words) || [];
    const order = [];
    for (let m = 0; m < M; m++) if (use[m] > 0) order.push(m);
    order.sort((a, b) => use[b] - use[a]);
    return order.slice(0, topC).map(function (m) {
      const arr = Array.from(wordAct[m].entries()).sort((a, b) => b[1] - a[1]).slice(0, topW)
        .map(function (e) { return { word: words[e[0]] != null ? words[e[0]] : ("#" + e[0]), score: e[1] }; });
      return { concept: m, use: use[m], words: arr };
    });
  }
  function peekConcepts(model, ids, vocab, opts) {
    opts = opts || {};
    const N = opts.N || 500, Lc = model.Lc, block = opts.block;
    const M = opts.M || 24, k = opts.k || 6, steps = opts.steps || 4000;
    const A = [], tok = [];
    const maxStart = Math.max(1, ids.length - Lc);
    for (let s = 0; s < N; s++) {
      const start = (Math.random() * maxStart) | 0;
      const win = ids.slice(start, start + Lc);
      if (win.length < 1) continue;
      const f = model.forward(win);
      const vec = (block != null && f.cals) ? f.cals[block] : f.X2;
      A.push(Float64Array.from(vec[f.n - 1]));
      tok.push(win[win.length - 1]);
    }
    const sae = trainSAE(A, { M: M, k: k, steps: steps });
    const concepts = characterizeByInput(sae, A, tok, vocab, { topConcepts: opts.topConcepts || 14, topWords: opts.topWords || 4 });
    return { block: (block == null ? -1 : block), M: sae.M, k: sae.k, dead: sae.dead, avgActive: sae.avgActive, concepts: concepts, saeRaw: saeToRaw(sae) };
  }
  
  /* TRACCIAMENTO CAUSALE PROFONDO: spengo un primitivo del calloso intermedio,
     lascio propagare l'effetto (riproiezione → blocco/i a valle → testa) e guardo
     se la PAROLA finale cambia. È l'intervento causale applicato in profondità.
     `peek` è la SAE dei primitivi ricostruita (buildSAEInterface(peek.saeRaw)).
     Onestà: spengo solo all'ultima posizione, come l'intervento al calloso terminale. */
  function deepTrace(model, ctxIds, peek, vocab, opts) {
    opts = opts || {};
    const block = opts.block == null ? 0 : opts.block;
    const L = model.L || 1;
    if (typeof model.forwardFromCal !== "function" || block >= L - 1) return null;   // niente blocco intermedio a valle
    const head = (vecD) => { const lg = model.logitsAt(vecD); if (vocab.unk != null) lg[vocab.unk] = -1e9; const p = softmaxInto(lg); return { w: argmax(p), p }; };
    const f = model.forward(ctxIds);
    const calLast = Float64Array.from(f.cals[block][f.n - 1]);   // calloso intermedio reale, ultima posizione
    const base = head(f.X2[f.n - 1]);                            // parola realmente predetta
    const baseConf = base.p[base.w];
    const drop = (after) => baseConf > 0 ? Math.max(0, Math.min(1, (baseConf - after) / baseConf)) : 0;
    const z = peek.norm(calLast), code = peek.encode(z);
    const fired = [];
    for (let m = 0; m < peek.M; m++) if (code[m] > 1e-6) fired.push({ m, act: code[m] });
    fired.sort((a, b) => b.act - a.act);
    const ablate = (ms) => {
      const zz = z.slice();
      for (const m of ms) for (let k = 0; k < peek.d; k++) zz[k] -= code[m] * peek.Wd[k][m];
      return head(model.forwardFromCal(ctxIds, block, peek.denorm(zz)));
    };
    const rows = fired.map(function (x) {
      const r = ablate([x.m]);
      return { m: x.m, act: x.act, newWord: vocab.words[r.w] || "□", causal: r.w !== base.w, dropPct: drop(r.p[base.w]) };
    });
    const fidelitySingle = fired.length ? rows.filter(r => r.causal).length / fired.length : 0;
    let allConf = baseConf, allWord = base.w;
    if (fired.length) { const r = ablate(fired.map(x => x.m)); allConf = r.p[base.w]; allWord = r.w; }
    return { block, baseWord: vocab.words[base.w] || "□", baseConf, rows, fidelitySingle, fidelityGraded: drop(allConf), allWord: vocab.words[allWord] || "□" };
  }

  const API = {
      makeVocab, encode, decode, LM, train, perplexity, generate,
      lastPosMatrix, trainSAE, saeQuality, characterizeConcepts, extractGenome, derive, intervene, seal,
      serialize, deserialize, saeToRaw, buildSAEInterface, argmax,
      decoderGeometry, causalCensus, peekConcepts, characterizeByInput, deepTrace
    };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else root.SapereDNAGeneratore = API;

})(typeof window !== "undefined" ? window : this);