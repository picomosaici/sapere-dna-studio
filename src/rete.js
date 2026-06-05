/* ============================================================
   Sapere-DNA Studio · RETE (motore neurale + ponte di trasparenza)
   - una piccola rete addestrabile (classificatore a quadranti),
     evoluzione del progetto originale dell'utente
   - estrae un "genoma dei concetti" dai neuroni nascosti
   - deriva il tracciato (quali concetti si accendono), lo incide
     come filamento SDNA, lo mette alla prova con l'INTERVENTO
     (ablazione) e lo SIGILLA con le impronte del motore Genoma.
   Espone window.SapereDNARete  (module.exports in Node)
   ============================================================ */
(function (root) {
  "use strict";

  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize, learningRate) {
      this.inputSize = inputSize; this.hiddenSize = hiddenSize; this.outputSize = outputSize;
      this.learningRate = (learningRate == null) ? 0.3 : learningRate;
      this.weightsInputToHidden = Array.from({ length: hiddenSize }, () =>
        Array.from({ length: inputSize }, () => Math.random() * 2 - 1));
      this.biasHidden = Array(hiddenSize).fill(0);
      this.weightsHiddenToOutput = Array.from({ length: outputSize }, () =>
        Array.from({ length: hiddenSize }, () => Math.random() * 2 - 1));
      this.biasOutput = Array(outputSize).fill(0);
      this.hiddenLayer = new Array(hiddenSize);
    }
    feedForward(inputs, ablate) {
      for (let i = 0; i < this.hiddenSize; i++) {
        let s = 0;
        for (let j = 0; j < this.inputSize; j++) s += this.weightsInputToHidden[i][j] * inputs[j];
        this.hiddenLayer[i] = sigmoid(s + this.biasHidden[i]);
      }
      if (ablate != null) { // INTERVENTO: spegne uno o più concetti
        if (Array.isArray(ablate)) ablate.forEach(k => { this.hiddenLayer[k] = 0; });
        else this.hiddenLayer[ablate] = 0;
      }
      const out = new Array(this.outputSize);
      for (let i = 0; i < this.outputSize; i++) {
        let s = 0;
        for (let j = 0; j < this.hiddenSize; j++) s += this.weightsHiddenToOutput[i][j] * this.hiddenLayer[j];
        out[i] = sigmoid(s + this.biasOutput[i]);
      }
      return out;
    }
    train(inputs, target) {
      // forward
      for (let i = 0; i < this.hiddenSize; i++) {
        let s = 0;
        for (let j = 0; j < this.inputSize; j++) s += this.weightsInputToHidden[i][j] * inputs[j];
        this.hiddenLayer[i] = sigmoid(s + this.biasHidden[i]);
      }
      const out = new Array(this.outputSize);
      for (let i = 0; i < this.outputSize; i++) {
        let s = 0;
        for (let j = 0; j < this.hiddenSize; j++) s += this.weightsHiddenToOutput[i][j] * this.hiddenLayer[j];
        out[i] = sigmoid(s + this.biasOutput[i]);
      }
      // backprop corretto (con i delta — versione ripulita dell'originale)
      const deltaO = new Array(this.outputSize);
      for (let i = 0; i < this.outputSize; i++) {
        const err = target[i] - out[i];
        deltaO[i] = err * out[i] * (1 - out[i]);
      }
      const deltaH = new Array(this.hiddenSize);
      for (let i = 0; i < this.hiddenSize; i++) {
        let s = 0;
        for (let j = 0; j < this.outputSize; j++) s += this.weightsHiddenToOutput[j][i] * deltaO[j];
        deltaH[i] = s * this.hiddenLayer[i] * (1 - this.hiddenLayer[i]);
      }
      for (let i = 0; i < this.outputSize; i++) {
        for (let j = 0; j < this.hiddenSize; j++) this.weightsHiddenToOutput[i][j] += this.learningRate * deltaO[i] * this.hiddenLayer[j];
        this.biasOutput[i] += this.learningRate * deltaO[i];
      }
      for (let i = 0; i < this.hiddenSize; i++) {
        for (let j = 0; j < this.inputSize; j++) this.weightsInputToHidden[i][j] += this.learningRate * deltaH[i] * inputs[j];
        this.biasHidden[i] += this.learningRate * deltaH[i];
      }
    }
  }

  const LABELS = ["blu", "rosso", "verde", "viola"];
  const COLORS = { blu: "#4f8bd0", rosso: "#e08a8a", verde: "#7fce9a", viola: "#b89ae8" };
  function labelOf(x, y) { if (x <= 0 && y < 0) return "blu"; if (x <= 0 && y > 0) return "verde"; if (x > 0 && y <= 0) return "rosso"; return "viola"; }
  function oneHotEncode(l) { return { blu: [1, 0, 0, 0], rosso: [0, 1, 0, 0], verde: [0, 0, 1, 0], viola: [0, 0, 0, 1] }[l]; }
  function oneHotDecode(o) { let mi = 0; for (let i = 1; i < o.length; i++) if (o[i] > o[mi]) mi = i; return LABELS[mi]; }
  function makeData(n) { const d = []; for (let i = 0; i < n; i++) { const x = Math.random() * 2 - 1, y = Math.random() * 2 - 1; d.push({ x, y, label: labelOf(x, y) }); } return d; }

  // nome leggibile del neurone: rilevatore di semipiano, letto dal peso dominante
  function conceptName(net, i) {
    const a = net.weightsInputToHidden[i][0], b = net.weightsInputToHidden[i][1];
    if (Math.abs(a) >= Math.abs(b)) return a >= 0 ? "semipiano-x-positivo" : "semipiano-x-negativo";
    return b >= 0 ? "semipiano-y-positivo" : "semipiano-y-negativo";
  }
  const geneKey = (neuron, name) => "n" + neuron + "_" + name;

  // estrae il genoma dei concetti dai neuroni nascosti (ordinati per frequenza di accensione)
  function extractGenome(net, data, G, thresh) {
    thresh = (thresh == null) ? 0.5 : thresh;
    const freq = Array(net.hiddenSize).fill(0);
    for (const d of data) { net.feedForward([d.x, d.y]); for (let i = 0; i < net.hiddenSize; i++) if (net.hiddenLayer[i] > thresh) freq[i]++; }
    const order = [...Array(net.hiddenSize).keys()].sort((p, q) => freq[q] - freq[p]);
    const genes = order.map(i => ({
      neuron: i, name: conceptName(net, i),
      weights: [net.weightsInputToHidden[i][0], net.weightsInputToHidden[i][1]],
      bias: net.biasHidden[i], freq: freq[i]
    }));
    const words = genes.map(g => geneKey(g.neuron, g.name));
    const dict = G.loadDictionary(words, { id: "concetti-rete", version: "q1" });
    return { dict, order, freq, genes, words, thresh };
  }

  // deriva il tracciato per un punto: quali concetti si accendono, in che ordine
  function derive(net, point, genome, G) {
    const out = net.feedForward(point);
    const pred = oneHotDecode(out);
    const fired = [];
    for (let i = 0; i < net.hiddenSize; i++) if (net.hiddenLayer[i] > genome.thresh) fired.push({ neuron: i, name: conceptName(net, i), act: net.hiddenLayer[i] });
    fired.sort((a, b) => b.act - a.act);
    const SP = G.CTRL.SP;
    const filament = fired.map(f => G.addressOf(genome.dict.index[geneKey(f.neuron, f.name)])).join(SP);
    return { out, pred, fired, filament };
  }

  // INTERVENTO: tre letture complementari
  //  (1) singola  — spegne un concetto per volta: severa, penalizza la ridondanza
  //  (2) gruppo   — raggruppa i concetti con lo stesso nome e li spegne insieme: scioglie la ridondanza
  //  (3) graduata — spegne TUTTI i concetti accesi e misura il crollo di sicurezza: lettura morbida
  function intervene(net, point, trace) {
    const predIdx = LABELS.indexOf(trace.pred);
    const predConf = trace.out[predIdx];
    const norm = (after) => predConf > 0 ? Math.max(0, Math.min(1, (predConf - after) / predConf)) : 0;

    // (1) singola
    const rows = trace.fired.map(f => {
      const out2 = net.feedForward(point, f.neuron);
      const newPred = oneHotDecode(out2);
      const after = out2[predIdx];
      return { neuron: f.neuron, name: f.name, newPred, causal: newPred !== trace.pred, confAfter: after, dropPct: norm(after) };
    });
    const fidelitySingle = trace.fired.length ? rows.filter(r => r.causal).length / trace.fired.length : 0;

    // (2) di gruppo (per nome di concetto)
    const byName = {};
    trace.fired.forEach(f => { (byName[f.name] = byName[f.name] || []).push(f.neuron); });
    const groups = Object.keys(byName).map(name => {
      const neurons = byName[name];
      const out2 = net.feedForward(point, neurons);
      const newPred = oneHotDecode(out2);
      const after = out2[predIdx];
      return { name, neurons, newPred, causal: newPred !== trace.pred, confAfter: after, dropPct: norm(after) };
    });
    const fidelityGroup = groups.length ? groups.filter(g => g.causal).length / groups.length : 0;

    // (3) graduata: spegne tutti i concetti accesi insieme
    let fidelityGraded = 0, allPred = trace.pred, allConf = predConf;
    if (trace.fired.length) {
      const outAll = net.feedForward(point, trace.fired.map(f => f.neuron));
      allConf = outAll[predIdx]; allPred = oneHotDecode(outAll); fidelityGraded = norm(allConf);
    }

    return {
      rows, groups, predConf, allPred, allConf,
      fidelitySingle, fidelityGroup, fidelityGraded,
      fidelity: fidelitySingle, causali: rows.filter(r => r.causal).length // compatibilità
    };
  }

  function seal(genome, filament, G) { return { genomeHash: genome.dict.hash, traceHash: G.fingerprint(filament) }; }

  // ---- LENTI per sciogliere la ridondanza (la "sovrapposizione") ----
  // matrice delle attivazioni dei neuroni nascosti su tutto il dataset (N×H)
  function activationMatrix(net, data) {
    const A = [];
    for (const d of data) { net.feedForward([d.x, d.y]); A.push(net.hiddenLayer.slice()); }
    return A;
  }
  // autovalori/autovettori di matrice simmetrica (Jacobi ciclico)
  function jacobiEig(S) {
    const n = S.length, A = S.map(r => r.slice());
    const V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0));
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0; for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < 1e-12) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-14) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
        for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
      }
    }
    const eig = []; for (let i = 0; i < n; i++) eig.push({ val: A[i][i], vec: V.map(r => r[i]) });
    eig.sort((a, b) => b.val - a.val); return eig;
  }
  // PCA: riscrive i concetti in una base ORTOGONALE (niente doppioni) e ne misura la ridondanza
  function pcaConcepts(A) {
    const N = A.length, H = A[0].length;
    const mean = Array(H).fill(0); for (const r of A) for (let j = 0; j < H; j++) mean[j] += r[j] / N;
    const C = Array.from({ length: H }, () => Array(H).fill(0));
    for (const r of A) for (let i = 0; i < H; i++) for (let j = 0; j < H; j++) C[i][j] += (r[i] - mean[i]) * (r[j] - mean[j]) / (N - 1);
    const eig = jacobiEig(C);
    const total = eig.reduce((s, e) => s + Math.max(0, e.val), 0) || 1;
    const ratio = eig.map(e => Math.max(0, e.val) / total);
    let cum = 0, effDim = 0; for (; effDim < ratio.length;) { cum += ratio[effDim++]; if (cum >= 0.95) break; }
    let maxOff = 0, pairs = 0;
    for (let i = 0; i < H; i++) for (let j = i + 1; j < H; j++) {
      const a = Math.abs(C[i][j] / (Math.sqrt(C[i][i] * C[j][j]) || 1));
      if (a > maxOff) maxOff = a; if (a > 0.9) pairs++;
    }
    return { H, effDim, ratio, maxOff, pairs };
  }
  // dizionario sparso minimale (autoencoder con penalità L1): concetti rari e puliti
  function sparseDictionary(A, opts) {
    opts = opts || {}; const H = A[0].length, N = A.length;
    const M = opts.M || 2 * H, steps = opts.steps || 30000, lr = opts.lr || 0.05, l1 = opts.l1 || 0.02;
    const We = Array.from({ length: M }, () => Array.from({ length: H }, () => Math.random() * 0.2 - 0.1));
    const Wd = Array.from({ length: H }, () => Array.from({ length: M }, () => Math.random() * 0.2 - 0.1));
    const be = Array(M).fill(0), bd = Array(H).fill(0), relu = x => x > 0 ? x : 0;
    for (let s = 0; s < steps; s++) {
      const a = A[(Math.random() * N) | 0], z = Array(M), code = Array(M);
      for (let m = 0; m < M; m++) { let acc = be[m]; for (let h = 0; h < H; h++) acc += We[m][h] * a[h]; z[m] = acc; code[m] = relu(acc); }
      const rec = Array(H);
      for (let h = 0; h < H; h++) { let acc = bd[h]; for (let m = 0; m < M; m++) acc += Wd[h][m] * code[m]; rec[h] = acc; }
      const dRec = Array(H); for (let h = 0; h < H; h++) dRec[h] = 2 * (rec[h] - a[h]) / H;
      const dCode = Array(M).fill(0);
      for (let m = 0; m < M; m++) { for (let h = 0; h < H; h++) dCode[m] += Wd[h][m] * dRec[h]; dCode[m] += l1 * Math.sign(code[m]); }
      for (let h = 0; h < H; h++) { for (let m = 0; m < M; m++) Wd[h][m] -= lr * dRec[h] * code[m]; bd[h] -= lr * dRec[h]; }
      for (let m = 0; m < M; m++) { const g = z[m] > 0 ? dCode[m] : 0; for (let h = 0; h < H; h++) We[m][h] -= lr * g * a[h]; be[m] -= lr * g; }
    }
    let active = 0, rmse = 0;
    for (const a of A) {
      const code = Array(M); for (let m = 0; m < M; m++) { let acc = be[m]; for (let h = 0; h < H; h++) acc += We[m][h] * a[h]; code[m] = relu(acc); }
      active += code.filter(v => v > 1e-2).length;
      let e = 0; for (let h = 0; h < H; h++) { let acc = bd[h]; for (let m = 0; m < M; m++) acc += Wd[h][m] * code[m]; e += (acc - a[h]) ** 2; } rmse += Math.sqrt(e / H);
    }
    return { M, avgActive: active / N, rmse: rmse / N };
  }
  // analisi completa della ridondanza: PCA + dizionario sparso
  function analyzeRedundancy(net, data, opts) {
    const A = activationMatrix(net, data);
    return { pca: pcaConcepts(A), sparse: sparseDictionary(A, opts) };
  }

  const API = { NeuralNetwork, LABELS, COLORS, labelOf, oneHotEncode, oneHotDecode, makeData, conceptName, geneKey, extractGenome, derive, intervene, seal, sigmoid, analyzeRedundancy };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else root.SapereDNARete = API;

})(typeof window !== "undefined" ? window : this);