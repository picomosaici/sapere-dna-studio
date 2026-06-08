/* ============================================================
   Sapere-DNA Studio · GENERATORE / CASSANDRA — Web Worker
   ------------------------------------------------------------
   Esegue i lavori PESANTI fuori dal thread dell'interfaccia,
   così la finestra resta viva con una barra di avanzamento.
     · job "train" — addestra il modello; perplessità prima/dopo
       e rispedisce i PESI.
     · job "sae"   — attivazioni dell'ultima posizione + dizionario
       sparso (trainSAE); rispedisce A e i DATI della SAE.

   Due motori, stessa interfaccia:
     · classico   → Gen.LM            (generatore.js)
     · "emisferi" → Mot.LMEmisferi    (motore-emisferi.js, Cassandra)
   Si sceglie con msg.engine === "emisferi". Tutto il resto
   (perplexity, train, lastPosMatrix, trainSAE, saeToRaw) è
   condiviso: funziona identico sui due modelli.

   Protocollo (main → worker):  { job, engine?, ... }
   Protocollo (worker → main):  { type: "progress" | "done" | "error", ... }
   ============================================================ */
"use strict";

importScripts("generatore.js", "emisferi.js", "motore-emisferi.js");
var Gen = self.SapereDNAGeneratore;
var Mot = self.SapereDNAMotoreEmisferi;

function makeModel(engine, vocab, cfg) {
  return engine === "emisferi" ? new Mot.LMEmisferi(vocab, cfg) : new Gen.LM(vocab, cfg);
}
function loadModel(engine, weights, vocab) {
  return engine === "emisferi" ? Mot.deserialize(weights, vocab) : Gen.deserialize(weights, vocab);
}
function dumpModel(engine, model, vocab) {
  return engine === "emisferi" ? Mot.serialize(model, vocab) : Gen.serialize(model, vocab);
}

self.onmessage = function (e) {
  var msg = e.data || {};
  try {
    if (!Gen) { self.postMessage({ type: "error", job: msg.job, message: "motore generatore non caricato nel worker" }); return; }
    if (msg.engine === "emisferi" && !Mot) { self.postMessage({ type: "error", job: msg.job, message: "motore a emisferi (Cassandra) non caricato nel worker" }); return; }

    if (msg.job === "train") {
      // perplexity è condivisa; l'ADDESTRAMENTO di Cassandra usa Mot.train,
      // che scala il passo con la profondità (i modelli a più lastre vanno in
      // stallo a passo pieno). Il generatore classico resta su Gen.train.
      var model = makeModel(msg.engine, msg.vocab, msg.cfg);
      var trainFn = (msg.engine === "emisferi") ? Mot.train : Gen.train;
      var slice = msg.pplSlice ? msg.ids.slice(0, msg.pplSlice) : msg.ids;
      var ppl0 = Gen.perplexity(model, slice);
      var r = trainFn(model, msg.ids, {
        steps: msg.steps,
        lr: msg.lr,
        onProgress: function (frac, loss, ppl) {
          self.postMessage({ type: "progress", job: "train", frac: frac, loss: loss, ppl: ppl });
        }
      });
      var ppl1 = Gen.perplexity(model, slice);
      var weights = dumpModel(msg.engine, model, msg.vocab);
      self.postMessage({ type: "done", job: "train", weights: weights, loss: r.loss, ppl0: ppl0, ppl1: ppl1 });

    } else if (msg.job === "sae") {
      // dizionario sparso (concetti) sull'ultima posizione (= il calloso, per Cassandra)
      var m = loadModel(msg.engine, msg.weights, msg.vocab);
      var A = Gen.lastPosMatrix(m, msg.ids, { N: msg.N });
      self.postMessage({ type: "progress", job: "sae", frac: 0.45 });
      var sae = Gen.trainSAE(A, { M: msg.M, k: msg.k, steps: msg.steps, lr: msg.lr, l1: msg.l1 });
      self.postMessage({ type: "progress", job: "sae", frac: 0.95 });
      self.postMessage({ type: "done", job: "sae", A: A, saeRaw: Gen.saeToRaw(sae) });

    } else if (msg.job === "peek") {
      // sbirciata: concetti "primitivi" del calloso di un blocco intermedio (msg.block)
      var mp = loadModel(msg.engine, msg.weights, msg.vocab);
      self.postMessage({ type: "progress", job: "peek", frac: 0.25 });
      var peek = Gen.peekConcepts(mp, msg.ids, msg.vocab, { N: msg.N, block: msg.block, M: msg.M, k: msg.k, steps: msg.steps });
      self.postMessage({ type: "done", job: "peek", peek: peek });

    } else {
      self.postMessage({ type: "error", job: msg.job, message: "lavoro sconosciuto: " + msg.job });
    }
  } catch (err) {
    self.postMessage({ type: "error", job: msg.job, message: String((err && err.message) || err) });
  }
};