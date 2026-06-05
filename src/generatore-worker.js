/* ============================================================
   Sapere-DNA Studio · GENERATORE — Web Worker
   ------------------------------------------------------------
   Esegue i lavori PESANTI fuori dal thread dell'interfaccia,
   così la finestra resta viva con una barra di avanzamento:
     · job "train" — addestra il Transformer causale a parole;
       calcola perplessità prima/dopo e rispedisce i PESI.
     · job "sae"   — raccoglie le attivazioni dell'ultima posizione
       (lastPosMatrix) e addestra il dizionario sparso (trainSAE);
       rispedisce la matrice A e i DATI della SAE (senza closure).

   Il motore è l'UNICA fonte di verità: questo Worker importa
   generatore.js (nessun codice duplicato). L'estrazione del
   genoma dei concetti resta sul thread principale, perché usa
   il motore Genoma (G) e produce closure non trasferibili.

   Protocollo (main → worker):  { job, ... }
   Protocollo (worker → main):  { type: "progress" | "done" | "error", ... }
   ============================================================ */
"use strict";

importScripts("generatore.js");
var Gen = self.SapereDNAGeneratore;

self.onmessage = function (e) {
  var msg = e.data || {};
  try {
    if (!Gen) { self.postMessage({ type: "error", job: msg.job, message: "motore generatore non caricato nel worker" }); return; }

    if (msg.job === "train") {
      // --- addestramento del modello linguistico ---
      var model = new Gen.LM(msg.vocab, msg.cfg);
      var slice = msg.pplSlice ? msg.ids.slice(0, msg.pplSlice) : msg.ids;
      var ppl0 = Gen.perplexity(model, slice);
      var r = Gen.train(model, msg.ids, {
        steps: msg.steps,
        lr: msg.lr,
        onProgress: function (frac, loss, ppl) {
          self.postMessage({ type: "progress", job: "train", frac: frac, loss: loss, ppl: ppl });
        }
      });
      var ppl1 = Gen.perplexity(model, slice);
      var weights = Gen.serialize(model, msg.vocab);
      self.postMessage({ type: "done", job: "train", weights: weights, loss: r.loss, ppl0: ppl0, ppl1: ppl1 });

    } else if (msg.job === "sae") {
      // --- dizionario sparso (concetti) sull'ultima posizione ---
      var m = Gen.deserialize(msg.weights, msg.vocab);
      var A = Gen.lastPosMatrix(m, msg.ids, { N: msg.N });
      self.postMessage({ type: "progress", job: "sae", frac: 0.45 });   // raccolta attivazioni: ~prima metà
      var sae = Gen.trainSAE(A, { M: msg.M, steps: msg.steps, lr: msg.lr, l1: msg.l1 });
      self.postMessage({ type: "progress", job: "sae", frac: 0.95 });
      self.postMessage({ type: "done", job: "sae", A: A, saeRaw: Gen.saeToRaw(sae) });

    } else {
      self.postMessage({ type: "error", job: msg.job, message: "lavoro sconosciuto: " + msg.job });
    }
  } catch (err) {
    self.postMessage({ type: "error", job: msg.job, message: String((err && err.message) || err) });
  }
};
