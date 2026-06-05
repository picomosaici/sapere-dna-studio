/* ============================================================
   Sapere-DNA Studio · DECODIFICA (controller)
   Espone window.initDecodifica(APP)
   ============================================================ */
(function () {
  "use strict";

  window.initDecodifica = function (APP) {
    const G = APP.G, C = APP.C;
    const panel = document.getElementById("dec-panel");
    let mode = "genoma";
    let loadedEntry = null;   // voce caricata da .json (se presente)
    let restored = "";

    panel.innerHTML = `
      <div class="seg" id="dec-seg">
        <button data-m="genoma" class="on">Genoma</button>
        <button data-m="classica">Classica</button>
      </div>
      <div class="row" style="margin-bottom:12px">
        <button class="btn violet ghost" id="dec-load">Carica voce .json…</button>
        <span id="dec-loaded" style="font-family:var(--font-mono);font-size:.68rem;color:var(--parch-dim)"></span>
      </div>

      <div id="dec-genoma">
        <div class="flabel"><span>Il filamento di DNA</span><span class="flink" id="dec-load-strand">apri file filamento…</span></div>
        <textarea id="dec-strand" class="mono" placeholder="incolla le basi ATGC del filamento Genoma, oppure carica una voce .json…"></textarea>
      </div>

      <div id="dec-classica" style="display:none">
        <div class="flabel"><span>La sequenza di DNA</span><span class="flink" id="dec-load-dna">apri file DNA…</span></div>
        <textarea id="dec-dna" class="mono" placeholder="incolla le basi ATGC…"></textarea>
        <div class="flabel" style="margin-top:14px"><span>L'indice di recupero (primi × gematria)</span><span class="flink" id="dec-load-idx">apri file indice…</span></div>
        <textarea id="dec-idx" class="mono" placeholder="incolla i numeri separati da spazio…"></textarea>
      </div>

      <div class="row mt">
        <button class="btn violet" id="dec-go">Rileggi il filo &#10038;</button>
        <button class="btn violet ghost" id="dec-clear">Pulisci</button>
      </div>
      <div class="progress" id="dec-prog"><div class="bar"><div class="fill" id="dec-fill"></div></div><div class="lab" id="dec-plab"></div></div>
      <div id="dec-out"></div>`;

    const $ = (id) => document.getElementById(id);
    const out = $("dec-out");

    function setMode(m) {
      mode = m;
      [...$("dec-seg").children].forEach(c => c.classList.toggle("on", c.dataset.m === m));
      $("dec-genoma").style.display = m === "genoma" ? "block" : "none";
      $("dec-classica").style.display = m === "classica" ? "block" : "none";
    }

    $("dec-seg").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      setMode(b.dataset.m); out.innerHTML = "";
    });
    $("dec-clear").addEventListener("click", () => {
      $("dec-strand").value = ""; $("dec-dna").value = ""; $("dec-idx").value = "";
      $("dec-loaded").textContent = ""; loadedEntry = null; out.innerHTML = "";
    });
    $("dec-go").addEventListener("click", run);
    $("dec-load").addEventListener("click", loadEntry);
    $("dec-load-strand").addEventListener("click", async () => { const r = await APP.openText(); if (r && r.ok) $("dec-strand").value = r.content; });
    $("dec-load-dna").addEventListener("click", async () => { const r = await APP.openText(); if (r && r.ok) $("dec-dna").value = r.content; });
    $("dec-load-idx").addEventListener("click", async () => { const r = await APP.openText(); if (r && r.ok) $("dec-idx").value = r.content; });

    async function loadEntry() {
      const r = await APP.openText(); if (!r || !r.ok) return;
      let e;
      try { e = JSON.parse(r.content); } catch (err) { APP.toast("File non valido: serve un .json di questo strumento.", "bad"); return; }
      if (!e || !e.dna) { APP.toast("Voce non valida: manca il campo dna.", "bad"); return; }
      loadedEntry = e;
      if (e.mode === "genoma") {
        setMode("genoma"); $("dec-strand").value = e.dna;
      } else { // classica (anche voci vecchie senza campo mode)
        setMode("classica"); $("dec-dna").value = e.dna; $("dec-idx").value = e.idx || "";
      }
      const reqInfo = e.mode === "genoma" ? ("genoma " + e.dict_id + " v" + e.dict_version) : "classica";
      $("dec-loaded").textContent = "voce caricata: «" + (e.title ? e.title.slice(0, 40) : "(senza titolo)") + "» · " + reqInfo;
      APP.toast("Voce caricata", "ok");
    }

    async function run() {
      const go = $("dec-go"); go.disabled = true; out.innerHTML = "";
      const prog = $("dec-prog"), fill = $("dec-fill"), plab = $("dec-plab");
      prog.classList.add("on");
      const pcb = (f) => { fill.style.width = (f * 100).toFixed(1) + "%"; plab.textContent = "decodifica… " + ((f * 100) | 0) + "%"; };
      const t0 = performance.now();

      if (mode === "genoma") {
        const dna = $("dec-strand").value;
        if (dna.replace(/[^ATGCatgc]/g, "").length < 3) { fail("Manca il filamento di DNA."); return; }
        // se ho la voce caricata uso le sue impronte; altrimenti solo il filamento
        const entry = (loadedEntry && loadedEntry.mode === "genoma" && loadedEntry.dna.replace(/\s+/g, "") === dna.replace(/\s+/g, ""))
          ? loadedEntry : { mode: "genoma", dna: dna };
        const res = await G.decode(entry, APP.dict, pcb);
        const t1 = performance.now();
        finishGenoma(res, entry, t1 - t0);
      } else {
        const dna = $("dec-dna").value;
        const idxArr = $("dec-idx").value.trim().split(/\s+/).filter(x => x.length).map(Number);
        if (dna.replace(/[^ATGCatgc]/g, "").length < 3) { fail("Manca la sequenza di DNA."); return; }
        if (idxArr.some(isNaN)) { fail("L'indice contiene valori non numerici."); return; }
        const res = await C.decode(dna, idxArr, pcb);
        const t1 = performance.now();
        finishClassica(res, t1 - t0);
      }
      go.disabled = false;
    }

    function fail(msg) {
      $("dec-prog").classList.remove("on"); $("dec-fill").style.width = "0";
      out.innerHTML = `<p class="err">${msg}</p>`; $("dec-go").disabled = false;
    }

    function finishGenoma(res, entry, ms) {
      $("dec-prog").classList.remove("on"); $("dec-fill").style.width = "0";
      if (res.error) { out.innerHTML = `<p class="err">${APP.esc(res.error)}</p>`; $("dec-go").disabled = false; return; }
      restored = res.text;
      const vBadge = res.verified === true ? '<span class="badge ok">&#10003; integrità verificata</span>'
        : res.verified === false ? '<span class="badge bad">&#9888; impronta non corrispondente</span>'
        : '<span class="badge warn">verifica non disponibile (manca l\'impronta)</span>';
      out.innerHTML = `
        <div class="out-label"><span>Conoscenza ricostruita (${APP.fmt(res.text.length)} caratteri)</span>${vBadge}</div>
        <div class="box text">${APP.esc(res.text)}</div>
        <div class="statbar"><span>codoni letti: <b>${APP.fmt(res.count)}</b></span><span>caratteri: <b>${APP.fmt(res.text.length)}</b></span><span>tempo: <b>${ms.toFixed(0)} ms</b></span></div>
        ${res.problems ? `<p class="err">${res.problems} simboli non ricostruiti (□): filamento e genoma potrebbero non combaciare, o il file è troncato.</p>`
          : res.verified === false ? '' : '<p class="ok-note">⟡ Il filo è stato riletto.</p>'}
        <div class="row mt"><button class="btn violet" id="dec-save">Salva testo</button></div>`;
      $("dec-save").addEventListener("click", () =>
        APP.saveNative(restored, "ricostruito-" + APP.stamp(), ".txt", "Sto preparando il testo ricostruito (" + APP.fmt(restored.length) + " caratteri)"));
      $("dec-go").disabled = false;
    }

    function finishClassica(res, ms) {
      $("dec-prog").classList.remove("on"); $("dec-fill").style.width = "0";
      if (res.error) { out.innerHTML = `<p class="err">${APP.esc(res.error)}</p>`; $("dec-go").disabled = false; return; }
      restored = res.text;
      out.innerHTML = `
        <div class="out-label"><span>Conoscenza ricostruita (${APP.fmt(res.text.length)} caratteri)</span></div>
        <div class="box text">${APP.esc(res.text)}</div>
        <div class="statbar"><span>codoni letti: <b>${APP.fmt(res.count)}</b></span><span>caratteri: <b>${APP.fmt(res.text.length)}</b></span><span>tempo: <b>${ms.toFixed(0)} ms</b></span></div>
        ${res.problems ? `<p class="err">${res.problems} caratteri non ricostruiti (□): DNA e indice potrebbero non combaciare.</p>` : '<p class="ok-note">⟡ Ricostruzione completa.</p>'}
        <div class="row mt"><button class="btn violet" id="dec-save">Salva testo</button></div>`;
      $("dec-save").addEventListener("click", () =>
        APP.saveNative(restored, "ricostruito-" + APP.stamp(), ".txt", "Sto preparando il testo ricostruito (" + APP.fmt(restored.length) + " caratteri)"));
      $("dec-go").disabled = false;
    }
  };
})();
