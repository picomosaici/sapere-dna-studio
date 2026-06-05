/* ============================================================
   Sapere-DNA · MOTORE unificato (Classico + Genoma)
   ------------------------------------------------------------
   CLASSICO : testo → lettera ebraica → codone DNA + indice gematrico
              (doppio binario, reversibile al 100%)
   GENOMA   : parola → "gene" = indirizzo a larghezza variabile,
              filamento unico, niente indice, auto-verifica con impronte.

   Espone tre globali (browser) / module.exports (Node):
     SapereDNA        → API Classico
     SapereDNAGenoma  → API Genoma
     Motore           → { classic, genoma } (comodità)
   ============================================================ */
(function (root) {
  "use strict";

  /* ============================================================
     PARTE 1 · CLASSICO  (tavola condivisa dei caratteri + gematria)
     ============================================================ */
  const HEB = [
    {l:"א",g:1,codon:"TTA"},{l:"ב",g:2,codon:"GCA"},{l:"ג",g:3,codon:"GGA"},
    {l:"ד",g:4,codon:"GTA"},{l:"ה",g:5,codon:"GAA"},{l:"ו",g:6,codon:"TCA"},
    {l:"ז",g:7,codon:"ATT"},{l:"ח",g:8,codon:"AAA"},{l:"ט",g:9,codon:"CGA"},
    {l:"י",g:10,codon:"GAT"},{l:"כ",g:20,codon:"ACA"},{l:"ל",g:30,codon:"CCA"},
    {l:"מ",g:40,codon:"AAT"},{l:"נ",g:50,codon:"CAA"},{l:"ס",g:60,codon:"TTT"},
    {l:"ע",g:70,codon:"TAT"},{l:"פ",g:80,codon:"ATG"},{l:"צ",g:90,codon:"CAT"},
    {l:"ק",g:100,codon:"TGT"},{l:"ר",g:200,codon:"TGG"},{l:"ש",g:300,codon:"TAA"},
    {l:"ת",g:400,codon:"TGA"}
  ];
  const codon2idx = {};
  HEB.forEach((h, i) => { codon2idx[h.codon] = i; });

  const ORDER = [];
  for (let c = 97; c <= 122; c++) ORDER.push(String.fromCharCode(c));
  for (let c = 65; c <= 90; c++)  ORDER.push(String.fromCharCode(c));
  for (let c = 48; c <= 57; c++)  ORDER.push(String.fromCharCode(c));
  ORDER.push(" ", "\n", "\t");
  [".", ",", ";", ":", "!", "?", "'", "\"", "(", ")", "[", "]", "{", "}",
   "-", "_", "/", "\\", "|", "@", "#", "%", "&", "*", "+", "=", "<", ">",
   "~", "`", "^", "$"].forEach(c => ORDER.push(c));
  ["–", "—", "«", "»", "°", "§", "…", "’", "“", "”"].forEach(c => ORDER.push(c));
  ["à","è","é","ì","í","î","ò","ó","ù","ú","ü","À","È","É","Ì","Ò","Ù","ç","ñ"]
    .forEach(c => ORDER.push(c));

  const ORD = [], _seen = new Set();
  for (const c of ORDER) { if (!_seen.has(c)) { _seen.add(c); ORD.push(c); } }
  const char2pos = {};
  ORD.forEach((c, i) => { char2pos[c] = i; });

  const PRIMES = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97];
  const CHUNK = 40000;
  const yieldUI = () => new Promise(r => setTimeout(r, 0));

  async function classicEncode(text, onProgress) {
    const chars = Array.from(text);
    const n = chars.length;
    const dna = new Array(n);
    const idx = new Array(n);
    const unmapped = new Set();
    let w = 0;
    for (let i = 0; i < n; i++) {
      const p = char2pos[chars[i]];
      if (p === undefined) { unmapped.add(chars[i]); continue; }
      const hi = p % 22, rank = (p / 22) | 0;
      dna[w] = HEB[hi].codon;
      idx[w] = PRIMES[rank] * HEB[hi].g;
      w++;
      if (i % CHUNK === 0 && onProgress) { onProgress(i / n); await yieldUI(); }
    }
    dna.length = w; idx.length = w;
    if (onProgress) onProgress(1);
    return { dna: dna.join(""), idx: idx.join(" "), count: w, bases: w * 3, unmapped: [...unmapped] };
  }

  async function classicDecode(dnaStr, idxArr, onProgress) {
    const dna = String(dnaStr).toUpperCase().replace(/[^ATGC]/g, "");
    const m = (dna.length / 3) | 0;
    if (m !== idxArr.length) {
      return { error: `Disallineamento: ${m} codoni nel DNA ma ${idxArr.length} numeri nell'indice. ` +
        `Devono essere in egual numero — probabilmente DNA e indice non provengono dalla stessa ` +
        `codifica, oppure uno dei due è stato troncato.` };
    }
    const out = new Array(m);
    let problems = 0;
    for (let i = 0; i < m; i++) {
      const cod = dna.substr(i * 3, 3);
      const hi = codon2idx[cod];
      if (hi === undefined) { out[i] = "□"; problems++; }
      else {
        const g = HEB[hi].g;
        const prime = idxArr[i] / g;
        const rank = PRIMES.indexOf(prime);
        const pos = rank >= 0 ? rank * 22 + hi : -1;
        if (rank < 0 || ORD[pos] === undefined) { out[i] = "□"; problems++; }
        else out[i] = ORD[pos];
      }
      if (i % CHUNK === 0 && onProgress) { onProgress(i / m); await yieldUI(); }
    }
    if (onProgress) onProgress(1);
    return { text: out.join(""), problems, count: m };
  }

  function classicHebPreview(dnaStr, limit) {
    limit = limit || 6000;
    const m = (dnaStr.length / 3) | 0;
    const take = (a, b) => {
      let s = "";
      for (let i = a; i < b; i++) {
        const h = HEB[codon2idx[dnaStr.substr(i * 3, 3)]];
        s += h ? h.l : "·";
      }
      return s;
    };
    if (m <= limit) return { txt: take(0, m), truncated: false, total: m };
    return { txt: take(0, limit), truncated: true, total: m, shown: limit };
  }

  const SapereDNA = {
    HEB, ORD, PRIMES, char2pos, codon2idx,
    encode: classicEncode, decode: classicDecode, hebPreview: classicHebPreview,
    charsetSize: ORD.length
  };

  /* ============================================================
     PARTE 2 · GENOMA  (indirizzi a larghezza variabile, filamento unico)
     ============================================================ */
  const BASES = ["A", "T", "G", "C"];
  const ALL = [];
  for (const a of BASES) for (const b of BASES) for (const c of BASES) ALL.push(a + b + c);

  const CTRL = {
    SHIFT: "CTA",  // apre/chiude la parentesi letterale (geni <-> caratteri)
    SP:    "CTT",  // spazio
    NL:    "CTC",  // a-capo
    CAP:   "CTG",  // iniziale maiuscola alla parola seguente
    UP:    "CAC",  // parola seguente tutta maiuscola
    RAW:   "CAG"   // via di fuga: i 4 codoni seguenti = punto-codice Unicode
  };
  const RESERVED = ["CGT", "CGG"];
  const CTRL_SET = new Set(Object.values(CTRL).concat(RESERVED));
  const CTRL_BY_CODON = {};
  Object.keys(CTRL).forEach(k => { CTRL_BY_CODON[CTRL[k]] = k; });

  const DIGITS = ALL.filter(c => !CTRL_SET.has(c)); // 56
  const DIGIT_VAL = {};
  DIGITS.forEach((c, i) => { DIGIT_VAL[c] = i; });
  const RADIX = DIGITS.length; // 56

  const T1 = 28, T2 = 44, T3 = 52;
  const CAP1 = T1;
  const CAP2 = (T2 - T1) * RADIX;
  const CAP3 = (T3 - T2) * RADIX * RADIX;
  const CAP4 = (RADIX - T3) * RADIX * RADIX * RADIX;
  const BASE2 = CAP1;
  const BASE3 = CAP1 + CAP2;
  const BASE4 = CAP1 + CAP2 + CAP3;
  const MAX_RANK = CAP1 + CAP2 + CAP3 + CAP4 - 1;

  function rankToCodons(r) {
    if (r < BASE2) return DIGITS[r];
    if (r < BASE3) {
      const off = r - BASE2;
      return DIGITS[T1 + ((off / RADIX) | 0)] + DIGITS[off % RADIX];
    }
    if (r < BASE4) {
      const off = r - BASE3;
      return DIGITS[T2 + ((off / (RADIX * RADIX)) | 0)] +
             DIGITS[((off / RADIX) | 0) % RADIX] + DIGITS[off % RADIX];
    }
    const off = r - BASE4;
    return DIGITS[T3 + ((off / (RADIX * RADIX * RADIX)) | 0)] +
           DIGITS[((off / (RADIX * RADIX)) | 0) % RADIX] +
           DIGITS[((off / RADIX) | 0) % RADIX] + DIGITS[off % RADIX];
  }

  function tierOf(rank) {
    if (rank < BASE2) return 1;
    if (rank < BASE3) return 2;
    if (rank < BASE4) return 3;
    return 4;
  }

  function readAddress(codons, i) {
    const d0 = DIGIT_VAL[codons[i]];
    if (d0 === undefined) return null;
    if (d0 < T1) return { rank: d0, next: i + 1 };
    if (d0 < T2) {
      const d1 = DIGIT_VAL[codons[i + 1]];
      if (d1 === undefined) return null;
      return { rank: BASE2 + (d0 - T1) * RADIX + d1, next: i + 2 };
    }
    if (d0 < T3) {
      const d1 = DIGIT_VAL[codons[i + 1]], d2 = DIGIT_VAL[codons[i + 2]];
      if (d1 === undefined || d2 === undefined) return null;
      return { rank: BASE3 + (d0 - T2) * RADIX * RADIX + d1 * RADIX + d2, next: i + 3 };
    }
    const d1 = DIGIT_VAL[codons[i + 1]], d2 = DIGIT_VAL[codons[i + 2]], d3 = DIGIT_VAL[codons[i + 3]];
    if (d1 === undefined || d2 === undefined || d3 === undefined) return null;
    return { rank: BASE4 + (d0 - T3) * RADIX * RADIX * RADIX + d1 * RADIX * RADIX + d2 * RADIX + d3, next: i + 4 };
  }

  function fingerprint(str, seed) {
    seed = seed >>> 0 || 0;
    let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
    for (let i = 0; i < str.length; i++) {
      const ch = str.charCodeAt(i);
      h1 = Math.imul(h1 ^ ch, 2654435761);
      h2 = Math.imul(h2 ^ ch, 1597334677);
    }
    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
    h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
    h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
    const n = 4294967296 * (2097151 & h2) + (h1 >>> 0);
    return n.toString(16).padStart(14, "0");
  }

  const isLetter = (ch) => {
    try { return /\p{L}/u.test(ch); } catch (e) { return /[a-zA-Z\u00C0-\u024F]/.test(ch); }
  };

  function loadDictionary(words, meta) {
    const list = [];
    const index = Object.create(null);
    for (const w of words) {
      if (w == null || w === "") continue;
      if (!(w in index)) { index[w] = list.length; list.push(w); }
    }
    return {
      id: (meta && meta.id) || "genoma",
      version: (meta && meta.version) || "0",
      words: list,
      index: index,
      hash: fingerprint(list.join("\u0000")),
      size: list.length,
      capacity: MAX_RANK + 1,
      over: list.length > MAX_RANK + 1
    };
  }

  /* tokenizza come fa l'encoder: parole minuscole + singoli segni (per costruire genomi) */
  function genomeTokens(text) {
    const chars = Array.from(text);
    const n = chars.length;
    const tokens = [];
    let i = 0;
    while (i < n) {
      const ch = chars[i];
      if (ch === " " || ch === "\n" || ch === "\t" || ch === "\r") { i++; continue; }
      if (isLetter(ch)) {
        let j = i;
        while (j < n && isLetter(chars[j])) j++;
        tokens.push(chars.slice(i, j).join("").toLowerCase());
        i = j;
      } else { tokens.push(ch); i++; }
    }
    return tokens;
  }

  /* costruisce un genoma ordinato per frequenza da uno o più testi */
  function buildGenome(text, opts) {
    opts = opts || {};
    const counts = new Map();
    for (const t of genomeTokens(text)) counts.set(t, (counts.get(t) || 0) + 1);
    if (opts.seed) for (const w of opts.seed) if (!counts.has(w)) counts.set(w, 0);
    const minCount = opts.minCount || 1;
    const arr = [...counts.entries()].filter(e => e[1] >= minCount || e[1] === 0);
    arr.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    return arr.map(e => e[0]);
  }

  async function genomaEncode(text, dict, onProgress) {
    const chars = Array.from(text);
    const n = chars.length;
    const out = [];
    let mode = "GENE";
    let geneHits = 0, litChars = 0, ctrlCount = 0, rawChars = 0;

    const pushCtrl = (name) => { out.push(CTRL[name]); ctrlCount++; };
    const ensure = (m) => { if (mode !== m) { out.push(CTRL.SHIFT); ctrlCount++; mode = m; } };
    const fixed4 = (v) =>
      DIGITS[((v / 175616) | 0) % RADIX] + DIGITS[((v / 3136) | 0) % RADIX] +
      DIGITS[((v / RADIX) | 0) % RADIX] + DIGITS[v % RADIX];
    const emitRaw = (cp) => { out.push(CTRL.RAW); out.push(fixed4(cp)); rawChars++; ctrlCount++; };
    const emitGene = (rank, capFlag) => {
      ensure("GENE");
      if (capFlag) pushCtrl(capFlag);
      out.push(rankToCodons(rank));
      geneHits++;
    };
    const emitLitChar = (ch) => {
      const p = char2pos[ch];
      if (p === undefined) { emitRaw(ch.codePointAt(0)); return; }
      ensure("LIT");
      out.push(rankToCodons(p));
      litChars++;
    };

    let i = 0;
    while (i < n) {
      const ch = chars[i];
      if (ch === " ") { pushCtrl("SP"); i++; }
      else if (ch === "\n") { pushCtrl("NL"); i++; }
      else if (isLetter(ch)) {
        let j = i;
        while (j < n && isLetter(chars[j])) j++;
        const word = chars.slice(i, j).join("");
        i = j;
        const lower = word.toLowerCase();
        let capFlag = null, lookup = null;
        if (word === lower) lookup = lower;
        else if (word.length > 1 && word === lower.toUpperCase()) { capFlag = "UP"; lookup = lower; }
        else if (word === (lower.charAt(0).toUpperCase() + lower.slice(1))) { capFlag = "CAP"; lookup = lower; }
        const rank = (lookup != null) ? dict.index[lookup] : undefined;
        if (rank !== undefined) emitGene(rank, capFlag);
        else for (const c of Array.from(word)) emitLitChar(c);
      }
      else {
        const rank = dict.index[ch];
        if (rank !== undefined) emitGene(rank, null);
        else emitLitChar(ch);
        i++;
      }
      if ((i & 16383) === 0 && onProgress) { onProgress(i / n); await yieldUI(); }
    }
    if (onProgress) onProgress(1);

    const dna = out.join("");
    return {
      mode: "genoma", format: 1,
      dict_id: dict.id, dict_version: dict.version, dict_hash: dict.hash,
      dna, codons: dna.length / 3, chars: n, text_hash: fingerprint(text),
      gene_hits: geneHits, lit_chars: litChars, ctrl_count: ctrlCount, raw_chars: rawChars
    };
  }

  async function genomaDecode(entry, dict, onProgress) {
    if (entry.dict_hash && dict && dict.hash !== entry.dict_hash) {
      return { error: "Genoma non corrispondente: questo filamento è stato scritto con un dizionario " +
        "diverso o di un'altra versione (atteso «" + entry.dict_id + " " + entry.dict_version +
        "», impronta " + entry.dict_hash + "; attivo impronta " + dict.hash + "). " +
        "Carica il genoma giusto per rileggerlo correttamente." };
    }
    const clean = String(entry.dna).toUpperCase().replace(/[^ATGC]/g, "");
    const m = (clean.length / 3) | 0;
    if (entry.codons != null && m !== entry.codons) {
      return { error: "Disallineamento: il filamento contiene " + m + " codoni ma ne erano attesi " +
        entry.codons + " — probabile troncamento o corruzione." };
    }
    const codons = new Array(m);
    for (let k = 0; k < m; k++) codons[k] = clean.substr(k * 3, 3);

    let mode = "GENE", pendingCase = null, problems = 0;
    const parts = [];
    let i = 0;
    while (i < m) {
      const cod = codons[i], ctrl = CTRL_BY_CODON[cod];
      if (ctrl === "RAW") {
        const d0 = DIGIT_VAL[codons[i + 1]], d1 = DIGIT_VAL[codons[i + 2]],
              d2 = DIGIT_VAL[codons[i + 3]], d3 = DIGIT_VAL[codons[i + 4]];
        if (d0 === undefined || d1 === undefined || d2 === undefined || d3 === undefined) {
          parts.push("□"); problems++; i++;
        } else {
          const cp = ((d0 * RADIX + d1) * RADIX + d2) * RADIX + d3;
          try { parts.push(String.fromCodePoint(cp)); } catch (e) { parts.push("□"); problems++; }
          pendingCase = null; i += 5;
        }
      } else if (ctrl) {
        if (ctrl === "SHIFT") mode = (mode === "GENE") ? "LIT" : "GENE";
        else if (ctrl === "SP") parts.push(" ");
        else if (ctrl === "NL") parts.push("\n");
        else if (ctrl === "CAP") pendingCase = "CAP";
        else if (ctrl === "UP") pendingCase = "UP";
        i++;
      } else if (CTRL_SET.has(cod)) { parts.push("□"); problems++; i++; }
      else {
        const a = readAddress(codons, i);
        if (!a) { parts.push("□"); problems++; i++; continue; }
        let s;
        if (mode === "GENE") {
          s = dict ? dict.words[a.rank] : undefined;
          if (s === undefined) { parts.push("□"); problems++; pendingCase = null; i = a.next; continue; }
          if (pendingCase === "CAP") s = s.charAt(0).toUpperCase() + s.slice(1);
          else if (pendingCase === "UP") s = s.toUpperCase();
        } else {
          s = ORD[a.rank];
          if (s === undefined) { parts.push("□"); problems++; pendingCase = null; i = a.next; continue; }
          if (pendingCase) s = s.toUpperCase();
        }
        parts.push(s); pendingCase = null; i = a.next;
      }
      if ((i & 16383) === 0 && onProgress) { onProgress(i / m); await yieldUI(); }
    }
    if (onProgress) onProgress(1);

    const text = parts.join("");
    const verified = (entry.text_hash == null) ? null : (fingerprint(text) === entry.text_hash);
    return { text, verified, problems, count: m, chars: text.length };
  }

  function trace(dna, dict) {
    const clean = String(dna).toUpperCase().replace(/[^ATGC]/g, "");
    const m = (clean.length / 3) | 0;
    const codons = new Array(m);
    for (let k = 0; k < m; k++) codons[k] = clean.substr(k * 3, 3);
    const segs = [];
    let mode = "GENE", pendingCase = null, i = 0;
    while (i < m) {
      const cod = codons[i], ctrl = CTRL_BY_CODON[cod];
      if (ctrl === "RAW") {
        const d0 = DIGIT_VAL[codons[i + 1]], d1 = DIGIT_VAL[codons[i + 2]],
              d2 = DIGIT_VAL[codons[i + 3]], d3 = DIGIT_VAL[codons[i + 4]];
        let ch = "\u25a1";
        if (d0 !== undefined && d1 !== undefined && d2 !== undefined && d3 !== undefined) {
          try { ch = String.fromCodePoint(((d0 * RADIX + d1) * RADIX + d2) * RADIX + d3); } catch (e) {}
        }
        segs.push({ role: "raw", codons: 5, text: ch }); pendingCase = null; i += 5;
      } else if (ctrl) {
        if (ctrl === "SHIFT") { mode = (mode === "GENE") ? "LIT" : "GENE"; segs.push({ role: "shift", codons: 1, text: "" }); }
        else if (ctrl === "SP") segs.push({ role: "space", codons: 1, text: " " });
        else if (ctrl === "NL") segs.push({ role: "space", codons: 1, text: "\n" });
        else { pendingCase = ctrl; segs.push({ role: "mod", codons: 1, text: "" }); }
        i++;
      } else if (CTRL_SET.has(cod)) { segs.push({ role: "raw", codons: 1, text: "\u25a1" }); i++; }
      else {
        const a = readAddress(codons, i);
        if (!a) { segs.push({ role: "raw", codons: 1, text: "\u25a1" }); i++; continue; }
        if (mode === "GENE") {
          let w = dict ? dict.words[a.rank] : ("#" + a.rank);
          if (w === undefined) w = "\u25a1";
          if (pendingCase === "CAP") w = w.charAt(0).toUpperCase() + w.slice(1);
          else if (pendingCase === "UP") w = w.toUpperCase();
          segs.push({ role: "gene", codons: a.next - i, text: w, rank: a.rank });
        } else {
          let c = ORD[a.rank];
          if (c === undefined) c = "\u25a1";
          if (pendingCase) c = c.toUpperCase();
          segs.push({ role: "lit", codons: a.next - i, text: c });
        }
        pendingCase = null; i = a.next;
      }
    }
    return segs;
  }

  async function classicSize(text) {
    const r = await classicEncode(text);
    return { dnaBytes: r.dna.length, idxBytes: r.idx.length, total: r.dna.length + r.idx.length, count: r.count };
  }

  const SapereDNAGenoma = {
    encode: genomaEncode, decode: genomaDecode, trace,
    loadDictionary, buildGenome, genomeTokens, classicSize, fingerprint,
    rankToCodons, readAddress, tierOf, addressOf: (r) => rankToCodons(r),
    CTRL, RESERVED, DIGITS, RADIX,
    tiers: { T1, T2, T3, CAP1, CAP2, CAP3, CAP4, MAX_RANK,
      ranges: [ {bases:3, from:0, to:CAP1-1}, {bases:6, from:CAP1, to:BASE3-1},
                {bases:9, from:BASE3, to:BASE4-1}, {bases:12, from:BASE4, to:MAX_RANK} ] },
    info: { controls: Object.keys(CTRL).length, reserved: RESERVED.length, digits: RADIX, capacity: MAX_RANK + 1 }
  };

  /* ---- esportazione ---- */
  const Motore = { classic: SapereDNA, genoma: SapereDNAGenoma };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = Motore;
  } else {
    root.SapereDNA = SapereDNA;
    root.SapereDNAGenoma = SapereDNAGenoma;
    root.Motore = Motore;
  }
})(typeof window !== "undefined" ? window : this);
