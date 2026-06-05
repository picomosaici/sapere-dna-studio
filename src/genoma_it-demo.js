/* ============================================================
   Sapere-DNA · Genoma dimostrativo "it-demo"
   ------------------------------------------------------------
   Lista CONGELATA e ORDINATA PER FREQUENZA (la più comune prima).
   L'ordine È l'assegnazione degli indirizzi: le prime ~28 voci
   costano 1 codone (3 basi), poi 2, poi 3...
   In produzione si sostituirà con un elenco grande ricavato da
   un vero corpus di frequenza. NON modificare sul posto: per
   cambiarlo si crea una nuova versione (vedi campo "version").
   ============================================================ */
(function (root) {
  "use strict";

  const WORDS = [
    // ---- tier 1 (1 codone / 3 basi): le 28 più frequenti ----
    "di", "e", "il", "la", "che", "è", "a", "in", "un", "per",
    ".", ",", "non", "una", "con", "le", "si", "lo", "i", "da",
    "sono", "come", "più", "anche", "ma", "ha", "se", "o",

    // ---- tier 2 (2 codoni / 6 basi): comuni ----
    "gli", "mi", "questo", "tutto", "ci", "quando", "essere", "fare",
    "ti", "ne", "so", "sua", "suo", "loro", "noi", "voi", "io", "tu",
    "lei", "lui", "questa", "quello", "quella", "stato", "stata", "molto",
    "bene", "dove", "perché", "mentre", "però", "quindi", "ancora", "sempre",
    "mai", "poco", "ogni", "due", "tre", "prima", "dopo", "senza", "tra",
    "fra", "sotto", "sopra", "dentro", "fuori", "contro", "verso", "fino",
    "già", "qui", "qua", "lì", "là", "così", "cioè", "oppure", "anzi",
    "nel", "nella", "nei", "nelle", "del", "della", "dei", "delle", "dal",
    "dalla", "al", "alla", "ai", "alle", "sul", "sulla", "col", "cogli",
    "essere", "avere", "ho", "hai", "abbiamo", "avete", "hanno", "era",
    "erano", "sarà", "saranno", "fu", "siamo", "siete", "sei",
    "l", "d", "un", "dell", "nell", "all", "sull", "dall", "quell", "gliel",
    "che", "chi", "cui", "ciò", "tutti", "tutte", "tutta", "nulla", "niente",
    "qualcosa", "qualcuno", "ognuno", "stesso", "stessa", "altro", "altra",
    "altri", "altre", "grande", "grandi", "piccolo", "piccola", "nuovo",
    "nuova", "buono", "buona", "primo", "ultimo", "vero", "vera", "certo",
    "tempo", "anno", "anni", "giorno", "giorni", "volta", "volte", "casa",
    "uomo", "donna", "mano", "occhi", "vita", "mondo", "parte", "parti",
    "cosa", "cose", "modo", "punto", "caso", "fatto", "lavoro", "parola",
    "parole", "nome", "numero", "acqua", "fuoco", "aria", "terra", "cielo",
    "sole", "luna", "mare", "monte", "montagna", "fiume", "strada", "città",
    "paese", "luce", "ombra", "voce", "suono", "colore", "forma", "filo",
    "libro", "storia", "idea", "pensiero", "mente", "cuore", "corpo",
    "testa", "piede", "gente", "amico", "amore", "padre", "madre", "figlio",
    "bambino", "ragazzo", "scuola", "lettera", "segno", "senso", "verità",
    "ragione", "forza", "potere", "morte", "pace", "guerra", "dio",
    "natura", "scienza", "numero", "punto", "linea", "spazio", "energia",
    "materia", "calore", "freddo", "peso", "massa", "moto", "quiete",
    "pressione", "gradi", "bolle", "bassa", "alta", "alto", "basso",
    "perde", "perché", "quanto", "quanti", "quale", "quali", "questo",
    "molto", "troppo", "abbastanza", "appena", "soltanto", "solo", "sola",
    "insieme", "vicino", "lontano", "destra", "sinistra", "avanti", "indietro",
    "presto", "tardi", "oggi", "ieri", "domani", "adesso", "ora", "poi",
    "essere", "diventare", "rimanere", "restare", "andare", "venire",
    "vedere", "sentire", "sapere", "conoscere", "capire", "pensare",
    "credere", "dire", "parlare", "scrivere", "leggere", "vivere", "morire",
    "nascere", "dare", "prendere", "mettere", "tenere", "portare", "trovare",
    "cercare", "guardare", "ascoltare", "amare", "volere", "potere", "dovere",

    // ---- punteggiatura e simboli comuni come "geni" ----
    ";", ":", "!", "?", "'", "\u2019", "\u00ab", "\u00bb", "\u201c", "\u201d",
    "(", ")", "-", "\u2014", "\u2013", "\u2026", "/"
  ];

  const DICT = { id: "it-demo", version: "2026.1", words: WORDS };

  if (typeof module !== "undefined" && module.exports) module.exports = DICT;
  else root.GENOMA_IT_DEMO = DICT;

})(typeof window !== "undefined" ? window : this);
