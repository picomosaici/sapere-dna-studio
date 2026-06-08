# Sapere-DNA Studio

Un'app **Electron** per rendere l'intelligenza artificiale **interpretabile e _verificabile_** — e, sullo stesso principio, per archiviare la conoscenza in modo reversibile e tracciabile.

Al centro c'è un'idea sola, applicata due volte. Prima a un **archivio**: costruisci un dizionario di riferimento (il **genoma**), incidi la conoscenza in un filamento di **DNA simbolico** e la rileggi con controllo d'integrità — ogni parola risale al suo _locus_. Poi a una **rete neurale**: gli stessi tre gesti — **derivare** un tracciato dal calcolo, **dimostrarne la causa**, **sigillarlo** — diventano un banco di interpretabilità, dove il ragionamento di un modello si apre, si mette alla prova e si firma.

## Dove brilla

L'interpretabilità di frontiera sa estrarre concetti da modelli enormi, e perfino metterli alla prova spegnendoli. Ma su un modello opaco da miliardi di parametri due cose restano fuori portata: **misurare** quanto una spiegazione è davvero fedele (manca la verità di base) e **legarla** al calcolo così che chiunque possa ri-verificarla. Sapere-DNA prende l'altra strada — **piccolo e interamente ispezionabile** — proprio per ottenere quelle due cose, e ne aggiunge una terza:

- **Niente falsa attribuzione.** Quando un concetto _non_ è la causa, il sistema lo dichiara — «qui quel concetto è aggirato» — invece di raccontare una storia plausibile. È un'onestà costruita nel metodo, non una buona intenzione.
- **La spiegazione legata al calcolo.** Il **sigillo** vincola il tracciato a un'impronta del calcolo reale: non è una glossa scritta dopo, è una firma che si può ri-verificare. Una spiegazione che non può barare.
- **Verità di base e codice leggibile.** Tutto è scritto a mano, senza librerie opache: ogni peso è ispezionabile, e su modelli con risposta nota si _misura_ davvero la fedeltà della spiegazione — un lusso che su un colosso da miliardi di parametri non esiste.

Il metodo è dimostrato su tre banchi di realismo crescente, fino a **Cassandra**: un piccolo Transformer causale a due emisferi e più blocchi, che si può aprire, sondare blocco per blocco — e firmare.

Il principio guida, in una riga: **meglio un errore visibile che una spiegazione falsa silenziosa.**

> **Nota onesta.** Il lato archivio è un sistema _simbolico_ — non biologia reale, non crittografia sicura: il suo valore è la **reversibilità verificabile** e la **tracciabilità**, non la segretezza. E il lato interpretabilità non gareggia sulla scala: gareggia sulla _prova_.

---

## Compilare l'installer Windows (in locale)

Ti serve **Node.js 18+** (https://nodejs.org, versione "LTS"). Verifica con `node --version`.

Apri il Prompt dei comandi **nella radice del progetto** (la cartella che contiene questo
README e `package.json`) ed esegui:

```
npm install
npm run dist
```

Al termine troverai l'installer in:

```
dist\Sapere-DNA Studio Setup 1.0.2.exe
```

> Suggerimento: per aprire il Prompt nella cartella giusta, apri la cartella in Esplora File,
> clicca nella barra dell'indirizzo, scrivi `cmd` e premi Invio.

Per provare l'app senza creare l'installer:

```
npm install
npm start
```

> Nota: l'installer non è firmato digitalmente, quindi la prima volta Windows (SmartScreen)
> può mostrare un avviso. È normale per i progetti senza certificato: "Ulteriori informazioni"
> → "Esegui comunque".

---

## Compilazione automatica su GitHub

Il repo include un workflow (`.github/workflows/build.yml`) che compila l'installer da solo.

- **Da una versione:** crea e pubblica un tag, ad esempio
  ```
  git tag v1.0.2
  git push origin v1.0.2
  ```
  GitHub compila l'installer e lo **allega automaticamente alla Release** del tag.
- **A mano:** vai nella scheda **Actions** del repo → *Compila installer Windows* →
  *Run workflow*. Al termine l'installer è scaricabile come *artifact* della run.

---

## Struttura

```
└─ src/
   ├─ index.html              guscio: barra di stato + sidebar + sezioni
   ├─ styles.css              estetica (schema colori SDNA, layout a laboratorio)
   ├─ motore.js               MOTORE unificato: Classico + Genoma
   ├─ genoma_it-demo.js       genoma dimostrativo (sostituibile con uno grande)
   ├─ codifica.js             controller della Codifica
   ├─ decodifica.js           controller della Decodifica
   ├─ rete.js                 rete neurale + analisi di trasparenza (Mente)
   ├─ trasformatore.js        Transformer in miniatura (banco Transformer)
   ├─ mente-trasformatore.js  controller del banco Transformer
   ├─ emisferi.js             due emisferi cablati dalla regola del π + corpo calloso
   ├─ motore-emisferi.js      Transformer causale multi-blocco a emisferi (Cassandra)
   ├─ generatore.js           interpretabilità su Cassandra: SAE, derivazione, intervento, sigillo, sbirciata, tracciamento profondo
   ├─ generatore-worker.js    addestramento + SAE + sbirciata di Cassandra in Web Worker
   ├─ mente-generatore.js     controller del banco Cassandra
   ├─ mente.js                controller della sezione Mente
   └─ app.js                  orchestratore dell'app (sidebar, sezioni, archivio)
```

---

## Storico versioni

- **1.0.2 — Tracciamento causale profondo.** In Cassandra, dalla decisione di ogni parola si
  può spegnere un concetto «primitivo» del 1° blocco e seguirne l'effetto fino alla parola finale,
  attraverso il blocco terminale. È una **sonda d'onestà**: se la parola non cambia lo dichiara —
  «il 1° blocco qui è aggirato» — senza attribuire la decisione a concetti che non la causano.
- **1.0.1 — Sbirciata sul calloso intermedio.** Cassandra mostra i concetti «primitivi» formati
  dal 1° blocco, descritti da ciò che li accende: i mattoni che il blocco finale ricompone.
- **1.0.0 — Prima release.** Codifica, Decodifica, Laboratorio del genoma e il banco Mente.

---

## Le due modalità

- **Genoma** — *un solo filamento di DNA, niente indice.* Le parole note al genoma diventano
  "geni" (indirizzo a larghezza variabile, 1–4 codoni); ciò che non è nel genoma viene sillabato
  carattere per carattere; qualsiasi carattere fuori tavola (anche un'emoji) passa per la via di
  fuga `RAW`. Ogni voce porta le impronte di **dizionario** e **testo**: se apri un filamento col
  genoma sbagliato o troncato, l'app te lo dice invece di sbagliare in silenzio.
- **Classica** — *doppio binario.* Ogni carattere → lettera ebraica + codone DNA, più un indice
  gematrico `primo(rango) × gematria`. È il sistema originale, intatto, e fa da riferimento nei
  confronti di dimensione.

## Il laboratorio del genoma

La sezione **Genoma** mostra il genoma attivo (identità, versione, impronta, capienza,
distribuzione per costo dell'indirizzo), permette di esplorare i geni con i loro **loci**
(indirizzi), di **aggiungere parole**, di **costruire un genoma da un corpus** (incolli del
testo, l'app conta le frequenze e ordina le parole) e di **importare/esportare** il genoma come
`.json` da congelare e condividere. Genoma e archivio si salvano automaticamente nei dati utente.

## La Mente — banco di interpretabilità

La sezione **Mente** applica gli stessi tre strumenti del genoma (dizionario, espressione,
impronta) a una rete neurale, per renderne il ragionamento leggibile e *verificabile*: si
estrae dai neuroni un "genoma di concetti", si incide il **tracciato** delle decisioni, lo si
mette alla prova con l'**intervento causale** (spegnere un concetto e vedere se la risposta
cambia) e lo si **sigilla**. Il metodo è dimostrato su tre banchi di realismo crescente — il classificatore lineare (**Mente**)
e il **Transformer** in miniatura, entrambi con verità di base nota, e **Cassandra**, un Transformer
causale a due emisferi e più blocchi: niente verità di base, ma prova causale e sigillo ancora
validi. Su Cassandra si possono anche **sbirciare** i concetti primitivi del 1° blocco e farne il
**tracciamento causale profondo** fino alla parola finale. Il principio guida: **meglio un errore
visibile che una spiegazione falsa silenziosa.**

---

## Licenza

Software proprietario a **uso libero e gratuito, senza modifica**: chiunque può usarlo
gratis per qualsiasi scopo; la ridistribuzione è permessa solo in forma originale e immutata;
la modifica e le opere derivate non sono permesse senza permesso scritto. Vedi il file
[`LICENSE`](./LICENSE)
