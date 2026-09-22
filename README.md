# labscripts
Questa repo contiene script per effettuare l'analisi dei dati raccolti durante le esperienze di Lab 2

## Installazione rapida delle dipendenze

Dalla cartella dello script, esegui:

```bash
pip install numpy pandas matplotlib scipy openpyxl
```

`openpyxl` è necessario a `pandas` per leggere i file `.xlsx` (non è import
esplicito nello script, ma serve comunque come motore di lettura Excel).

## Struttura dei file di dati

Ogni file Excel di input deve avere **esattamente 4 colonne**, nell'ordine:

| x | y | dx | dy |
|---|---|----|----|

- Riga di intestazione con i nomi/etichette delle grandezze (es. `I_m (uA)`,
  `V_m (V)`, `dI_m (uA)`, `dV_m (V)`)
- Nessuna colonna di indice/numerazione a sinistra
- Tutti i valori numerici (niente celle vuote o testo nelle righe dati)

## Utilizzo

Lo script accetta i file di dati come argomenti da riga di comando.

### Un solo dataset

```bash
python3 main.py dati.xlsx
```

### Due dataset a confronto (es. configurazione a monte/a valle)

```bash
python3 main.py dati_configurazione_monte.xlsx dati_configurazione_valle.xlsx
```

Lo script:
1. Carica i dati da entrambi i file
2. Esegue il fit lineare con ODR (`parametriFitODR`), tenendo conto degli
   errori su entrambi gli assi
3. Stampa a schermo i parametri della retta ($m \pm \sigma_m$, $q \pm
   \sigma_q$)
4. Calcola e stampa il $\chi^2$ e il $\chi^2$ ridotto (rispetto ai gradi
   di libertà) per ciascun dataset
5. Genera un grafico con punti sperimentali (barre di errore su x e y) e
   le rette di fit sovrapposte

## Interpretare l'output

- **$\chi^2_{rid} \approx 1$**: il fit è statisticamente compatibile con
  i dati e gli errori dichiarati
- **$\chi^2_{rid} \ll 1$**: gli errori sperimentali dichiarati sono
  probabilmente sovrastimati
- **$\chi^2_{rid} \gg 1$**: il modello lineare non descrive bene i dati,
  oppure gli errori sono sottostimati, oppure sono presenti outlier —
  vale la pena controllare i residui punto per punto

## File dello script

- `retta(params, x)` — modello lineare $y = mx + q$ nella convenzione
  richiesta da `scipy.odr` (parametri come tupla, poi x)
- `parametriFitODR(x, y, dx, dy)` — esegue il fit e restituisce
  $m, q, \sigma_m, \sigma_q$
- `chiQuadroLinare(x, y, m, q, dx, dy)` — calcola il $\chi^2$ con il
  metodo della varianza effettiva
- `getData(filepath)` — legge il file Excel e restituisce gli array
  `x, y, dx, dy` più le etichette delle colonne
- `pulisci(nome)` — ripulisce le etichette delle colonne (rimuove
  eventuali prefissi tipo `nome=`)
