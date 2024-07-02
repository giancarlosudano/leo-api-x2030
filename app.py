import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

def get_gpt():
	azure_endpoint: str = os.getenv("AZURE_OPENAI_BASE") or ""
	api_key = os.getenv("AZURE_OPENAI_API_KEY") or ""
	api_version: str = os.getenv("AZURE_OPENAI_API_VERSION") or ""
	azure_openai_deployment : str = os.getenv("AZURE_OPENAI_MODEL_NAME") or ""
	llm = AzureChatOpenAI(azure_deployment=azure_openai_deployment, temperature=0, streaming=True, azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version)
	return llm

def read_file(file_name):
	with open(file_name, "r", encoding="utf-8") as file:
		return file.read()

@app.route('/ChatCompletion', methods=['POST'])
@cross_origin()
def chat_completion():
	
    template = """
Dati la seguente documentazione delimitata da ----------
rispondi alla seguente domanda usano solo i documenti che ti ho fornito:

domanda:
{question}


Documenti:

----------
Documento: scheda informativa popolazione 2006
```
Nome della società LAMPOGAS ROMANA S.r.l.
```
```
Deposito di Pantano di Grano (ROMA)
Via degli Oleodotti 25
```
```
Portavoce della società
(se diverso dal responsabile)
```
(^)
La società ha presentato la notifica prescritta
dall’art.6 del D.Lgs 334/^
La società ha presentato il rapporto di sicurezza
prescritto dall’art. 8 del D.Lgs 334/
La società ha presentato la relazione di cui all’art.
5 del D.Lgs 334/
Responsabile dello stabilimento
RUGGIERI ETTORE Gestore
COLONNA CESARE
Descrizione dell’attività svolta nel deposito


Il deposito di stoccaggio e travaso GPL è gestito dalla Società LAMPOGAS ROMANA S.r.l. e da lavoro a n°
14 dipendenti diretti.

L'attività svolta consiste nel travaso e stoccaggio di G.P.L. ( gas di petrolio liquefatti ) per il rifornimento dei
piccoli serbatoi per uso domestico, artigianale e industriale installati presso la clientela. L’attività è ad intensità
variabile con andamento stagionale in funzione delle temperature ambientali.


## Descrizione del territorio circostante

Lo stabilimento LAMPOGAS ROMANA S.r.l. è ubicato nel territorio del Comune di Roma, in località Pantano
di Grano.

Il deposito è ubicato in area a destinazione agricola e confina :

## A NORD : Via degli Oleodotti

## Ad OVEST: Gassificatore Soc Colari

## Ad EST : terreno agricolo

## A SUD : Discarica Giovi srl

### SITUAZIONE DEMOGRAFICA

Le frazioni più vicine al deposito, Ponte Galeria e Pantano di Grano, distano rispettivamente circa 4 km e 1,
km; a distanze inferiori troviamo solo abitazioni isolate.

### ALTRI IMPIANTI INDUSTRIALI

Alle distanze riportate dal baricentro del deposito, vi sono i seguenti impianti industriali:

## DISTANZA IN METRI

## DE.CO. – Deposito Comune scarl 150

## GIOVI Srl – trattamento rifiuti e

## discarica

## 100

## Raffineria di Roma 500

## Raffineria di Roma 500

## Gassificatore Soc.Colari 100


Natura dei rischi di incidente rilevante – Informazioni generali
I GPL sono caratterizzati da spiccate caratteristiche d’infiammabilità, sono prodotti stabili e non danno origine a
reazioni pericolose. Il propano ed il butano allo stato puro sono quasi inodori; i GPL possono avere odore
caratteristico solo se odorizzati per uso combustione.
Per il deposito LAMPOGAS ROMANA la natura dei rischi è strettamente associata alla natura intrinseca dei
GPL stessi (caratteristiche di infiammabilità ed esplodibilità) ed alla presenza costante di recipienti a pressione.

Tipi di effetti per la popolazione e l’ambiente

1. Sono possibili i seguenti effetti:
    - • Irraggiamento termico nel caso di incendio
    - • Onda d’urto e proiezione di frammenti nel caso molto remoto, in cui si dovesse verificare
       un esplosione
2. Nessun danno, ambientale dovuto ad inquinamento da GPL data la loro natura. In caso di incidente, le aree
    di impatto sicuro o molto probabile sono tutte comprese all’interno del perimetro dello stabilimento o nelle
    aree limitrofe
3. Gli effetti incidentali si esauriscono nel breve termine e non comportano effetti a medio e lungo termine.
4. Non esistono soggetti particolarmente vulnerabili agli effetti incidentali, né possibili effetti indiretti.
5. Le persone effettivamente in pericolo tra quelle presenti nell'area a rischio sono quelle che si trovano
    all'aperto; le persone all'interno di edifici e lontane da superfici vetrate sono ragionevolmente protette dagli
    effetti incidentali.


## Mezzi di segnalazione incidenti

Lo stato d'emergenza interna viene segnalato nel deposito mediante dispositivo acustico (sirena) e visivo
(lampada rotante) attivato da pulsanti manuali e rilevatori di gas.

L’allarme esterno viene attivato da sirena manuale e tramite linea telefonica secondo la procedura prevista nel
Piano di Emergenza Interno nel caso che l’emergenza non sia gestibile con mezzi interni o che possa provocare
effetti all’esterno del deposito.

La funzione incaricata di dare l’allarme esterno è il Responsabile del deposito o in sua assenza il Capo
Antincendio.

## Comportamento da seguire

### ESTERNAMENTE ALLO STABILIMENTO:

La responsabilità di suggerire alla popolazione uno specifico comportamento è affidata dalla Legge alle Autorità
preposte, alle quali la LAMPOGAS ROMANA assicura la più completa, trasparente e tempestiva
collaborazione.
Data la natura dei possibili rischi, è comunque presumibile che la popolazione possa essere chiamata per
prudenza ad alcune semplici precauzioni, quali:
♦♦ Evitare di affollare le strade con automezzi, per consentire l’agevole accesso ai mezzi di soccorso
♦♦ Porsi in ascolto di una determinata stazione radio

## Mezzi di Comunicazione previsti

I mezzi di comunicazione previsti all’interno del deposito sono i seguenti:
1) telefono/megafono
Comunicazioni con l’esterno: Rete telefonica pubblica secondo i modelli di comunicazione previsti nel Piano di
Emergenza Interno.




Documento: Informativa alla popolazione del Piano di Emergenza Esterna LAMPOGAS - Via degli Oleodotti 25 Roma (ora AGN Energia S.p.A)


### INFORMAZIONE ALLA POPOLAZIONE

## STABILIMENTO LAMPOGAS s.r.l.

```
ai sensi del D.M. 29 SETTEMBRE 2016 N. 200
```
La vigente normativa (D.Lgs. 105/2015 e s.m.i.), di derivazione comunitaria, in tema di controllo

dei pericoli di incidenti rilevanti connessi con determinate sostanze pericolose, prevede la

predisposizione del Piano di Emergenza Esterno (PEE) per gli stabilimenti in esso rientranti. Il PEE
viene predisposto dal Prefetto, d’intesa con gli enti territoriali interessati, previa consultazione

della popolazione.

Il PEE è stato redatto per la Società **LAMPOGAS SRL** Via degli Oleodotti 25, ROMA

### 1) DESCRIZIONE E CARATTERISTICHE DELL’AREA INTERESSATA DALLA PIANIFICAZIONE

Denominazione e ubicazione dello stabilimento e nominativo del Direttore responsabile

Il Deposito Lampogas S.r.l. è ubicato nel comune di Roma, in località Pantano di Grano.

Il Deposito è ubicato in un’area tra Ponte Galeria e l’agglomerato di Malagrotta, a circa 3,6 km dal
G.R.A. lato ovest di Roma.

**1.1 Descrizione del territorio circostante lo stabilimento**

Nel raggio di circa 2 km dal Deposito sono compresi gli agglomerati di Massimina e di Fontignani.

Altri dati relativi a linee ferroviarie, strade, autostrade, attività produttive e acquedotti compresi
nel raggio di 2000 m dal perimetro del Deposito, sono:

Linee ferroviarie: a circa 1,3 km la linea ferroviaria Roma-Pisa-Livorno;

Strade comunali: a circa 250 m via di Malagrotta;

Attività industriali: Raffineria di Roma (400 m);

DE.CO. (130 m);

EnerGas (650 m);

ENI (1,4 Km)

Discarica Malagrotta (50 m);

Acquedotti: a circa 200 m l’acquedotto ACEA;

Strade nazionali: a circa 3,0 km la S.S. Aurelia.


Lo stabilimento è ubicato in un'area industriale nella quale non risultano presenti, nel raggio di
1000 m, strutture strategiche (ospedali, scuole, caserme, ecc.) in prossimità della zona interessata
dagli effetti degli scenari incidentali di cui al presente PEE.

Al limite della Terza zona di danno risultano n. 3 residenti. Nel raggio di 1 Km sono presenti nr.
residenti e non risultano pazienti disabili secondo quanto comunicato dalla Asl Roma 3 in data 16
marzo 2018.

Il Deposito è delimitato dai seguenti confini:

```
^ a Nord con la Via degli Oleodotti;^
^ ad Ovest con l’impianto Gassificatore;^
 a Sud con la discarica di Malagrotta;
^ ad Est con la discarica di Malagrotta.^
```
**Rischio Natech**

**_Idrogeologico_** - Dalla relazione di valutazione del rischio idrogeologico effettuata dall’Azienda
emerge che, in base alle quote del sito, con riferimento alla cartografia PAI Tavola PB 73 Galeria, lo
stabilimento non ricade all’interno di campiture indicanti rischio idraulico, tranne una piccola
porzione che ricade nella zona a rischio R2 (associata a eventi con tempo di ritorno di 500 anni) in
corrispondenza del parcheggio automezzi, non interessando le aree operative di stoccaggio e
carico/scarico del GPL. Le aree classificate R3 corrispondono a Via degli Oleodotti e, pertanto, è
possibile individuare un rischio indotto sulla viabilità, rappresentato principalmente dalle
emergenze che possono verificarsi sulle strade adiacenti lo stabilimento.

A livello geomorfologico, l’area in oggetto non risulta caratterizzata da fenomeni franosi, ovvero
da un’analisi sul rischio frana non sono stati censiti fenomeni in atto. Dalla cartografia PAI, si
evidenza l’assenza di rischio.

Per quanto riguarda la vulnerabilità dello stabilimento rispetto al rischio inondazione, dalla
valutazione effettuata emerge che, ad essere interessate da inondazione risultano essere
solamente la zona antistante l’ingresso adibita a parcheggio ed una piccola area verde a ridosso
del muro perimetrale, ricadenti in area di pericolosità P2 e in minima parte in area P1 della
classificazione del PAI. Ciò è dovuto alla differenza delle quote altimetriche dello stabilimento e del
livello idrico che si instaura a seguito del rigurgito del Rio Galeria lungo il canale in calcestruzzo
adiacente a Via degli Oleodotti. Tale conclusione risulta avvalorata dall’indicazione ufficiale
dell’Autorità di Bacino Distrettuale dell’Appennino Centrale – Autorità di Bacino del Fiume Tevere
del tirante idrico nell’area dello stabilimento in occasione di un evento di piena con tempo di
ritorno pari a 200 anni. Per la presenza del muro di perimetrazione le uniche vie di accesso del
flusso d’acqua coincidono con i cancelli carrabili.

In caso di eventi meteorologici avversi, le aree di ammassamento mezzi di soccorso sono
individuate in maniera diversa rispetto a condizioni meteo ordinarie


**_Sismico_** - Il Municipio XII di Roma Capitale è classificato come sottozona sismica 3A (Ordinanza del

Presidente del Consiglio dei Ministri n. 3519 del 28 aprile 2006 e Delibera di Giunta Regionale n°
387 del 22 maggio 2009).

Il Municipio XII di Roma Capitale ha redatto, ai sensi della DGR Lazio n. 545/2010 (Determinazione

della Regione Lazio n. A0394 del 21/05/2013, anche una microzonazione sismica, da cui risulta che

l’area dove insiste lo stabilimento è classificata come (ZAS)

In relazione alla valutazione di vulnerabilità sismica, e a seguito delle valutazioni analitiche effettuate, si
può sostanzialmente riassumere che:

```
Struttura analizzata Conclusioni Misure da attuare
```
Muri di contenimento del
tumulato e selle di ancoraggio
serbatoi di stoccaggio GPL.

```
Complessivamente adeguato. Nessuna
```
Muro dei punti di travaso
autobotti.

```
Complessivamente adeguato. Si consiglia un irrigidimento della
struttura metallica della
copertura.
```
Struttura Torre faro e relativa
fondazione

```
Pienamente adeguato. Nessuna.
```
Edificio servizi tecnici Complessivamente adeguato. Si consiglia un irrigidimento della
struttura di copertura.

Piping GPL e antincendio Serbatoi pienamente adeguati;

```
Piping GPL e antincendio
complessivamente adeguati.
```
```
Revisione dei sistemi di
ancoraggio del piping alle
carpenterie metalliche di
sostegno.
```

**1.2 Infrastrutture stradali, ferroviarie, aeroportuali, portuali**

```
nome Distanza (m) tipo
Leonardo da Vinci
di Fiumicino
```
```
5.500 aeroporto
```
```
Via di Malagrotta 250 Strada comunale
Via della Pisana 1.600 Strada comunale
S.S. Aurelia 3.000 Strada Statale
Autostrada Roma -
Civitavecchia
```
```
4.500 Autostrada
```
```
Linea ferroviaria
Roma-Pisa, Livorno
```
```
1.300 Ferrovia
```
```
Stazione FS Ponte
Galeria
```
```
4.400 Linea Ferroviaria
```
### 2) NATURA DEI RISCHI

**2.1 Attività dello stabilimento**

L'attività consiste nel travaso e stoccaggio di GPL (gas di petrolio liquefatti) per il rifornimento dei
piccoli serbatoi per uso domestico (miscela uso domestico), artigianale e industriale installati
presso la clientela. L’attività è ad intensità variabile con andamento stagionale in funzione delle
temperature ambientali.
Nello stabilimento non avvengono processi di trasformazione, ma unicamente attività di

carico/scarico.

I GPL arrivano allo stabilimento a mezzo di autobotti (bilici) e vengono immessi nei serbatoi fissi di
stoccaggio con operazioni a ciclo chiuso, senza dispersione di gas nell'atmosfera. Il prodotto viene
successivamente ripreso per il caricamento di botticelle destinate al rifornimento dei piccoli
serbatoi della clientela. La materia prima entrante, GPL (gas di petrolio liquefatti), non subisce
modifiche o trattamenti per cui, dopo lo stoccaggio, diventa anche il prodotto uscente. La
temperatura del GPL è sostanzialmente quella ambiente e quindi la pressione del GPL nei serbatoi
e nelle tubazioni è quella corrispondente alla tensione di vapore alla temperatura ambiente.

**2.2 I rischi**

Eventi e scenari incidentali previsti nel Modulo di notifica e di informazione sui rischi di incidente
rilevante per i cittadini ed i lavoratori (Allegato 5 del suddetto Decreto n. 105/2015).


**Eventi e scenari incidentali**

```
Evento Frequenza
occ/anno
```
```
Localizzazione
```
```
1 * Rilascio da linea, foro 2 " 6,69E- 06 Area serbatoi, area compressori e area
travaso
6.1 Rilascio in fase di travaso ATB,
fessurazione
```
```
1,62E- 04 Area travaso
```
```
8 Rilascio da compressore, foro
1 "
```
```
2,01E- 04 Area compressori
```
```
9 Rilascio da serbatoietto vuoto
non bonificato
```
```
8,80E- 04 Area serbatoietti vuoti non bonificati
```
**Stima delle conseguenze incidentali**

Trattandosi di un deposito in cui non vengono applicati procedimenti di lavorazione e nel quale
non sono ipotizzabili reazioni chimiche fra le sostanze detenute, le sequenze di incidente risultano
connesse prevalentemente a forature o rotture casuali che comportano il rilascio delle sostanze
pericolose, mentre risultano meno verosimili guasti di strumentazione di processo o di controllo.

La stima delle conseguenze viene effettuata per i casi credibili di incidente, ovvero per gli eventi
con frequenza attesa > 1- 10 - 7 occasioni/anno riportati nella tabella precedente.

Per la definizione delle aree di danno dovuto all’irraggiamento e alla sovrappressione si è fatto
riferimento ai valori di soglia per la valutazione degli effetti come riportati nella seguente tabella,
che tiene conto delle linee guida per la “Pianificazione di Emergenza esterna per gli impianti
industriali a rischio di incidente rilevante” (D.P.C.M. 25/02/2005) e del D.M. del 15/05/
“Criteri di analisi e valutazione dei rapporti di sicurezza relativi ai depositi di gas e petrolio
liquefatto.

**Prescrizioni particolari per le aziende limitrofe**

```
Evento Azione gestore Azione delle aziende limitrofe (per i dipendenti in loco)
Incidente Allertamento (suono di
sirena continuo e
prolungato....o altra
modalità idonea per
allertare la popolazione)
```
```
Riparo al chiuso con porte e finestre chiuse
Divieto assoluto di avvicinamento all'impianto fino a
cessato allarme
```

Con le periodiche esercitazioni interne verrà attuato un coordinamento anche con i soggetti
limitrofi, che dovranno essere coinvolti nelle forme ritenute più idonee dai rispettivi referenti. A
tal fine, sarebbe opportuno che le aziende limitrofe, qualora a rischio incidente rilevante, siano
reciprocamente in possesso dei rispettivi piani di emergenza interni (pei).

**2.3 Aree di danno**

**Zone di danno ed elementi sensibili all’interno di ciascuna zona**

### I ZONA

```
(inizio letalità)
```
```
Dispersione di vapori infiammabili per rottura tubazione 2" di fase
liquida per la quale il gestore ipotizza che il limite inferiore del
campo di infiammabilità rimane all’interno dei 68 mt dalle tubazioni
antistanti il muro di contenimento dei serbatoi di stoccaggio GPL
```
### II ZONA

```
(elevata letalità)
```
```
Dispersione di vapori infiammabili per rottura tubazione 2" di fase
liquida per la quale il gestore ipotizza che la metà del limite inferiore
del campo di infiammabilità rimane all’interno dei 149 mt dalle
tubazioni antistanti il muro di contenimento dei serbatoi di
stoccaggio GPL
```
### III ZONA

```
(di attenzione)
```
```
La zona di attenzione, all’interno dei 298 mt (*) dalle tubazioni
antistanti il muro di contenimento dei serbatoi di stoccaggio GPL,
viene considerata come una zona, al di fuori della quale, nel caso
venga disposta la misura di autotutela (rifugio al chiuso), il personale
non interessato all’emergenza debba portarsi:
Lato Nord via degli Idrocarburi
Lato ovest fino a via di Malagrotta esclusa
Lato est limite esterno discarica attiva
Lato sud limite impianto Malagrotta 2
(*) La distanza è determinata come da indicazioni del DPCM
25/02/
```

### 3) AZIONI PREVISTE PER LA MITIGAZIONE E LA RIDUZIONE DEGLI EFFETTI E DELLE

### CONSEGUENZE DI UN INCIDENTE E AUTORITA’ PUBBLICHE COINVOLTE

**Funzioni di supporto**

Si riportano qui di seguito le funzioni di supporto con relativi compiti che generalmente vengono
attivate, in caso di evento, declinate in maniera specifica a seconda delle caratteristiche degli
stabilimenti interessati dalla pianificazione di emergenza

**GESTORE**
Evento Azione Gestore
Quasi
incidente

```
Incidente
```
```
Attiva con la sirena il pei
Allerta tempestivamente il comando prov. Vigili del fuoco
Attiva i livelli di allerta secondo la gravità dell’evento
Informa: prefetto, sindaco, presidente della regione e presidente della citta’
metropolitana
Segue costantemente l'evoluzione dell'incidente
Aggiorna le informazioni comunicando con il prefetto
Avvisa le aziende e i soggetti presenti all’interno delle aree di danno secondo i
pei
Resta a disposizione del responsabile dei vigili del fuoco intervenuto sul posto.
```

### PREFETTO (AP)

```
Evento AZIONE PREFETTO
```
```
Incidente
```
```
Coordina l'attuazione del pee secondo i livelli di allerta
```
```
Acquisisce dal gestore e altri soggetti ogni utile informazione
Attiva e presiede il centro coordinamento soccorsi (ccs)
```
```
Istituisce in loco, se ritiene, il centro operativo misto (com)
```
```
Informa il dipartimento della protezione civile, il ministero dell'ambiente, il
ministero dell'interno, i prefetti delle province limitrofe ed i sindaci dei comuni
limitrofi
Acquisisce i dati meteo locali avvalendosi delle stazioni meteo del territorio, dei
centri regionali funzionali e del dipartimento della protezione civile
```
```
Verifica che siano stati attivati i sistemi di allarme per le comunicazioni alla
popolazione e ai soccorritori
Valuta e decide con il sindaco, sentito il direttore tecnico dei soccorsi ed il
direttore dei soccorsi sanitari, le misure di protezione per la popolazione, in
base ai dati tecnico-scientifici forniti dagli organi competenti o dalle funzioni di
supporto
Sentiti il sindaco interessato e gli organi competenti, dirama comunicati
stampa/radio, gestendo la comunicazione in emergenza con il proprio addetto
stampa
Accerta l'attivazione delle misure di protezione collettiva
```
```
Valuta la necessità di adottare provvedimenti straordinari in materia di viabilità
e trasporti
```
```
Valuta costantemente con il sindaco, sentiti gli organi competenti, l'opportunità
di revocare lo stato di emergenza esterna e dichiara il cessato allarme
```

### SALA OPERATIVA PER LA^ GESTIONE DELL’EMERGENZA (SOE)

```
Evento SOE DI RIFERIMENTO
Incidente (pee non attivato) Sala operativa comando provinciale VVF (h24)
Incidente (pee attivato) Sala operativa della prefettura
```
### COMANDO PROVINCIALE DEI VIGILI DEL FUOCO

I Vigili del Fuoco porranno in essere le attività operative di propria competenza per il soccorso
tecnico urgente assumendo il coordinamento delle operazioni di soccorso in loco.
Evento AZIONE VVF

```
Incidente
```
```
Riceve dal gestore l'informazione sul preallertamento e la richiesta di
allertamento, secondo le previsioni del pei;
Se l'incidente ha rilevanza esterna, potenziale o reale, avvisa il prefetto per
l'attivazione del PEE;
Assume, su incarico del prefetto, la funzione di direttore tecnico dei soccorsi, cui
si rapportano tutte le funzioni;
Dirige il soccorso tecnico per il salvataggio delle persone e la risoluzione tecnica
dell’emergenza, avvalendosi del supporto del gestore e delle altre funzioni e
raccordandosi con il prefetto secondo quanto previsto dal PEE;
Tiene costantemente informato il prefetto sull’azione di soccorso e sulle misure
necessarie per la tutela della salute pubblica, valutando l’opportunità di
un'evacuazione o di altre misure suggerite dalle circostanze e previste nelle
pianificazioni operative di settore;
Individua le zone di danno per consentire la perimetrazione da parte delle forze
di polizia che impedisca l’accesso al personale non autorizzato
```
### SINDACO

Evento AZIONE SINDACO

Attiva le strutture comunali di Protezione Civile (Ufficio Protezione Civile, Polizia
Locale, Municipio competente) Come previsto dal PEE;

Informa la popolazione sull'incidente e comunica le misure di protezione da
adottare per ridurne le conseguenze;

Attua le azioni di competenza del piano operativo per la viabilità e per
l’evacuazione assistita;

Adotta ordinanze contingibili ed urgenti per la tutela dell’incolumità pubblica;

Segue l'evoluzione della situazione e informa la popolazione della revoca dello
stato di emergenza;

In caso di cessata emergenza, opera per il ripristino delle condizioni di normalità e
in particolare per l'ordinato rientro della popolazione nelle abitazioni ove


```
necessario
```
**POLIZIA LOCALE** (^) Il personale può operare solo in zona sicura (zona bianca).
Predispone e presidia i cancelli nella zona gialla di competenza;
Coadiuva la polstrada nel controllo dei blocchi stradali;
Presidia i percorsi alternativi individuati nel piano operativo per la viabilità,
garantendo un regolare flusso dei mezzi di soccorso
**QUESTURA**
Il personale può operare solo in zona sicura (zona bianca).
Evento AZIONE QUESTURA
Incidente Coordina le FF.OO. (carabinieri, guardia di finanza, corpo forestale), la polizia
Locale e, se attivate dal prefetto, le forze armate.
Controlla i flussi nelle aree dell’emergenza, anche ai fini del mantenimento
dell'ordine e della sicurezza pubblica
Predispone e presidia i cancelli, gli sbarramenti e le perimetrazioni della zona
gialla, avvalendosi di FF.OO, Polizia Locale e, se attivate dal prefetto, forze
armate.
Predispone e presidia, avvalendosi della polstrada, i percorsi alternativi di cui al
piano operativo di viabilità, per garantire il flusso dei soccorsi e l’evacuazione;
Coordina e vigila sull'evacuazione affinché avvenga in modo corretto ed ordinato,
come da piano operativo di evacuazione assistita.

### AZIENDA SANITARIA LOCALE (ASL RM 3 )

L’Asl porrà in essere le attività operative di propria competenza per il soccorso sanitario e le altre
iniziative collegate all’emergenza.
Il personale può operare solo in zona sicura (zona bianca)
EVENTO AZIONE ASL Rm 3
Incidente Invia il personale tecnico, che si raccorda con il prefetto come previsto dal PEE
per una valutazione della situazione;
Informa, sentito il direttore dei soccorsi sanitari, gli ospedali sugli aspetti sanitari
connessi all'incidente per la parte di competenza;
Effettua, di concerto con l'arpa, analisi, rilievi e misurazioni per identificare le
sostanze coinvolte e quantificare il rischio sulle matrici ambientali (aria, acqua,
suolo) per la parte di competenza. Se necessario, di concerto con le autorità
competenti, fornisce tutti gli elementi per l’emanazione di limitazioni o divieti
dell'uso di risorse idriche;
Fornisce al prefetto, sentite le altre autorità sanitarie, i dati su entità ed
estensione dei rischi per la salute pubblica e l’ambiente.


### SERVIZIO EMERGENZA SANITARIA 118

L’Ares-118 porrà in essere le attività operative di propria competenza per il soccorso sanitario,
l’evacuazione assistita e le altre iniziative collegate all’emergenza.
In particolare, il personale può operare, su specifica disposizione dei VV.F. in funzione delle
condizioni di sicurezza accertate, nella zona di danno (zona gialla) se adeguatamente formato e
dotato di dpi; in caso contrario può operare solo nella zona sicura (zona bianca).

```
Evento AZIONE 118
Acquisisce le informazioni necessarie per individuare farmaci, antidoti e
attrezzature per contrastare gli effetti sanitari degli incidenti individuati nel PEE.
Incidente Invia il personale sanitario che si raccorda con il prefetto come previsto dal PEE
per il soccorso sanitario urgente;
Assume, su incarico del prefetto, la funzione di direttore dei soccorsi sanitari, cui
si rapporteranno l’ASL e gli altri enti previsti;
Gestisce il soccorso sanitario e l'evacuazione assistita per la parte di
competenza;
Interviene nelle zone di danno (zona gialla) per il soccorso alle vittime, previa
specifica autorizzazione dei VV.F. e con adeguati dpi;
Assicura in caso di evacuazione il trasporto dei disabili e malati, e il ricovero di
eventuali feriti.
```
### CROCE ROSSA ITALIANA (CRI)^ ED ALTRI ENTI DI SOCCORSO SANITARIO

Il personale può operare solo in zona sicura (zona bianca)
Evento AZIONE CRI E ALTRI ENTI DI SOCCORSO SANITARIO
Incidente Invia il proprio personale che dipenderà funzionalmente dal responsabile del 118

```
Assicura, in caso di evacuazione il trasporto dei disabili e malati, e il ricovero di
eventuali feriti
```

### AGENZIA REGIONALE PER LA PROTEZIONE DELL’AMBIENTE (ARPA)

Il personale può operare solo in zona sicura (zona bianca).

```
Evento Azione arpa
Incidente Fornisce supporto tecnico in base alla conoscenza dei rischi associati agli
stabilimenti e ai controlli effettuati
```
```
Effettua di concerto con l’ASL ogni accertamento necessario sullo stato
dell’ambiente, le analisi chimico/fisiche per valutare l’evoluzione della situazione
nelle zone più critiche come previsto nel piano operativo di sicurezza ambientale
per la parte di competenza
```
```
Eliminare (è incluso nel primo punto).
```
```
Trasmette direttamente al prefetto i risultati delle analisi e delle rilevazioni
richieste
```
```
Fornisce, relativamente alle proprie competenze, supporto per la definizione
delle azioni da intraprendere a tutela dell’ambiente e della sicurezza della
popolazione e dei luoghi dove si è verificato l’incidente.
```
```
Nel caso in cui si sia determinato il rilascio di sostanze pericolose per l‘ambiente
provvede ad attività di monitoraggio e allo svolgimento delle attività di
competenza previste dalla normativa inerente ai siti contaminati (d.lgs. 152/
s.m.i. Parte iv titolo v, dgr 1 luglio 2008 n.451).
```
### REGIONE LAZIO AGENZIA REGIONALE PROTEZIONE CIVILE

La protezione civile regionale è allertata dal gestore ai sensi del D.lgs 334/99. Il personale può
operare solo in zona sicura (zona bianca).

```
Evento Azione protezione civile regionale
Incidente Se necessario, attiva i gruppi di volontariato di protezione civile dei comuni
limitrofi, di altri comuni.
```

### VOLONTARIATO

I volontari di Protezione Civile possono operare solo in zona sicura (zona bianca) secondo quanto
previsto dal PEE e adeguatamente formato ed equipaggiato.
Evento Azione volontariato
Incidente Supporta le FF.OO. e la Polizia Locale per il controllo del traffico all'esterno delle
zone di danno, come previsto dal piano operativo per la viabilità;
Assiste la popolazione ove ritenuto necessario.

(^) **CITTÀ METROPOLITANA DI ROMA CAPITALE**
Il personale può operare solo in zona sicura (zona bianca).
Evento Azione città metropolitana
Redazione
PEE
Assicura il supporto tecnico-scientifico alla stesura, revisione ed aggiornamento
del PEE
Incidente Assicura il supporto tecnico per la messa in sicurezza dell’area
**Informazione preventiva della popolazione**
In attesa che il Dipartimento della protezione civile stabilisca, d'intesa con la Conferenza Unificata,
le linee guida per la predisposizione del piano di emergenza esterna, e per la relativa informazione
alla popolazione (art. 21 comma 7 del D.lgs 105/2015) “Il comune ove è localizzato lo stabilimento
mette tempestivamente a disposizione del pubblico, anche in formato elettronico e mediante
pubblicazione sul proprio sito web, le informazioni fornite dal gestore ai sensi dell’articolo 13,
comma 5 ....”, l’Ufficio ha provveduto a pubblicare sul sito istituzionale, in un’apposita sezione le
informazioni suddette corredate da apposita cartografia, allegando un opuscolo con indicazioni sui
comportamenti da seguire in caso di incidente rilevante.
L’Ufficio intende promuovere una campagna capillare con materiale divulgativo distribuito delle
associazioni di volontariato ai cittadini presenti a vario titolo nelle zone a rischi.
Informazione sull’evento incidentale
La popolazione interessata dall’evento emergenziale è immediatamente informata sui fatti relativi
all’incendio, sul comportamento da adottare e sui provvedimenti di protezione sanitaria ad essa
applicabili nella fattispecie. In particolare, vengono fornite in modo rapido e ripetuto informazioni
riguardanti:
a) la sopravvenuta emergenza e, in base alle notizie disponibili, le sue caratteristiche: tipo, origine,
portata e prevedibile evoluzione;
b) le disposizioni da rispettare - in base al tipo di emergenza - ed eventuali suggerimenti di
cooperazione;


c) le autorità e gli enti cui rivolgersi per informazione, consiglio, assistenza, soccorso ed eventuali
forme di collaborazione.

Per quanto riguarda l’organizzazione della diffusione dell’informazione, l’obiettivo prioritario è
quello di informare tempestivamente la popolazione interessata da un evento incidentale già a
partire dalla fase di preallarme, in modo tale da evitare o contenere al massimo fenomeni di
inquietudine e reazioni imprevedibili.

Al fine di evitare la diffusione di notizie non sicure e non suffragate da dati certi, deve essere
designato un responsabile unico per la diffusione dell’informazione, con funzione di
coordinamento. Per le finalità del presente Piano e, in particolare, in caso di evento che preveda
l’attivazione del Servizio della protezione civile, responsabile dell’informazione è il Prefetto di
Roma.

In particolare, in caso di preallarme, alla popolazione devono essere fornite informazioni
riguardanti:

il tipo e l’origine dell’evento;

le principali caratteristiche delle sostanze rilasciate;

i tempi e le modalità con le quali sono diffusi gli aggiornamenti sull’evoluzione della situazione
emergenziale.

In caso di allarme, la popolazione deve ricevere in modo rapido e ripetuto informazioni
riguardanti:

il tipo di situazione di emergenza in atto;

la prevedibile evoluzione dell’evento e l’influenza dei fattori climatici e meteorologici;

le principali caratteristiche delle sostanze rilasciate;

la zona geografica del territorio eventualmente interessata;

le Autorità cui rivolgersi per ulteriori informazioni e consigli.

Nelle situazioni in cui si impongono provvedimenti e comportamenti di protezione per la salute
della popolazione, il Sindaco trasmetterà i messaggi RIFUGIO AL CHIUSO con una adeguata
campagna informativa. Devono, inoltre, essere diffuse informazioni su:

circolazione delle persone e utilizzo razionale delle abitazioni (per esempio chiusura di porte e
finestre, spegnimento degli impianti di aria condizionata e dei sistemi di presa d’aria esterna,
spostamento in ambienti seminterrati o interrati);

eventuali restrizioni e avvertimenti relativi al consumo degli alimenti e dell’acqua


### 4) FASI E RELATIVO CRONOPROGRAMMA DELLA PIANIFICAZIONE

**PIANIFICAZIONE**

```
Riunione di apertura Revisione p.e.e. 27 novembre 2017
```
```
Riunione con gestore e gruppo
generale di pianificazione
Richieste ai vari Enti:
```
```
A Roma Capitale:
```
```
Censimento della popolazione nel raggio di un
Km; aggiornamento zona ammassamento mezzi
di soccorso e viabilità con cartografia aggiornata.
```
```
27 novembre
```
```
ai gestori,
Arpa Lazio, ASL ed ARES 118
```
```
Aggiornamento scenari di danno,
dati meteo, ospedali coinvolti di zona e non, ed
elenco pazienti disabili.
```
```
27 novembre
```
```
Riunione ristretta
```
```
Esame documentazione presentata dal gestore
ed elaborazione cartografie; solleciti per i dati
mancanti alla ASL e alla Questura (viabilità,
pazienti disabili)
```
```
26 gennaio 2018
```
```
Riunione ristretta con Roma Capitale
```
```
Aggiornamento della documentazione ricevuta e
modifica delle cartografie secondo le ulteriori
indicazioni dei VV.F.(misure di autotutela e
sintesi delle zone di danno)
```
```
29 gennaio 2018
```
```
Riunione ristretta con Roma Capitale
```
- Dipartimento Sicurezza e
Protezione Civile – Direzione
Protezione Civile

```
Elaborazione della documentazione pervenuta e
aggiornamento secondo le indicazioni dei VV.F.
(necessità o meno di tenere due zone di
ammassamento mezzi di soccorso per la società
DE.CO) solleciti alla ASL per i pazienti disabili
```
```
30 gennaio 2018 e
12 febbraio 2018
16 marzo 2018
27 marzo 2018
28 marzo 2018
```
```
Riunione Gruppo Tecnico
```
```
Condivisione della documentazione pervenuta,
lettura dell’elaborato del P.E.E.
Sollecito alla ASL eventuali pazienti disabili o
legati ad apparecchiature salvavita.
```
```
15 febbraio 2018
```
**DIVULGAZIONE**

```
Pubblicazione delle informazioni da
rendere disponibili alla popolazione
(ex D.M. 24.7.2009, nr. 139)
```
- 30 marzo 2018

```
Eventuali osservazioni
pervenute dalla popolazione -^
```
```
Entro 30 gg.
dalla pubblicazione
```
```
Consultazione della popolazione
(assemblea pubblica) -^
```
```
Data presumibile:
15 maggio 2018
```
```
Riunione con gruppo generale di
pianificazione ed approvazione da
parte del Prefetto
```
- Data presumibile: 30 maggio 2018

### 1)


### 5) AZIONI PREVISTE DAL PEE CONCERNENTI IL SISTEMA DEGLI ALLARMI IN EMERGENZA E LE

### RELATIVE MISURE DI AUTOPROTEZIONE DA ADOTTARE

A parte le prescrizioni nei confronti delle aziende che rientrano nella disciplina del d.lgs.n.105 del

2015 , in caso di accadimento di incidente rilevante l’azienda interessata darà comunicazione alle
aziende limitrofe dell’evento in corso e del comportamento da tenere in caso di evacuazione e di

intervento dei VV.F.

Le aziende a rischio d’incidente rilevante circostanti l’area comprensoriale dovranno, ove non

ancora provveduto, preventivamente informare mediante la trasmissione del piano di emergenza

interno dei rischi e comportamenti da seguire in caso di emergenza.

La distinzione in livelli di allerta ha lo scopo di consentire ai Vigili del Fuoco di intervenire fin dai
primi momenti, e all’AP il tempo di attivare, in via precauzionale, le misure di protezione e

mitigazione delle conseguenze previste nel PEE per salvaguardare la salute della popolazione e la

tutela dell’ambiente.

I livelli di allerta sono:

```
ATTENZIONE
```
Stato conseguente ad un evento che, seppur privo di qualsiasi ripercussione all’esterno dell'attività
produttiva per il suo livello di gravità, può o potrebbe essere avvertito dalla popolazione creando,

così, in essa una forma incipiente di allarmismo e preoccupazione per cui si rende necessario

attivare una procedura informativa da parte dell’Amministrazione comunale.

In questa fase, il gestore informa l’AP e gli altri soggetti individuati nel PEE in merito agli eventi in

corso, al fine di consentir ne l'opportuna gestione.

```
PREALLARME
```
Si instaura uno stato di «preallarme» quando l’evento, pur sotto controllo, per la sua natura o per

particolari condizioni ambientali, spaziali, temporali e meteorologiche, possa far temere un

aggravamento o possa essere avvertito dalla maggior parte della popolazione esposta,

comportando la necessità di attivazione delle procedure di sicurezza e di informazione.

Tali circostanze sono relative a tutti quegli eventi che, per la vistosità o fragorosità dei loro effetti

(incendio, esplosione, fumi, rilasci o sversamenti di sostanze pericolose), vengono percepiti
chiaramente dalla popolazione esposta, sebbene i parametri fisici che li caratterizzano non

raggiungano livelli di soglia che dalla letteratura sono assunti come pericolosi per la popolazione

e/o l’ambiente.


In questa fase, il gestore richiede l’intervento di squadre esterne dei VVF, informa l’AP e gli altri

soggetti individuati nel PEE.

L’AP assume il coordinamento della gestione dell’emergenza al fine di consentire un’attivazione

preventiva delle strutture, affinché si tengano pronte a intervenire in caso di evoluzione di un

evento incidentale.

```
ALLARME - EMERGENZA ESTERNA ALLO STABILIMENTO
```
Si instaura uno stato di «allarme» quando l’evento incidentale richiede, per il suo controllo nel

tempo, l’ausilio dei VVF e, fin dal suo insorgere o a seguito del suo sviluppo incontrollato, può

coinvolgere, con i suoi effetti infortunistici, sanitari ed inquinanti, le aree esterne allo stabilimento.

Tali circostanze sono relative a tutti quegli eventi che possono dare origine esternamente allo
stabilimento a valori di irraggiamento, sovrappressione e tossicità superiori a quelli solitamente

presi a riferimento per la stima delle conseguenze (DM 9 maggio 2001).

In questa fase, si ha l’intervento di tutti i soggetti individuati nel PEE.

```
CESSATO ALLARME
```
La procedura di attivazione del cessato allarme è assunta dall’Autorità Preposta (Prefetto) sentite
le strutture operative e gli amministratori locali, quando è assicurata la messa in sicurezza del

territorio e dell’ambiente.

Comunicazione dell’allarme da parte del gestore alle aziende interne alle aree di danno

Al verificarsi di una qualunque situazione di emergenza, il coordinatore delle misure di emergenza,

(gestore dell’impianto) o il suo sostituto in caso di assenza (assistente di stabilimento), attiva la

procedura di comunicazione dell’emergenza alle aziende limitrofe che si trovano all’interno delle

aree di danno utilizzando la linea telefonica e i sistemi di allarme.

### 6) STRUTTURE RICETTIVE IN CASO DI INCIDENTE

Non essendo state censite strutture o abitazioni nella zona di sicuro impatto, in cui sarebbe

prevista la misura di protezione dell’evacuazione, non è stata individuata alcuna area di attesa.


### 7) LIVELLI DI AUTO PROTEZIONE DA FAR ASSUMERE ALLA POPOLAZIONE NELLE ZONE A

### RISCHI

### ZONA DI DANNO MISURE DI AUTOTUTELA

```
I (fino a 68 m)
```
```
II ( 68 - 149 m)
```
```
III ( 149 - 298 m)
```
```
Rifugio al chiuso
```
```
I residenti e le persone che si trovano a
qualsiasi titolo presenti in questa zona
dovranno permanere all’interno degli edifici,
avendo cura di tenersi lontani da porte e
finestre.
```
### ALLARME E MESSAGGIO ALLA POPOLAZIONE

l PEE è attivato a seguito di segnalazione anche con allarme proveniente dallo Stabilimento,

tramite il suono di sirena che indica contemporaneamente ai soccorritori e alla popolazione il

verificarsi di un incidente

I sistemi di allarme sono dislocati all’esterno della palazzina uffici

Il messaggio di allarme è UN SUONO LUNGO CONTINUATO

Il messaggio alla popolazione riguarda: RIFUGIO AL CHIUSO

Il messaggio di cessato allarme è UN SUONO LUNGO (ripetuto tre volte)


### COMPORTAMENTO DA SEGUIRE

```
I comportamenti specifici che la popolazione deve tenere, nell’eventualità
dell’accadimento di un incidente tale da interessare le aree esterne del Deposito, sono
riportati di seguito.
 seguire le indicazioni del PEE.
```
Si raccomanda alla popolazione di:

```
 mantenere la calma;
 non recarsi sul luogo dell'incidente;
 non occupare l’area limitrofa al Deposito (anche in caso di familiari coinvolti);
 lasciare libere le vie di comunicazione e gli accessi al Deposito;
 non occupare inutilmente le linee telefoniche;
 non usare ascensori;
 interrompere l'erogazione del gas; spegnere ogni tipo di fiamma;
 accendere radio/TV e sintonizzarsi sulle emittenti locali;
 attendere istruzioni dalle Autorità preposte su eventuali altre azioni;
 attendere il “cessata emergenza” comunicata dagli organi di informazione o da chi è
preposto nel Piano di Emergenza Esterno (PEE)
```
In caso di incendio la popolazione deve:

```
 cercare riparo dall’irraggiamento diretto;
 fermare i sistemi di condizionamento in caso di permanenza entro edifici.
```
In caso di esplosione:

```
 mantenere la calma, ricordando che generalmente non si tratta di un evento ripetitivo;
 accertarsi e portare i primi soccorsi (senza allontanarsi) ad eventuali feriti da schegge e
frammenti.
```
Le misure di autotutela previste nel PEE per le persone presenti nelle zone di danno, limitrofe al
Deposito, sono quelle di seguito descritte: RIFUGIO AL CHIUSO

Le persone che si trovano al chiuso DEVONO permanere all’interno degli edifici, mantenendo
accuratamente chiuse porte e finestre TENENDOSI lontane dalle stesse.


----------


"""
    try:
        data = request.json
        question = data.get('question')

        llm = get_gpt()

        input_variables = ["question"]
        prompt_text = template
        prompt_template = PromptTemplate(template=prompt_text, input_variables=input_variables)
        chain = prompt_template | llm | StrOutputParser()				
        generation = chain.invoke({"question": question})

        response = {
            'question': question,
            'answer': generation
        }

        return jsonify(response)

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# Quali sono gli impianti a distanza superiore ai 300 metri? 

# Quali sono le zone di danno e a che metratura si trovano? Che azioni vanno intraprese per ciascuna zona? 

# Quali sono gli insediamenti produttivi limitrofi all'impianto? 

# Quanti sono i dipendenti dell'impianto? 

# Che tipi di danni potrebbero verificarsi sulla popolazione e quindi vanno controllati? 

# Quali sono le misure di autotutela per le persone presenti? 