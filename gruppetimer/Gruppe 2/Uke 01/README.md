# Uke 1

Velkommen til første gruppetime i Søketeknologi høsten 2023! Under er en oppsummering av det vi går gjennom i gruppetimen :)

## Oversikt over avhengighetene i prekoden

Repoet er svært, og det er mye å sette seg inn i. Mye er veldig nyttig, noe er mindre nyttig. Noen filer skal dere endre, noen filer trenger dere ikke engang å se på. Jeg har laget en liten oversikt [her](https://github.com/aohrn/in3120-2023/blob/main/gruppetimer/Gruppe%202/Prekoden.md) som viser alle avhengighetene, inkludert de som er relevante for de tre første obligene.

Et annet tips er å kommentere ut testene, også prøve å få koden til å kjøre på én og én test om gangen :)

## Hva er egentlig "information retrieval"?

Information retrieval går ut på å finne materiale som oppfyller et _informasjonsbehov_ fra et _korpus_.

### Korpus?

Et korpus er en samling med dokumenter.

### Hva er strukturert, semi-strukturert og ustrukturert data?

**Strukturert data** har en rigid struktur. Vi kan tenke på tabeller og databaser.

**Semi-strukturert data** er fortsatt strukturert, men ikke like streng. Her kan vi tenke på JSON, der vi har bestemte nøkkel-verdi-par som kan være nøstet og ordnet vilkårlig.

De virker kanskje like, men poenget er at tabeller bryr seg om hvordan radene og kolonnene er posisjonert.

Denne tabellen:

| Navn       | Alder | Fylke    |
| ---------- | ----- | -------- |
| Aleksander | 53    | Oslo     |
| Sergey     | 23    | Nordland |
| Oliver     | 23    | Viken    |

og denne tabellen:

| Navn       | Fylke    | Alder |
| ---------- | -------- | ----- |
| Sergey     | Nordland | 23    |
| Aleksander | Oslo     | 53    |
| Oliver     | Viken    | 23    |

er ulike per definisjon, fordi tabeller kolonnene er posisjonert forskjellig, selv om selve dataen er lik. Derfor må de i teorien behandles ulikt.

Derimot er dette objektet:

```json
{
  "navn": "Sergey",
  "alder": 23,
  "fylke": "Nordland"
}
```

og dette objektet:

```json
{
  "fylke": "Nordland",
  "navn": "Sergey",
  "alder": 23
}
```

"identiske", fordi JSON-objekter bare avhenger av deres dataverdier, og ikke på noen spesiell rekkefølge, så eksemplene over kan behandles helt likt.

**Ustrukturert data** er fritekst, som f.eks. en vanlig .txt-fil. Det å bruke Ctrl+F/Cmd+F for å søke etter en streng i en et slikt dokument, er et _ustrukturert søk_.

### Hva bestemmer kvaliteten på svaret til søkemotoren?

Her er to objektive måter å måle dette på:

- **Precision**: av alle dokumenter som ble hentet, hvor mange er relevante for brukerens informasjonsbehov?
- **Recall**: av alle dokumenter som er relevante for brukerens informasjonsbehov, hvor mange hentet vi?

Ønsker vi bra recall, kan vi hente _alle_ tilgjengelige dokumenter. Da har vi, i teorien, også hentet _alle_ de relevante dokumentene! Men også en haug irrelevante, som svekker precision.

Ønsker vi bra precision, kan vi prøve å få tak i ett dokument som er relevant. Henter vi det ene dokumentet alene, får vi 100% precision. Men kanskje dårligere recall, om det finnes flere relevante dokumenter der ute.

Ofte ønsker vi seg en mellomting, og selve informasjonsbehovet bestemmer gjerne hva som er et akseptabel forhold.

### Hva er en "posting"?

En posting sier noe om forholdet mellom en **term** og et **dokument**. En posting kan inneholde ting som

- dokument-ID (hvilket dokument det er snakk om)
- frekvens (hvor mange ganger termen forekommer i dokumentet)
- posisjon (nøyaktig hvor termen forekommer i dokumentet)

I obligene jobber vi med [posting.py](https://github.com/aohrn/in3120-2023/blob/main/in3220/posting.py). Den klassen tar vare på to instansvariabler: dokument-ID og frekvens.

### Hva er en "invertert indeks"?

Moderne søkesystemer benytter en _invertert indeks_. Den tar vare på en ordbok med termer, og lister med postings som er relevante for hver term. Hver term mapper nemlig til en liste med postinger. Den enkleste typen postinger sier gjerne bare hvilket dokument termen forekommer i:

```python
invertert_indeks = {
    "søketek": [1, 3, 7, 12, 23],
    "informatikk": [1, 4, 8],
    "maps": [4, 70, 78, 81]
}
```

I eksemplet over, forekommer termen "søketek" i dokumentene med ID 1, 3, 7, 12 og 23.

Vi kan som sagt også fylle dem med enda mer data, f.eks. hvor mange ganger de forekommer i hvert dokument:

```python
invertert_indeks_med_frekvens = {
    "søketek": [
        { docID: 1, frekvens: 4 },
        { docID: 3, frekvens: 42 },
        { docID: 7, frekvens: 1 },
        { docID: 12, frekvens: 12 },
        { docID: 23, frekvens: 4 }
    ],
    "informatikk": [
        { docID: 1, frekvens: 1 },
        { docID: 4, frekvens: 2 },
        { docID: 8, frekvens: 3 }
    ],
    "maps": [
        { docID: 4, frekvens: 100 },
        { docID: 70, frekvens: 42 },
        { docID: 78, frekvens: 1 },
        { docID: 81, frekvens: 43 }
    ]
}
```

I eksemplet over, forekommer termen "søketek" 4 ganger i dokumentet med ID 1, 42 ganger i dokumentet med ID 3, 1 gang i dokumentet med ID 7, osv.

Hvis vi søker etter "søketek" i en søkemotor som benytter den inverterte indeksen over, vil det være naturlig å få opp dokumentet med ID 3 før dokumentet med ID 7, siden søkeordet forekommer 42 flere ganger i førstnevnte.

Vi kan også legge inn _hvor_ i dokumentet den ligger, f.eks. at "søketek" forekommer i tittelen, eller at "søketek" er det femte og tiende ordet i dokumentet, osv. osv. Det er egentlig ingen begrensning på hvor mye info du kan pakke inn i en posting.

### Vil ikke ord som "er" forekomme i kjempemange dokumenter?

Det stemmer, og "er" er et så kalt _stoppord_. Stoppord er ord som forekommer i veldig mange (om ikke alle) dokumenter. På en side kan man ignorere dem, fordi de er så vanlige, men på en annen side er de helt avgjørende:

- Fly _til_ Oslo
  - Uten stoppord: Fly Oslo
    - Ønsker vi resultater på fly til eller fra Oslo?
- Ta _på_ hatten din
  - Uten stoppord: Ta hatten din
    - Ønsker vi å ta på hatten? ta av hatten? ta ned hatten? osv.

Noen fraser/egennavn består utelukkende av stoppord, som

- [Let it be](<https://en.wikipedia.org/wiki/Let_It_Be_(Beatles_album)>)
- [The Who](https://en.wikipedia.org/wiki/The_Who)
- [To be or not to be](https://en.wikipedia.org/wiki/To_be,_or_not_to_be)

Ved hjelp av gode kompresjons- og optimaliseringsteknikker, blir det helt ufarlig å inkludere stoppord i inverterte indekser :)

### Boolean retrieval

Kanskje den enkleste formen for spørringer. Veldig vanlig før, og brukes blant annet fortsatt i mail-søk. Vi bruker boolske uttrykk i spørringene, så spørringen "informatikk er gøy" vil resultere i "informatikk **AND** er **AND** gøy". Da vil vi, helt overordnet, finne postinglista til hver av termene, og vise dokumentene der alle tre forekommer.

Si at vi kjører spørringen over på et søkesystem med følgende inverterte indeks:

```python
invertert_indeks =
{
    "informatikk": [1, 2, 4, 5, 7, 8, 9, 12, 14, 18],
    "er": [1, 4, 5, 6, 7, 9],
    "gøy": [1, 2, 4, 7, 8, 9]
}
```

Da vil vi ende opp med

```python
postings = [1, 4, 7, 8, 9]
```

Fordi dokumentene med ID 1, 4, 7, 8 og 9 forekommer i postinglistene til alle termene. Siden listene til "er" og "gøy" er kortest, optimaliserer vigjerne søket til ("informatikk" **AND** ("er" **AND** "gøy")).

Faktisk er det akkurat dette vi skal implementere i [Assignment A](https://github.com/aohrn/in3120-2023/blob/main/assignment-a.md)! :)

> _Side note:_ Om man er kjent med mengdelære, kan man tenke på "and" som snitt og "or" som union.

### Hva med frasespørringer?

Om vi søker på

> Informatikk er gøy

, vil også

> Informatikk er kjedelig, støvsuging er gøy

være en match gjennom boolean retrieval.

For å unngå dette, kan vi prøve å bruke en **bi-word index**!

### Bi-word index

Her lagrer vi to og to ord av gangen, som dette:

```python
biword_index =
{
    "informatikk er": [1, 3, 7, 8],
    "er gøy": [1, 2, 3, 8, 9]
}
```

Da vil vi... fortsatt matche på "**informatikk er** kjedelig, støvsuging **er gøy**". Så det problemet er ikke løst helt enda. Men! Denne typen indeks fungerer ganske bra når søket inneholder negasjon. Om vi prøver oss på

> kakeoppskrift uten egg

og søkemotoren vår bruker denne inverterte indeksen:

```python
biword_index =
{
    "kakeoppskrift uten": [1, 3, 7, 8],
    "uten egg": [1, 2, 3, 8, 9]
}
```

vil vi få alle resultatene som inneholder "uten egg"! Woo

**Så**, kort oppsummert:

Pros:

- En større garanti for at spørringer med negasjon virker

Cons:

- Vi vil matche på falske positiver som "**informatikk er** kjedelig, støvsuging **er gøy**"
- Indeksen blir giga (ca. 4 ganger så stor som en "vanlig" invertert indeks)

### Alternativ til bi-word index

Som tidligere nevnt, kan vi ekspandere den postinglistene med _hvor i dokumentet termen forekommer_:

```python
invertert_indeks_med_plassering =
{
    "informatikk": [
        { docID: 1, plassering: [20, 22] },
        { docID: 3, plassering: [1, 3, 19] },
        { docID: 7, plassering: [9, 11, 91] }
    ],
    "er": [
        { docID: 2, plassering: [2, 4, 6, 9] },
        { docID: 3, plassering: [4] },
        { docID: 7, plassering: [10, 58]}
    ]
    "gøy": [
        { docID: 2, plassering: [5, 7, 29] },
        { docID: 3, plassering: [5, 21, 25] },
        { docID: 8, plassering: [16, 74] }
    ]
}
```

Dette ser kanskje noe rotete ut, men tanken er at termen "informatikk" forekommer i dokumentene med ID 1, 3 og 7. I det første dokumentet, forekommer den som det 20. og 22. ordet. I den andre dokumentet forekommer den som det 1., 3. og 19. ordet. I det siste dokumentet forekommer den som det 9., 11. og 91. ordet.

Den legendariske frasen "informatikk er gøy" vil dermed skjule seg i dokumentet med ID 3! Dette ser vi fordi "informatikk" er på plass 3, "er" er på plass 4, og "gøy" er på plass 5.

### Kan metodene kombineres?

Ja, fordi det er naturlig å lagre egennavn som "Universitetet i Oslo" og "Ole-Johan Dahl" som egne termer, istedenfor å flette sammen postinglistene til henholdsvis "Universitetet", "i" og "Oslo", eller "Ole", "Johan" og "Dahl".

## Ukas bok

Jeg kommer også til å kjøre et lite innslag i gruppetimene som jeg kaller "ukas bok". Jeg tror det er smud og bra å lese, og forhåpentligvis vil noen av dere plukke opp en av disse og like dem like mye som det jeg gjorde. Begrunnelsen vil nok bare komme muntlig i gruppetimene, men jeg skal få lagt inn et bilde på slutten av disse READMEene. Denne ukas bok trenger ingen introduksjon!:))

![ukas bok](../images/intro_to_ir.jpg)
