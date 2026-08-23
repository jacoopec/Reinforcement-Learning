Model-free prediction

Qui l’obiettivo non è ancora trovare la policy migliore.
L’obiettivo è:
data una policy, stimare quanto è buona.
Cioè: l’agente segue una certa politica di comportamento, ad esempio:
in ogni stato, scegli l’azione secondo π
e vuole stimare la funzione di valore:
$Vπ(s)$
oppure:
$Qπ(s,a)$

quanto valore mi aspetto di ottenere partendo da questo stato, o da questa coppia stato-azione, seguendo la policy π?

In una griglia, se seguo sempre una certa policy, voglio capire:

quanto è buono trovarmi nella cella (1,2)?

oppure:

quanto è buono fare "destra" dalla cella (1,2)?

## Monte Carlo Learning

Aspetto la fine dell’episodio e poi aggiorno i valori usando il ritorno totale osservato.

Esempio:

parto dallo stato s
faccio varie azioni
arrivo al goal
osservo la ricompensa totale
uso questa ricompensa per aggiornare V(s)

Monte Carlo quindi aggiorna dopo aver visto un episodio completo.

## Temporal-Difference Learning

Aggiorno il valore passo dopo passo, senza aspettare la fine dell’episodio.

La forma tipica è: $V(s)←V(s)+α[r+γV(s′)−V(s)]$

## Model-free control

Qui l’obiettivo cambia.

Non voglio solo valutare una policy.

Voglio:

trovare una buona policy.

Quindi il problema di controllo cerca di imparare direttamente quali azioni scegliere.

Di solito si stima una funzione: $Q(s,a)$

perché se conosco il valore di ogni azione in ogni stato, posso scegliere l’azione migliore:

$π(s)=argamax​Q(s,a)$


Model-free significa che l’agente non conosce il modello dell’ambiente.

Cioè non conosce in anticipo:
P(s′∣s,a)

e non conosce perfettamente:

R(s,a)

Quindi non sa esattamente:

se faccio questa azione, dove finirò?
che ricompensa riceverò?


### On-policy Monte Carlo Control

L’agente valuta e migliora la stessa policy che sta usando per esplorare.

Cioè:

uso una policy ε-greedy
raccolgo episodi
aggiorno Q
miglioro la stessa policy ε-greedy

È “on-policy” perché la policy usata per generare i dati è la stessa che viene migliorata.


### On-policy Temporal-Difference Learning

Qui rientra soprattutto SARSA.

SARSA aggiorna così:

Q(s,a)←Q(s,a)+α[r+γQ(s′,a′)−Q(s,a)]

Dove a′ è l’azione realmente scelta dalla policy nello stato successivo.

Quindi SARSA impara il valore della policy che sta effettivamente seguendo.

Per questo è on-policy.

### Off-policy Learning

Qui la policy usata per esplorare può essere diversa dalla policy che si vuole imparare.

L’esempio classico è Q-learning.


| Concetto              | Obiettivo                                               | Esempi                            |
| --------------------- | ------------------------------------------------------- | --------------------------------- |
| Model-free prediction | Stimare il valore di una policy data                    | Monte Carlo, TD                   |
| Model-free control    | Trovare una buona policy                                | SARSA, Q-learning, Expected SARSA |
| On-policy             | Imparo la policy che sto usando                         | SARSA                             |
| Off-policy            | Imparo una policy diversa da quella usata per esplorare | Q-learning                        |

Prediction risponde alla domanda:

quanto è buona questa policy?

Control risponde alla domanda:

quale policy dovrei usare?