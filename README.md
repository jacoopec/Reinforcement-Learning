### Reward
### episode
sequence of steps come to end.
At each step:
 - the agent: receives a reward and an observation, emits an action
 - the environment: receives an action, emits a reward and an observation

# History
An agent will have  a sequence of observation, reward,  action

# State
Information used to decide the next action and the reward

# Policy
Decides  the agent's bhaviour and maps states to actions.

# State value function
expected return  being in state s and following policy pi.

# Action valu function 
expected return  being in state s, taking action a and following policy pi.

#  Model
It explicitly describeshow the environment works.

# Model-free prediction methods
They estimate the valu function given a policy in a non-observable environment
- montecarlo learning
- temporal diff learning



# Markov decision processes 
formally deescribe an environment for reinforcement learning


# L’equazione di Bellman 
nel Reinforcement Learning descrive il valore di uno stato come:

ricompensa immediata + valore atteso degli stati futuri

State value  function 
$$v_pi(s)=e[Gt|s]$$

Action value  function 
$$q_pi(s)=e[Gt|St=s,At=a]$$

Bellman's equations allow to convert infos about the environment into improvement of the agent's behaviour 


---

### Value function
misura quanto è buono trovarsi in uno stato s, seguendo una certa policy π.
Si indica con Vπ(s) e rappresenta il ritorno atteso partendo dallo stato s.
$Vπ(s)=a∑​π(a∣s)s′∑​P(s′∣s,a)[R(s,a,s′)+γVπ(s′)]$
---

### Bellman equation per la Q-function

Invece di valutare solo uno stato, possiamo valutare una coppia:
Qπ(s,a) cioè: quanto è buono fare l’azione a nello stato s, seguendo poi la policy π
$Qπ(s,a)=s′∑​P(s′∣s,a)[R(s,a,s′)+γa′∑​π(a′∣s′)Qπ(s′,a′)]$

---
### Bellman optimality equation

Quando vogliamo trovare la policy migliore, non usiamo più una policy fissata π, ma scegliamo sempre l’azione migliore.

Per la value function ottimale:
$V∗(s)=amax​s′∑​P(s′∣s,a)[R(s,a,s′)+γV∗(s′)]$
Per la Q-function ottimale:
$Q∗(s,a)=s′∑​P(s′∣s,a)[R(s,a,s′)+γa′max​Q∗(s′,a′)]$
Questa è la base di algoritmi come Q-learning.

---
###  MonteCarlo
The return is given by the sum  of th rewards divided by the number of states
MC si basa sul ritorno finale che ha molta varianza perch´e dipende da tante transizioni, azioni, rewards
quindi per avere una stima corretta ho bisogno di tantissimi episodi

---
### Temporal difference
Il TD target dipende da una sola transizione, una sola azione e un solo reward quindi ha varianza inferiore
al ritorno finale e impara in meno tempo

---
### Model-free control
Not estimate but evaluate a policy 

• On-Policy Monte-Carlo Control
• On-Policy Temporal-Difference Control (SARSA)
• Off-Policy Learning (Q-learning)

2 ways  to estimate a policy are state value funtction and action-value function
it is possible to improve the policy from the value function by acting greedily:
$$π′ = greedy(vπ)$$
I choose the policy which maximizes the reward.
Nei contesti model-free abbiamo a disposizione solo Q(s, a) perch´e per essere greedy rispetto alla state-value
function ho bisogno del modello e in particolare sapere quale azione mi porta in quale stato (probabilit`a di
transizione).

Per i metodi di model-free control abbiamo in generale due possibilit`a:
• On-policy learning: impara la policy migliore basandosi sui suoi episodi
• Off-policy learning: impara la policy migliore basandosi su episodi di un’altra policy


-----------
### Q-learning

Nel Q-learning, la regola di aggiornamento è:
$Q(s,a)←Q(s,a)+α[r+γa′max	​Q(s′,a′)−Q(s,a)]$

Dove:
$r+γa′maxQ(s′,a′)$

è il target di Bellman.

La differenza:
$r+γa′maxQ(s′,a′)−Q(s,a)$
si chiama TD error, cioè errore di temporal difference.

Il Q-learning è un algoritmo di Reinforcement Learning che serve a imparare quale azione conviene fare in ogni stato, anche senza conoscere in anticipo il modello dell’ambiente.

L’idea è imparare una tabella o funzione:

Q(s,a)

che significa:

quanto è buona l’azione a quando mi trovo nello stato s.

####  Cosa impara il Q-learning?

Impara la Q-function ottimale:
$Q∗(s,a)$

cioè il valore migliore possibile associato a ogni coppia stato-azione.
Una volta imparata questa funzione, la policy ottimale è semplice:
$π(s)=argamax​Q(s,a)$
nello stato s, scegli l’azione con il valore Q più alto.


Il Q-learning funziona così:

L’agente si trova in uno stato.
Sceglie un’azione.
Riceve una ricompensa.
Finisce in un nuovo stato.
Aggiorna il valore Q(s,a).
Ripete molte volte.

Con l’esperienza, i valori Q diventano sempre più accurati.

Il Q-learning è un algoritmo off-policy.

Vuol dire che può esplorare usando una strategia, per esempio casuale o ϵ-greedy, ma aggiorna i valori assumendo di seguire la miglior azione futura:
$a′max​Q(s′,a′)$
Quindi impara la policy ottimale anche mentre si comporta in modo esplorativo.

Il Q-learning è un metodo per imparare una funzione Q(s,a), che dice quanto conviene fare una certa azione in un certo stato.

La regola fondamentale è:

nuovo valore=vecchio valore+α⋅errore

cioè l’agente corregge gradualmente le sue stime in base all’esperienza.

----------------------

### SARSA On policy Temporal Difference Control
è un algoritmo di Reinforcement Learning molto simile al Q-learning, ma con una differenza importante:

SARSA aggiorna Q(s,a) usando l’azione che l’agente sceglierà davvero nel prossimo stato.

Il nome SARSA viene dalla sequenza: $S,A,R,S′,A′$

La formula di SARSA è:
$Q(s,a)←Q(s,a)+α[r+γQ(s′,a′)−Q(s,a)]$

SARSA confronta: $Q(s,a)$ cioè il valore stimato attuale, con $r+γQ(s′,a′)$

Quindi SARSA corregge Q(s,a) usando l’esperienza appena osservata.

####  Differenza tra SARSA e Q-learning

La differenza principale è nel termine futuro.
Q-learning:
$Q(s,a)←Q(s,a)+α[r+γa′max​Q(s′,a′)−Q(s,a)]$
Q-learning usa:
$max​Q(s′,a′)$
cioè assume che nel prossimo stato verrà scelta l’azione migliore possibile.

SARSA usa:
Q(s′,a′)
cioè il valore dell’azione che la policy sceglie davvero.

SARSA è un algoritmo on-policy.

Significa che impara il valore della stessa policy che sta usando per agire.

Per esempio, se l’agente usa una strategia ϵ-greedy, ogni tanto fa azioni casuali per esplorare. SARSA tiene conto anche di queste azioni esplorative nell’aggiornamento.

Q-learning invece è off-policy, perché aggiorna sempre assumendo la migliore azione futura, anche se nella pratica l’agente potrebbe non sceglierla.


SARSA e Q-learning sono entrambi algoritmi model-free.

Significa che non hanno bisogno di conoscere il modello dell’ambiente, cioè non richiedono esplicitamente: $P(s′∣s,a)$ nè $R(s,a,s′)$

L’agente impara direttamente dall’esperienza, osservando transizioni del tipo:

(s,a,r,s′)   nel q-learning
oppure, nel caso di SARSA:
(s,a,r,s′,a′)

SARSA è:

model-free
on-policy
value-based
TD learning

È model-free perché aggiorna Q(s,a) usando esperienze osservate, senza conoscere le probabilità di transizione.

Q-learning è:

model-free
off-policy
value-based
TD learning

Expected SARSA è:

model-free
value-based
TD learning
on-policy

se la policy usata per calcolare l’aspettazione è la stessa policy usata dall’agente per agire, per esempio ϵ-greedy.

può anche essere usato in versione off-policy, se l’agente esplora con una policy ma calcola l’aspettazione rispetto a un’altra policy target.


###  Expected SARSA aggiorna così:
$Q(s,a)←Q(s,a)+α[r+γa′∑​π(a′∣s′)Q(s′,a′)−Q(s,a)]$

$a′∑​π(a′∣s′)Q(s′,a′)$
media pesata dei valori Q delle azioni possibili nello stato successivo
La media è pesata dalla policy π.

La differenza principale è che Expected SARSA sostituisce la singola azione futura con una media pesata su tutte le azioni possibili.