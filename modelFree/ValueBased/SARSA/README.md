
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

