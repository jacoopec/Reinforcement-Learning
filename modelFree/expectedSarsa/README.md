###  Expected SARSA 
aggiorna così:
$$Q(s,a)←Q(s,a)+α[r+γa′∑​π(a′∣s′)Q(s′,a′)−Q(s,a)]$$

$$a′∑​π(a′∣s′)Q(s′,a′)$$
media pesata dei valori Q delle azioni possibili nello stato successivo
La media è pesata dalla policy π.

La differenza principale è che Expected SARSA sostituisce la singola azione futura con una media pesata su tutte le azioni possibili.