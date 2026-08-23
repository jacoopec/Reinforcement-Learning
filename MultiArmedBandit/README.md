# MULTI-HARMED BANDIT 
slot machine with multiple levers.
you have ,k actions that each gives a  random reward. There is no state or sequence.
The goal  is to learn which lever has the  highest everage reward, while still trying others
enough to eb sure  

The central RL problem is about  exploitation vs exploration
Exploit: pick the lever that seems bbest so far 
explore:  try other to discover if they are better.

Algorithms:
 choose a random lever with probability eps (explore) or the best estimated lever (exploit)
 then update each lever's estimated value with the evarage reward you have seen.