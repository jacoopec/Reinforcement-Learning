# Maze

Model it as a gridworld Markov Decision Process (MDP).

This maze is modeled in RL as a finite gridworld MDP, where each reachable cell is a state, the agent moves with actions like up/down/left/right, gets penalized for time or invalid moves, and gets a positive reward for reaching the goal.

For this maze:

States: each reachable white square is a state.
S: all valid white cells
Each open cell can be represented by its grid coordinates:
s=(row,col)
You can also number the valid cells:
S={0,1,2,…,n−1}
Start state: the white square marked “Start”.
Goal state: the white square marked “Goal”.

Actions: usually up, down, left, right.
A={↑,↓,←,→}



Transitions:
If the move goes to another white square, the agent moves there.
If the move hits a black wall or goes outside the maze, either:
the agent stays in the same state, or
you give an invalid move penalty.
Terminal state: the goal square.
Transition model
Deterministic version
This is the simplest:
move succeeds exactly as chosen if the destination is open
otherwise remain in place
So:
P(s′∣s,a)=1



A clean formalization is:

M=(S,A,P,R,γ)



P(s′∣s,a): transition probability

R(s,a,s′): reward

R(s,a,s′)={+10−1​if s′=goal otherwise​}


γ: discount factor






Episode definition
One episode:
reset agent to Start
let it move until:
it reaches Goal, or
a max step limit is reached



Objective

Learn a policy:
π(a∣s)

that maximizes expected discounted return:
𝐺𝑡=∑𝑘=0∞𝛾𝑘𝑟𝑡+𝑘+1Gt
	​
Suitable RL methods

Because this maze is small and discrete, the most natural methods are:

Q-learning
SARSA
Value iteration or policy iteration if you already know the model