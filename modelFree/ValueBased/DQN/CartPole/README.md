In CartPole, the agent controls a cart that can move left or right while trying to keep a pole balanced vertically.

The state is a vector of 4 values: $s=[x,x˙,θ,θ˙]$

where:

x: cart position 
x˙: cart velocity
θ: pole angle 
θ˙: pole angular velocity

There are only two actions:

$a∈{0,1}$

0 → push cart left
1 → push cart right

4 state values
      ↓
Linear layer
      ↓
ReLU
      ↓
Linear layer
      ↓
ReLU
      ↓
2 Q-values

---
The important flow in this script is:

state
  │
  ▼
DQN
  │
  ├── Q(left)
  │
  └── Q(right)
        │
        ▼
   ε-greedy action
        │
        ▼
     CartPole
        │
        ▼
(s, a, r, s')
        │
        ▼
  Replay Buffer
        │
        ▼
   Random batch
        │
        ▼
   DQN training


For every sampled transition, the network computes

Q(s,a;θ)

while the target network computes:

y=r+γ(1−terminated)a′max ​Qtarget​(s′,a′)

and the policy network is trained to make

Q(s,a;θ)≈y.

One detail worth noticing is that I store terminated, rather than terminated or truncated, for the Bellman target. A terminated episode represents an actual terminal MDP state, whereas truncated can simply mean CartPole reached its time limit; Gymnasium explicitly separates these because that distinction matters for bootstrapping algorithms such as DQN.

This is essentially the classic setting used to teach DQN in the official PyTorch/TorchRL material as well.