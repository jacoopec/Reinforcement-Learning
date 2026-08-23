Multi-armed Bandits
The most important feature distinguishing reinforcement learning from other types of
learning is that it uses training information that evaluates the actions taken rather
than instructs by giving correct actions. This is what creates the need for active
exploration, for an explicit search for good behavior. Purely evaluative feedback indicates
how good the action taken was, but not whether it was the best or the worst action
possible. Purely instructive feedback, on the other hand, indicates the correct action to
take, independently of the action actually taken. This kind of feedback is the basis of
supervised learning, which includes large parts of pattern classification, artificial neural
networks, and system identification. In their pure forms, these two kinds of feedback
are quite distinct: evaluative feedback depends entirely on the action taken, whereas
instructive feedback is independent of the action taken.
In this chapter we study the evaluative aspect of reinforcement learning in a simplified
setting, one that does not involve learning to act in more than one situation. This
nonassociative setting is the one in which most prior work involving evaluative feedback
has been done, and it avoids much of the complexity of the full reinforcement learning
problem. Studying this case enables us to see most clearly how evaluative feedback di↵ers
from, and yet can be combined with, instructive feedback.
The particular nonassociative, evaluative feedback problem that we explore is a simple
version of the k-armed bandit problem. We use this problem to introduce a number
of basic learning methods which we extend in later chapters to apply to the full reinforcement learning problem. At the end of this chapter, we take a step closer to the full
reinforcement learning problem by discussing what happens when the bandit problem
becomes associative, that is, when actions are taken in more than one situation.
2.1 A k-armed Bandit Problem
Consider the following learning problem. You are faced repeatedly with a choice among
k di↵erent options, or actions. After each choice you receive a numerical reward chosen
from a stationary probability distribution that depends on the action you selected. Your