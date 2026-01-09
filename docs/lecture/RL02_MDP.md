# RL02_MDP

**Source:** c:\Users\catalin.butacu\Downloads\RL\resourses\lecture\RL02_MDP.pdf

**Pages:** 64

---


## Page 1

Reinforcement Learning
2. Markov Decision Processes 
Florin Leon
“Gheorghe Asachi” Technical University of Iași, Romania
Faculty of Automatic Control and Computer Engineering
https://florinleon.byethost24.com/lect_rl.html
2025


## Page 2

Markov Decision Processes 
1. Formalization
2. Value Iteration
3. Policy Iteration
2


## Page 3

Markov Decision Processes 
1. Formalization
2. Value Iteration
3. Policy Iteration
3


## Page 4

Markov Decision Processes
Markov Decision Processes (MDPs) are a formal framework for 
sequential decision making
Unlike bandit problems, actions affect both immediate and future 
rewards by influencing the next state
The goal is to learn a series of actions that maximize cumulative 
reward over time
Actions influence not just immediate rewards, but also subsequent 
states, and through those future rewards 
MDPs involve delayed reward and the need to trade off immediate 
and delayed reward
4


## Page 5

The Agent-Environment Interaction Loop
The agent selects an action
The environment presents a new state and a reward
This loop continues until a terminal state is reached
This results in a trajectory of the form: S0, A0, R1, S1, A1, R2, S2, A2, R3, ...
5


## Page 6

Key Components of an RL Problem
Agent: the decision maker that takes actions
Environment: the system with which the agent interacts
State: a representation of the environment at a given time
Action: a choice available to the agent at each state
Reward: a numerical value that indicates the desirability of 
taking an action in a state; it guides the agent’s learning
6


## Page 7

Agent-Environment Boundary

The boundary between agent and environment is not always the physical 
boundary (e.g., of a robot)

Mechanical components and sensors of a robot are typically considered part 
of the environment rather than the agent

Rewards are computed “inside” the agent but are still considered part of the 
environment because they define the task and cannot be arbitrarily changed 
by the agent

The agent controls its actions but cannot arbitrarily modify its environment

The agent-environment boundary represents the limit of the agent’s control, 
not of its knowledge

The agent may have knowledge about how rewards are computed or how the 
environment works, but this does not change the fundamental distinction

Even with full knowledge of the environment, a task can still be challenging, like 
solving a Rubik’s cube despite knowing all the rules
7


## Page 8

Formalization
An MDP is defined by: 
A finite set of states S
A finite set of actions A available in each state
A transition probability function p
A reward function r
A discount factor γ ∊ [0, 1]  (explained later)
The solution of this problem is a policy π
E.g., π(s) is the action that should be taken in state s
8


## Page 9

Function Definitions

The functions for transition probability, reward, and policy can be defined in 
different ways, depending on the problem

Starting from the general form of the transition probability function (the 
first in the next slide), one can obtain and use different forms for transitions 
and rewards by marginalization (eliminating some variable from the joint 
probability distribution by summing them out) 

The equations in the next slide look complex, but in fact they only show that 
we have the flexibility to define, e.g., the reward function as r(s, a, s'), 
r(s, a), or r(s'), as we see fit depending on the problem, and different 
representations can be converted using the general transition probabilities
9


## Page 10

Possible Function Definitions
10


## Page 11

Reward Hypothesis
All of what we mean by goals and purposes can be understood 
as maximizing the expected value of the cumulative sum of a 
received scalar signal, called the reward
Any goal can be reduced to preferences over outcomes, and 
preferences can be ranked numerically
Numbers allow consistent comparison and accumulation over time, 
and make complex objectives manageable
The reward function defines what is desirable, while specific RL 
algorithms focus on how to achieve it
11


## Page 12

Designing Rewards
Designing an effective reward function is critical for agent learning
Bad reward signals lead to undesired behaviors, for example, a chess 
agent should be rewarded for winning, not capturing pieces
Rewards can be sparse (only at goal) or dense (frequent feedback)
Sparse: a chess agent gets a positive reward for winning and a negative 
reward for losing
Dense: a navigation agent gets a small negative reward for each step it 
takes, encouraging it to reach the goal quickly
12


## Page 13

The Markov Property

The future is independent of the past given the present 

A process is Markovian if the next state depends only on the current state

The state captures all relevant information from the history; once the state 
is known, the history may be ignored

E.g., in a chess game only the current board position matters, not how it 
was reached

The Markov assumption allows efficient dynamic programming solutions

Most real scenarios are unlikely to be Markov, but we usually can transform 
or approximate this property, e.g., by introducing sensors for the relevant 
features of the environment and taking actions based on these current 
observations
13
p(St+1 | St ) = p(St+1 | S1 , …, St )


## Page 14

Example: Bioreactor Optimization
Reinforcement learning controls temperature and stirring rates
Actions: target temperatures and stirring intensities
States: sensor readings on chemical production levels
Rewards: positive for optimal production rate, negative for 
system failure
14


## Page 15

Example: Pick-and-Place Robot
Robot picks and places objects in an assembly line
Actions: motor torque commands for arm movement
States: joint angles, velocities, and object positions
Rewards: positive for a successful placement, negative for jerky 
motion
15


## Page 16

Example: Self-Driving Car Navigation
Autonomous vehicles use MDPs to optimize driving decisions
Actions: speed control, lane changes, braking decisions
States: traffic conditions, GPS location, speed limits
Rewards: positive for safe travel and reaching destination, 
negative for delays, large negative for near-collisions
16


## Page 17

Example: Recycling Robot
A mobile robot collects empty soda cans in an office environment
It has sensors to detect cans, an arm and gripper to pick them up, 
and operates on a rechargeable battery
The control system handles sensory input, navigation, and arm 
control
States: battery level (high or low) ⇒S = {high, low} 
Actions: search, wait, or return to recharge ⇒
A(high) = {search, wait}, A(low) = {search, wait, recharge}
17


## Page 18

Example: Recycling Robot

Rewards: +1 for each can collected, –3 if battery depletes

Transition probabilities: 

A period of searching that begins with a high energy level leaves the energy level 
high with probability α and reduces it to low with probability 1 – α

A period of searching undertaken when the energy level is low leaves it low with 
probability β and depletes the battery with probability 1 – β


## Page 19

Episodic and Continuing Tasks
An episodic task is one where the agent-environment interaction is 
broken into discrete episodes. Each episode starts from an initial 
state and ends in a terminal state after a finite number of steps
Examples: games, maze solving, car parking
A continuing task has no terminal state. The interaction 
continues indefinitely
Examples: industrial control, stock trading, pole balancing (it can fall, but 
ideally, it could be kept in equilibrium forever)
19


## Page 20

The Return Function
The return is the total reward from time step t onward: 
where T is a final time step 
The agent must maximize the expected return
For continuing tasks, T = ∞ 
20
1
2
3
t
t
t
t
T
G
R
R
R
R
+
+
+
=
+
+
+…+


## Page 21

Discounted Rewards
In continuing tasks, rewards can be discounted over time to avoid 
infinite sums
The discounted return is: 
γ (0 ≤ γ ≤ 1) is the discount factor
γ = 0 ⇒Short-sighted agent (only considers immediate rewards)
γ ≈ 1 ⇒Far-sighted agent (values future rewards more strongly)
21


## Page 22

Finite Returns in Continuing Tasks
Suppose that all rewards are bounded
In this case, returns are finite, and the algorithms can compare them 
to determine the optimal expected value
22
t
max
R
R
t
≤
∀
1
max
max
0
0
0
k
k
k
t
t k
k
k
k
G
R
R
R
γ
γ
γ
∞
∞
∞
+ +
=
=
=
≤
≤
=



0
1
[0,1)
1
k
k
if
γ
γ
γ
∞
=
=
∈
−

max
1
t
R
G
γ

≤
−
γ ≠1


## Page 23

Unified Notation for Episodic and 
Continuing Tasks
The unified notation treats episodic tasks as entering an absorbing 
state, which transitions only to itself and generates only zero rewards
This convention allows both episodic and continuing tasks to be 
described using a single mathematical framework
In episodic tasks, γ can be 1 (but it can be < 1, too)
In continuing tasks, T can be considered ∞ (but from the practical 
point of view, it will be in fact finite)
23


## Page 24

Policy Functions
A policy is the agent’s decision-making rule. It defines how the 
agent chooses actions based on the current state to maximize 
expected return
π(s) denotes a deterministic policy

It gives the action that the agent will take in state s
π(a∣s) denotes a stochastic policy

It gives the probability of taking action a in state s
24
( ) :
s
π
→
S
A
(
) :
[0,1]
(
)
1
a
a s
with
a s
s
π
π
∈
×
→
=
∀∈

A
S
A
S
∣
∣


## Page 25

Value Function or State-Value Function
The value function or state-value function of a state s under a 
policy π is the expected return when starting in s and following 
π thereafter 
25


## Page 26

Quality Function or Action-Value Function 
The quality function or action-value function of taking action a 
in state s under a policy π is the expected return starting from 
s, taking the action a, and thereafter following policy π
From the practical point of view, agents (RL algorithms) can choose 
actions by computing argmaxa q(s, a)
In contrast, v(s) gives no information about which action to take. 
Especially in non-deterministic environments (but not only), 
“wanting to reach” a state is not enough. The agent must still know 
which actions can lead to the next states
26


## Page 27

Bellman Equation
The expectation term is complicated to assess (assume t = 0)
The Bellman equation expresses vπ(s) as a recursive relationship 
which is easier to compute
It decomposes long-term value into local steps, which forms the basis 
for learning algorithms
In the fully deterministic case (both policy and environment)
27
0
0
1
0
0
( )
[
]
k
k
k
v
s
G
S
s
R
S
s
π
π
π
γ
∞
+
=


=
=
=
=





E
E
∣
∣
[
]
( )
(
)
(
, )
( )
( )
a
s
v
s
a s
p s
s a
r s
v
s
π
π
π
γ
′
′
′
′
=
+


∣
∣
( )
( )
( )
v
s
r s
v
s
π
π
γ
′
′
=
+


## Page 28

Bellman Equation
Similar expressions can be deduced for the action-value 
function
Return-based definition (t = 0)
General recursive form (stochastic policy and environment)
Deterministic case (policy and environment)
28
[
]
0
0
0
1
0
0
0
( , )
,
,
k
k
k
q
s a
G
S
s A
a
R
S
s A
a
π
π
π
γ
∞
+
=


=
=
=
=
=
=





∣
∣
E
E
( , )
(
, )
( )
(
)
( ,
)
s
a
q
s a
p s
s a
r s
a
s q
s a
π
π
γ
π
′
′


′
′
′
′
′
′
=
+






∣
∣
( , )
( )
( , ( ))
q
s a
r s
q
s
s
π
π
γ
π
′
′
′
=
+


## Page 29

Optimal Policies and Value Functions 
An optimal policy π*maximizes the expected return from every 
state
The optimal value function is: 
Similarly, the optimal action-value function is:
Relation between v* and q*:
29
*( )
max
( )
v s
v
s
π
π
=
*( , )
max
( , )
q s a
q
s a
π
π
=
*
*
( )
max
( , )
a
v s
q s a
=


## Page 30

Optimal Policies
All optimal policies achieve the optimal value function:
∗ = ∗()
The value of a state under an optimal policy is the highest expected 
reward achievable
All optimal policies achieve the optimal action-value function:
∗,  = ∗(, )
Solving the Belmann equations (one for each state) provides a way to 
compute the optimal policy
Still, it can be unfeasible in practice if the number of states is very 
large
Approximation methods are used in this case, e.g., using (deep) 
neural networks
30


## Page 31

Markov Decision Processes 
1. Formalization
2. Value Iteration
3. Policy Iteration
31


## Page 32

Dynamic Programming
Dynamic programming (DP) refers to a set of algorithms for 
computing optimal policies given a perfect, known MDP model
DP provides a theoretical foundation for many RL methods, which 
aim to achieve similar results with less computation and without a 
perfect model
There are two important DP algorithms for solving an MDP:
Value iteration
Policy iteration
32


## Page 33

Value Iteration Algorithm Outline
It is an algorithm for computing the optimal policy π*
The value of each state V(s) is initialized to 0
V(s) approximates v*(s)
State values are iteratively updated
The state values are used to select the optimal action for each 
state
The state with the highest value is chosen
33


## Page 34

Solving an MDP
There are n states
There is one Bellman equation for each state

⇒n equations with n unknowns: V(s)
It cannot be solved as a system of linear equations due to the max
operator
Therefore, it is solved iteratively (k is the solving iteration)
For each state s:
34


## Page 35

35


## Page 36

Synchronous vs. Asynchronous VI
Synchronous updates: compute all the new values of V(s) from all the 
old values of V(s), then update V(s) with the new values
Asynchronous updates: compute and update V(s) for each state one 
at a time 
The previous VI pseudocode is an asynchronous in-place variant
There is no temporary copy of the V array
Once a V(s) is changed, it is used in the other updates
The asynchronous version uses less memory and usually converges 
faster
The synchronous version is easier to parallelize
36


## Page 37

Example: Simple Mars Rover
The environment models a Mars rover navigating a linear grid with 
six states, indexed from 0 to 5
The rover can perform two actions at each step:
Move, which advances the rover by one cell with probability 90%, or stays in 
place with probability 10%
Speed, which advances the rover two cells forward with probability 80%, 
one cell forward with probability 10%, or stays in place with probability 10%
Rewards are assigned as follows: 
Reaching the goal state (state 4) yields +100 and terminates the episode
Overshooting the goal (state 5) yields –100 and terminates
Landing on an obstacle (state 2) incurs a penalty of –20
All other transitions cost –1
37


## Page 39

Example: Simple Mars Rover
SimpleMarsEnv.py  (Gymnasium environment)
1 SimpleMarsRoverAgent_HP.py (with hardcoded policy)
39
initial 
state
obstacle (–20)
agent
goal state 
(+100)
unsafe 
termination 
(–100)


## Page 40

Simple Mars Rover with VI
SimpleMarsEnv.py  (the same environment)
2 SimpleMarsRoverAgent_VI.py (an agent implementing the 
value iteration algorithm)
40


## Page 41

VI Application Example
V0(s) = 0 for all states (S0 to S5)
γ = 1 (no discounting; the environment is episodic, so that is ok)
41
move
speed
The value of terminal states is always 0 
because there is no reward from there onward


## Page 42

VI Application Example
42
move
speed
Action move: p1 = 0.9 to go 1 step to S4 (+100), p2 = 0.1 to remain in S3 (–1)
Action speed: p1 = 0.8 to go 2 steps to S5 (–100), p2 = 0.1 to go 1 step to S4 (+100),
p3 = 0.1 to remain in S3 (–1)
S3
S4
S5


## Page 43

Iteration 1, S0
43
Evaluate Q(s=0, a=0):
p = 0.9, R(1) = -1, V[1] = 0.00  =>  0.9 * (-1 + 1.0 * 0.00) = -0.90
p = 0.1, R(0) = -1, V[0] = 0.00  =>  0.1 * (-1 + 1.0 * 0.00) = -0.10
Q(s=0, a=0) = -1.00
Evaluate Q(s=0, a=1):
p = 0.8, R(2) = -20, V[2] = 0.00  =>  0.8 * (-20 + 1.0 * 0.00) = -16.00
p = 0.1, R(1) = -1, V[1] = 0.00  =>  0.1 * (-1 + 1.0 * 0.00) = -0.10
p = 0.1, R(0) = -1, V[0] = 0.00  =>  0.1 * (-1 + 1.0 * 0.00) = -0.10
Q(s=0, a=1) = -16.20
Update V[s=0] = max Q = -1.00


## Page 44

Iteration 1, S3
44
Evaluate Q(s=3, a=0):
p = 0.9, R(4) = 100, V[4] = 0.00  =>  0.9 * (100 + 1.0 * 0.00) = 90.00
p = 0.1, R(3) = -1, V[3] = 0.00  =>  0.1 * (-1 + 1.0 * 0.00) = -0.10
Q(s=3, a=0) = 89.90
Evaluate Q(s=3, a=1):
p = 0.8, R(5) = -100, V[5] = 0.00  =>  0.8 * (-100 + 1.0 * 0.00) = -80.00
p = 0.1, R(4) = 100, V[4] = 0.00  =>  0.1 * (100 + 1.0 * 0.00) = 10.00
p = 0.1, R(3) = -1, V[3] = 0.00  =>  0.1 * (-1 + 1.0 * 0.00) = -0.10
Q(s=3, a=1) = -70.10
Update V[s=3] = max Q = 89.90


## Page 45

Iteration 10, S3
45
Evaluate Q(s=3, a=0):
p = 0.9, R(4) = 100, V[4] = 0.00  =>  0.9 * (100 + 1.0 * 0.00) = 90.00
p = 0.1, R(3) = -1, V[3] = 99.89  =>  0.1 * (-1 + 1.0 * 99.89) = 9.89
Q(s=3, a=0) = 99.89
Evaluate Q(s=3, a=1):
p = 0.8, R(5) = -100, V[5] = 0.00  =>  0.8 * (-100 + 1.0 * 0.00) = -80.00
p = 0.1, R(4) = 100, V[4] = 0.00  =>  0.1 * (100 + 1.0 * 0.00) = 10.00
p = 0.1, R(3) = -1, V[3] = 99.89  =>  0.1 * (-1 + 1.0 * 99.89) = 9.89
Q(s=3, a=1) = -60.11
Update V[s=3] = max Q = 99.89


## Page 46

Results
46
Optimal Value Function:
State 0: V = 95.31
State 1: V = 96.42
State 2: V = 97.65
State 3: V = 99.89
State 4: V = 0.00
State 5: V = 0.00
Optimal Policy (0: move, 1: speed):
State 0: 0
State 1: 1
State 2: 1
State 3: 0
State 4: None
State 5: None


## Page 47

Markov Decision Processes 
1. Formalization
2. Value Iteration
3. Policy Iteration
47


## Page 48

Policy Iteration
The policy vector π is initialized randomly and modified only 
when necessary
The algorithm alternates between two steps: 
Policy evaluation: computes the values of all states given policy πi 
Policy improvement: computes a new policy πi+1 based on state values Vi
48


## Page 49

Policy Evaluation
The policy is arbitrarily initialized
Unlike in case of value iteration, where max is used
Here we know the action given by the policy (the policy may be bad 
at first, but it is known)
So we have a linear system of Bellman equations, one for every state
49


## Page 50

Policy Evaluation
This system of n equations with n unknowns can be solved 
algebraically or iteratively
The iterative form is more efficient for large MDPs
50


## Page 51

51


## Page 52

Policy Improvement Theorem 
The policy improvement theorem says that if we evaluate the current 
policy and then act greedily (i.e., choose actions that maximize the 
expected return under the current value estimates), the new policy 
will be at least as good
If we have a current policy π and a new policy π' such that for all 
states s
then the new policy π' is at least as good as the old one
If the inequality is strict for at least one state, then π′ is strictly better 
than π
Iterating this improvement process leads to the optimal policy
52
( ,
( ))
( )
q
s
s
v
s
π
π
π ′
≥
( )
( )
v
s
v
s
s
π
π
′
≥
∀


## Page 53

Policy Improvement Step
The goal is to improve the current policy π using the value function 
Vπ obtained from the policy evaluation step
For each state s, we identify the action that maximizes expected 
return (the “optimal so far” action):
If ∗() ≠(), the policy is updated: 
  ←∗

This two steps (policy evaluation and policy improvement) are repeated 
until the policy no longer changes, meaning that the optimal policy has been 
found
53
*
,
( )
argmax
( ,
, )
( )
a
s r
a
s
p s r s a
r
V
s
π
γ
′


′
′
=
+



∣


## Page 55

Simple Mars Rover with PI
SimpleMarsEnv.py  (the same environment)
3 SimpleMarsRoverAgent_PIi.py (an agent implementing the 
policy iteration algorithm with iterative policy evaluation)
4 SimpleMarsRoverAgent_PIs.py (policy evaluation with exact 
solving of the linear equation system)
55


## Page 56

Iteration 1
56
assume pi(all) = 0
but this can be arbitrary
Current policy:
π(0) = 0
π(1) = 0
π(2) = 0
π(3) = 0
π(4) = None
π(5) = None
Policy evaluation:
State values Vπ:
V[0] = 75.44
V[1] = 76.56
V[2] = 96.67
V[3] = 99.89
V[4] = 0.00
V[5] = 0.00
VI after I1
V[0] = -1.00
V[1] = -2.90
V[2] = 77.90
V[3] = 89.90


## Page 57

Iteration 1
57
Policy improvement:
Q(s=1, a=0):
p=0.9, R(2)=-20, V[2]=96.67  =>  69.00
p=0.1, R(1)= -1, V[1]=76.56  =>  7.56
Q = 76.56
Q(s=1, a=1):
p=0.8, R(3)= -1, V[3]=99.89  =>  79.11
p=0.1, R(2)=-20, V[2]=96.67  =>  7.67
p=0.1, R(1)= -1, V[1]=76.56  =>  7.56
Q = 94.33
Best action for state 1 = 1 (changed)
Current policy:
π(0) = 0
π(1) = 0
π(2) = 0
π(3) = 0
π(4) = None
π(5) = None


## Page 58

Iteration 3
58
Current policy:
π(0) = 0
π(1) = 1
π(2) = 1
π(3) = 0
π(4) = None
π(5) = None
State values Vπ:
V[0] = 95.31
V[1] = 96.42
V[2] = 97.65
V[3] = 99.89
V[4] = 0.00
V[5] = 0.00
Policy improvement:
Q(s=0, a=0):
p=0.9, R(1)= -1, V[1]=96.42  =>  85.88
p=0.1, R(0)= -1, V[0]=95.31  =>  9.43
Q = 95.31
Q(s=0, a=1):
p=0.8, R(2)=-20, V[2]=97.65  =>  62.12
p=0.1, R(1)= -1, V[1]=96.42  =>  9.54
p=0.1, R(0)= -1, V[0]=95.31  =>  9.43
Q = 81.10
Best action for state 0 = 0
Q(s=3, a=0):
p=0.9, R(4)=100, V[4]=0.00  =>  90.00
p=0.1, R(3)= -1, V[3]=99.89  =>  9.89
Q = 99.89
Q(s=3, a=1):
p=0.8, R(5)=-100, V[5]=0.00  =>  -80.00
p=0.1, R(4)=100, V[4]=0.00  =>  10.00
p=0.1, R(3)= -1, V[3]=99.89  =>  9.89
Q = -60.11
Best action for state 3 = 0


## Page 59

Policy Iteration vs. Value Iteration
Value iteration
Updates values directly using Bellman optimality equations
It may need more iterations to converge, but each iteration is simple
Policy iteration
Alternates between policy evaluation and policy improvement
Converges in fewer iterations but requires solving a set of equations
This can be expensive in large state spaces, but it is very fast and accurate in 
small ones
Value iteration scales better for large MDPs
Policy iteration is better for small MDPs
59


## Page 60

Generalized Policy Iteration
In GPI, the evaluation and improvement steps 
are interleaved
This is a general framework, not a specific 
algorithm
The granularity of the updates is not specified
The policy is continually improved with respect 
to the value function, while the value function 
is continually updated toward the value of the 
current policy
60


## Page 61

Generalized Policy Iteration
This can involve a single or a few evaluation operations
And a single or a few improvement operations
GPI allows approximate or partial updates, which reduce 
computation
It allows to trade off accuracy and speed
GPI is a “conceptual umbrella” that unifies many RL algorithms
61


## Page 62

Generalized Policy Iteration

The evaluation and improvement are both competing and cooperating

When the policy is improved based on the current value estimates, those value 
estimates become outdated

When the value function is updated to match the current policy, it may reveal that 
some actions are better than what the policy currently chooses

But the continuous updates of π and V lead in the long run to a single joint solution: 
the optimal value function and an optimal policy 
62


## Page 63

Conclusions
MDPs formalize sequential decisions with states, actions, 
transitions, rewards, and discounting
The Markov property means the next state depends only on the 
current state, not the full history
Value iteration finds optimal policies through iterative Bellman 
updates; it is scalable to large MDPs
Policy iteration alternates between evaluation and 
improvement and needs fewer iterations but costs more per 
step
63


## Page 64

Main Reference
Sutton, R.S. and Barto, A.G. (2018). Reinforcement Learning: An 
Introduction. 2nd edition. MIT Press, Cambridge, MA. 
http://incompleteideas.net/book/the-book-2nd.html
64
