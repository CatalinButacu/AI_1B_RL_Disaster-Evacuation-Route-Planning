# RL03_MC

**Source:** c:\Users\catalin.butacu\Downloads\RL\resourses\lecture\RL03_MC.pdf

**Pages:** 57

---


## Page 1

Reinforcement Learning 
3. Monte Carlo Methods 
 
Florin Leon 
 
“Gheorghe Asachi” Technical University of Iași, Romania 
Faculty of Automatic Control and Computer Engineering 
 
https://florinleon.byethost24.com/lect_rl.html 
 
2025 
 


## Page 2

Monte Carlo Methods 
1. Introduction 
2. Monte Carlo Prediction (Passive RL) 
 
2.1. First-Visit and Every-Visit Policy Evaluation 
3. Monte Carlo Control (Active RL) 
 
3.1. MC Exploring Starts 
 
3.2. On-Policy First-Visit MC Control for Soft Policies 
4. Off-Policy Methods 
 
4.1. Off-Policy Prediction via Importance Sampling 
 
4.2 Off-Policy MC Control 
2 


## Page 3

Monte Carlo Methods 
1. Introduction 
2. Monte Carlo Prediction (Passive RL) 
 
2.1. First-Visit and Every-Visit Policy Evaluation 
3. Monte Carlo Control (Active RL) 
 
3.1. MC Exploring Starts 
 
3.2. On-Policy First-Visit MC Control for Soft Policies 
4. Off-Policy Methods 
 
4.1. Off-Policy Prediction via Importance Sampling 
 
4.2 Off-Policy MC Control 
3 


## Page 4

Reinforcement Learning 
Markov Decision Process 
State space, action space 
The transition model is 
known 
The reward function is 
known 
Computes an optimal 
policy  
Reinforcement Learning 
Is based on MDPs, but: 
The transition model is 
unknown 
The reward function is 
unknown 
Learns an optimal policy 
4 


## Page 5

The Learning Model 

Monte Carlo (MC) methods estimate value functions and determine 
policies by using sample sequences of states, actions, and rewards 

MC is model-free: direct experience eliminates the need for knowledge of 
environment dynamics 

Experience demands only a model that generates sample transitions 

MC methods sample and average returns for each state-action pair 

Return values for a given state-action pair depend on subsequent actions 
in the same episode 

MC methods solve RL tasks by averaging complete returns from 
episodes 

Each episode terminates and value estimates update only after an episode 
finishes 
5 


## Page 6

Formalization 
6 


## Page 7

Types of Algorithms 
In SB’s terminology: 
Prediction = policy evaluation (AIMA: passive reinforcement learning) 
Control = learning an optimal policy (AIMA: active reinforcement learning) 
MC methods estimate the value function vπ(s) by averaging observed 
returns following visits to state s under policy π 
MC methods are better than other RL methods when: 
The environment has high variability (stochastic transitions or rewards) 
The rewards are sparse and delayed 
Episodes are short and fully accessible 
 

SB: Sutton, R.S. and Barto, A.G. (2018). Reinforcement Learning: An Introduction 

AIMA: Russell, S. J. and Norvig, P. (2022). Artificial Intelligence: A Modern Approach 
7 


## Page 8

Monte Carlo Methods 
1. Introduction 
2. Monte Carlo Prediction (Passive RL) 
 
2.1. First-Visit and Every-Visit Policy Evaluation 
3. Monte Carlo Control (Active RL) 
 
3.1. MC Exploring Starts 
 
3.2. On-Policy First-Visit MC Control for Soft Policies 
4. Off-Policy Methods 
 
4.1. Off-Policy Prediction via Importance Sampling 
 
4.2 Off-Policy MC Control 
8 


## Page 9

Monte Carlo Prediction 

The first-visit MC method averages returns only from the first 
occurrence of s per episode, while every-visit MC uses returns from all 
occurrences of s 

First-visit MC provides independent, identically distributed estimates of 
vπ(s) with finite variance, and thus convergence by the law of large 
numbers 

First-visit MC converges more slowly in practice since it uses fewer 
samples per episode, but each estimate is unbiased (converges to the 
correct value on average) and based on independent returns 

Every-visit MC typically converges faster because it uses all visits to a 
state, though early estimates may be slightly biased due to correlated 
returns within episodes 
9 


## Page 10

First-Visit MC Policy Evaluation 
The policy is known (given) 
We are interested in policy evaluation, not learning 
A complete episode is run to the end 
 
 
 
 
 
 
Several episodes are considered (the more, the better) 
10 


## Page 11

Every-Visit MC Policy Evaluation 
11 


## Page 12

12 


## Page 13

Example: Mars 2 Environment 

Linear grid with 21 states (0 to 20) 

Start: state 0; Goal: state 19 (+500, terminal); Unsafe: state 20 (−100, 
terminal) 

Actions:  

move (states 0–9: 90% → +1 cell, states 10–18: 70% → +1 cell) 

speed (states 0–9: 80% → +2, 10% → +1, 10% → stay;  
states 10–18: 60% → +2, 20% → +1, 20% → stay) 

Obstacles: states 4, 8 → −20, states 13, 14 → −50 

Traps: states 5, 10, 15 (20% termination, −100) 

Default reward: −1 per step 
 
13 


## Page 14

14 


## Page 15

Mars 2 Environment 
SimpleMarsEnv2.py 
1 SimpleMarsRoverAgent_HP.py (hardcoded policy, for 
evaluation) 
2 FirstVisit.py (including both forward and backward 
implementations of the algorithm) 
15 


## Page 16

Value Function (10k Episodes) 
16 


## Page 17

MC Estimation of Action Values 
Without a model, estimating action values q*(s, a) is essential because 
state values alone are insufficient for determining which actions to 
select in each state 
Monte Carlo methods estimate qπ(s, a), the expected return starting 
from state s with action a and then following policy π, by averaging 
sampled returns 
The first-visit MC method averages returns following the first time  
(s, a) is encountered in each episode 
The every-visit method averages returns over all such occurrences 
Both first-visit and every-visit MC methods converge to true expected 
values as the number of visits to each state-action pair goes to infinity 
17 


## Page 18

Monte Carlo Methods 
1. Introduction 
2. Monte Carlo Prediction (Passive RL) 
 
2.1. First-Visit and Every-Visit Policy Evaluation 
3. Monte Carlo Control (Active RL) 
 
3.1. MC Exploring Starts 
 
3.2. On-Policy First-Visit MC Control for Soft Policies 
4. Off-Policy Methods 
 
4.1. Off-Policy Prediction via Importance Sampling 
 
4.2 Off-Policy MC Control 
18 


## Page 19

Monte Carlo Control 

Monte Carlo control approximates optimal policies 
through generalized policy iteration (GPI) 

It alternates between evaluating the current policy  
and improving it using greedy action-value  
selection 
 
 
 

In MC policy iteration, we start with an arbitrary policy π₀ and alternate 
between complete policy evaluation and greedy policy improvement steps 
until convergence to π* 
 
19 


## Page 20

Policy Improvement Theorem 
The greedy policy π(s) = argmax a q(s, a) uses the current action-
value estimates to choose actions that appear most valuable – this is 
used for policy improvement 
The policy improvement theorem ensures πk+1 is always as good as or 
better than πk , thus repeated improvement steps guarantee 
convergence to the optimal policy 
Monte Carlo methods can learn optimal policies from experience 
alone, without knowledge of the transition or reward models 
20 


## Page 21

Exploring Starts 

Deterministic policies fail to estimate values for all actions in each state 

They never explore alternatives and this makes comparisons between actions 
impossible 

To ensure all state-action pairs are evaluated, continual exploration is 
needed 

This is analogous to the exploration issue in the k-armed bandit problem 

Exploring starts assumes the episodes start in a random state-action 
pair, and that every pair has a nonzero probability of being selected as 
the start 

This guarantees that all state-action pairs will be visited an infinite 
number of times in the limit of an infinite number of episodes 
21 


## Page 22

22 


## Page 23

Mars 2 Environment 
SimpleMarsEnv2.py (the same environment) 
3 ExploringStarts.py (including both forward and backward 
implementations of the algorithm) 
23 


## Page 24

Action-Value Function (10k Episodes) 
Max Q can be different (sometimes higher) from the values of First Visit because FV used a 
fixed suboptimal policy, while ES is learning a (near-)optimal policy. Lower Q(move) in 4, 9, 14 
24 


## Page 25

Monte Carlo Methods 
1. Introduction 
2. Monte Carlo Prediction (Passive RL) 
 
2.1. First-Visit and Every-Visit Policy Evaluation 
3. Monte Carlo Control (Active RL) 
 
3.1. MC Exploring Starts 
 
3.2. On-Policy First-Visit MC Control for Soft Policies 
4. Off-Policy Methods 
 
4.1. Off-Policy Prediction via Importance Sampling 
 
4.2 Off-Policy MC Control 
25 


## Page 26

On-Policy vs. Off-Policy Methods. 
Soft Policies 

On-policy methods attempt to evaluate or improve the policy that is used 
to make decisions 

Off-policy methods evaluate or improve a policy different from that used 
to generate the data 
 

Exploring starts are often unrealistic in real environments; instead, 
using stochastic policies with nonzero probability for all actions ensures 
all pairs are eventually sampled 

To avoid relying on exploring starts, the on-policy MC control method 
uses a soft policy: π(a|s) > 0, ∀ s, a 

All actions have nonzero probability and this allows continued 
exploration 
26 


## Page 27

є-Greedy Policy 
An є-greedy policy selects the action with highest estimated value 
most of the time (with probability 1 – є) and chooses randomly 
among all actions with probability є to ensure exploration 
Each nongreedy action has the probability of selection є / |A(s)| 
The best (greedy) action has the probability 1 – є + є / |A(s)| 
Exercise: prove that the sum of probabilities is 1 
The є-greedy policy is an example of an є-soft policy, defined as a 
policy for which π(a|s) ≥ є / |A(s)|, є > 0 
27 


## Page 28

MC Control without Exploring Starts 
In many cases, the idea of exploring starts is impractical, but without 
it, we cannot improve the policy by making it greedy with respect to 
the current value function 
That would prevent further exploration of nongreedy actions 
GPI does not require that the policy be taken all the way to a greedy 
policy, only that it be moved toward a greedy policy 
In this on-policy method we will move it only to an є-greedy 
policy 
For any є-soft policy π, any є-greedy policy with respect to qπ is 
guaranteed to be better than or equal to π 
28 


## Page 29

29 


## Page 30

Program 
4 EpsSoftControl.py 
30 


## Page 31

Monte Carlo Methods 
1. Introduction 
2. Monte Carlo Prediction (Passive RL) 
 
2.1. First-Visit and Every-Visit Policy Evaluation 
3. Monte Carlo Control (Active RL) 
 
3.1. MC Exploring Starts 
 
3.2. On-Policy First-Visit MC Control for Soft Policies 
4. Off-Policy Methods 
 
4.1. Off-Policy Prediction via Importance Sampling 
 
4.2 Off-Policy MC Control 
31 


## Page 32

On-Policy vs. Off-Policy 
The agent needs to learn optimal action values, but needs to behave 
non-optimally to explore all actions (to find the optimal ones) 
The on-policy approach is actually a compromise: it learns action 
values not for the optimal policy, but for a near-optimal policy that 
still explores 
Another approach is to use two policies, one that is learned and 
becomes the optimal policy, and one that is more exploratory and is 
used to generate behavior 
32 


## Page 33

Off-Policy Prediction 
The target policy π represents the policy being learned, while the 
behavior policy b generates data 
Off-policy learning enables learning from different data sources, 
including human demonstrations or non-learning controllers 
Off-policy learning generalizes on-policy learning 
It includes the special case where the target and behavior policies are 
identical 
Off-policy prediction: both policies (π and b) are considered known 
and fixed 
b can change between episodes, or even within episodes 
33 


## Page 34

Importance Sampling 

To ensure coverage, every action selected by the target policy should also 
appear in the behavior policy: π(a|s) > 0 ⇒ b(a|s) > 0 

The behavior policy b must be stochastic in states where it differs from the 
target policy to ensure all necessary actions are sampled, but π can be 
deterministic 

Importance sampling adjusts returns from the behavior policy to estimate 
values under the target policy by weighting returns according to the 
probability ratio of both policies 

Given a starting state St , the probability of the subsequent state-action 
trajectory At, St+1, At+1, . . . , ST  is: 
 
34 


## Page 35

Importance Sampling 
The importance sampling ratio is defined as: 
 
 
 
Expected returns under the behavior policy have the wrong 
expectation, so importance sampling corrects this by transforming 
returns to match the expected values of the target policy 
This ratio tells us how much more (or less) likely the trajectory is 
under π compared to b 
 
35 


## Page 36

Justification 
We want to estimate the expected returns (values) under π, but we 
have the returns Gt under b  
These returns have the wrong expectation 𝔼[Gt | St = s] = vb(s) and 
cannot be averaged to obtain vπ 
The ratio ρt:T – 1 transforms the returns to have the correct expected 
value:  
 
 
 
36 
we assume that time steps 
continue between episodes 
(if an episode ends at time 
step t, the next episode 
starts at t+1) 


## Page 37

Ordinary Importance Sampling 
Unbiased: the estimate converges to the correct expected value 
High variance: can suffer from very large fluctuations, especially if  
ρ is large (e.g., long episodes, rare trajectories) 
37 


## Page 38

Weighted Importance Sampling 
WIS also introduces bias, but the bias converges asymptotically to 
zero as more episodes are observed 
But it has much lower variance 
It is preferred in practice due to better stability and faster 
convergence 
38 


## Page 39

Incremental Implementation 
39 


## Page 40

Example: Non-Incremental 
40 


## Page 41

Example: Incremental 
41 


## Page 42

42 


## Page 43

Example: Simple Mars Rover 
SimpleMarsEnv.py 
Actions: move (1 step), speed (2 steps) 
 
43 
initial  
state 
obstacle (–20) 
agent 
goal state  
(+100) 
unsafe  
termination  
(–100) 


## Page 44

Example 
44 
Target policy π: 
In state 0: prefer move (action 0), π(0|0) = 0.9, π(1|0) = 0.1 
In state 1: prefer speed (action 1), π(0|1) = 0.1, π(1|1) = 0.9 
In state 2: prefer speed (action 1), π(0|2) = 0.1, π(1|2) = 0.9 
In state 3: prefer move (action 0), π(0|3) = 0.9, π(1|3) = 0.1 
 
Behavior policy b: 
For all states s: b(0|s) = 0.5, b(1|s) = 0.5 
 
Objective: 
Estimate Q(s,a) using WIS 
   0           1            2         3           4          5 


## Page 45

Episode 1 
45 
Start at state 0 
Action: 0 (move), goes to state 1 
Reward: -1 
Action: 1 (speed), goes to state 3 
Reward: -1 
Action: 0 (move), goes to state 4 
Reward: +100 (goal) 
Episode: [(0,0,-1), (1,1,-1), (3,0,+100)] 
Returns: G2 = 100, G1 = -1 + 100 = 99, G0 = -1 + 99 = 98 
Importance Sampling Ratio (ρ): 
ρ = π(0|0)/b(0|0) × π(1|1)/b(1|1) × π(0|3)/b(0|3) 
  = (0.9/0.5) × (0.9/0.5) × (0.9/0.5) 
  = 1.8 × 1.8 × 1.8 ≈ 5.832 
   0           1            2         3           4          5 


## Page 46

Episode 2 
46 
Start at state 0 
Action: 1 (speed), goes to state 2 
Reward: -20 (obstacle) 
Action: 0 (move), goes to state 3 
Reward: -1 
Action: 1 (speed), goes to state 5 
Reward: -100 (unsafe, ends) 
Episode: [(0,1,-20), (2,0,-1), (3,1,-100)] 
Returns: G2 = -100, G1 = -1 - 100 = -101, G0 = -20 - 101 = -121 
Importance Sampling Ratio (ρ): 
ρ = π(1|0)/b(1|0) × π(0|2)/b(0|2) × π(1|3)/b(1|3) 
  = (0.1/0.5) × (0.1/0.5) × (0.1/0.5) 
  = 0.2 × 0.2 × 0.2 = 0.008 
   0           1            2         3           4          5 


## Page 47

Episode 3 
47 
Start at state 0 
Action: 0 (move), goes to state 1 
Reward: -1 
Action: 0 (move), goes to state 2 
Reward: -20 (obstacle) 
Action: 1 (speed), goes to state 4 
Reward: +100 
Episode: [(0,0,-1), (1,0,-20), (2,1,+100)] 
Returns: G2 = 100, G1 = -20 + 100 = 80, G0 = -1 + 80 = 79 
Importance Sampling Ratio (ρ): 
ρ = π(0|0)/b(0|0) × π(0|1)/b(0|1) × π(1|2)/b(1|2) 
  = (0.9/0.5) × (0.1/0.5) × (0.9/0.5) 
  = 1.8 × 0.2 × 1.8 ≈ 0.648 
   0           1            2         3           4          5 


## Page 48

WIS for Q 
Q(s, a) is computed for all state-action pairs using: 
 
 
 
 
We will only compute estimates for the first pair of (state, action) of 
each episode – assume we are doing First-Visit MC for Q(s, a) 
 
48 
( , )
( , )
i
i
i
i
i
G s a
Q s a






## Page 49

Action-Value Function 
Q(0,0) 
Appears in Episode 1: ρ = 5.832, G = 98 
Appears in Episode 3: ρ = 0.648, G = 79 
 
 
 
 
Q(0,1) 
Appears in Episode 2: ρ = 0.008, G = -121 
 
49 
5.832 98
0.648 79
571.536
 ×
51.192
622.728
(0,0)
96.07
5.832
0.648
6.4
 
8
6
×
 
 
.48
Q







 
0.008
121
(0,1)
121
0.
×
008
Q





## Page 50

Program 
5 OffPolicyPrediction.py 
50 
Q(0,0) = 78.000 
Q(0,1) = 74.917 
 
Q(1,0) = 73.870 
Q(1,1) = 79.677 
 
Q(2,0) = 80.120 
Q(2,1) = 95.472 
 
Q(3,0) = 98.253 
Q(3,1) = -61.574 
 
Q(4/5, 0/1) = 0.000 
   0           1            2         3           4          5 


## Page 51

Monte Carlo Methods 
1. Introduction 
2. Monte Carlo Prediction (Passive RL) 
 
2.1. First-Visit and Every-Visit Policy Evaluation 
3. Monte Carlo Control (Active RL) 
 
3.1. MC Exploring Starts 
 
3.2. On-Policy First-Visit MC Control for Soft Policies 
4. Off-Policy Methods 
 
4.1. Off-Policy Prediction via Importance Sampling 
 
4.2 Off-Policy Control 
51 


## Page 52

Off-Policy MC Control 
Off-policy MC control methods rely on WIS to adjust returns 
generated by the behavior policy 
Generalized policy iteration (GPI) guides learning to ensure that the 
estimated target policy converges toward optimality even when 
trained using a different behavior policy 
The target policy can be deterministic 
The behavior policy must be є-soft to ensure convergence  

Nonzero probability for all actions 
52 


## Page 53

53 


## Page 54

Limitations 
Learning primarily occurs from the latter part of episodes, when 
remaining actions align with the greedy policy 
If nongreedy actions dominate early in episodes, value estimation for 
earlier states slows down significantly 
This may potentially hinder learning efficiency in long episodes 
54 


## Page 55

Program 
6 OffPolicyControl.py 
RaceTrackEnv.py 
55 


## Page 56

Conclusions 

Monte Carlo methods learn from complete episodes without the need for a 
model 

They work well in uncertain environments with sparse rewards and 
complex dynamics 

Monte Carlo methods estimate value functions by averaging returns from 
either the first visit or all visits to each state in an episode 

Control methods improve policies through exploration and incremental 
learning based on sampled returns 

On-policy methods learn from episodes generated by the same policy that is 
being improved 

Off-policy methods learn about a target policy using episodes generated by a 
different behavior policy through importance sampling 
56 


## Page 57

Main Reference 
Sutton, R.S. and Barto, A.G. (2018). Reinforcement Learning: An 
Introduction. 2nd edition. MIT Press, Cambridge, MA. 
http://incompleteideas.net/book/the-book-2nd.html 
57 
