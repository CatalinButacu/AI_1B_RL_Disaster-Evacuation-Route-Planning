# RL04_TD

**Source:** c:\Users\catalin.butacu\Downloads\RL\resourses\lecture\RL04_TD.pdf

**Pages:** 61

---


## Page 1

Reinforcement Learning 
4. Temporal-Difference Learning 
 
Florin Leon 
 
“Gheorghe Asachi” Technical University of Iași, Romania 
Faculty of Automatic Control and Computer Engineering 
 
https://florinleon.byethost24.com/lect_rl.html 
 
2025 
 


## Page 2

Temporal-Difference Learning 
1. Introduction. Brief History 
2. Temporal-Difference Prediction 
3. Temporal-Difference Control  
 
3.1. Sarsa 
 
3.2. Q-Learning 
 
3.3. Expected Sarsa 
4. n-step Bootstrapping 
 
4.1. n-step Sarsa 
 
4.2 n-step Q-Learning 
2 


## Page 3

Temporal-Difference Learning 
1. Introduction. Brief History 
2. Temporal-Difference Prediction 
3. Temporal-Difference Control  
 
3.1. Sarsa 
 
3.2. Q-Learning 
 
3.3. Expected Sarsa 
4. n-step Bootstrapping 
 
4.1. n-step Sarsa 
 
4.2 n-step Q-Learning 
3 


## Page 4

The Law of Effect 
Edward Thorndike’s experiments with cats in puzzle boxes led to the 
law of effect: behaviors followed by rewards become more likely in 
the future 
Food  ← 
4 


## Page 5

The First RL System 
In 1951, Marvin Minsky and Dean Edmonds built the Stochastic 
Neural-Analog Reinforcement Calculator (SNARC), a neural network 
with 40 neurons and 400 connections that used reward signals to 
learn how to navigate simulated mazes 
The SNARC worked in simple mazes but failed in complex ones 
It showed that direct outcome reinforcement could not scale 
Delayed rewards create a difficult credit assignment problem because 
no clear link exists between actions and outcomes 
E.g., if only a final reward exists (solved or failed), which of the 50 
successive actions actually helped the agent to solve the task? 
5 


## Page 6

Temporal Difference Learning 
In 1984, Richard Sutton introduced a new solution to the temporal 
credit assignment problem in his PhD dissertation, advised by 
Andrew Barto 
More advanced animals, such as mammals, learn by updating 
behavior based on moment-to-moment changes in expected future 
rewards 
Sutton proposed using predicted rewards to reinforce actions instead 
of waiting for actual outcomes at the end of a task 
Temporal-difference (TD) learning strengthens or weakens actions 
during a task based on ongoing changes in expected future success 
The system is taught from its own predictions 
6 


## Page 7

TD-Gammon 
Gerald Tesauro at IBM Research was working on Neurogammon 
A backgammon program trained on human expert moves 
1989: Neurogammon was beating other backgammon programs,  
but not people 
In the 1990s, he proposed TD-Gammon, using TD RL with a neural 
network 
The final version approached the level of world-class human players 
Sutton proved that TD works in theory 
Tesauro was one of the first to show that it works in practice 
7 


## Page 8

TD-Gammon 
8 


## Page 9

Temporal-Difference Learning 
1. Introduction. Brief History 
2. Temporal-Difference Prediction 
3. Temporal-Difference Control  
 
3.1. Sarsa 
 
3.2. Q-Learning 
 
3.3. Expected Sarsa 
4. n-step Bootstrapping 
 
4.1. n-step Sarsa 
 
4.2 n-step Q-Learning 
9 


## Page 10

Monte Carlo 
γ = 1 
Action: move 1 step,  
deterministically 
Eventually: V(A) → 9, V(B) → 10, V(C) = 0 
10 


## Page 11

Temporal Difference 
The target for the MC update is Gt , whereas the target for the 
TD update is Rt+1 + γ · V(St+1), an estimation of Gt 
 
11 


## Page 12

12 


## Page 13

TD Learning 

Temporal-difference learning combines Monte Carlo sampling with the 
bootstrapping of dynamic programming for estimating value functions 

Like Monte Carlo methods, TD learns from experience without a model of 
environment 

Like DP, TD updates value estimates using other learned estimates, without 
waiting for episode termination 

One-step TD (TD(0)) updates values at each step using the immediate 
reward and the estimated value of the next state 

TD methods enable learning from partial sequences with intermediate 
updates 

Both TD and MC both use sampling, but TD integrates prediction steps into 
ongoing experience immediately 
13 


## Page 14

TD Error 
The TD error quantifies the difference between the current value 
estimate and the improved, bootstrapped target: 
 
 
This TD error becomes available at time t+1, and allows online 
correction of V(St) 
If the value function does not change during an episode, the MC error 
can be expressed as a sum of TD errors: 
​ 
 
 
In TD(0), the update changes V during the episode, so this identity 
only holds approximately 
 
1
1
(
)
(
)
t
t
t
t
R
V S
V S







1
(
)
T
k t
t
t
k
k t
G
V S







14 


## Page 15

Example 1: Driving Home 

Leave office at 6 PM 

Begin with an initial estimate of 30-minute commute based on time, day, 
and weather 

Adjust estimate to 40 minutes after noticing rain and slow traffic 

Revise estimate to 35 minutes after quick highway segment 

Encounter delay behind slow truck on narrow road 

Final arrival at home occurs at 6:43 


## Page 16

Example 1: Driving Home 
MC                                                               TD 

MC waits until the episode ends to compute the total return and update 
earlier predictions 

TD immediately updates each estimate toward the one that follows it 


## Page 17

Advantages of TD Methods 
TD methods generally converge faster than MC methods, although 
this has not been formally proven 
They converge on the value function with a sufficiently small step 
size parameter, or with a decreasing step size 
They are very useful for continuing tasks that cannot be broken 
down in episodes as required by MC methods 
E.g.: pole balancing, power grid balancing, elevator dispatching, stock 
trading 


## Page 18

Example 2: Random Walk 
18 


## Page 19

Program 
1 RandomWalk.py 
Estimated state values (α = 0.1) 
19 
RMS error averaged over states 


## Page 20

Questions for Discussion 
In the left graph, the first episode results in a change in only V(A). 
What happened on the first episode? Why was only the estimate for 
this one state changed? By exactly how much was it changed? 
 
In the right graph of the random walk example, the RMS error of the 
TD method seems to go down and then up again, particularly at high 
α’s. What could have caused this? Do you think this always occurs, or 
might it be a function of how the approximate value function was 
initialized? 
 
20 


## Page 21

Example 3: You Are the Predictor 
Place yourself in the role of the predictor of returns for an unknown 
Markov reward process 
Suppose you observe the following 8 episodes: 
 
 
 
The first episode starts in state A, transitions to B with a reward of 0, 
and then terminates from B with a reward of 0 
The other 7 episodes start from B and terminate immediately 
What would you say are V(A) and V(B)? 
21 


## Page 22

Example 3: You Are the Predictor 

V(B) = (6 · 1 + 2 · 0) / 8 = 3/4 

V(A) = 3/4 (TD) 

100% of the times the process was in 
state A it traversed immediately to B 
with a reward of 0; because V(B) = 3/4, 
V(A) = 3/4 as well 

V(A) = 0 (MC) 

We have seen A once and the return 
that followed it was 0 ⇒ V(A) = 0 
22 

MC tends to minimize the training MSE of returns 

TD tends to compute the value function of the MLE model of the environment 
(as if the empirical model from the observed data is the true environment) 


## Page 23

Temporal-Difference Learning 
1. Introduction. Brief History 
2. Temporal-Difference Prediction 
3. Temporal-Difference Control  
 
3.1. Sarsa 
 
3.2. Q-Learning 
 
3.3. Expected Sarsa 
4. n-step Bootstrapping 
 
4.1. n-step Sarsa 
 
4.2 n-step Q-Learning 
23 


## Page 24

On-Policy Learning of Action-Values 
An episode: 
 
 
TD update for action-values: 
 
 
This update is done after every transition from a nonterminal state St 
If St+1 is terminal, Q(St+1, At+1) = 0  
This rule uses every element of the quintuple that makes up a 
transition from one state-action pair to the next 
Sarsa uses the same policy to generate actions and make policy 
updates; thus, it is an on-policy method 
24 


## Page 25

25 


## Page 26

Q-Learning 
Q-learning is an off-policy control algorithm  
 
 
 
Unlike Sarsa, Q-learning uses the maximum over next actions as its 
target, regardless of which action was actually taken 
The max over a defines the target as if the next action were chosen 
greedily, while the data can come from any behavior policy that 
explores 
Target policy: the greedy policy πtarget(s) = argmaxa Q(s, a) 
Behavior policy: whatever selects actions, usually є-greedy w.r.t. Q 
26 


## Page 27

27 


## Page 28

Question for Discussion 
Suppose action selection is greedy. Is Q-learning then exactly 
the same algorithm as Sarsa? Will they make exactly the same 
action selections and weight updates? 
28 


## Page 29

Expected Sarsa 
Expected Sarsa updates Q using the expected value over the possible 
next actions rather than a sampled action 
 
 
 
 
The target is the policy-weighted average of action-values at the next 
state 
This decreases the variance from stochastic action selection 
Compared to Sarsa, Expected Sarsa reduces learning noise, especially 
when action selection is highly stochastic 
Compared to Q-learning, Expected Sarsa uses the same experience 
but applies a softer, expectation-based target rather than max 
29 


## Page 30

Example 1: Windy Gridworld 
A grid with start and goal, and 4 actions: up, down, left, right 
R = –1 for each step 
There is a crosswind running upward through the middle of the grid 
The strength of the wind is given below each column, in number of 
cells shifted upward 
30 


## Page 31

Program 
2 WindyGridWorld.py 
The number of episodes completed as learning 
progresses. Q-Learning learns the fastest 
31 


## Page 32

Example 2: Sarsa vs. Q-Learning in  
Risky Settings 
Environment 
 
Chain: A → B → C → D 
Rewards: r(B) = −1, r(C) = +10, r(D) = −10 
є = 0.1, γ = 1. C and D are terminal. 
 
Actions at A and B: 
    • Safe (S): move exactly one cell. 
    • Risky (R): jump 2 cells with probability 0.7, or stay with probability 0.3 (reward −1, 
       stay in place). 
32 


## Page 33

Q-Learning 
Q(B, S) ← Q(B, S) + 0.1 [10 − Q(B, S)] 
Success to D: Q(B, R) ← Q(B, R) + 0.1 [−10 − Q(B, R)] 
Failure, stay in B: Q(B, R) ← Q(B, R) + 0.1 [−1 + max(Q(B, S), Q(B, R)) − Q(B, R)] 
  
Q(A, S) ← Q(A, S) + 0.1 [−1 + max(Q(B, S), Q(B, R)) − Q(A, S)] 
Success to C: Q(A, R) ← Q(A, R) + 0.1 [10 − Q(A, R)] 
Failure, stay in A: Q(A, R) ← Q(A, R) + 0.1 [−1 + max(Q(A, S), Q(A, R)) − Q(A, R)] 
  
Assume that during training, we have these values:  
Q(A, S) = 9.0, Q(A, R) = 9.5, Q(B, S) = 10.0, Q(B, R) = −5 
  
State A, action S: 
Q(A, S) ← Q(A, S) + 0.1 [−1 + max(Q(B, S), Q(B, R)) − Q(A, S)] 
Q(A, S) ← 9.0 + 0.1 [−1 + max(10.0, −5) − 9.0] = 9.0 
  
State B, action R, success to D: 
Q(B, R) ← Q(B, R) + 0.1 [−10 − Q(B, R)] 
Q(B, R) ← −5 + 0.1 [−10 − (−5)] = −5.5 
  
Q values after this episode: 
Q(A, S) = 9.0, Q(A, R) = 9.5, Q(B, S) = 10.0, Q(B, R) = −5.5 
  
Final Q values: 
Q(A, S) = 9.0, Q(A, R) = 9.6, Q(B, S) = 10.0, Q(B, R) = −4.5 
33 


## Page 34

Sarsa 
34 
Q(B, S) ← Q(B, S) + 0.1 [10 − Q(B, S)] 
Success to D: Q(B, R) ← Q(B, R) + 0.1 [−10 − Q(B, R)] 
Failure, stay in B, next action a': Q(B, R) ← Q(B, R) + 0.1 [−1 + Q(B, a') − Q(B, R)] 
  
Q(A, S), next action a': Q(A, S) ← Q(A, S) + 0.1 [−1 + Q(B, a') − Q(A, S)] 
Success to C: Q(A, R) ← Q(A, R) + 0.1 [10 − Q(A, R)] 
Failure, stay in A, next action a': Q(A, R) ← Q(A, R) + 0.1 [−1 + Q(A, a') − Q(A, R)] 
  
Assume that during training, we have these values:  
Q(A, S) = 9.0, Q(A, R) = 9.5, Q(B, S) = 10.0, Q(B, R) = −5 
  
State A, action S (next action a' = R): 
Q(A, S) ← Q(A, S) + 0.1 [−1 + Q(B, a') − Q(A, S)] 
Q(A, S) ← 9.0 + 0.1 [−1 + Q(B, R) − 9.0] = 7.5 
  
State B, action R, success to D: 
Q(B, R) ← Q(B, R) + 0.1 [−10 − Q(B, R)] 
Q(B, R) ← −5 + 0.1 [−10 − (−5)] = −5.5 
  
Q values after this episode: 
Q(A, S) = 7.5, Q(A, R) = 9.5, Q(B, S) = 10.0, Q(B, R) = −5.5 
  
A seemingly safe action leads to a state where exploration sometimes can bring a bad outcome. 
  
Final Q values: 
Q(A, S) = 8.4, Q(A, R) = 9.5, Q(B, S) = 10.0, Q(B, R) = −3.9 


## Page 35

Example 3: Cliff Walking 
35 
Actions: up, down, right, and left 
Reward is −1 on all transitions except those into the region marked “The Cliff”  
Stepping into this region incurs a reward of −100 and sends the agent 
instantly back to the start 


## Page 36

Programs 
3 CliffWalk.py 
3 CliffWalk_scaled.py (from episode 100, moving average 100) 
36 


## Page 37

Interim and asymptotic performance as a function of α 
All algorithms use an є-greedy policy with є = 0.1 
Asymptotic performance is an average over 100,000 episodes  
Interim performance is an average over the first 100 episodes 
37 


## Page 38

Discussion 
Q-learning learns the optimal policy along the cliff edge, minimizing 
steps and maximizing reward 
The greedy policy learned by Q-learning results in frequent high 
penalties due to exploratory steps falling into the cliff 
Sarsa learns a safer policy that sacrifices optimality in return for 
lower variance and better performance under exploration 
Despite converging to q∗, Q-learning may perform worse during 
training if exploration causes frequent catastrophic failures 
This example highlights the difference between learning an optimal 
policy and achieving optimal behavior during learning 
38 


## Page 39

Discussion 
Expected Sarsa achieves better asymptotic and interim performance 
than both Sarsa and Q-learning 
With deterministic transitions and stochastic policies, Expected Sarsa 
can safely use large α (even 1), unlike standard Sarsa 
Its average reward per episode is closer to optimal, even in early 
training, due to reduced variance in updates 
Expected Sarsa combines reliable convergence with improved 
learning efficiency 
39 


## Page 40

Temporal-Difference Learning 
1. Introduction. Brief History 
2. Temporal-Difference Prediction 
3. Temporal-Difference Control  
 
3.1. Sarsa 
 
3.2. Q-Learning 
 
3.3. Expected Sarsa 
4. n-step Bootstrapping 
 
4.1. n-step Sarsa 
 
4.2 n-step Q-Learning 
40 


## Page 41

Motivation for n-step Methods 
n-step methods unify one-step TD (bootstrapping) and MC methods 
(sampling) 
MC uses complete returns, while TD(0) uses bootstrapped one-step 
returns 
The update target in n-step TD is a truncated return over n steps plus 
a bootstrapped value at step t+n 
n-step methods are more flexible especially when immediate updates 
are too myopic or delayed returns are too noisy 
41 


## Page 42

Learning Tradeoffs 
One-step TD methods require a single-step transition before 
learning; Monte Carlo methods wait for episode completion 
n-step methods wait for n steps before updating, which allows a 
balance between sample variance and bias 
With a large n, the method relies more on actual rewards; this 
reduces bias but increases variance 
With a small n, the method relies more on existing estimates, which 
increases bias but reduces variance 
42 


## Page 43

Returns 
Monte Carlo – complete return 
 
 
TD(0) – one-step return 
 
 
Two-step return 
 
 
n-step return 
43 


## Page 45

Example 
45 


## Page 46

Computing Returns 
46 


## Page 47

Computing Values 
47 


## Page 48

Computing Values 
48 


## Page 49

Computing Values 


## Page 50

Main Ideas 
Updates only start at t = 2. At that time the agent has observed 
three rewards R1, R2, R3, enough to form the first full 3-step 
return from A 
Updates continue after the episode terminates. No new actions 
occur after t = 3, yet updates at t = 4 and t = 5 adjust V(C) and 
V(D) so every reward R1, …, R4 influences some n-step return 
50 


## Page 51

Example: Random Walk 
51 


## Page 52

n-step Control: n-step Sarsa  
n-step Sarsa generalizes the one-step Sarsa algorithm by updating 
action-value estimates from n-step returns 
 
 
 
Updates are made once the necessary n-step transition has been 
observed, with bootstrapping handled as in value prediction 
 
 
 
 
52 


## Page 53

n-step Sarsa 
n-step Sarsa is also on-policy 
It selects At ∼ π(⋅ ∣ St) and updates toward returns aligned with π 
If the episode terminates before t+n, the return target is truncated 
without bootstrapping from Q(St+n, At+n) 
During each episode, updates begin only after observing enough 
transitions to construct full n-step returns 
After the episode ends, the agent keeps updating for a few more steps 
to ensure that all rewards are considered 
53 


## Page 55

Speedup of Policy Learning 
55 


## Page 56

56 


## Page 57

n-step Q-Learning 
It uses an off-policy target by applying the greedy action-value 
backup, that is, max over a of Q(s, a), instead of following the action 
actually taken 
The max operator is applied only once, at the final state sτ+n, and not 
at any earlier steps within the return 
Although the behavior policy generates the entire trajectory, 
including actions after τ, the Q-learning update at τ depends only on 
the observed rewards and on the greedy estimate at the final state, 
not on the specific actions taken 
The target is more biased than in SARSA, because it assumes the 
agent behaves greedily after τ+n, but it has lower variance 
57 


## Page 58

Program 
4 WindyGridWorld_nStep.py (n-step Sarsa) 
58 


## Page 59

Choosing n 
Empirical results suggest intermediate n values often give the best 
tradeoff between speed and accuracy 
The optimal n can depend on episode length, reward sparsity, and 
variance in transitions 
n-step methods provide flexibility to tune learning to the temporal 
structure of the environment 
 
59 


## Page 60

Conclusions 
Temporal-difference learning updates predictions from successive 
estimates; it solves temporal credit assignment by learning from 
incomplete episodes 
TD methods combine sampling with bootstrapping, which enables 
model-free prediction and often faster convergence 
Sarsa learns on-policy value estimates, Q-learning learns off-policy 
greedy values, and Expected Sarsa reduces variance through 
expectation 
Sarsa favors safer exploratory policies, while Q-learning may allow 
risky optimal policies 
n-step methods interpolate between TD(0) and Monte Carlo and thus 
trade bias and variance via truncated returns 
60 


## Page 61

Main References 
Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: 
An Introduction. 2nd edition. MIT Press, Cambridge, MA. 
http://incompleteideas.net/book/the-book-2nd.html 
Bennett, M. (2023). A Brief History of Intelligence. Mariner 
Books 
61 
